"""Learning again every morning from what people decided on the screen.

At HOUR o'clock (or a minute after the service starts, if the PC was switched on later),
the database is copied (store.backup) and, if anyone has included or skipped a page since the last time, both
models are trained again (learn.py --again) in a process of their own at low priority,
into model/candidate/. They replace the models in use only if, tested on manuals they
never saw, the page model misses no more spares pages than the one in use or than
learn.KEEP allows (within SLACK), and the picture model sends no more of the spares pages
it is asked about to "unlikely" than learn.LOOKS_KEEP allows (within PICTURE_SLACK);
otherwise they wait in model/candidate/ and the models in use stay. Nothing shows on the
screen. Each morning's outcome is a line in data/logs/retrain.log, what learn.py printed is
in data/logs/retrain-last.txt, and the last outcome is in /api/health.

    FINDER_RETRAIN_HOUR    the hour to learn again at (default 9: the office PC is on from
                           08:30 to 18:00, never at night; -1: never)
    python -m finder.retrain    learn again now, with the same safeguard (retrain.bat)
    python -m finder.retrain "sheet.xlsx" "vessel folder" [more pairs]
                                a finished vessel's spreadsheet as well, with the same safeguard
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from finder import learn, store

HOUR = int(os.environ.get("FINDER_RETRAIN_HOUR", 9))
SLACK = 0.002   # how much more often the new page model may miss a spares page, as a share
PICTURE_SLACK = 0.01   # the picture model's share is set on some manuals and tested on others: it wanders
HERE = Path(__file__).parent.parent


def candidate() -> Path:
    return learn.MODEL.parent / "candidate"


def state_file() -> Path:
    return store.DATA / "retrain.json"


def last() -> dict:
    """{"at", "outcome", "decisions"} of the last retraining, or {}: also when the file cannot
    be read (on 2026-11-03 it was found empty), which only means learning again tonight."""
    try:
        return json.loads(state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def remember(state: dict) -> None:
    """Keep `state` as the last retraining: written whole beside it, then swapped in."""
    new = state_file().with_name(state_file().name + ".new")
    new.write_text(json.dumps(state), encoding="utf-8")
    os.replace(new, state_file())


def decisions() -> str:
    """A fingerprint of every page people have decided: it changes with every Include or Skip.
    No database (learning a vessel from a folder of its own): nobody has decided anything."""
    marks = []
    if (store.DATA / "finder.db").exists():
        with store.db() as con:
            marks = con.execute("SELECT f.sha, p.page, p.label FROM pages p JOIN files f ON f.id = p.file_id "
                                "WHERE p.label IS NOT NULL AND f.status = 'done' AND f.sha IS NOT NULL "
                                "ORDER BY 1, 2, 3").fetchall()
    return hashlib.sha256(json.dumps([list(m) for m in marks]).encode()).hexdigest()


def _learn(pairs=()) -> int:
    """learn.py into model/candidate/, in its own process: again, or with the spreadsheet and
    folder `pairs` as well; its exit code."""
    folder = candidate()
    folder.mkdir(parents=True, exist_ok=True)
    if learn.training_file().exists():   # learning again starts from everything learned before
        shutil.copy(learn.training_file(), folder / "training.joblib")
    logs = store.DATA / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / "retrain-last.txt", "w", encoding="utf-8") as out:
        return subprocess.run([sys.executable, "-m", "finder.learn", *(pairs or ["--again"])], cwd=HERE, stdout=out,
                              stderr=subprocess.STDOUT, env={**os.environ, "FINDER_MODEL": str(folder / "pages.joblib"),
                                                             "PYTHONIOENCODING": "utf-8"},
                              creationflags=0x4000 if os.name == "nt" else 0).returncode   # below normal priority


def _verdict(new: dict, old: dict | None) -> str:
    """Why the new models may not replace the old ones, or "" if they may."""
    if old is None:
        return ""
    missed = lambda report: report["tested on manuals left out"]["spares pages missed"]
    allowed = max(missed(old), 1 - learn.KEEP) + SLACK   # no more than now, or than the share agreed (KEEP)
    if missed(new) > allowed:
        return f"the page model would miss {missed(new):.1%} of spares pages, against {missed(old):.1%} now"
    picture = new.get("picture model")
    if old.get("picture model") and not picture:
        return "no picture model could be trained"
    tested = picture["tested on manuals left out"] if picture else {}
    for sent, which in ((tested.get("spares check pages sent to unlikely"), "the spares pages it is asked about"),
                        (tested.get("check pages with lines", {}).get("spares sent to unlikely"),
                         "the spares pages with lines it is asked about")):
        if sent is not None and sent > 1 - learn.LOOKS_KEEP + PICTURE_SLACK:
            return (f"the picture model would send {sent:.1%} of {which} to 'probably not spares', "
                    f"more than the {1 - learn.LOOKS_KEEP:.0%} agreed")
    return ""


def _promote() -> None:
    """The candidate's files become the models in use; the ones they replace are kept as
    model/pages.previous.*. Whole files only, and never a moment without a model: the
    reader may load it at any time."""
    live = learn.MODEL.parent
    for name in ("pages.joblib", "pages.json"):
        if (live / name).exists():
            shutil.copy2(live / name, live / name.replace("pages", "pages.previous"))
    for name in ("training.joblib", "pages.json", "pages.joblib"):   # the model itself last
        shutil.copy(candidate() / name, live / (name + ".new"))
        os.replace(live / (name + ".new"), live / name)


def train(pairs=()) -> str:
    """Learn again, keep the new models only if they are no worse; what happened."""
    if _learn(pairs):
        return "learning failed: see data/logs/retrain-last.txt"
    new = json.loads((candidate() / "pages.json").read_text(encoding="utf-8"))
    report = learn.MODEL.with_suffix(".json")
    why = _verdict(new, json.loads(report.read_text(encoding="utf-8")) if report.exists() else None)
    if why:
        return f"kept the models in use: {why}. The new ones wait in model/candidate."
    _promote()
    tested = new["tested on manuals left out"]
    return (f"now using models learned from {new['pages']:,} pages: they miss {tested['spares pages missed']:.1%} of "
            f"spares pages and send {tested['other pages sent to check']:.1%} of other pages to a person")


def daily(force: bool = False, pairs=()) -> str:
    """Learn again if anyone has decided a page since last time (or if `force`); what happened."""
    now = decisions()
    outcome = train(pairs) if force or now != last().get("decisions") else "nothing new decided"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    remember({"at": stamp, "outcome": outcome, "decisions": now})
    (store.DATA / "logs").mkdir(parents=True, exist_ok=True)
    with open(store.DATA / "logs" / "retrain.log", "a", encoding="utf-8") as log:
        log.write(f"{stamp} {outcome}\n")
    return outcome


def wait(now: datetime) -> float:
    """Seconds until the morning's run: a minute, if HOUR has passed and today has had no
    run yet (the PC was switched on late); else until HOUR today or tomorrow."""
    at = now.replace(hour=HOUR, minute=0, second=0, microsecond=0)
    if now >= at:
        if last().get("at", "")[:10] == now.strftime("%Y-%m-%d"):
            at += timedelta(days=1)
        else:
            return 60
    return (at - now).total_seconds()


def start() -> None:
    """Every morning: a copy of the database, then learning again."""
    if HOUR < 0:
        return

    def run():
        while True:
            pause = wait(datetime.now())
            if pause > 300:
                # a few minutes at a time, checked against the clock: Windows' fast start-up hibernates
                # the service overnight, and one long sleep does not count those hours (on 2026-11-03 a
                # run due at 01:00 started at 10:35)
                time.sleep(300)
                continue
            time.sleep(pause)
            try:
                store.backup()
            except Exception as exc:
                print(f"backup failed: {exc}", flush=True)
            try:
                daily()
            except Exception as exc:   # tried again in an hour
                print(f"retraining failed: {exc}", flush=True)
                time.sleep(3600)
    threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    print(daily(force=True, pairs=[str(Path(a).resolve()) for a in sys.argv[1:]]))   # learn.py runs in the tool's folder
