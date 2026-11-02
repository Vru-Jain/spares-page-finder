"""Learning again every night from what people decided on the screen.

At HOUR o'clock, if anyone has included or skipped a page since the last time, both
models are trained again (learn.py --again) in a process of their own at low priority,
into model/candidate/. They replace the models in use only if, tested on manuals they
never saw, the page model misses no more spares pages (within SLACK) and the picture
model sends no known spares page to "unlikely"; otherwise they wait in model/candidate/
and the models in use stay. Nothing shows on the screen. Each night's outcome is a line
in data/logs/retrain.log, what learn.py printed is in data/logs/retrain-last.txt, and the
last outcome is in /api/health.

    FINDER_RETRAIN_HOUR    the hour to learn again at (default 1, before the 02:00
                           backup; -1: never)
    python -m finder.retrain    learn again now, with the same safeguard (retrain.bat)
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

HOUR = int(os.environ.get("FINDER_RETRAIN_HOUR", 1))
SLACK = 0.002   # how much more often the new page model may miss a spares page, as a share
HERE = Path(__file__).parent.parent


def candidate() -> Path:
    return learn.MODEL.parent / "candidate"


def state_file() -> Path:
    return store.DATA / "retrain.json"


def last() -> dict:
    """{"at", "outcome", "decisions"} of the last retraining, or {}."""
    return json.loads(state_file().read_text(encoding="utf-8")) if state_file().exists() else {}


def decisions() -> str:
    """A fingerprint of every page people have decided: it changes with every Include or Skip."""
    with store.db() as con:
        marks = con.execute("SELECT f.sha, p.page, p.label FROM pages p JOIN files f ON f.id = p.file_id "
                            "WHERE p.label IS NOT NULL AND f.status = 'done' AND f.sha IS NOT NULL "
                            "ORDER BY 1, 2, 3").fetchall()
    return hashlib.sha256(json.dumps([list(m) for m in marks]).encode()).hexdigest()


def _learn() -> int:
    """learn.py --again into model/candidate/, in its own process; its exit code."""
    folder = candidate()
    folder.mkdir(parents=True, exist_ok=True)
    if learn.training_file().exists():   # learning again starts from everything learned before
        shutil.copy(learn.training_file(), folder / "training.joblib")
    logs = store.DATA / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / "retrain-last.txt", "w", encoding="utf-8") as out:
        return subprocess.run([sys.executable, "-m", "finder.learn", "--again"], cwd=HERE, stdout=out,
                              stderr=subprocess.STDOUT, env={**os.environ, "FINDER_MODEL": str(folder / "pages.joblib"),
                                                             "PYTHONIOENCODING": "utf-8"},
                              creationflags=0x4000 if os.name == "nt" else 0).returncode   # below normal priority


def _verdict(new: dict, old: dict | None) -> str:
    """Why the new models may not replace the old ones, or "" if they may."""
    if old is None:
        return ""
    missed = lambda report: report["tested on manuals left out"]["spares pages missed"]
    if missed(new) > missed(old) + SLACK:
        return f"the page model would miss {missed(new):.1%} of spares pages, against {missed(old):.1%} now"
    picture = new.get("picture model")
    if old.get("picture model") and not picture:
        return "no picture model could be trained"
    if picture and picture["tested on manuals left out"]["spares pages sent to unlikely"]:
        return (f"the picture model would send {picture['tested on manuals left out']['spares pages sent to unlikely']}"
                " known spares pages to 'probably not spares'")
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


def train() -> str:
    """Learn again, keep the new models only if they are no worse; what happened."""
    if _learn():
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


def nightly(force: bool = False) -> str:
    """Learn again if anyone has decided a page since last time (or if `force`); what happened."""
    now = decisions()
    outcome = train() if force or now != last().get("decisions") else "nothing new decided"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    state_file().write_text(json.dumps({"at": stamp, "outcome": outcome, "decisions": now}), encoding="utf-8")
    (store.DATA / "logs").mkdir(parents=True, exist_ok=True)
    with open(store.DATA / "logs" / "retrain.log", "a", encoding="utf-8") as log:
        log.write(f"{stamp} {outcome}\n")
    return outcome


def start() -> None:
    """Learn again every night at HOUR."""
    if HOUR < 0:
        return

    def run():
        while True:
            now = datetime.now()
            at = now.replace(hour=HOUR, minute=0, second=0, microsecond=0)
            time.sleep(((at if at > now else at + timedelta(days=1)) - now).total_seconds())
            try:
                nightly()
            except Exception as exc:   # tried again the next night
                print(f"retraining failed: {exc}", flush=True)
    threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    print(nightly(force=True))
