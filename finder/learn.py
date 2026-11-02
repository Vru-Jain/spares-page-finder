"""Teach the tool which pages are spares pages.

    python -m finder.learn "MERGE ALL FILES.xlsx" "manuals folder" [more pairs]
                                        add a vessel's finished spreadsheet, and learn again
    python -m finder.learn --again      learn again: everything learned before, plus what
                                        people have decided since (retrain.bat does this)

Every page it has learned from is kept, with its answer and what its text layer said,
in model/training.joblib. So learning again needs nothing outside the tool's folder:
not the spreadsheets, not the manuals they came from, not even the database. The model
it replaces is kept as model/pages.previous.joblib: to go back, copy it over
model/pages.joblib. The reader picks up a new model before its next manual.

A finished merge spreadsheet lists every spare part typed from a vessel's manuals:
the page in "Ref Page" ("17/56/93" means several pages), and the manual's name either
in an "Internal Remark" column or, in older files ("with group heading"), in a one-cell
heading row above each manual's parts. A tracking sheet (TORM's) lists each file of
a vessel with its "Marked Pages" ("6,8,16-36"). Those pages are spares pages; every other page
of a listed manual is not. Manuals the spreadsheet does not list are left out, since
nobody said what is in them, and so is a manual the spreadsheet lists pages beyond the
end of (the name matched the wrong file). Include and Skip decisions made on the screen
are added on top and outrank the spreadsheets.

The model sees what text.py's page_reading gives it: the words of a page's text layer,
and which parts-table columns Parsel's table passes found there. It judges only pages
whose text describes them (text.USABLE_TEXT); the rest are judged by their picture. It
is tested on whole manuals it was not trained on, and two cut-offs are set from that test:
  unsure_from  KEEP of known spares pages score at least this; below it, not spares
  spares_from  of pages scoring at least this, SURE really are spares pages; these
               are marked spares without anyone checking
Between the two, a page is also read as a picture, and a person checks it.

A second, smaller model learns from the pages judged by their picture: what the picture
reader said, and how the page looks as a whole (looks.py). It only ever moves a page the
picture reader left to a person into "unlikely". Its cut-off is the lowest score any
known spares page got from a model that never saw that page's manual. On H1868 and
BERGE DAISEN, tested that way, it put 2 of 490 spares pages under "unlikely" (both in
H1868, when it had learned from BERGE DAISEN alone) and took 23% of the other pages off
the "check" pile, so a person still looks "unlikely" pages over. The numbers it learns
from are kept in model/training.joblib too.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pymupdf
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from finder import looks, store, tables, text
from finder.tables import CHECK

MODEL = Path(os.environ.get("FINDER_MODEL", Path(__file__).parent.parent / "model" / "pages.joblib"))
KEEP = 0.99   # share of spares pages that must not be ruled out: "only pages we are sure of may be left out"
SURE = 0.90   # how often a page marked spares unchecked must really be one


# ---- what the model sees ---------------------------------------------------

def frame(readings: list[dict]) -> pd.DataFrame:
    """One row per page reading, in the columns the model is trained on."""
    return pd.DataFrame([{"text": r["text"], **{c: float(c in r["signs"]) for c in tables.COLUMNS},
                          "table score": min(max((r["confidence"] or 0) / 50, 0), 1)}
                         for r in readings])


def pipeline() -> Pipeline:
    """Words and word pieces of the page (pieces cope with scanner misreadings and
    Japanese), the table columns found, then a logistic regression."""
    return Pipeline([
        ("features", ColumnTransformer([
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                                      max_features=200_000), "text"),
            ("pieces", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                       sublinear_tf=True, max_features=300_000), "text"),
            ("table", "passthrough", tables.COLUMNS + ["table score"]),
        ])),
        ("model", LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)),
    ])


_LOADED: dict = {}


def load() -> dict | None:
    """The trained model, or None if none has been trained. Reloaded when the file
    changes, so a retrained model is used from the next manual on. Only this tool's
    own model file is ever loaded: a joblib file can run code when it is opened."""
    if not MODEL.exists():
        return None
    stamp = (MODEL, MODEL.stat().st_mtime)
    if _LOADED.get("stamp") != stamp:
        _LOADED.update(stamp=stamp, model=joblib.load(MODEL))
    return _LOADED["model"]


def score(model: dict, readings: dict[int, dict]) -> dict[int, float]:
    """{page: chance it is a spares page}."""
    if not readings:
        return {}
    numbers = list(readings)
    chance = model["pipeline"].predict_proba(frame([readings[n] for n in numbers]))[:, 1]
    return dict(zip(numbers, chance.tolist()))


def looks_pipeline() -> Pipeline:
    """The picture model: a heavily held-back logistic regression on looks.features."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))


def looks_score(head: dict, vectors, verdicts) -> np.ndarray:
    """The picture model's chance that each page is a spares page."""
    return head["pipeline"].predict_proba(looks.features(np.stack(vectors), verdicts))[:, 1]


# ---- what it learns from ---------------------------------------------------

def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def spreadsheet_pages(xlsx) -> dict[str, set[int]]:
    """{manual named in the spreadsheet: its spares pages}. A tracking sheet (TORM's
    "Allocation Sheet" / "Manual List": "Filename" and "Marked Pages", one sheet per
    vessel) counts every such sheet; a file marked "NA" has no list and is left out."""
    import openpyxl
    from finder.outputs import parse_pages
    book = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    tracked = defaultdict(set)
    try:
        for sheet in book.worksheets:
            rows = sheet.iter_rows(values_only=True)
            for header in rows:
                names = [_norm(c) for c in header]
                if "refpage" in names or {"filename", "markedpages"} <= set(names):
                    break
            else:
                continue
            if {"filename", "markedpages"} <= set(names):
                file_at, marked_at = names.index("filename"), names.index("markedpages")
                for row in rows:
                    if len(row) > max(file_at, marked_at) and row[file_at] and row[marked_at]:
                        tracked[str(row[file_at]).strip()].update(parse_pages(row[marked_at]))
                continue
            page_at = names.index("refpage")
            manual_at = names.index("internalremark") if "internalremark" in names else None
            pages, heading = defaultdict(set), None
            for row in rows:
                if manual_at is None:   # the older layout: a heading row names the manual below it
                    filled = [c for c in row if c is not None and str(c).strip()]
                    if len(filled) == 1 and isinstance(filled[0], str):
                        heading = filled[0].strip()
                        continue
                    manual = heading
                else:
                    manual = row[manual_at] if len(row) > manual_at else None
                if not manual or len(row) <= page_at or row[page_at] is None:
                    continue
                for part in re.split(r"[/,;&]+", str(row[page_at])):
                    part = part.strip().removesuffix(".0")
                    if part.isdigit():
                        pages[str(manual).strip()].add(int(part))
            if pages:
                return pages
    finally:
        book.close()
    if any(tracked.values()):
        return {manual: pages for manual, pages in tracked.items() if pages}
    raise ValueError(f"{xlsx}: no manuals found. It needs a 'Ref Page' column, and the manual named in an "
                     "'Internal Remark' column or in heading rows (use the file 'with group heading'), "
                     "or 'Filename' and 'Marked Pages' columns")


def _name(manual) -> str:
    """A manual's name as the spreadsheet gives it, ready to compare with file names: the
    last part when it is a path, without ".pdf"."""
    return re.sub(r"\.pdf$", "", re.split(r"[\\/]", str(manual).strip())[-1].strip(), flags=re.I)


def _words(text) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def match(manuals, folder) -> dict[str, Path]:
    """{manual name: its PDF}. The name is the file name without .pdf, compared without
    spaces and punctuation; a name cut short in the spreadsheet matches if only one
    file starts with it; a name typed a little differently matches the file name whose
    words are closest, if it is clearly closer than any other. Several files of one
    name match only if they are copies of each other (same size): Nanoq has two
    different "1A Operators manual, engine.pdf"."""
    from rapidfuzz import fuzz, process
    pdfs, by_words = defaultdict(list), defaultdict(list)
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() == ".pdf":
            pdfs[_norm(p.stem)].append(p)
            by_words[_words(p.stem)].append(p)
    one = lambda paths: paths[0] if len({p.stat().st_size for p in paths}) == 1 else None
    found = {}
    for manual in manuals:
        key = _norm(_name(manual))
        if not key:
            continue
        names = [key] if key in pdfs else [n for n in pdfs if n.startswith(key) or key.startswith(n)]
        if len(names) == 1:
            hit = one(pdfs[names[0]])
        elif not names and by_words:
            best = process.extract(_words(_name(manual)), list(by_words), scorer=fuzz.token_set_ratio, limit=2)
            clear = best[0][1] >= 88 and (len(best) < 2 or best[0][1] - best[1][1] >= 6)
            hit = one(by_words[best[0][0]]) if clear else None
        else:
            hit = None
        if hit:
            found[manual] = hit
    return found


def _sha(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _readings(path, numbers) -> dict[int, dict]:
    got = text.read_pages(str(path), list(numbers), lambda *_: None)
    return {n: r for n, r in got.items() if r is not None}   # a page that cannot be read teaches nothing


def from_spreadsheet(xlsx, folder) -> tuple[dict, dict, list[str]]:
    """({(manual's sha256, page): (label, reading)}, {(sha256, page): (label, picture
    verdict, numbers)} for manuals the tool has read, manuals that matched no PDF)."""
    listed = spreadsheet_pages(xlsx)
    found = match(listed, folder)
    rows, seen = {}, {}
    for manual, path in sorted(found.items()):
        with pymupdf.open(path) as doc:
            count = doc.page_count
        if max(listed[manual]) > count:
            print(f"  left out {manual[:70]}: it lists page {max(listed[manual])}, {path.name} has {count}")
            continue
        print(f"  reading {manual[:70]}", flush=True)
        sha = _sha(path)
        for number, reading in _readings(path, range(1, count + 1)).items():
            rows[(sha, number)] = (int(number in listed[manual]), reading)
        for number, (verdict, vector) in looks.load(sha).items():
            seen[(sha, number)] = (int(number in listed[manual]), verdict, vector)
    return rows, seen, sorted(set(listed) - set(found))


def from_corrections() -> tuple[dict, dict]:
    """({(manual's sha256, page): (label, reading)}, {(sha256, page): (label, picture
    verdict, numbers)}) from what people decided on the screen: Include, Skip and Leave
    all out."""
    if not (store.DATA / "finder.db").exists():
        return {}, {}
    with store.db() as con:
        picked = con.execute(
            "SELECT f.id, f.path, f.sha, p.page, p.label FROM pages p JOIN files f ON f.id = p.file_id "
            "WHERE p.label IS NOT NULL AND f.status = 'done' AND f.sha IS NOT NULL ORDER BY f.id").fetchall()
        kept = {(r["file_id"], r["page"]): json.loads(r["reading"])
                for r in con.execute("SELECT file_id, page, reading FROM readings")}
    by_file = defaultdict(dict)
    for fid, path, sha, page, label in picked:   # later decisions on the same page win
        by_file[(fid, path, sha)][page] = label
    rows, seen = {}, {}
    for (fid, path, sha), labels in by_file.items():
        if path and store.pdf(path).exists():
            got = _readings(store.pdf(path), labels)
        else:   # the manual was removed (archive.py): what its pages said was kept
            got = {n: kept[(fid, n)] for n in labels if (fid, n) in kept}
        for number, reading in got.items():
            rows[(sha, number)] = (labels[number], reading)
        for number, (verdict, vector) in looks.load(sha).items():
            if number in labels:
                seen[(sha, number)] = (labels[number], verdict, vector)
    return rows, seen


# ---- learning --------------------------------------------------------------

def _sure_from(chance, label, sure) -> float:
    """The lowest score above which at least `sure` of pages are spares pages."""
    order = np.argsort(-chance)
    hits = np.cumsum(label[order])
    precision = hits / np.arange(1, len(order) + 1)
    good = np.nonzero(precision >= sure)[0]
    return float(chance[order][good.max()]) if len(good) else 1.01


def fit(rows: dict, keep=KEEP, sure=SURE) -> dict:
    """Test on manuals left out, set the two cut-offs from that test, then train the
    model that is kept on everything."""
    rows = {k: v for k, v in rows.items() if len(v[1]["text"].strip()) >= text.USABLE_TEXT}
    keys = list(rows)
    data = frame([rows[k][1] for k in keys])
    label = np.array([rows[k][0] for k in keys])
    groups = [k[0] for k in keys]
    if len(set(groups)) < 2 or label.sum() == 0:
        raise ValueError("Need at least two manuals and some spares pages to learn from")
    chance = np.zeros(len(keys))
    for train, test in GroupKFold(n_splits=min(5, len(set(groups)))).split(data, label, groups):
        chance[test] = pipeline().fit(data.iloc[train], label[train]).predict_proba(data.iloc[test])[:, 1]
    unsure_from = float(np.quantile(chance[label == 1], 1 - keep))
    spares_from = max(_sure_from(chance, label, sure), unsure_from)
    spares, other = chance[label == 1], chance[label == 0]
    share = lambda part, low, high=9.9: round(float(((part >= low) & (part < high)).mean()), 4)
    report = {
        "trained": datetime.datetime.now().isoformat(timespec="minutes"),
        "pages": len(keys), "spares pages": int(label.sum()), "manuals": len(set(groups)),
        "unsure_from": round(unsure_from, 4), "spares_from": round(spares_from, 4),
        "tested on manuals left out": {
            "spares pages marked spares": share(spares, spares_from),
            "spares pages sent to check": share(spares, unsure_from, spares_from),
            "spares pages missed": share(spares, -1, unsure_from),
            "other pages marked spares": share(other, spares_from),
            "other pages sent to check": share(other, unsure_from, spares_from),
        },
    }
    return {"pipeline": pipeline().fit(data, label), "unsure_from": unsure_from,
            "spares_from": spares_from, "report": report}


def _held_out(data, label, groups) -> np.ndarray:
    """Each page's picture-model score from a model trained without its manual."""
    chance = np.zeros(len(label))   # 0 where no model could be trained: a cut-off of 0 moves nothing
    for train, test in GroupKFold(n_splits=min(5, len(set(groups)))).split(data, label, groups):
        if len(set(label[train])) == 2:
            chance[test] = looks_pipeline().fit(data[train], label[train]).predict_proba(data[test])[:, 1]
    return chance


def fit_looks(rows: dict) -> dict | None:
    """The picture model, from {(manual's sha256, page): (label, picture verdict, numbers)};
    None until there are two manuals with spares pages and other pages to learn from.
    Tested the way it is used: each manual left out in turn, with its cut-off set from
    the other manuals alone."""
    keys = list(rows)
    label = np.array([rows[k][0] for k in keys])
    groups = np.array([k[0] for k in keys])
    if len(set(groups[label == 1])) < 2 or label.all():
        return None
    verdict = np.array([rows[k][1] for k in keys])
    data = looks.features(np.stack([rows[k][2] for k in keys]), verdict)
    moved = np.zeros(len(keys), bool)   # pages the test sent to "unlikely"
    for train, test in GroupKFold(n_splits=min(5, len(set(groups)))).split(data, label, groups):
        if len(set(groups[train][label[train] == 1])) < 2 or label[train].all():
            continue
        cut = _held_out(data[train], label[train], groups[train])[label[train] == 1].min()
        chance = looks_pipeline().fit(data[train], label[train]).predict_proba(data[test])[:, 1]
        moved[test] = (verdict[test] == CHECK) & (chance < cut)
    cut = float(_held_out(data, label, groups)[label == 1].min())
    asked = (verdict == CHECK) & (label == 0)
    report = {
        "pages": len(keys), "spares pages": int(label.sum()), "manuals": len(set(groups)),
        "cut": float(f"{cut:.3g}"),   # often tiny: most pages that are not spares score far below any that are
        "tested on manuals left out": {
            "spares pages sent to unlikely": int((moved & (label == 1)).sum()),
            "other check pages sent to unlikely": round(float(moved[asked].mean()) if asked.any() else 0.0, 4),
        },
    }
    return {"pipeline": looks_pipeline().fit(data, label), "cut": cut, "report": report}


def training_file() -> Path:
    """{"rows": {(manual's sha256, page): (answer, reading)}, "looks": {(sha256, page):
    (answer, picture verdict, numbers)}, "sources": spreadsheets used}."""
    return MODEL.with_name("training.joblib")


def main(args: list[str]) -> None:
    learned = joblib.load(training_file()) if training_file().exists() else {"rows": {}, "sources": []}
    learned.setdefault("looks", {})
    if args == ["--again"]:
        pairs = []
        report = MODEL.with_suffix(".json")
        if not learned["rows"] and report.exists():   # a model from before training.joblib: its spreadsheets once more
            old = json.loads(report.read_text(encoding="utf-8")).get("sources", [])
            pairs = list(zip(old[::2], old[1::2]))
    elif args and not len(args) % 2:
        pairs = list(zip(args[::2], args[1::2]))
    else:
        sys.exit(__doc__)
    rows = learned["rows"]
    for xlsx, folder in pairs:
        print(f"{xlsx}:", flush=True)
        found, seen, missing = from_spreadsheet(xlsx, folder)
        rows.update(found)
        learned["looks"].update(seen)
        learned["sources"] = sorted(set(learned["sources"]) | {Path(xlsx).name})
        if missing:
            print(f"  no PDF found for: {missing}")
    corrections, seen = from_corrections()
    rows.update(corrections)   # people's latest decisions outrank everything else
    learned["looks"].update(seen)
    if not rows:
        sys.exit("Nothing to learn from yet: give a merge spreadsheet and its manuals folder.")
    print(f"learning from {len(rows):,} pages, {len(corrections)} of them decided on the screen; "
          f"the picture model from {len(learned['looks']):,}", flush=True)
    model = fit(rows)
    model["looks"] = fit_looks(learned["looks"])
    if model["looks"]:
        model["report"]["picture model"] = model["looks"]["report"]
    model["report"]["sources"] = learned["sources"]
    MODEL.parent.mkdir(exist_ok=True)
    for old in (MODEL, MODEL.with_suffix(".json")):   # the model it replaces is kept, to go back to
        if old.exists():
            old.replace(old.with_name(old.stem + ".previous" + old.suffix))
    joblib.dump(model, MODEL)
    joblib.dump(learned, training_file())
    MODEL.with_suffix(".json").write_text(json.dumps(model["report"], indent=2), encoding="utf-8")
    print(json.dumps(model["report"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
