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
the manual's name in its "Internal Remark" column and the page in "Ref Page"
("17/56/93" means several pages). Those pages are spares pages; every other page of
a listed manual is not. Manuals the spreadsheet does not list are left out, since
nobody said what is in them. Include and Skip decisions made on the screen are added
on top and outrank the spreadsheets.

The model sees what text.py's page_reading gives it: the words of a page's text layer,
and which parts-table columns Parsel's table passes found there. It judges only pages
whose text describes them (text.USABLE_TEXT); the rest are judged by their picture. It
is tested on whole manuals it was not trained on, and two cut-offs are set from that test:
  unsure_from  KEEP of known spares pages score at least this; below it, not spares
  spares_from  of pages scoring at least this, SURE really are spares pages; these
               are marked spares without anyone checking
Between the two, a page is also read as a picture, and a person checks it.
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
from sklearn.pipeline import Pipeline

from finder import store, tables, text

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


# ---- what it learns from ---------------------------------------------------

def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def spreadsheet_pages(xlsx) -> dict[str, set[int]]:
    """{manual named in the spreadsheet: its spares pages}."""
    import openpyxl
    book = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        for sheet in book.worksheets:
            rows = sheet.iter_rows(values_only=True)
            for header in rows:
                names = [_norm(c) for c in header]
                if "internalremark" in names and "refpage" in names:
                    break
            else:
                continue
            manual_at, page_at = names.index("internalremark"), names.index("refpage")
            pages = defaultdict(set)
            for row in rows:
                if len(row) <= max(manual_at, page_at) or not row[manual_at] or row[page_at] is None:
                    continue
                for part in re.split(r"[/,;&]+", str(row[page_at])):
                    part = part.strip().removesuffix(".0")
                    if part.isdigit():
                        pages[str(row[manual_at]).strip()].add(int(part))
            return pages
    finally:
        book.close()
    raise ValueError(f"{xlsx}: no sheet with 'Internal Remark' and 'Ref Page' columns")


def match(manuals, folder) -> dict[str, Path]:
    """{manual name: its PDF}. The name is the file name without .pdf, compared without
    spaces and punctuation; a name cut short in the spreadsheet matches if only one
    file starts with it."""
    pdfs = {_norm(p.stem): p for p in Path(folder).rglob("*") if p.suffix.lower() == ".pdf"}
    found = {}
    for manual in manuals:
        key = _norm(manual)
        hits = [pdfs[key]] if key in pdfs else [p for n, p in pdfs.items() if n.startswith(key) or key.startswith(n)]
        if len(hits) == 1:
            found[manual] = hits[0]
    return found


def _sha(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _readings(path, numbers) -> dict[int, dict]:
    got = text.read_pages(str(path), list(numbers), lambda *_: None)
    return {n: r for n, r in got.items() if r is not None}   # a page that cannot be read teaches nothing


def from_spreadsheet(xlsx, folder) -> tuple[dict, list[str]]:
    """({(manual's sha256, page): (label, reading)}, manuals that matched no PDF)."""
    listed = spreadsheet_pages(xlsx)
    found = match(listed, folder)
    rows = {}
    for manual, path in sorted(found.items()):
        print(f"  reading {manual[:70]}", flush=True)
        sha = _sha(path)
        with pymupdf.open(path) as doc:
            count = doc.page_count
        for number, reading in _readings(path, range(1, count + 1)).items():
            rows[(sha, number)] = (int(number in listed[manual]), reading)
    return rows, sorted(set(listed) - set(found))


def from_corrections() -> dict:
    """{(manual's sha256, page): (label, reading)} from what people decided on the screen:
    Include, Skip and pasted checked page lists."""
    if not (store.DATA / "finder.db").exists():
        return {}
    with store.db() as con:
        picked = con.execute(
            "SELECT f.id, f.path, f.sha, p.page, p.label FROM pages p JOIN files f ON f.id = p.file_id "
            "WHERE p.label IS NOT NULL AND f.status = 'done' AND f.sha IS NOT NULL ORDER BY f.id").fetchall()
        kept = {(r["file_id"], r["page"]): json.loads(r["reading"])
                for r in con.execute("SELECT file_id, page, reading FROM readings")}
    by_file = defaultdict(dict)
    for fid, path, sha, page, label in picked:   # later decisions on the same page win
        by_file[(fid, path, sha)][page] = label
    rows = {}
    for (fid, path, sha), labels in by_file.items():
        if path and store.pdf(path).exists():
            got = _readings(store.pdf(path), labels)
        else:   # the manual was removed (archive.py): what its pages said was kept
            got = {n: kept[(fid, n)] for n in labels if (fid, n) in kept}
        for number, reading in got.items():
            rows[(sha, number)] = (labels[number], reading)
    return rows


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


def training_file() -> Path:
    """{"rows": {(manual's sha256, page): (answer, reading)}, "sources": spreadsheets used}."""
    return MODEL.with_name("training.joblib")


def main(args: list[str]) -> None:
    learned = joblib.load(training_file()) if training_file().exists() else {"rows": {}, "sources": []}
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
        found, missing = from_spreadsheet(xlsx, folder)
        rows.update(found)
        learned["sources"] = sorted(set(learned["sources"]) | {Path(xlsx).name})
        if missing:
            print(f"  no PDF found for: {missing}")
    corrections = from_corrections()
    rows.update(corrections)   # people's latest decisions outrank everything else
    if not rows:
        sys.exit("Nothing to learn from yet: give a merge spreadsheet and its manuals folder.")
    print(f"learning from {len(rows):,} pages, {len(corrections)} of them decided on the screen", flush=True)
    model = fit(rows)
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
