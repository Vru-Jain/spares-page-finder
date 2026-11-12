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
  spares_from  of pages scoring at least this, SURE are pages the spreadsheets list; these
               are marked spares without anyone checking
Between the two, a page is also read as a picture, and a person checks it.

Two things learned on 2026-11-02, held out by manual on 61,581 pages:
  - Parts pages come in runs. A second step sees each page's score beside its neighbours'
    (neighbours), and decides with its own two cut-offs.
  - The spreadsheets skip pages whose parts were typed from another page, so some real
    parts pages are taught as "not spares". A "not spares" page that scores DOUBTFUL or
    more within NEAR pages of a listed spares page is left out of learning and of the
    cut-offs: all 8 looked at were parts lists.
Together they sent 37% of other pages to a person instead of 40%, and marked 16% of
spares pages by themselves instead of 2.7%.

A second, smaller model learns from the pages judged by their picture: what the picture
reader said, and how the page looks as a whole (looks.py). It only ever moves a page the
picture reader left to a person into "unlikely", which a person still looks over. Its
cut-off keeps LOOKS_KEEP of the spares pages in that pile, scored by models that never
saw their manuals. The numbers it learns from are kept in model/training.joblib too.
From 2026-11-12 it also reads what the picture reader read on such a page (its lines): a
lines model, words and word pieces as the page model reads text, and a small step that
weighs the two. Pages with lines get that pair's own cut-off (LINES_KEEP); pages without
(kept before then) the picture model's alone. Nine vessels, each scored by models that
never saw it (44,356 check scans): the picture model alone moved 41.7% of the other check
scans and 2.8% of the spares ones (11.1% on one vessel); the pair at LINES_KEEP 51.3% and
1.5% (4.8%). classify.py also leaves in "check" a page next to one marked spares, which
kept most of the rest: exploded drawings beside their parts lists.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from finder import looks, store, tables, text
from finder.tables import CHECK

MODEL = Path(os.environ.get("FINDER_MODEL", Path(__file__).parent.parent / "model" / "pages.joblib"))
KEEP = 0.98   # share of spares pages that must not be ruled out. 0.99 until 2026-11-02: keeping the
              # last 1% sent a person 37% of other pages instead of 27%, and about half of the
              # listed pages it kept for were not parts pages (whole page ranges in the spreadsheets)
SURE = 0.85   # how often a page marked spares unchecked must be a page the spreadsheets list. 0.90 until
              # 2026-11-06: the spreadsheets leave out parts lists that were not typed (repeats, makers'
              # standard sheets), and on two vessels never learned 22 of 24 marked pages they did not
              # list were parts pages. The aim is about 97% really spares pages: there that was a bar
              # near 0.95, marking 31% of text spares pages instead of 10%. 0.80 set the bar at 0.926,
              # which marked ramp and stanchion drawings too: about 90% right.
DOUBTFUL, NEAR = 0.5, 2   # a "not spares" page scoring this high this near a listed one is left out
LOOKS_KEEP = 0.98   # share of the spares pages the picture reader left to a person that the picture model
                    # must leave there too. Until 2026-11-03 none was ever moved: that took 2% of the scanned
                    # pages off a person's pile, 0.98 takes 40% (held out by manual, five vessels). The rest
                    # go to "probably not spares", which a person still looks over.
LINES_KEEP = 0.99   # the same for the picture model and the lines model together. What a page says is more
                    # a client's own than how it looks: a cut-off keeping 98% of spares pages held out by
                    # manual kept 96.8% on nine vessels never seen (10% lost on one); 0.99 keeps 98.5% there.
SECOND_STEP = 50   # manuals the second step needs to learn runs of parts pages from


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


def second_step() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)


def neighbours(chance: dict[int, float], readings: dict[int, dict], count: int) -> tuple[list[int], np.ndarray]:
    """(pages, what the second step sees of each): its score as log-odds, its neighbours'
    within 3 pages (missing where a neighbour's text was not read), the highest and mean of
    those, where it sits among the manual's `count` pages, and its table columns."""
    clipped = {n: min(max(c, 1e-6), 1 - 1e-6) for n, c in chance.items()}
    logit = {n: float(np.log(c / (1 - c))) for n, c in clipped.items()}
    numbers = sorted(chance)
    got = []
    for n in numbers:
        near = [logit.get(n + d, np.nan) for d in (-3, -2, -1, 1, 2, 3)]
        seen = [v for v in near if v == v]
        r = readings[n]
        got.append([logit[n], *near, max(seen) if seen else np.nan, float(np.mean(seen)) if seen else np.nan,
                    n / max(count, 1), *(float(c in r["signs"]) for c in tables.COLUMNS),
                    min(max((r["confidence"] or 0) / 50, 0), 1)])
    return numbers, np.array(got, dtype=float)


def judge(model: dict, readings: dict[int, dict], count: int) -> tuple[dict[int, float], float, float]:
    """({page: chance it is a spares page}, unsure_from, spares_from) for the text pages of a
    manual of `count` pages: the second step's, when the model has one."""
    chance = score(model, readings)
    step = model.get("second")
    if not step or not chance:
        return chance, model["unsure_from"], model["spares_from"]
    numbers, seen = neighbours(chance, readings, count)
    return (dict(zip(numbers, step["model"].predict_proba(seen)[:, 1].tolist())),
            step["unsure_from"], step["spares_from"])


def looks_pipeline() -> Pipeline:
    """The picture model: a heavily held-back logistic regression on looks.features."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))


def looks_score(head: dict, vectors, verdicts) -> np.ndarray:
    """The picture model's chance that each page is a spares page."""
    return head["pipeline"].predict_proba(looks.features(np.stack(vectors), verdicts))[:, 1]


def lines_pipeline() -> Pipeline:
    """The lines model: words and word pieces of what the picture reader read on a page."""
    return make_pipeline(FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=150_000)),
        ("pieces", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True,
                                   max_features=200_000)),
    ]), LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000))


def _logit(chance) -> np.ndarray:
    chance = np.clip(np.asarray(chance, float), 1e-6, 1 - 1e-6)
    return np.log(chance / (1 - chance))


def _pair(picture, said) -> np.ndarray:
    """What the step weighing the picture model and the lines model sees: both as log-odds."""
    return np.column_stack([_logit(picture), _logit(said)])


def looks_unlikely(head: dict, vectors, verdicts, lines=None) -> np.ndarray:
    """Which pages the picture model sends to "unlikely": by the picture model and the lines
    model together where the picture reader left the page to a person and read lines on it,
    by the picture model alone elsewhere (and with a model from before the lines model)."""
    picture = looks_score(head, vectors, verdicts)
    out = picture < head["cut"]
    both = head.get("lines")
    if both and lines is not None:
        have = np.array([bool(t) and v == CHECK for t, v in zip(lines, verdicts)], bool)
        if have.any():
            said = both["pipeline"].predict_proba(np.array(lines, dtype=object)[have])[:, 1]
            out[have] = both["step"].predict_proba(_pair(picture[have], said))[:, 1] < both["cut"]
    return out


# ---- what it learns from ---------------------------------------------------

def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def spreadsheet_pages(xlsx) -> dict[str, set[int]]:
    """{manual named in the spreadsheet: its spares pages}. A tracking sheet (TORM's
    "Allocation Sheet" / "Manual List": "Filename" and "Marked Pages", one sheet per
    vessel) counts every such sheet; a file marked "NA" has no list and is left out.
    A typing allocation sheet ("Path", "File", "DATA") names each manual by
    its folder and file. "No Data" means a typist looked and found no spares page: its
    pages are all taught as not spares (an empty set). Anything else in DATA that is not
    a page list ("we don't take this Spares", "ask ...") is a decision about the manual,
    not about its pages, and is left out."""
    import openpyxl
    from finder.outputs import parse_pages
    book = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    tracked = defaultdict(set)
    try:
        for sheet in book.worksheets:
            rows = sheet.iter_rows(values_only=True)
            for header in rows:
                names = [_norm(c) for c in header]
                if ("refpage" in names or {"filename", "markedpages"} <= set(names)
                        or {"path", "file", "data"} <= set(names)):
                    break
            else:
                continue
            if {"path", "file", "data"} <= set(names):
                folder_at, file_at, data_at = names.index("path"), names.index("file"), names.index("data")
                allocated = {}
                for row in rows:
                    if len(row) <= max(folder_at, file_at, data_at) or not row[file_at] or row[data_at] is None:
                        continue
                    name = str(row[file_at]).strip()
                    if not name.lower().endswith(".pdf"):
                        continue
                    pages = parse_pages(row[data_at])
                    if pages or _norm(row[data_at]) == "nodata":
                        allocated[f"{str(row[folder_at] or '').strip().rstrip(chr(92) + '/')}\\{name}"] = pages
                if allocated:
                    return allocated
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
    pdfs, by_words, by_place = defaultdict(list), defaultdict(list), defaultdict(list)
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() == ".pdf":
            pdfs[_norm(p.stem)].append(p)
            by_words[_words(p.stem)].append(p)
            by_place[(_norm(p.parent.name), _norm(p.stem))].append(p)
    one = lambda paths: paths[0] if len({p.stat().st_size for p in paths}) == 1 else None
    found = {}
    for manual in manuals:
        key = _norm(_name(manual))
        if not key:
            continue
        parts = re.split(r"[\\/]", str(manual).strip())
        place = (_norm(parts[-2]), key) if len(parts) > 1 else None
        if place in by_place:   # named with its folder: that folder's file, whatever other folders hold
            if hit := one(by_place[place]):
                found[manual] = hit
            continue
        if re.match(r"([A-Za-z]:|\\\\)", str(manual).strip()):
            continue   # a full path names one file; without its folder here, a file of a like name is another manual
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
    verdict, numbers, lines)} for manuals the tool has read, manuals that matched no PDF)."""
    listed = spreadsheet_pages(xlsx)
    found = match(listed, folder)
    rows, seen = {}, {}
    for manual, path in sorted(found.items()):
        try:
            with pymupdf.open(path) as doc:
                count = doc.page_count
        except Exception as exc:   # a damaged file: the rest of the vessel still teaches
            print(f"  left out {manual[:70]}: {path.name} cannot be opened ({exc.__class__.__name__})")
            continue
        if max(listed[manual], default=0) > count:
            print(f"  left out {manual[:70]}: it lists page {max(listed[manual])}, {path.name} has {count}")
            continue
        print(f"  reading {manual[:70]}", flush=True)
        sha = _sha(path)
        try:
            readings = _readings(path, range(1, count + 1))
        except Exception as exc:
            print(f"  left out {manual[:70]}: {path.name} cannot be read ({exc.__class__.__name__})")
            continue
        for number, reading in readings.items():
            rows[(sha, number)] = (int(number in listed[manual]), reading)
        for number, (verdict, vector, said) in looks.load(sha).items():
            seen[(sha, number)] = (int(number in listed[manual]), verdict, vector, said)
    return rows, seen, sorted(set(listed) - set(found))


def from_corrections() -> tuple[dict, dict]:
    """({(manual's sha256, page): (label, reading)}, {(sha256, page): (label, picture
    verdict, numbers, lines)}) from what people decided on the screen: Include, Skip and
    Leave all out."""
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
        for number, (verdict, vector, said) in looks.load(sha).items():
            if number in labels:
                seen[(sha, number)] = (labels[number], verdict, vector, said)
    return rows, seen


# ---- learning --------------------------------------------------------------

def _sure_from(chance, label, sure) -> float:
    """The lowest score above which at least `sure` of pages are spares pages."""
    order = np.argsort(-chance)
    hits = np.cumsum(label[order])
    precision = hits / np.arange(1, len(order) + 1)
    good = np.nonzero(precision >= sure)[0]
    return float(chance[order][good.max()]) if len(good) else 1.01


def _first_held_out(data, label, groups, learn_from) -> np.ndarray:
    """Each page's first-step score from a model that never saw its manual, learning only
    from the `learn_from` pages."""
    chance = np.zeros(len(label))
    for train, test in GroupKFold(n_splits=min(5, len(set(groups)))).split(data, label, groups):
        train = train[learn_from[train]]
        chance[test] = pipeline().fit(data.iloc[train], label[train]).predict_proba(data.iloc[test])[:, 1]
    return chance


def _neighbours_of(keys, chance, rows, answers) -> np.ndarray:
    """neighbours() of every page, manual by manual, in the order of `keys`."""
    where = defaultdict(dict)
    for i, (sha, number) in enumerate(keys):
        where[sha][number] = i
    seen = None
    for sha, at in where.items():
        numbers, got = neighbours({n: chance[i] for n, i in at.items()}, {n: rows[(sha, n)][1] for n in at},
                                  max(answers[sha]))
        seen = np.full((len(keys), got.shape[1]), np.nan) if seen is None else seen
        seen[[at[n] for n in numbers]] = got
    return seen


def _cuts(chance, label, keep, sure) -> tuple[float, float]:
    unsure_from = float(np.quantile(chance[label == 1], 1 - keep))
    return unsure_from, max(_sure_from(chance, label, sure), unsure_from)


def fit(rows: dict, keep=KEEP, sure=SURE) -> dict:
    """Test on manuals left out, set the cut-offs from that test, then train what is kept
    on everything: the first step on each page's text and, from SECOND_STEP manuals on,
    the second step on runs of pages. Doubtful "not spares" pages take part in neither.
    "pipeline", "unsure_from" and "spares_from" stay the first step's, so a reader from
    before the second step still judges correctly with a model that has one."""
    answers = defaultdict(dict)   # every page's answer, text or not: is a neighbour a listed page?
    for (sha, number), (answer, _) in rows.items():
        answers[sha][number] = answer
    rows = {k: v for k, v in rows.items() if len(v[1]["text"].strip()) >= text.USABLE_TEXT}
    keys = list(rows)
    data = frame([rows[k][1] for k in keys])
    label = np.array([rows[k][0] for k in keys])
    groups = np.array([k[0] for k in keys])
    if len(set(groups)) < 2 or label.sum() == 0:
        raise ValueError("Need at least two manuals and some spares pages to learn from")
    chance = _first_held_out(data, label, groups, np.ones(len(keys), bool))
    doubtful = np.array([answer == 0 and c >= DOUBTFUL and any(answers[sha].get(n + d) == 1
                                                               for d in range(-NEAR, NEAR + 1) if d)
                         for (sha, n), answer, c in zip(keys, label, chance)])
    kept = ~doubtful
    if doubtful.any():   # learning again without them
        chance = _first_held_out(data, label, groups, kept)
    unsure_from, spares_from = _cuts(chance[kept], label[kept], keep, sure)
    model = {"pipeline": pipeline().fit(data.iloc[kept], label[kept]),
             "unsure_from": unsure_from, "spares_from": spares_from}
    decisive, cuts = chance, (unsure_from, spares_from)
    if len(set(groups)) >= SECOND_STEP:
        seen = _neighbours_of(keys, chance, rows, answers)
        decisive = np.zeros(len(keys))
        for train, test in GroupKFold(n_splits=min(5, len(set(groups)))).split(seen, label, groups):
            train = train[kept[train]]
            decisive[test] = second_step().fit(seen[train], label[train]).predict_proba(seen[test])[:, 1]
        cuts = _cuts(decisive[kept], label[kept], keep, sure)
        model["second"] = {"model": second_step().fit(seen[kept], label[kept]),
                           "unsure_from": cuts[0], "spares_from": cuts[1]}
    spares, other = decisive[kept & (label == 1)], decisive[kept & (label == 0)]
    share = lambda part, low, high=9.9: round(float(((part >= low) & (part < high)).mean()), 4)
    model["report"] = {
        "trained": datetime.datetime.now().isoformat(timespec="minutes"),
        "pages": len(keys), "spares pages": int(label.sum()), "manuals": len(set(groups)),
        "doubtful pages left out": int(doubtful.sum()), "keep": keep, "second step": "second" in model,
        "unsure_from": round(cuts[0], 4), "spares_from": round(cuts[1], 4),
        "tested on manuals left out": {
            "spares pages marked spares": share(spares, cuts[1]),
            "spares pages sent to check": share(spares, cuts[0], cuts[1]),
            "spares pages missed": share(spares, -1, cuts[0]),
            "other pages marked spares": share(other, cuts[1]),
            "other pages sent to check": share(other, cuts[0], cuts[1]),
        },
    }
    return model


def _held_out(data, label, groups) -> np.ndarray:
    """Each page's picture-model score from a model trained without its manual."""
    chance = np.zeros(len(label))   # 0 where no model could be trained: a cut-off of 0 moves nothing
    for train, test in GroupKFold(n_splits=min(5, len(set(groups)))).split(data, label, groups):
        if len(set(label[train])) == 2:
            chance[test] = looks_pipeline().fit(data[train], label[train]).predict_proba(data[test])[:, 1]
    return chance


def _looks_cut(chance, label, verdict) -> float:
    """The picture model's cut-off: LOOKS_KEEP of the spares pages the picture reader left
    to a person score at least this (the lowest any spares page scored, if none were left)."""
    asked = chance[(verdict == CHECK) & (label == 1)]
    return float(np.quantile(asked, 1 - LOOKS_KEEP)) if len(asked) else float(chance[label == 1].min())


def _held_out_lines(texts, label, groups) -> np.ndarray:
    """Each page's lines-model score from a model trained without its manual."""
    chance = np.full(len(label), 0.5)
    for train, test in GroupKFold(n_splits=min(5, len(set(groups)))).split(texts, label, groups):
        if len(set(label[train])) == 2:
            chance[test] = lines_pipeline().fit(texts[train], label[train]).predict_proba(texts[test])[:, 1]
    return chance


def _lines_of(row) -> str:
    """The lines of a training row: rows kept before 2026-11-12 have none."""
    return row[3] if len(row) > 3 else ""


def looks_head(data, label, groups, verdict, texts) -> tuple[dict, np.ndarray, np.ndarray]:
    """(the picture model and, where there are pages to learn it from, the lines model with
    the step weighing the two; the pages with lines; their held-out score from the pair).
    Cut-offs keep LOOKS_KEEP and LINES_KEEP of the spares pages, scored by models that never
    saw them."""
    picture = _held_out(data, label, groups)
    head = {"pipeline": looks_pipeline().fit(data, label), "cut": _looks_cut(picture, label, verdict)}
    said = (verdict == CHECK) & (texts != "")
    paired = np.zeros(int(said.sum()))
    if len(set(groups[said & (label == 1)])) >= 2 and (label[said] == 0).any():
        lines = _held_out_lines(texts[said], label[said], groups[said])
        step = LogisticRegression(class_weight="balanced").fit(_pair(picture[said], lines), label[said])
        paired = step.predict_proba(_pair(picture[said], lines))[:, 1]
        head["lines"] = {"pipeline": lines_pipeline().fit(texts[said], label[said]), "step": step,
                         "cut": float(np.quantile(paired[label[said] == 1], 1 - LINES_KEEP))}
    return head, said, paired


def fit_looks(rows: dict) -> dict | None:
    """The picture model, from {(manual's sha256, page): (label, picture verdict, numbers[,
    lines])}; None until there are two manuals with spares pages and other pages to learn
    from. Tested the way it is used: each manual left out in turn, with its cut-off set
    from the other manuals alone."""
    keys = list(rows)
    label = np.array([rows[k][0] for k in keys])
    groups = np.array([k[0] for k in keys])
    if len(set(groups[label == 1])) < 2 or label.all():
        return None
    verdict = np.array([rows[k][1] for k in keys])
    data = looks.features(np.stack([rows[k][2] for k in keys]), verdict)
    texts = np.array([_lines_of(rows[k]) for k in keys], dtype=object)
    moved = np.zeros(len(keys), bool)   # pages the test sent to "unlikely" by the picture alone
    folds = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(data, label, groups))
    for train, test in folds:
        if len(set(groups[train][label[train] == 1])) < 2 or label[train].all():
            continue
        cut = _looks_cut(_held_out(data[train], label[train], groups[train]), label[train], verdict[train])
        chance = looks_pipeline().fit(data[train], label[train]).predict_proba(data[test])[:, 1]
        moved[test] = (verdict[test] == CHECK) & (chance < cut)
    head, said, paired = looks_head(data, label, groups, verdict, texts)
    used = moved.copy()   # as used: the pair decides the pages with lines
    if "lines" in head:
        # held-out scores already; each fold's cut-off is set on the other folds' pages
        at = np.flatnonzero(said)
        fold = np.zeros(len(keys), int)
        for i, (_, test) in enumerate(folds):
            fold[test] = i
        for i in range(len(folds)):
            here = fold[at] == i
            spares = paired[~here & (label[at] == 1)]
            if len(spares):
                used[at[here]] = paired[here] < np.quantile(spares, 1 - LINES_KEEP)
    asked = (verdict == CHECK) & (label == 0)
    spares_asked = (verdict == CHECK) & (label == 1)
    share = lambda which, where: round(float(which[where].mean()) if where.any() else 0.0, 4)
    report = {
        "pages": len(keys), "spares pages": int(label.sum()), "manuals": len(set(groups)), "keep": LOOKS_KEEP,
        "lines keep": LINES_KEEP, "cut": float(f"{head['cut']:.3g}"), "check pages with lines": int(said.sum()),
        "lines cut": float(f"{head['lines']['cut']:.3g}") if "lines" in head else None,
        "tested on manuals left out": {
            "spares pages sent to unlikely": int((used & (label == 1)).sum()),
            "spares check pages sent to unlikely": share(used, spares_asked),
            "other check pages sent to unlikely": share(used, asked),
            "check pages with lines": {
                "spares sent to unlikely": share(used, spares_asked & said),
                "others sent to unlikely": share(used, asked & said),
                "others sent to unlikely by the picture alone": share(moved, asked & said),
            },
        },
    }
    return {**head, "report": report}


def _keep_lines(learned: dict, seen: dict) -> None:
    """learned.update(seen), keeping the lines a page has when it comes again without."""
    for key, row in seen.items():
        old = learned.get(key)
        keep = old is not None and _lines_of(old) and not _lines_of(row)
        learned[key] = (*row[:3], _lines_of(old)) if keep else row


def training_file() -> Path:
    """{"rows": {(manual's sha256, page): (answer, reading)}, "looks": {(sha256, page):
    (answer, picture verdict, numbers, lines)}, "sources": spreadsheets used}."""
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
        _keep_lines(learned["looks"], seen)
        learned["sources"] = sorted(set(learned["sources"]) | {Path(xlsx).name})
        if missing:
            print(f"  no PDF found for: {missing}")
    corrections, seen = from_corrections()
    rows.update(corrections)   # people's latest decisions outrank everything else
    _keep_lines(learned["looks"], seen)
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
