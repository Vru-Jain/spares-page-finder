"""Parsel's PDF table passes, as used by the Spares Page Finder.

Copied from D:/parsel/engine/pdf_extractor.py at commit 9b17de6 (2026-08-21) and
trimmed to what finder/classify.py reaches: whole functions and constants, each
unchanged. Left out: extract_pdf and its scan, OCR, Tesseract, table-structure-model
and page-context code, which the tool replaces with its own reader.
"""
from __future__ import annotations
import re
import pandas as pd


def _rows_to_df(rows: list[list]) -> pd.DataFrame:
    """First non-empty row becomes the header; remaining rows are data."""
    cleaned = []
    for row in rows:
        cleaned.append(["" if c is None else str(c).strip() for c in row])
    cleaned = [r for r in cleaned if any(cell for cell in r)]
    if len(cleaned) < 2:
        return pd.DataFrame()

    header = cleaned[0]
    # de-duplicate / fill blank header cells so pandas doesn't choke
    seen: dict[str, int] = {}
    norm_header = []
    for i, name in enumerate(header):
        name = name or f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        norm_header.append(name)

    width = len(norm_header)
    body = []
    for r in cleaned[1:]:
        if len(r) < width:
            r = r + [""] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        body.append(r)

    return pd.DataFrame(body, columns=norm_header)


def _best_df(candidates: list[pd.DataFrame]) -> pd.DataFrame:
    """Pick the candidate with the most cells filled (rows*cols, weighted by fill)."""
    best = pd.DataFrame()
    best_score = 0.0
    for df in candidates:
        if df.empty:
            continue
        cells = df.shape[0] * df.shape[1]
        if cells == 0:
            continue
        filled = (df.astype(str).apply(lambda s: s.str.strip() != "")).sum().sum()
        score = filled  # raw count of non-empty cells
        if score > best_score:
            best_score = score
            best = df
    return best


# NOTE the optional underscore in col_?\d+: the text path names unnamed columns
# "col_{i}" (see _rows_to_df) but both OCR paths name them "col{i}" without it
# (_slanet_boxes_to_table, _boxes_to_table). Matching only the underscored form
# meant a scanned page whose columns were all unnamed dodged this penalty
# entirely and could out-score a genuinely better candidate — on a real 46-page
# scanned manual it surfaced "col0".."col12" as if they were column headers.
_GARBAGE_HEADER = re.compile(
    r"^(col(umn)?_?\d+|unnamed:?\s*\d*|man\s*b&w|\d[\d\-/]{4,}|\d+\s*\(\d+\))$"
)


def _norm_header(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def build_known_headers(config: dict) -> set[str]:
    """Normalized set of every schema column + every alias, for header scoring."""
    known: set[str] = set()
    for col in config.get("target_schema", []):
        known.add(_norm_header(col))
    for col, aliases in (config.get("header_aliases", {}) or {}).items():
        known.add(_norm_header(col))
        for a in aliases:
            known.add(_norm_header(a))
    known.discard("")
    return known


def score_dataframe(df: pd.DataFrame, known_headers: set[str]) -> float:
    """
    Higher = better extraction. Rewards recognizable headers + filled cells,
    penalizes garbage headers (drawing numbers, 'MAN B&W', 'col_N', single chars).
    """
    if df is None or df.empty:
        return 0.0

    score = 0.0
    headers = [str(c) for c in df.columns]
    for h in headers:
        nh = _norm_header(h)
        hl = h.strip().lower()
        if not nh:
            score -= 2
        elif _GARBAGE_HEADER.match(hl):
            score -= 8                     # strong penalty: clearly wrong header
        elif nh in known_headers:
            score += 12                    # exact recognizable column name
        elif len(nh) <= 1:
            score -= 3                     # fragment like "s", "t"
        else:
            # partial credit if a known header is contained / contains it
            if any(nh in k or k in nh for k in known_headers if len(k) > 2):
                score += 4
            else:
                score += 0.5

    # cell-fill reward (normalized so a huge garbage table can't dominate)
    cells = df.shape[0] * df.shape[1]
    if cells:
        filled = (df.astype(str).apply(lambda s: s.str.strip() != "")).sum().sum()
        score += 3.0 * (filled / cells)

    # penalize tables that are basically one fragmented column of noise
    if df.shape[1] == 1 and df.shape[0] < 2:
        score -= 5

    return score


def _extract_pymupdf_page(mu_page: "fitz.Page") -> tuple[pd.DataFrame, str]:
    """PyMuPDF's built-in table finder as the final fallback."""
    try:
        finder = mu_page.find_tables()
    except Exception:
        return pd.DataFrame(), ""

    candidates = []
    for tbl in getattr(finder, "tables", []) or []:
        try:
            rows = tbl.extract()
        except Exception:
            continue
        df = _rows_to_df(rows)
        if not df.empty:
            candidates.append(df)

    best = _best_df(candidates)
    return best, ("pymupdf:find_tables" if not best.empty else "")


# header tokens we expect on a parts page — used to find the header row
_OCR_HEADER_TOKENS = {
    "no", "name", "type", "qt", "qty", "quantity", "plate", "remaker", "remark",
    "remarks", "designation", "description", "item", "itemno", "part", "partno",
    "partnumber", "pos", "posno", "drawing", "ref", "refno", "code", "material",
    "maker", "unit", "spec",
}


# A scanned row whose text begins with one of these is a caution/instruction/
# note block, not a spare part — dropped so it doesn't pollute the output.
_OCR_ROW_SKIP_PREFIXES = (
    "caution", "warning", "danger", "notice", "note", "attention",
    "instruction", "please", "do not", "important", "remark",
)


# Prepositions/articles that only make sense as part of a multi-word column
# header (e.g. OCR splits "NAME OF PART" into three separate boxes wide enough
# apart that they form distinct column centers → phantom "OF" column).
_OCR_COL_STOPWORDS = {
    "of", "in", "by", "at", "to", "for", "and", "or", "the", "a", "an", "with",
}


def _find_header_by_onset(rows: list, current_best: int, headings=_OCR_HEADER_TOKENS) -> int:
    """
    Data-onset anchoring: verify or improve the header-row index by locating the
    first onset of sequential item numbers (two consecutive rows each carrying a
    short integer token). The header is the last row with ≥2 recognized header
    tokens just above that onset.

    Returns current_best unchanged when no better candidate is identified — this
    is purely additive and never degrades an already-good detection.
    """
    int_rows = [
        ri for ri, row in enumerate(rows)
        if any(re.fullmatch(r"\d{1,4}[A-Za-z]?", c["t"].strip()) for c in row)
    ]
    if len(int_rows) < 2:
        return current_best
    # Find the first pair of integer-carrying rows within 3 rows of each other.
    onset = None
    for k in range(len(int_rows) - 1):
        if int_rows[k + 1] - int_rows[k] <= 3:
            onset = int_rows[k]
            break
    if onset is None or onset == 0:
        return current_best
    # Walk up from the onset to find the nearest row with ≥2 header tokens.
    for back in range(onset - 1, max(-1, onset - 6), -1):
        sc = sum(1 for c in rows[back] if _norm_header(c["t"]) in headings)
        if sc >= 2:
            return back
    return current_best


def _boxes_to_table(res, page_w: int, headings=_OCR_HEADER_TOKENS) -> pd.DataFrame:
    """Reconstruct a table from RapidOCR boxes: cluster cells into rows (by y),
    locate the header row, derive columns from the table region, return a df.
    `headings`: the normalised cell texts that count as column headings.

    Columns are clustered from the table region ONLY (the header row and below),
    not the whole page — otherwise sparse title-block metadata above the table
    (e.g. "Assembly Name", "Assembly Dwg. No.") sits between the item-number and
    part-number columns and bridges the x-gap, merging two real columns into one
    and destroying the header detection."""
    cells, heights = [], []
    for box, text, score in res:
        t = (text or "").strip()
        if not t or (score is not None and score < 0.4):
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cells.append({"cx": sum(xs) / 4.0, "cy": sum(ys) / 4.0, "t": t})
        heights.append(max(ys) - min(ys))
    if len(cells) < 4:
        return pd.DataFrame()

    hmed = sorted(heights)[len(heights) // 2] or 12
    ytol = max(8.0, hmed * 0.7)

    # ---- cluster rows by y (running mean) ----
    cells.sort(key=lambda c: c["cy"])
    rows, cur, cur_y = [], [cells[0]], cells[0]["cy"]
    for c in cells[1:]:
        if abs(c["cy"] - cur_y) <= ytol:
            cur.append(c)
            cur_y = sum(x["cy"] for x in cur) / len(cur)
        else:
            rows.append(cur)
            cur, cur_y = [c], c["cy"]
    rows.append(cur)

    # ---- find the header row: the band with the most cells whose OWN text is a
    # known header token (checked per raw cell, BEFORE any column merge) ----
    best_i, best_score = -1, 0
    for ri, row in enumerate(rows):
        sc = sum(1 for c in row if _norm_header(c["t"]) in headings)
        if sc > best_score:
            best_score, best_i = sc, ri

    # No recognizable parts header (caution page, nameplate/spec sheet, prose)
    # -> not a parts table. Returning empty makes the page report "no table"
    # instead of dumping generic, jumbled col0/col1 rows.
    if best_score < 2:
        return pd.DataFrame()

    # Data-onset refinement: verify the token-based header against the actual
    # start of sequential integer item numbers.  Only replaces best_i when a
    # better-scored header row is found just above the integer onset.
    best_i = _find_header_by_onset(rows, best_i, headings)

    # ---- cluster columns by x using only the table region (header + below) ----
    if best_i == len(rows) - 1:
        # Header is the last row — nothing beneath it; treat as no parseable table.
        return pd.DataFrame()
    region_rows = rows[best_i:]
    # Use header cells only for column clustering.  Using the full region
    # (header + all below) lets data cells bridge the x-gaps between real
    # columns when a spec/title block sits just below the parts header (common
    # on engineering general-view sheets), collapsing multiple columns into one.
    centers = sorted(c["cx"] for c in rows[best_i])
    colgap = max(40.0, page_w * 0.035)
    grp, colcenters = [centers[0]], []
    for x in centers[1:]:
        if x - grp[-1] <= colgap:
            grp.append(x)
        else:
            colcenters.append(sum(grp) / len(grp))
            grp = [x]
    colcenters.append(sum(grp) / len(grp))

    # ---- Recover columns with no header cell ("unheadered" columns) ----------
    # A column whose header was never OCR'd (common for unlabeled description/
    # name columns) has no x-center above, so every one of its data cells gets
    # mis-snapped into whichever neighboring header column happens to be
    # nearest, silently destroying that column. Detect data-cell x-positions
    # that sit clearly outside every known header's gap (further than colgap
    # from the nearest header center) and recurring across multiple data rows,
    # and promote them to real, unlabeled column centers.
    orphan_xs = [
        c["cx"] for r in region_rows[1:] for c in r
        if min((abs(c["cx"] - cc) for cc in colcenters), default=1e9) > colgap
    ]
    if orphan_xs:
        orphan_xs.sort()
        ogrp, ocenters = [orphan_xs[0]], []
        for x in orphan_xs[1:]:
            if x - ogrp[-1] <= colgap:
                ogrp.append(x)
            else:
                ocenters.append((sum(ogrp) / len(ogrp), len(ogrp)))
                ogrp = [x]
        ocenters.append((sum(ogrp) / len(ogrp), len(ogrp)))
        # Require the cluster to recur in at least 2 data rows so a single
        # stray OCR box (noise) can't fabricate a phantom column.
        colcenters.extend(cc for cc, n in ocenters if n >= 2)
        colcenters.sort()

    ncol = len(colcenters)

    def col_of(cx):
        return min(range(ncol), key=lambda i: abs(cx - colcenters[i]))

    grid = []
    for row in region_rows:
        bycol = {}
        for c in sorted(row, key=lambda c: c["cx"]):
            i = col_of(c["cx"])
            bycol[i] = (bycol.get(i, "") + " " + c["t"]).strip()
        grid.append(bycol)

    header, data_rows = grid[0], grid[1:]

    colnames, seen = [], {}
    for i in range(ncol):
        nm = (header.get(i, "") or f"col{i}").strip() or f"col{i}"
        if nm in seen:
            seen[nm] += 1
            nm = f"{nm}_{seen[nm]}"
        else:
            seen[nm] = 0
        colnames.append(nm)

    # ---- Merge orphan stop-word columns into adjacent neighbors --------------
    # OCR sometimes places each word of a multi-word column header ("NAME OF
    # PART") in a separate bounding box far enough apart that they form distinct
    # column centers, producing phantom "OF" or "IN" columns.
    #
    # Strategy: when a column whose entire header is a preposition/article is
    # SANDWICHED between two content columns (both > 3 chars, neither a stop
    # word), absorb left, center, and right into one group in a single step.
    # This handles "NAME | OF | PART" → "NAME OF PART" without mistakenly
    # merging unrelated adjacent columns.  Edge-case stop words (leftmost or
    # rightmost) are merged with whichever neighbor exists.
    col_groups: list[list[int]] = [[i] for i in range(ncol)]
    cg_names: list[str] = list(colnames)
    ci = 0
    while ci < len(col_groups):
        nm = cg_names[ci].strip().lower()
        if nm in _OCR_COL_STOPWORDS:
            left_ok = (ci > 0
                       and len(cg_names[ci - 1].strip()) > 3
                       and cg_names[ci - 1].strip().lower() not in _OCR_COL_STOPWORDS)
            right_ok = (ci + 1 < len(col_groups)
                        and len(cg_names[ci + 1].strip()) > 3
                        and cg_names[ci + 1].strip().lower() not in _OCR_COL_STOPWORDS)
            if left_ok and right_ok:
                # sandwich: absorb stop word AND right neighbor into left group
                col_groups[ci - 1].extend(col_groups[ci])
                col_groups[ci - 1].extend(col_groups[ci + 1])
                cg_names[ci - 1] = (
                    cg_names[ci - 1] + " " + cg_names[ci] + " " + cg_names[ci + 1]
                ).strip()
                col_groups.pop(ci + 1); cg_names.pop(ci + 1)  # pop right first
                col_groups.pop(ci);     cg_names.pop(ci)        # then stop word
                # ci stays: may be another stop word to process at the same index
            elif ci > 0:
                col_groups[ci - 1].extend(col_groups[ci])
                cg_names[ci - 1] = (cg_names[ci - 1] + " " + cg_names[ci]).strip()
                col_groups.pop(ci); cg_names.pop(ci)
            elif ci + 1 < len(col_groups):
                col_groups[ci + 1] = col_groups[ci] + col_groups[ci + 1]
                cg_names[ci + 1] = (cg_names[ci] + " " + cg_names[ci + 1]).strip()
                col_groups.pop(ci); cg_names.pop(ci)
            else:
                ci += 1
        else:
            ci += 1
    ngrp = len(col_groups)
    colnames, seen = [], {}
    for nm in cg_names:
        nm = nm or f"col{len(colnames)}"
        if nm in seen:
            seen[nm] += 1
            nm = f"{nm}_{seen[nm]}"
        else:
            seen[nm] = 0
        colnames.append(nm)
    # --------------------------------------------------------------------------

    # Compute row centre-y values for gap detection.  A gap wider than
    # 5 × median cell height indicates a footer/letterhead block separated from
    # the data (common in landscape manuals after a 90° rotation pass).
    region_ys = [sum(c["cy"] for c in rrow) / len(rrow) for rrow in region_rows]
    gap_limit = 5.0 * hmed
    prev_y = region_ys[0]  # header row y

    records = []
    for ri, g in enumerate(data_rows):
        cur_y = region_ys[ri + 1]
        if cur_y - prev_y > gap_limit:
            break   # large gap → letterhead / footer block; stop here
        prev_y = cur_y
        cells = [
            " ".join(g.get(oi, "").strip() for oi in grp).strip()
            for grp in col_groups
        ]
        nonempty = [c for c in cells if c]
        if not nonempty:
            continue
        low = [c.lower() for c in nonempty]
        # Drop a leaked secondary header row (e.g. a repeated "TYPE / QT." band).
        # Threshold is 3 to avoid false-drops: tokens like "code", "type", "unit"
        # are common real data values in marine parts lists.
        if sum(1 for c in low if _norm_header(c) in _OCR_HEADER_TOKENS) >= 3:
            continue
        # drop caution / instruction / note prose
        if any(c.startswith(_OCR_ROW_SKIP_PREFIXES) for c in low):
            continue
        records.append({colnames[k]: cells[k] for k in range(ngrp)})

    # --- inverted-table pass -----------------------------------------------
    # Some engineering drawing sheets print a reference parts table with data
    # rows ABOVE the column header and unrelated content (specs, title block)
    # below.  Signal: the item-number column has very few integers in the rows
    # extracted from below the header.  When triggered, re-extract using rows
    # above best_i, filtered to the table's x-range so drawing labels are
    # excluded.  Only replaces the normal result when numeric fraction is better.
    _NO_COL_TOKENS = {"no", "item", "itemno", "ref", "refno"}
    _no_idx = next(
        (i for i, nm in enumerate(colnames)
         if _norm_header(nm) in _NO_COL_TOKENS),
        None,
    )
    if _no_idx is not None and records and best_i > 0:
        _no_name = colnames[_no_idx]
        _num_frac = sum(
            1 for r in records
            if str(r.get(_no_name, "")).strip().isdigit()
        ) / len(records)
        if _num_frac < 0.3:
            # Build column centers from header cells only (same as main pass)
            _hcxs = sorted(c["cx"] for c in rows[best_i])
            _igap = max(40.0, page_w * 0.035)
            _ig, _ics = [_hcxs[0]], []
            for _x in _hcxs[1:]:
                if _x - _ig[-1] <= _igap:
                    _ig.append(_x)
                else:
                    _ics.append(sum(_ig) / len(_ig))
                    _ig = [_x]
            _ics.append(sum(_ig) / len(_ig))
            _inc = len(_ics)
            _ix_lo = _ics[0] - _igap * 2
            _ix_hi = _ics[-1] + _igap * 2

            def _icol(cx):
                return min(range(_inc), key=lambda k: abs(cx - _ics[k]))

            _ihdr: dict = {}
            for _c in sorted(rows[best_i], key=lambda c: c["cx"]):
                _ci = _icol(_c["cx"])
                _ihdr[_ci] = (_ihdr.get(_ci, "") + " " + _c["t"]).strip()
            _inm: list = []
            _iseen: dict = {}
            for _i in range(_inc):
                _n = (_ihdr.get(_i, "") or f"col{_i}").strip() or f"col{_i}"
                if _n in _iseen:
                    _iseen[_n] += 1
                    _n = f"{_n}_{_iseen[_n]}"
                else:
                    _iseen[_n] = 0
                _inm.append(_n)

            # Merge stop-word columns in the inverted pass (sandwich, same as main pass)
            _icg: list = [[_j] for _j in range(_inc)]
            _icgn: list = list(_inm)
            _ij = 0
            while _ij < len(_icg):
                _inm_word = _icgn[_ij].strip().lower()
                if _inm_word in _OCR_COL_STOPWORDS:
                    _ileft = (_ij > 0
                              and len(_icgn[_ij - 1].strip()) > 3
                              and _icgn[_ij - 1].strip().lower() not in _OCR_COL_STOPWORDS)
                    _iright = (_ij + 1 < len(_icg)
                               and len(_icgn[_ij + 1].strip()) > 3
                               and _icgn[_ij + 1].strip().lower() not in _OCR_COL_STOPWORDS)
                    if _ileft and _iright:
                        _icg[_ij - 1].extend(_icg[_ij])
                        _icg[_ij - 1].extend(_icg[_ij + 1])
                        _icgn[_ij - 1] = (
                            _icgn[_ij - 1] + " " + _icgn[_ij] + " " + _icgn[_ij + 1]
                        ).strip()
                        _icg.pop(_ij + 1); _icgn.pop(_ij + 1)
                        _icg.pop(_ij);     _icgn.pop(_ij)
                    elif _ij > 0:
                        _icg[_ij - 1].extend(_icg[_ij])
                        _icgn[_ij - 1] = (_icgn[_ij - 1] + " " + _icgn[_ij]).strip()
                        _icg.pop(_ij); _icgn.pop(_ij)
                    elif _ij + 1 < len(_icg):
                        _icg[_ij + 1] = _icg[_ij] + _icg[_ij + 1]
                        _icgn[_ij + 1] = (_icgn[_ij] + " " + _icgn[_ij + 1]).strip()
                        _icg.pop(_ij); _icgn.pop(_ij)
                    else:
                        _ij += 1
                else:
                    _ij += 1
            _ingrp = len(_icg)
            _inm = []
            _inseen: dict = {}
            for _nn in _icgn:
                _nn = _nn or f"col{len(_inm)}"
                if _nn in _inseen:
                    _inseen[_nn] += 1; _nn = f"{_nn}_{_inseen[_nn]}"
                else:
                    _inseen[_nn] = 0
                _inm.append(_nn)

            _irecs: list = []
            _ipy = region_ys[0]
            for _irow in reversed(rows[:best_i]):
                _ir = [_c for _c in _irow if _ix_lo <= _c["cx"] <= _ix_hi]
                if not _ir:
                    continue
                _icy = sum(_c["cy"] for _c in _ir) / len(_ir)
                if _ipy - _icy > gap_limit:
                    break
                _ipy = _icy
                _ib: dict = {}
                for _c in sorted(_ir, key=lambda c: c["cx"]):
                    _ci = _icol(_c["cx"])
                    _ib[_ci] = (_ib.get(_ci, "") + " " + _c["t"]).strip()
                _ic = [
                    " ".join(_ib.get(_ok, "").strip() for _ok in _og).strip()
                    for _og in _icg
                ]
                _ine = [_v for _v in _ic if _v]
                if not _ine:
                    continue
                _il = [_v.lower() for _v in _ine]
                if sum(1 for _v in _il
                       if _norm_header(_v) in _OCR_HEADER_TOKENS) >= 3:
                    continue
                if any(_v.startswith(_OCR_ROW_SKIP_PREFIXES) for _v in _il):
                    continue
                _irecs.append({_inm[_k]: _ic[_k] for _k in range(_ingrp)})

            if _irecs:
                # Use token search so the index is valid after the stop-word merge
                _ino_name = next(
                    (nm for nm in _inm if _norm_header(nm) in _NO_COL_TOKENS), ""
                )
                _ino_frac = sum(
                    1 for _r in _irecs
                    if str(_r.get(_ino_name, "")).strip().isdigit()
                ) / len(_irecs)
                if _ino_frac > _num_frac:
                    records = _irecs
    # -----------------------------------------------------------------------

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # An un-headered column that is mostly free text is the description/part-name
    # column (common in parts lists where the name column has no caption). Rename
    # it "Description" so the column mapper recognises it as the Part Name.
    df = _name_description_column(df)
    return df


def _name_description_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a single auto-named ('colN') column to 'Description' when it holds
    mostly alphabetic free text (texty >= 0.6, avglen >= 4) — so it maps to
    Part Name downstream instead of being dropped as unmapped. Selection uses
    texty * avglen as a content metric; column geometry is not measured."""
    auto = [c for c in df.columns if re.fullmatch(r"col\d+", str(c))]
    if not auto:
        return df
    # already have a name/description/designation column -> nothing to rescue
    _name_like = ("desc", "name", "designation", "denomination", "nomenclature")
    if any(any(k in str(c).lower() for k in _name_like) for c in df.columns):
        return df
    best, best_metric = None, 0.0
    for c in auto:
        vals = [str(v).strip() for v in df[c] if str(v).strip()]
        if not vals:
            continue
        texty = sum(1 for v in vals if re.search(r"[A-Za-z]", v)) / len(vals)
        avglen = sum(len(v) for v in vals) / len(vals)
        if texty >= 0.6 and avglen >= 4:
            metric = texty * avglen
            if metric > best_metric:
                best, best_metric = c, metric
    if best is not None and "Description" not in df.columns:
        df = df.rename(columns={best: "Description"})
    return df


_ITEM_NO_RE = re.compile(r"^\d{1,4}[A-Za-z]?$")


def _extract_textblocks_page(mu_page: "fitz.Page") -> tuple[pd.DataFrame, str]:
    """
    Reconstruct a parts table from positioned words for the common MAN B&W
    "Item No. / Item Designation" layout.

    LAYOUT-ANCHORED (not line-anchored): we find the item-number COLUMN by the
    x-position of the short numeric tokens, then read each item's description as
    the words to the RIGHT of its number on the same row. This recovers rows
    that a naive "line must start with a number" rule loses when a left-column
    sub-heading (e.g. "Bolts") clusters onto the row, and it ignores left-margin
    group headings and page footers automatically.

    Rows are clustered by vertical center with a running mean (robust to the
    small y-jitter that a fixed-bin rounding would split across two rows).
    """
    try:
        words = mu_page.get_text("words")  # (x0,y0,x1,y1,word,block,line,wordno)
    except Exception:
        return pd.DataFrame(), ""
    toks = [(w[0], w[1], w[3], str(w[4]).strip()) for w in words if str(w[4]).strip()]
    if len(toks) < 4:
        return pd.DataFrame(), ""

    # ---- cluster tokens into rows by vertical center (running mean) ----
    toks.sort(key=lambda t: (t[1] + t[2]) / 2.0)
    heights = sorted(t[2] - t[1] for t in toks if t[2] > t[1])
    hmed = heights[len(heights) // 2] if heights else 10.0
    ytol = max(4.0, hmed * 0.6)
    bands: list[list[tuple[float, str]]] = []
    cur = [toks[0]]
    cur_y = (toks[0][1] + toks[0][2]) / 2.0
    for t in toks[1:]:
        cy = (t[1] + t[2]) / 2.0
        if abs(cy - cur_y) <= ytol:
            cur.append(t)
            cur_y = sum((x[1] + x[2]) / 2.0 for x in cur) / len(cur)
        else:
            bands.append(sorted([(c[0], c[3]) for c in cur], key=lambda p: p[0]))
            cur = [t]
            cur_y = cy
    bands.append(sorted([(c[0], c[3]) for c in cur], key=lambda p: p[0]))

    # ---- locate the item-number column (median x of numeric tokens) ----
    num_xs = sorted(x for b in bands for (x, t) in b if _ITEM_NO_RE.match(t))
    if len(num_xs) < 2:
        return pd.DataFrame(), ""
    numcol = num_xs[len(num_xs) // 2]
    xtol = 30.0

    # ---- build rows; description = words to the right of the number ----
    rows: list[list[str]] = []
    cur_row: list[str] | None = None
    for b in bands:
        numtok = next(((x, t) for (x, t) in b
                       if _ITEM_NO_RE.match(t) and abs(x - numcol) <= xtol), None)
        if numtok:
            desc = " ".join(t for (x, t) in b if x > numtok[0] + 2)
            cur_row = [numtok[1], desc.strip()]
            rows.append(cur_row)
        elif cur_row is not None:
            # continuation line: a wrapped description sits in the description
            # column (right of the number column) with no number of its own.
            cont = " ".join(t for (x, t) in b
                            if x > numcol + 2 and not _ITEM_NO_RE.match(t))
            if cont:
                cur_row[1] = (cur_row[1] + " " + cont).strip()

    rows = [r for r in rows if r[1]]   # drop bare numbers with no description
    if len(rows) >= 2:
        return pd.DataFrame(rows, columns=["No.", "Description"]), "textblocks:item-designation"
    return pd.DataFrame(), ""

