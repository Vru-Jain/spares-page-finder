"""What a page looks like: our own loop around the OCR engine, looking for a parts table.

Reading text is where OCR spends its time, and a long line costs more than a short one.
A table is made of short cells, so:
  - lines are found on a shrunken copy of the page first (see SIZES) and cut from the
    full-size page to be read;
  - only short lines are read, column headings first (see PEEK), and a page with no two
    headings among them has no table: Parsel's table builder would find no header row;
  - running text is never read;
  - a page is turned only when its lines run sideways.
Measured on a 464-page scanned Yanmar manual and on BERGE DAISEN, on an i7-12700; the
knobs that would differ on another PC can be set from outside (FINDER_...).
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from engine.pdf_extractor import _OCR_HEADER_TOKENS, _boxes_to_table, _norm_header, score_dataframe
from finder import looks, pool, tables
from finder.tables import CHECK, NO, SPARES

# Worker processes reading pictures, and threads each. More workers do not help on this
# PC: 47 Yanmar pages took 26 s with 1 worker of 8 threads, 20 s with 2 of 4, 19 s with
# 4 of 2 and 22 s with 8 of 1. 0 threads = the fast cores, shared out.
WORKERS = int(os.environ.get("FINDER_WORKERS", 2))
THREADS = int(os.environ.get("FINDER_THREADS", 0))
# The sizes at which a page's lines are found, tried in turn until one finds a parts
# table. Neither finds everything: on the Yanmar manual the small one found 22 parts
# pages the full size missed and missed 2 it found.
SIZES = [float(x) for x in os.environ.get("FINDER_SIZES", "0.6,1").split(",")]
# A page the model is unsure of is read only to see whether a part-number table confirms
# it; it is never ruled out, whatever its picture shows. The first size is enough for
# that: on 300 BERGE DAISEN picture pages, 2.25x faster, and 5 of 44 confirmations
# went to a person instead. Pages the picture alone decides get every size.
QUICK = SIZES[:1]
# "No table found" is no proof: the reader missed about 1 in 5 scanned parts pages of
# H1868 and BERGE DAISEN (small print on big sheets, tables upside down, headings in
# German). So a picture with no table found goes to a person unless it is nearly blank,
# showing fewer than BLANK short lines: a cover, a photo, an empty sheet. Of 490 listed
# scanned pages of the two vessels, none showed fewer than 8.
BLANK = 5
# Headings first: once a row of column headings and PEEK rows under it have been read, a
# page whose table already has a part-number column is decided. Short lines are read
# CHUNK at a time from the top, so the engine still gets full batches.
PEEK = 4
CHUNK = 30
HEADER = 0.25         # a column heading is never wider than this share of the page's short side
RUNNING_TEXT = 0.5    # a line longer than this share is running text, and is never read
# Column headings: the single words Parsel's table builder looks for, and every name of
# two words or more that Parsel gives a parts-table column. PP-OCRv6 reads "Part
# Description" as one line, so it never came as the two words. Single-word names such as
# "Spare" or "Index" are left out: they also head title blocks and revision boxes, and
# were taken for the table's headings. A heading split over two lines ("Part" over "No.")
# is joined (_stacked). Measured on 6,897 scanned TORM and SVITZER pages: 98 more listed
# spares pages found among the 2,760 sent to a person, none lost; of 3,000 others sent to
# a person, 8 now marked spares, 6 of them parts lists the spreadsheets left out.
HEADINGS = _OCR_HEADER_TOKENS | {
    _norm_header(name) for column, aliases in tables.CONFIG["header_aliases"].items() if column in tables.COLUMNS
    for name in (column, *aliases) if len(re.findall(r"[A-Za-z0-9]+", name)) > 1} - {""}
# The OCR models: PP-OCRv6 tiny, from RapidOCR 3.10's model hub (det SHA-256 f42c0fbd...,
# rec e16e242d...), run by rapidocr_onnxruntime 1.4.4's own pre- and post-processing.
# 0.44 s a page on 300 BERGE DAISEN pictures, against 0.75 s for PP-OCRv4.
MODELS = Path(__file__).parent.parent / "models"
DET, REC = MODELS / "PP-OCRv6_det_tiny.onnx", MODELS / "PP-OCRv6_rec_tiny.onnx"
# They run on OpenVINO when it is installed: 3.3x faster than onnxruntime on this PC's
# CPU, with the same answers. (Its graphics chip was slower, and found nothing.)
RUNTIME = "openvino" if importlib.util.find_spec("openvino") else "onnxruntime"

_ENGINE = None   # this worker's OCR engine
_LINES: list[str] = []   # what the engine read on the page being read now, for the picture model (learn.py)


def start_engine(threads: int) -> None:
    """Load the OCR models, and the network of looks.py, into this worker process."""
    global _ENGINE
    looks.start(threads)
    import rapidocr_onnxruntime
    # The recogniser reads its character list from the model file itself.
    _ENGINE = rapidocr_onnxruntime.RapidOCR(det_model_path=str(DET), rec_model_path=str(REC),
                                            intra_op_num_threads=threads, inter_op_num_threads=1)
    if RUNTIME == "openvino":
        _ENGINE.text_det.infer = _OpenVINO(DET, threads)
        _ENGINE.text_rec.session = _OpenVINO(REC, threads)


class _OpenVINO:
    """One of the OCR models run by OpenVINO, called the way RapidOCR calls onnxruntime."""

    def __init__(self, model: Path, threads: int):
        import openvino
        self.model = openvino.Core().compile_model(str(model), "CPU", {
            "INFERENCE_NUM_THREADS": threads, "ENABLE_CPU_PINNING": False})

    def __call__(self, batch):
        return [self.model(batch)[0]]


def _read(page, boxes) -> list:
    """[box, text, score] for each box of the page image, None where the engine is unsure."""
    if not boxes:
        return []
    read, _ = _ENGINE.text_rec(_ENGINE.get_crop_img_list(page, boxes))
    got = [[box.tolist(), text, score] if score >= _ENGINE.text_score else None
           for box, (text, score) in zip(boxes, read)]
    _LINES.extend(line[1] for line in got if line is not None)
    return got


def _is_heading(line) -> bool:
    return line is not None and _norm_header(line[1]) in HEADINGS


def _stacked(cells) -> list:
    """`cells` with each line joined to the one right under it when the two make a heading
    together ("Part" over "No."). The joined cell keeps the upper line's height, so it
    stays in the row of headings."""
    span = lambda cell, axis: (min(p[axis] for p in cell[0]), max(p[axis] for p in cell[0]))
    cells = sorted(cells, key=lambda cell: span(cell, 1)[0])
    out, used = [], set()
    for i, (box, text, score) in enumerate(cells):
        if i in used:
            continue
        (x0, x1), (y0, y1) = span(cells[i], 0), span(cells[i], 1)
        for j in range(i + 1, len(cells)):
            (u0, u1), (v0, _) = span(cells[j], 0), span(cells[j], 1)
            if v0 > y1 + (y1 - y0):
                break   # further down than a line
            joined = f"{text} {cells[j][1]}"
            if (j not in used and v0 >= y1 - 0.3 * (y1 - y0)   # under it, not beside it
                    and min(x1, u1) - max(x0, u0) > 0.5 * min(x1 - x0, u1 - u0)
                    and _norm_header(cells[j][1]) and _norm_header(joined) in HEADINGS):
                left, right = min(x0, u0), max(x1, u1)
                box, text, score = [[left, y0], [right, y0], [right, y1], [left, y1]], joined, min(score, cells[j][2])
                used.add(j)
                break
        out.append([box, text, score])
    return out


def _rows(boxes) -> list[int]:
    """Row number of each box (boxes sorted top to bottom), grouped the way Parsel's
    table builder groups cells into rows."""
    heights = sorted(float(np.ptp(box[:, 1])) for box in boxes)
    tolerance = max(8.0, 0.7 * (heights[len(heights) // 2] or 12))
    rows, row, members = [], 0, []
    for box in boxes:
        y = float(box[:, 1].mean())
        if members and abs(y - sum(members) / len(members)) > tolerance:
            row, members = row + 1, []
        members.append(y)
        rows.append(row)
    return rows


def _judge_once(img, size: float) -> tuple[str, bool, int]:
    """(answer, whether the page's lines mostly run sideways, how many short lines it
    has) for one way up, one size."""
    page, ratio_h, ratio_w = _ENGINE.preprocess(img)   # at most 2000 pixels a side
    found, _ = _ENGINE.auto_text_det(page if size == 1 else cv2.resize(
        page, None, fx=size, fy=size, interpolation=cv2.INTER_AREA))
    if found is None:
        return NO, False, 0
    boxes = [box / size for box in found]
    along = lambda box: float(np.linalg.norm(box[0] - box[1]))    # the way the text runs
    across = lambda box: float(np.linalg.norm(box[0] - box[3]))
    middle = lambda box: float(box[:, 1].mean())
    sideways = sum(across(b) > 1.5 * along(b) for b in boxes) > len(boxes) / 2
    side = min(page.shape[:2])
    length = [max(along(b), across(b)) / side for b in boxes]
    short = sorted((b for b, n in zip(boxes, length) if n < HEADER), key=middle)
    medium = [b for b, n in zip(boxes, length) if HEADER <= n < RUNNING_TEXT]

    def verdict(lines) -> str:
        # back to `img`'s own pixels, which Parsel's builder is tuned for
        cells = [[[[x * ratio_w, y * ratio_h] for x, y in line[0]], line[1], line[2]]
                 for line in lines if line is not None]
        df = _boxes_to_table(_stacked(cells), img.shape[1], HEADINGS) if cells else None
        return NO if df is None or df.empty else tables.bucket(df, score_dataframe(df, tables.KNOWN))

    rows, got, near, header = _rows(short) if short else [], [], [], None
    for start in range(0, len(short), CHUNK):
        got += _read(page, short[start:start + CHUNK])
        complete = rows[len(got)] if len(got) < len(short) else rows[-1] + 1   # rows read in full
        if header is None:
            headings = [r for line, r in zip(got, rows) if r < complete and _is_heading(line)]
            header = next((r for r in sorted(set(headings)) if headings.count(r) >= 2), None)
            if header is not None and complete > min(header + PEEK, rows[-1]):
                bottom = max(middle(b) for b, r in zip(short, rows) if r < complete)
                near = [b for b in medium if middle(b) <= bottom]
                got += _read(page, near)
                if verdict(got) == SPARES:
                    return SPARES, sideways, len(short)   # decided: the rest of the page is never read
    if sum(map(_is_heading, got)) < 2:
        return NO, sideways, len(short)   # no two column headings: Parsel's builder would find no header row
    return verdict(got + _read(page, [b for b in medium if not any(b is n for n in near)])), sideways, len(short)


def _judge_at(img, size: float) -> tuple[str, int]:
    found, sideways, lines = _judge_once(img, size)
    if found == NO and sideways:   # clockwise, the way Parsel turns it
        found, _, turned = _judge_once(np.ascontiguousarray(np.rot90(img, k=3)), size)
        lines = max(lines, turned)
    return found, lines


def judge(img, sizes=SIZES) -> str:
    """What a page image shows: spares, check, or no (nearly blank, see BLANK)."""
    best, lines = NO, 0
    for size in sizes:
        found, seen = _judge_at(img, size)
        best, lines = max(best, found, key=tables.strength), max(lines, seen)
        if best == SPARES:
            break
    return CHECK if best == NO and lines >= BLANK else best


def read_picture(path: str, number: int, sizes=SIZES, numbers=False) -> tuple[int, str, np.ndarray | None, str]:
    """(number, what page `number` shows, its numbers from looks.py if `numbers` is
    asked for and the network is there, the lines read on it, one a line), in a worker
    process."""
    _LINES.clear()
    try:
        page = pool.open_pdf(path)[number - 1]
        # 200 dpi, but never more than 10,000 pixels a side (an A0 drawing is 9,362): a PDF
        # can declare a page 200 inches wide, and drawing that would exhaust the memory.
        zoom = min(200 / 72, 10_000 / max(page.rect.width, page.rect.height, 1))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
        img = np.ascontiguousarray(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3))
        found = judge(img, sizes)
        return number, found, looks.vector(img) if numbers else None, "\n".join(dict.fromkeys(_LINES))
    except Exception:
        return number, CHECK, None, ""   # could not be read: a person looks at it


_POOL = pool.Pool(WORKERS, start_engine,
                  lambda cores: (THREADS or max(1, (cores or os.cpu_count() or 4) // WORKERS),))


def read_pictures(path, pages, progress, quick=()) -> tuple[dict[int, str], dict[int, np.ndarray], dict[int, str]]:
    """({page: what its picture shows}, {page: its numbers from looks.py}, {page: the lines
    read on it}). Pages in `quick` are read at the first size only, and get no numbers:
    their text layer describes them. progress(done, total) after each page."""
    got, numbers, lines = {}, {}, {}
    for number, found, vector, said in _POOL.each(read_picture, [(str(path), n, QUICK if n in quick else SIZES, n not in quick)
                                                                 for n in sorted(pages)]):
        got[number], lines[number] = found, said
        if vector is not None:
            numbers[number] = vector
        progress(len(got), len(pages))
    return got, numbers, lines
