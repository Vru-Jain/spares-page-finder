"""What the server hands out: page pictures for the screen (drawn once and kept, the ones a
person will look at drawn as soon as a manual is read), and the downloads: each
manual whole, every page under a band saying whether it was included, the allocation
sheet, and a zip of them all.

The pages included are what the tool marked spares, as corrected by people. Pages still
waiting under "To check" are left out (and named in the allocation sheet's Note), and
so are the pages the picture model thinks are probably not spares, unless a person
included them. An archived drop keeps only each manual's included pages (archive.py).
"""
from __future__ import annotations

import ctypes
import datetime
import io
import os
import queue
import re
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath

import pymupdf as fitz
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from PIL import Image

from finder import store

_PDF = threading.Lock()   # PyMuPDF is not thread-safe


# The sizes a page is drawn at, as (dots per inch, most pixels a side): for the list, the
# enlarged view, and zoomed into a large drawing. The cap matters: a PDF can declare a
# page 200 inches wide, and drawing that at full size would exhaust the memory. An A0
# sheet at the zoom size is 7,000 pixels wide, 150 dpi: small print reads.
PAGE_SIZES = {0: (60, 1000), 1: (150, 3000), 2: (200, 7000)}
# The list and the enlarged view are WebP: on 60 pages of two real drops (20 Nov) a list
# picture went from 167 KB as PNG to 24 KB, an enlarged page from 804 KB to 96 KB, the print
# as sharp at 1:1. The zoom size stays a JPEG.
WEBP_QUALITY = {0: 75, 1: 80}


def page_png(path: str, page: int, size: int, turn: int) -> tuple[bytes, str] | None:
    """(picture, media type) of one page at one of PAGE_SIZES, turned by `turn` degrees
    (for tables scanned sideways); None if there is no such page. The list and enlarged
    sizes are WebP (WEBP_QUALITY), the zoom size a JPEG."""
    with _PDF, fitz.open(path) as doc:
        if not 1 <= page <= doc.page_count or turn % 90:
            return None
        shown = doc[page - 1]
        dpi, most = PAGE_SIZES.get(size, PAGE_SIZES[0])
        scale = min(dpi / 72, most / max(shown.rect.width, shown.rect.height, 1))
        pix = shown.get_pixmap(matrix=fitz.Matrix(scale, scale).prerotate(turn), alpha=False)
        if size == 2:
            return pix.tobytes("jpeg", jpg_quality=85), "image/jpeg"
        drawn = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    out = io.BytesIO()   # outside the lock: the page is drawn, and another can be meanwhile
    drawn.save(out, "WEBP", quality=WEBP_QUALITY.get(size, 75), method=4)
    return out.getvalue(), "image/webp"


def pictures_of(manual) -> Path:
    """Where a manual's page pictures are kept once drawn: data/pictures/<its sha256>/ (a manual
    is kept as uploads/<sha256>.pdf). archive.py removes them with the manual."""
    return store.DATA / "pictures" / PurePosixPath(str(manual).replace("\\", "/")).stem


def page_picture(path, page: int, size: int, turn: int = 0) -> tuple[bytes, str] | None:
    """page_png, drawn once: the list and enlarged sizes are kept (pictures_of), so a page
    opens at once the next time, for anyone. The zoom size, big and seldom asked for, is
    drawn each time."""
    if size not in WEBP_QUALITY:
        return page_png(path, page, size, turn)
    kept = pictures_of(path) / f"{page}-{size}-{turn % 360}.webp"
    if kept.exists():
        return kept.read_bytes(), "image/webp"
    got = page_png(path, page, size, turn)
    if got:
        kept.parent.mkdir(parents=True, exist_ok=True)
        part = kept.with_name(f"{kept.name}.{threading.get_ident()}.part")
        part.write_bytes(got[0])
        os.replace(part, kept)
        if not Path(path).exists():   # the manual was deleted while its page was being drawn
            kept.unlink(missing_ok=True)
    return got


_ahead: queue.Queue[tuple[int, int]] = queue.Queue()


def draw_ahead(fid: int) -> None:
    """Draw a manual's pictures before anyone asks, so a drop opened from far away does not wait
    for each page to be drawn: the list pictures of the pages the screen shows (to check first),
    then, once every manual queued before has its list pictures, the enlarged view of each page
    still to check."""
    _ahead.put((fid, 0))


def _drawing() -> None:
    if os.name == "nt":   # behind the screen's own requests, and the PC's other work
        ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), -1)
    while True:
        fid, size = _ahead.get()
        try:
            with store.db() as con:
                row = con.execute("SELECT path FROM files WHERE id=? AND status='done' AND path IS NOT NULL",
                                  (fid,)).fetchone()
                pages = [r["page"] for r in con.execute(
                    "SELECT page FROM pages WHERE file_id=? AND (bucket != 'no' OR label IS NOT NULL) "
                    "ORDER BY bucket != 'check', page" if size == 0 else
                    "SELECT page FROM pages WHERE file_id=? AND bucket = 'check' AND label IS NULL ORDER BY page", (fid,))]
            if row and store.pdf(row["path"]).exists():
                for n in pages:
                    page_picture(store.pdf(row["path"]), n, size)
                if size == 0:
                    _ahead.put((fid, 1))
        except Exception as exc:   # a manual deleted meanwhile, say: its pages are drawn when asked for
            print(f"drawing the pages of manual {fid} ahead failed: {exc}", flush=True)


def start_drawing() -> None:
    """Draw ahead the manuals of the open drops (the newest first), then each manual as its
    reading finishes (jobs.py)."""
    with store.db() as con:
        for r in con.execute("SELECT f.id FROM files f JOIN jobs j ON j.id = f.job_id WHERE j.archived IS NULL "
                             "AND j.deleted IS NULL AND f.status = 'done' AND f.path IS NOT NULL ORDER BY j.id DESC, f.id"):
            _ahead.put((r["id"], 0))
    threading.Thread(target=_drawing, daemon=True).start()


def drop_name(rels: list[str]) -> str:
    """What a drop is called on the screen: the folder it was dropped as, its one
    manual's name, or the first manual's name and how many more."""
    tops = {rel.split("/")[0] for rel in rels}
    if rels and len(tops) == 1 and all("/" in rel for rel in rels):
        return tops.pop()
    names = sorted(PurePosixPath(rel).name for rel in rels)
    return names[0] if len(names) == 1 else f"{names[0]} and {len(names) - 1} more"


def page_count(path) -> int:
    """How many pages a PDF has; 0 if it cannot be opened."""
    try:
        with _PDF, fitz.open(path) as doc:
            return 0 if doc.needs_pass else doc.page_count
    except Exception:
        return 0


def parse_pages(text: str, most: int = 100_000) -> set[int]:
    """A page list as people write it, e.g. "4-6,20-21,37,1171 (Type not Conform
    1171-1185 Type 3K140-3A_1),1297, dont take Double Sparts": page numbers and ranges;
    notes in brackets and words are ignored. A range is never filled in past `most` (no
    manual is that long; its end is kept, to be reported as outside the manual), so
    "1-2000000000" cannot fill the server's memory."""
    text = re.sub(r"\([^)]*\)", " ", str(text))
    pages = set()
    for first, last in re.findall(r"(?<![\w-])(\d{1,9})(?:\s*-\s*(\d{1,9}))?(?![\w-])", text):
        start, end = int(first), int(last or first)
        pages.update(range(start, min(end, most) + 1))
        if end > most:
            pages.add(end)
    return pages


def chosen(con, fid: int) -> list[int]:
    """The pages of a manual that go in the downloads."""
    return [r["page"] for r in con.execute(
        "SELECT page FROM pages WHERE file_id=? AND COALESCE(label, bucket = 'spares') "
        "ORDER BY page", (fid,))]


def standing(con, fid: int) -> dict[int, int | None]:
    """{page: 1 included, 0 left out, None still to check} for the pages the tool or a
    person said anything about; any other page is left out."""
    return {r["page"]: r["label"] if r["label"] is not None else 1 if r["bucket"] == "spares"
            else None if r["bucket"] == "check" else 0
            for r in con.execute("SELECT page, bucket, label FROM pages WHERE file_id=?", (fid,))}


def scratch(suffix: str) -> Path:
    """A new, empty file for one download. Each request gets its own, so two colleagues
    downloading at once never receive each other's half-written file."""
    handle, name = tempfile.mkstemp(suffix=suffix, dir=store.DATA / "out")
    os.close(handle)
    return Path(name)


def spares_pdf(src: str, pages: list[int]) -> Path:
    """A PDF of just these pages of a manual."""
    dest = scratch(".pdf")
    with _PDF, fitz.open(src) as doc:
        doc.select([page - 1 for page in pages])
        doc.save(dest, garbage=3, deflate=True)
    return dest


# The band on top of each page of a tagged manual: what it says, and its colour.
TAGS = {1: ("INCLUDED", (0.09, 0.45, 0.27)), 0: ("LEFT OUT", (0.42, 0.45, 0.5)),
        None: ("LEFT OUT - NOT CHECKED", (0.72, 0.42, 0.04))}


def _upright(page) -> None:
    """Set a turned page upright in its content, showing the same part of it."""
    media, crop, turn = page.mediabox, page.cropbox, page.rotation   # the crop in top-left coordinates
    w, h = media.width, media.height
    x0, y0, x1, y1 = crop.x0 - media.x0, crop.y0, crop.x1 - media.x0, crop.y1
    shown = {90: (h - y1, x0, h - y0, x1), 180: (w - x1, h - y1, w - x0, h - y0), 270: (y0, w - x1, y1, w - x0)}[turn % 360]
    page.set_cropbox(fitz.Rect(media.x0, 0, media.x1, h))   # uncropped, or remove_rotation moves the content
    page.remove_rotation()
    media = page.mediabox
    page.set_cropbox(fitz.Rect(shown) + (media.x0, 0, media.x0, 0) & fitz.Rect(media.x0, 0, media.x1, media.height))


def _band(page, words: str, colour, number: int, total: int) -> None:
    """Make the page taller by a band on top and write in it. Below the band the page
    looks exactly as before: a turned page is set upright first, and its crop is kept."""
    if page.rotation % 360:
        _upright(page)
    media = page.mediabox
    crop = page.cropbox & fitz.Rect(media.x0, 0, media.x1, media.height)   # a crop wider than the page is cut to it
    band = round(max(18, min(crop.width, crop.height) * 0.04))   # A4 24 points; an A0 drawing's band reads zoomed out
    page.set_mediabox(fitz.Rect(media.x0, media.y0, media.x1, media.y1 + band))   # which drops the crop
    grown = page.mediabox   # as written to the file, rounded
    page.set_cropbox(fitz.Rect(crop.x0, crop.y0, min(crop.x1, grown.x1), min(crop.y1 + band, grown.height)))
    top = page.rect
    size = band * 0.5
    page.draw_rect(fitz.Rect(top.x0, top.y0, top.x1, top.y0 + band), color=None, fill=colour, overlay=True)
    page.insert_text((top.x0 + band * 0.5, top.y0 + band * 0.68), words, fontsize=size, fontname="hebo", color=(1, 1, 1))
    where = f"PAGE {number} OF {total}"
    page.insert_text((top.x1 - band * 0.5 - fitz.get_text_length(where, "helv", size), top.y0 + band * 0.68), where,
                     fontsize=size, fontname="helv", color=(1, 1, 1))


def tagged_pdf(src: str, pages: dict[int, int | None]) -> Path:
    """The whole manual, each page under a band saying what became of it (TAGS); `pages`
    as standing() gives them. The lock is let go between pieces, so pages keep coming to
    the screen while a vessel's zip is made."""
    dest = scratch(".pdf")
    with _PDF:
        doc = fitz.open(src)
    try:
        for first in range(0, doc.page_count, 25):
            with _PDF:
                for n in range(first, min(first + 25, doc.page_count)):
                    _band(doc[n], *TAGS[pages.get(n + 1, 0)], n + 1, doc.page_count)
        with _PDF:
            doc.save(dest, garbage=1, deflate=True)
    finally:
        with _PDF:
            doc.close()
    return dest


# ---- the allocation sheet -------------------------------------------------------
# The sheet the typists work from, as D:\AllocationTool makes it (its colours and layout), with
# the typing dates and Done the team added and no Type column (2026-11-20), and Pages and DATA
# filled in: DATA the pages included, written as the typists write
# them ("1-3,5-11"), or "No Data". A note there has no figures in it, so that reading the
# sheet back (learn.spreadsheet_pages) can never take it for a page list. Pages still to check
# go in a last column, Note: Excel's cell notes (comments) stopped VS Code's viewer opening it.
# A browser tells the page where a dropped file sits only inside a dropped folder.
NO_FOLDER = "(dropped without its folder)"
ALLOCATION = ["Path", "File", "Pages", "DATA", "Typing Name", "Rows", "Typing Start Date", "Typing End Date",
              "Typing Done", "QC Name", "QC Rows", "Equipment Name", "Note"]
WIDTHS = {"Path": 55, "File": 42, "Pages": 18, "DATA": 22, "Typing Name": 18, "Rows": 10,
          "Typing Start Date": 19, "Typing End Date": 18, "Typing Done": 14,
          "QC Name": 18, "QC Rows": 10, "Equipment Name": 38, "Note": 40}
ACCENT, ACCENT_LIGHT, ZEBRA, FLAG_FILL, FLAG_TEXT, BORDER = "1E6F82", "EAF3F4", "F5F5F5", "FCEFC7", "8A5A00", "D9D9D9"
MADE_BY = "Spares Page Finder"   # the sheet's author, so learning knows its pages are the tool's own answers
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")   # what openpyxl refuses to write in a cell


def page_ranges(pages) -> str:
    """[1, 2, 3, 5, 6, 9] -> "1-3,5-6,9", as the typists write a manual's pages."""
    runs, pages = [], sorted(pages)
    for n in pages:
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ",".join(f"{a}-{b}" if b > a else str(a) for a, b in runs)


def equipment_name(path) -> str | None:
    """The biggest line of text on page 1, 13 points or more: the AllocationTool's guess at
    what the manual is about. None when page 1 has no such line (a scan, say). Text only:
    with its pictures decoded too, a vessel of 226 manuals took 175 s instead of 3 s."""
    try:
        with _PDF, fitz.open(path) as doc:
            info = doc[0].get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP) \
                if doc.page_count and not doc.needs_pass else {}
    except Exception:
        return None
    best, size = None, 0.0
    for block in info.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if spans and 3 <= len(text) <= 150 and not re.fullmatch(r"[\d\s.\-/,:]+", text):
                if max(span.get("size", 0.0) for span in spans) > size:
                    best, size = text, max(span.get("size", 0.0) for span in spans)
    return _ILLEGAL.sub("", best) if best and size >= 13.0 else None


def allocation_name(job: int) -> str:
    with store.db() as con:
        rels = [r["rel"] for r in con.execute("SELECT rel FROM files WHERE job_id=?", (job,))]
    name = re.sub(r'[\\/:*?"<>|]', "_", re.sub(r"\.pdf\b", "", drop_name(rels), flags=re.I) if rels else "drop")
    return f"Allocation Sheet - {name} - {datetime.date.today().isoformat()}.xlsx"


def allocation(job: int) -> Path:
    """Every manual of a drop, one row each in the folders it was dropped in: its pages, and
    in DATA the pages included or "No Data". Pages still to check are not in DATA: that cell
    is marked, and Note says which they are. A manual not read says why, marked the same way."""
    with store.db() as con:
        files = con.execute("SELECT id, rel, path, status, note, pages_total FROM files WHERE job_id=?", (job,)).fetchall()
        picked = {f["id"]: chosen(con, f["id"]) for f in files if f["status"] == "done"}
        waiting = {f["id"]: [r["page"] for r in con.execute(
            "SELECT page FROM pages WHERE file_id=? AND bucket='check' AND label IS NULL ORDER BY page", (f["id"],))]
            for f in files if f["status"] == "done"}
    files = sorted(files, key=lambda f: f["rel"].lower())
    name = drop_name([f["rel"] for f in files]) if files else "Empty drop"
    book = Workbook()
    ws = book.active
    ws.title = "Allocation Sheet"
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.tabColor = ACCENT
    book.properties.title, book.properties.creator = f"Allocation Sheet - {_ILLEGAL.sub('', name)}", MADE_BY
    book.properties.subject = "Spares pages of a drop"
    last = get_column_letter(len(ALLOCATION))
    thin = Side(style="thin", color=BORDER)
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    left = Alignment(horizontal="left", vertical="center", indent=1)
    right = Alignment(horizontal="right", vertical="center", indent=1)
    flag = lambda cell: (setattr(cell, "fill", PatternFill("solid", fgColor=FLAG_FILL)),
                         setattr(cell, "font", Font(color=FLAG_TEXT)))

    rows, flagged = [], 0
    for f in files:
        folder = PurePosixPath(f["rel"]).parent.as_posix()
        pages = f["pages_total"] or None
        data, note, marked = None, None, False
        if f["status"] == "done":
            data = page_ranges(picked[f["id"]]) or "No Data"
            if waiting[f["id"]]:
                marked, note = True, f"Still to check: {page_ranges(waiting[f['id']])}"
        elif f["status"] == "skipped":
            reason = "too large" if "GB" in (f["note"] or "") else (f["note"] or "by its name")
            data, marked = f"Not read: {reason}", True
        elif f["status"] == "failed":
            data, marked = "Could not be read", True
        else:
            data, marked = "Not read yet", True
        flagged += marked
        rows.append(([NO_FOLDER if folder == "." else folder, PurePosixPath(f["rel"]).name, pages, data,
                      None, None, None, None, None, None, None, equipment_name(store.pdf(f["path"])) if f["path"] and store.pdf(f["path"]).exists() else None,
                      note], marked))

    ws.merge_cells(f"A1:{last}1")
    ws["A1"] = "Allocation Sheet"
    ws.row_dimensions[1].height = 30
    for column in range(1, len(ALLOCATION) + 1):
        ws.cell(row=1, column=column).fill = PatternFill("solid", fgColor=ACCENT)
        ws.cell(row=2, column=column).fill = PatternFill("solid", fgColor=ACCENT_LIGHT)
    ws["A1"].font, ws["A1"].alignment = Font(size=18, bold=True, color="FFFFFF"), left
    ws.merge_cells(f"A2:{last}2")
    total = sum(f["pages_total"] or 0 for f in files)
    marked_pages = sum(len(v) for v in picked.values())
    subtitle = (f"{_ILLEGAL.sub('', name)}   \u2022   {len(files)} files   \u2022   {total:,} pages total   \u2022   "
                f"{marked_pages:,} spares pages marked")
    if flagged:
        subtitle += f"   \u2022   {flagged} flagged"
    ws["A2"] = subtitle + f"   \u2022   Generated {datetime.date.today().isoformat()} by {MADE_BY}"
    ws.row_dimensions[2].height = 20
    ws["A2"].font, ws["A2"].alignment = Font(size=10, italic=True, color=ACCENT), left
    for column, text in enumerate(ALLOCATION, start=1):
        cell = ws.cell(row=3, column=column, value=text)
        cell.font, cell.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor=ACCENT)
        cell.alignment, cell.border = left, border
    ws.row_dimensions[3].height = 20

    for i, (values, marked) in enumerate(rows):
        row = 4 + i
        for column, value in enumerate(values, start=1):
            cell = ws.cell(row=row, column=column, value=_ILLEGAL.sub("", value) if isinstance(value, str) else value)
            cell.border, cell.alignment = border, left
            if isinstance(value, str):
                cell.data_type = "s"   # always text: a file named "=..." must not become a formula
            if i % 2:
                cell.fill = PatternFill("solid", fgColor=ZEBRA)
        pages = ws.cell(row=row, column=3)
        if isinstance(values[2], int):
            pages.alignment, pages.number_format = right, "#,##0"
        data = ws.cell(row=row, column=4)
        if marked:
            flag(data)
    ws.freeze_panes = "A4"
    ws.auto_filter.ref = f"A3:{last}{3 + max(len(rows), 1)}"
    for column, header in enumerate(ALLOCATION, start=1):
        ws.column_dimensions[get_column_letter(column)].width = WIDTHS[header]
    dest = scratch(".xlsx")
    book.save(dest)
    return dest


def bundle(job: int, whole: bool = True, progress=lambda done, total: None) -> Path:
    """A zip of every manual read, in the folders it was dropped in, and the allocation sheet:
    each manual whole and tagged, or (`whole` False, for an archived drop) only the pages
    included. progress(manuals done, manuals) as it goes, for the screen."""
    dest = scratch(".zip")
    with store.db() as con:
        done = con.execute("SELECT id, path, rel FROM files WHERE job_id=? AND status='done' "
                           "ORDER BY rel", (job,)).fetchall()
        picked = [(f, standing(con, f["id"]) if whole else chosen(con, f["id"])) for f in done]
    with zipfile.ZipFile(dest, "w") as box:
        for i, (f, pages) in enumerate(picked):
            progress(i, len(picked))
            if (pages or whole) and store.pdf(f["path"]).exists():
                part = tagged_pdf(store.pdf(f["path"]), pages) if whole else spares_pdf(store.pdf(f["path"]), pages)
                rel = PurePosixPath(f["rel"])
                box.write(part, str(rel.with_name(f"{rel.stem} - {'tagged' if whole else 'spares'}.pdf")))
                part.unlink()
        progress(len(picked), len(picked))   # then the allocation sheet
        made = allocation(job)
        box.write(made, allocation_name(job))
        made.unlink()
    return dest
