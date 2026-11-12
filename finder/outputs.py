"""What the server hands out: page pictures for the screen, and the downloads: each
manual whole, every page under a band saying whether it was included, the Excel list,
and a zip of both.

The pages included are what the tool marked spares, as corrected by people. Pages still
waiting under "To check" are left out (and listed in the Excel's "Needs attention"), and
so are the pages the picture model thinks are probably not spares, unless a person
included them. An archived drop keeps only each manual's included pages (archive.py).
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath

import pymupdf as fitz
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from finder import store

_PDF = threading.Lock()   # PyMuPDF is not thread-safe


# The sizes a page is drawn at, as (dots per inch, most pixels a side): for the list, the
# enlarged view, and zoomed into a large drawing. The cap matters: a PDF can declare a
# page 200 inches wide, and drawing that at full size would exhaust the memory. An A0
# sheet at the zoom size is 7,000 pixels wide, 150 dpi: small print reads.
PAGE_SIZES = {0: (60, 1000), 1: (150, 3000), 2: (200, 7000)}


def page_png(path: str, page: int, size: int, turn: int) -> tuple[bytes, str] | None:
    """(picture, media type) of one page at one of PAGE_SIZES, turned by `turn` degrees
    (for tables scanned sideways); None if there is no such page. The zoom size is a
    JPEG, a quarter the bytes of a PNG for a scanned sheet; the others are PNG."""
    with _PDF, fitz.open(path) as doc:
        if not 1 <= page <= doc.page_count or turn % 90:
            return None
        shown = doc[page - 1]
        dpi, most = PAGE_SIZES.get(size, PAGE_SIZES[0])
        scale = min(dpi / 72, most / max(shown.rect.width, shown.rect.height, 1))
        pix = shown.get_pixmap(matrix=fitz.Matrix(scale, scale).prerotate(turn), alpha=False)
        return (pix.tobytes("jpeg", jpg_quality=85), "image/jpeg") if size == 2 else (pix.tobytes("png"), "image/png")


def page_list(pages) -> str:
    """[1, 2, 3, 7] -> "1, 2, 3, 7". Parsel's page box reads this back as it is."""
    return ", ".join(map(str, pages))


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


def excel(job: int) -> Path:
    """Sheet "Spares pages": each manual's file name (without the folders it was dropped
    in), and beside it its spares pages, comma-separated. Sheet "Needs attention" exists
    only when something is outstanding: pages nobody has checked, or manuals not read."""
    book = Workbook()
    listing = book.active
    listing.title = "Spares pages"
    listing.append(["Manual", "Spares pages"])
    outstanding = []
    with store.db() as con:
        for f in con.execute("SELECT id, rel, status, note FROM files WHERE job_id=? ORDER BY rel", (job,)).fetchall():
            name = PurePosixPath(f["rel"]).name
            if f["status"] != "done":
                outstanding.append([name, {"skipped": f"Not read: {f['note']}",
                                           "failed": f"Could not be read: {f['note']}",
                                           "stopped": "Stopped before it was read",
                                           "unsent": "Not sent: drop the folder again to finish",
                                           }.get(f["status"], "Not finished yet")])
                continue
            listing.append([name, page_list(chosen(con, f["id"])) or None])   # None: an empty cell
            waiting = [r["page"] for r in con.execute(
                "SELECT page FROM pages WHERE file_id=? AND bucket='check' AND label IS NULL ORDER BY page", (f["id"],))]
            if waiting:
                outstanding.append([name, "Pages still to check: " + page_list(waiting)])
    sheets = [listing]
    if outstanding:
        sheets.append(book.create_sheet("Needs attention"))
        sheets[1].append(["Manual", "What is outstanding"])
        for row in outstanding:
            sheets[1].append(row)
    for sheet in sheets:
        sheet.freeze_panes = "A2"
        sheet.column_dimensions["A"].width = 60
        sheet.column_dimensions["B"].width = 90
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="0F2D4A")
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                if cell.value is not None:
                    cell.data_type = "s"   # always text: a file named "=..." must not become a formula
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    dest = scratch(".xlsx")
    book.save(dest)
    return dest


def bundle(job: int, whole: bool = True) -> Path:
    """A zip of every manual read, in the folders it was dropped in, and the Excel list:
    each manual whole and tagged, or (`whole` False, for an archived drop) only the pages
    included."""
    dest = scratch(".zip")
    with store.db() as con:
        done = con.execute("SELECT id, path, rel FROM files WHERE job_id=? AND status='done' "
                           "ORDER BY rel", (job,)).fetchall()
        picked = [(f, standing(con, f["id"]) if whole else chosen(con, f["id"])) for f in done]
    with zipfile.ZipFile(dest, "w") as box:
        for f, pages in picked:
            if (pages or whole) and store.pdf(f["path"]).exists():
                part = tagged_pdf(store.pdf(f["path"]), pages) if whole else spares_pdf(store.pdf(f["path"]), pages)
                rel = PurePosixPath(f["rel"])
                box.write(part, str(rel.with_name(f"{rel.stem} - {'tagged' if whole else 'spares'}.pdf")))
                part.unlink()
        part = excel(job)
        box.write(part, "Spares pages.xlsx")
        part.unlink()
    return dest
