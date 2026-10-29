"""What the server hands out: page pictures for the screen, and the downloads, each
manual's spares pages as a PDF, the Excel list, and a zip of everything.

The pages that go out are what the tool marked spares, as corrected by people; pages
still waiting under "check" or "probably not spares" are left out (and listed in the
Excel's "Needs attention").
"""
from __future__ import annotations

import os
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath

import pymupdf as fitz
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from finder import store

_PDF = threading.Lock()   # PyMuPDF is not thread-safe


def page_png(path: str, page: int, big: bool, turn: int) -> bytes | None:
    """One page as a PNG, small for the list or big for the enlarged view, turned by
    `turn` degrees (for tables scanned sideways); None if there is no such page."""
    with _PDF, fitz.open(path) as doc:
        if not 1 <= page <= doc.page_count or turn % 90:
            return None
        shown = doc[page - 1]
        # 150 or 60 dpi, but never more than 3000 or 1000 pixels a side: a PDF can declare a
        # page 200 inches wide, and drawing that at full size would exhaust the memory.
        scale = min((150 if big else 60) / 72, (3000 if big else 1000) / max(shown.rect.width, shown.rect.height, 1))
        return shown.get_pixmap(matrix=fitz.Matrix(scale, scale).prerotate(turn)).tobytes("png")


def page_list(pages) -> str:
    """[1, 2, 3, 7] -> "1, 2, 3, 7". Parsel's page box reads this back as it is."""
    return ", ".join(map(str, pages))


def chosen(con, fid: int) -> list[int]:
    """The pages of a manual that go in the downloads."""
    return [r["page"] for r in con.execute(
        "SELECT page FROM pages WHERE file_id=? AND COALESCE(label, bucket = 'spares') "
        "ORDER BY page", (fid,))]


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
            for bucket, what in (("check", "Pages still to check: "),
                                 ("unlikely", "Pages probably not spares, nobody has looked at yet: ")):
                waiting = [r["page"] for r in con.execute(
                    "SELECT page FROM pages WHERE file_id=? AND bucket=? AND label IS NULL "
                    "ORDER BY page", (f["id"], bucket))]
                if waiting:
                    outstanding.append([name, what + page_list(waiting)])
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


def bundle(job: int) -> Path:
    """A zip of every manual's spares PDF, in the folders it was dropped in, and the Excel list."""
    dest = scratch(".zip")
    with store.db() as con:
        done = con.execute("SELECT id, path, rel FROM files WHERE job_id=? AND status='done' "
                           "ORDER BY rel", (job,)).fetchall()
        picked = [(f, chosen(con, f["id"])) for f in done]
    with zipfile.ZipFile(dest, "w") as box:
        for f, pages in picked:
            if pages and store.pdf(f["path"]).exists():
                part = spares_pdf(store.pdf(f["path"]), pages)
                rel = PurePosixPath(f["rel"])
                box.write(part, str(rel.with_name(f"{rel.stem} - spares.pdf")))
                part.unlink()
        part = excel(job)
        box.write(part, "Spares pages.xlsx")
        part.unlink()
    return dest
