"""What a page's text layer says: its words, and which parts-table columns Parsel's two
table passes find in it. The page model (learn.py) learns from, and scores, exactly this.
"""
from __future__ import annotations

import os

import pymupdf as fitz

from engine.pdf_extractor import _extract_pymupdf_page, _extract_textblocks_page, score_dataframe
from finder import pool, tables

# A text layer shorter than this does not describe the page: a stamp, a page number,
# scanner noise over a tool plate, or a footer over a drawn parts list ("Page 41
# Reciprocating Compressors...", 168 characters, on 11 Grasso parts pages of H1868 that
# the model left out). Such pages are judged by their picture, not by the model.
USABLE_TEXT = 500
# Big manuals' text layers are read by this many processes at once: 3.5x faster with 4
# on 1,814 BERGE DAISEN pages, with the same answers. Smaller ones are read here.
TEXT_WORKERS = int(os.environ.get("FINDER_TEXT_WORKERS", 4))
PARALLEL_FROM = 40   # text pages

_POOL = pool.Pool(TEXT_WORKERS)


def page_reading(page) -> dict:
    """{"text": the page's words, "signs": the parts-table columns of the better table
    from Parsel's quick text-block pass and PyMuPDF's table finder (the second only when
    the first is not confident), "confidence": that table's header score, None if no
    table}, as Parsel's extractor reads a text page."""
    best, table = float("-inf"), None
    for extract in (_extract_textblocks_page, _extract_pymupdf_page):
        df, _ = extract(page)
        if not df.empty and (score := score_dataframe(df, tables.KNOWN)) > best:
            best, table = score, df
        if best >= tables.CONFIDENT:
            break
    return {"text": page.get_text("text"),
            "signs": sorted(tables.signs(table)) if table is not None else [],
            "confidence": best if table is not None else None}


def reading(doc, number: int) -> dict | None:
    """page_reading of page `number`, or None if it could not be read."""
    try:
        return page_reading(doc[number - 1])
    except Exception:
        return None


def _part(path: str, pages: list[int]) -> dict[int, dict | None]:
    """One worker's share of a manual."""
    doc = pool.open_pdf(path)
    return {number: reading(doc, number) for number in pages}


def read_pages(path, pages, progress) -> dict[int, dict | None]:
    """{page: reading} for these pages; progress(done, total) after each page or share."""
    pages = sorted(pages)
    got = {}
    if len(pages) < PARALLEL_FROM or TEXT_WORKERS < 2:
        with fitz.open(path) as doc:
            for number in pages:
                got[number] = reading(doc, number)
                progress(len(got), len(pages))
        return got
    for share in _POOL.each(_part, [(str(path), pages[k:k + 20]) for k in range(0, len(pages), 20)]):
        got.update(share)
        progress(len(got), len(pages))
    return got
