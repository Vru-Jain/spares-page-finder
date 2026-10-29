"""The reader: decides every page of a manual, in the reader process the server keeps.

For each manual:
  1. census: which pages carry a text layer, and which are pictures
  2. every text layer is read (text.py)
  3. the page model (learn.py) scores each page whose text describes it: a high score
     is spares, a low score is not spares, anything between is unsure
  4. pictures are read (pictures.py) for the unsure pages, which can only be confirmed
     as spares or go to a person, and for pages whose text does not describe them,
     where the picture decides
  5. the picture model (learn.py, looks.py) looks again at the pages the picture left
     to a person: those it says are probably not spares pages go to "unlikely", which
     a person still sees, apart from the rest
Without a trained model, the fixed column rule (tables.py) decides text pages and every
picture page is read. A page that could not be read is never reported as a page
without spares: it goes to "check", or the whole manual fails.

    python -m finder.classify "manual.pdf"     one manual, to try things out
    python -m finder.classify --serve          the server's reader (see jobs.py)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import pymupdf as fitz

from finder import learn, looks, pictures, tables, text
from finder.tables import CHECK, NO, SPARES, UNLIKELY


def census(path) -> tuple[set[int], set[int], int]:
    """(pages with a text layer, pages that are pictures, page count). A page is in
    both when it is mostly a picture yet carries text: a scan with a stamp on it, or a
    scanner's own text layer. Reading only that text once hid page 13 of
    S1934_Alternator_Final.pdf, its spare parts list, behind a 55-character stamp."""
    least = int(tables.CONFIG["options"]["min_text_chars_per_page"])
    with_text, pictured = set(), set()
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise RuntimeError("This PDF is password-protected")
        for number, page in enumerate(doc, start=1):
            has_text = len(page.get_text("text").strip()) >= least
            if has_text:
                with_text.add(number)
            if not has_text or (sum(abs(fitz.Rect(image["bbox"])) for image in page.get_image_info())
                                >= 0.5 * abs(page.rect)):
                pictured.add(number)
        return with_text, pictured, doc.page_count


def classify_pdf(path, progress=None) -> dict[int, str]:
    """{page: spares | check | no} for every page of the PDF. progress(done, total)
    counts one run through the manual: text layers first, then the pictures needed."""
    progress = progress or (lambda *_: None)
    with_text, pictured, total = census(path)
    if not total:   # e.g. a download cut off before the PDF's page list
        raise RuntimeError("No pages could be found in it: the PDF is damaged or incomplete")
    pages = dict.fromkeys(range(1, total + 1), NO)
    work = [len(with_text) + len(pictured)]   # the total shrinks once it is known which pictures are needed
    readings = text.read_pages(path, with_text, lambda done, _: progress(done, work[0]))
    pages.update({n: CHECK for n, r in readings.items() if r is None})   # unreadable: a person looks
    model = learn.load()
    unsure = set()
    if model is None:   # the fixed rule, and every picture page is read
        pages.update({n: tables.rule(set(r["signs"]), r["confidence"]) for n, r in readings.items() if r})
        look = pictured
    else:
        usable = {n: r for n, r in readings.items() if r and len(r["text"].strip()) >= text.USABLE_TEXT}
        for number, chance in learn.score(model, usable).items():
            if chance >= model["spares_from"]:
                pages[number] = SPARES
            elif chance >= model["unsure_from"]:
                unsure.add(number)
        look = (set(pages) - set(usable)) | unsure   # text that does not describe the page, or unsure
    work[0] = len(with_text) + len(look)
    seen, numbers = pictures.read_pictures(path, look, lambda done, _: progress(len(with_text) + done, work[0]),
                                           quick=unsure)
    for number, found in seen.items():
        if number in unsure:   # a part-number table confirms it; otherwise a person looks
            pages[number] = SPARES if found == SPARES else CHECK
        else:
            pages[number] = max(pages[number], found, key=tables.strength)
    head = model and model.get("looks")
    doubtful = [n for n in numbers if pages[n] == CHECK]
    if head and doubtful:
        chance = learn.looks_score(head, [numbers[n] for n in doubtful], [seen[n] for n in doubtful])
        pages.update({n: UNLIKELY for n, c in zip(doubtful, chance) if c < head["cut"]})
    if numbers:   # kept, so people's marks on these pages can teach the picture model
        with open(path, "rb") as fh:
            looks.save(hashlib.file_digest(fh, "sha256").hexdigest(), {n: (seen[n], v) for n, v in numbers.items()})
    return pages


def serve() -> None:
    """Read manuals named on stdin, one path per line, keeping the worker processes from
    one to the next. Prints @progress lines, then @pages or @failed, for each."""
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            got = classify_pdf(path, lambda done, total: print(f"@progress {done} {total}", flush=True))
            print("@pages " + json.dumps(got), flush=True)
        except Exception as exc:
            print("@failed " + " ".join(str(exc).split())[:300], flush=True)


if __name__ == "__main__":   # the guard matters: worker processes import this module too
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        started = time.time()
        got = classify_pdf(sys.argv[1])
        for answer in (SPARES, CHECK, UNLIKELY):
            print(f"{answer}: {[p for p, b in got.items() if b == answer]}")
        print(f"{len(got)} pages in {time.time() - started:.0f} s")
