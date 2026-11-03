"""Which PDFs are worth opening, judged by filename alone.

The skip list started as D:\\pms\\pms\\intake.py's and keeps its stance: a file
that is skipped is always shown on screen with "include anyway", because
dropping a real manual is worse than reading a useless one.
"""
import re

_SKIP = [  # (pattern, reason shown to the user)
    # Lighting is not taken, even a "spare part list" (asked for on 2026-11-03; one vessel's
    # allocation sheet marks all 17 of its lighting manuals "we don't take this Spares" or
    # "No Data"). Older work did take some ("MARINE LIGHTING", "LIGHTING FIXTURE", signal
    # lamps), so the file still shows, with "include anyway".
    # Not "lamp": "Suez Canal Lamp Crane" is a crane, and UV lamps are water-treatment spares.
    (r"lighting|\blight (fittings?|fixtures?|signal|column)|signal (lights?|column)"
     r"|(navigation|side|search|flood) ?lights?\b", "lighting"),
    # "certif..." also catches yards' misspellings ("MISCELLANEOUS CERTIFCATES"). The
    # wordings added on 2026-10-26 were 13 BERGE DAISEN files, 4.8 minutes and 359 pages
    # to check, with no spares page; across D:\KIRAN they skip 468 more files, none of
    # them a manual (the one that says "manual" is an oil purifier's test report).
    (r"eiapp|nox tech|technical file|\bcert(ificate)?s?\b|certif\w*|certf\w*", "certificate"),
    (r"test record|shop (trial|test)|sea trial|test results?|test reports?|record of inspection"
     r"|inspection (and )?test", "test record"),
    # The ship's own stability papers. On BERGE DAISEN these 8 files were 55 of the
    # 167 minutes of reading and 777 pages to check, with no spares page among them.
    # Checked against the 138,184 PDF names under D:\KIRAN: 348 match, all of them
    # stability or tank documents. "sounding" alone would also catch sounding-pipe
    # drawings and valve manuals, hence the full phrases.
    (r"(sounding|ullage|gauge|tank|capacity|calibration) tables?|sounding (table )?(manual|book)"
     r"|loading (manual|guidance)|stability (booklet|book|manual|information)", "stability document"),
    # Wiring diagrams count as drawings, so _BOOKLET still saves "Catalogue, Instruction
    # Manual & Wiring Diagram of Gallery Equipment" and the like.
    (r"\bdrawings?\b|\bdwgs?\b|\bga\b|general arrangement|\bplans?\b|(connection|wiring) diagrams?",
     "drawing or plan"),
    (r"packing list|invoice|purchase order", "commercial document"),
    (r"radar|ecdis|gyro|auto ?pilot|\bd?gps\b|\bais\b|\bvdr\b|echo ?sounder|speed log"
     r"|gmdss|navtex|\bvhf\b|mf ?hf|\bradio|inmarsat|epirb|\bsart\b|bnwas|security alert"
     r"|navigation", "navigation manual"),
    # These makers also build machinery systems, so the name alone only raises the question.
    (r"furuno|\bjrc\b|sperry|ansch(ue?|ü)tz|simrad|tokyo keiki|\bsailor\b|transas|danelec|koden",
     "navigation maker"),
]
# Yards call a maker's whole booklet "FINAL DRAWINGS" (D:\KIRAN\SOECHI LINES has a
# 122-page "01 FINAL DRAWINGS MAIN DIESEL GENERATOR ENGINE"), and those carry parts lists.
_BOOKLET = r"final|finished|complete|manual|instruction|spare|parts"


def skip_reason(filename: str) -> str:
    """Why this file is not read, or "" to read it."""
    name = re.sub(r"[\W_]+", " ", filename.lower().removesuffix(".pdf"))
    reason = next((reason for pattern, reason in _SKIP if re.search(pattern, name)), "")
    return "" if reason == "drawing or plan" and re.search(_BOOKLET, name) else reason
