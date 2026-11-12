"""Which PDFs are worth opening, judged by filename alone.

The skip list started as D:\\pms\\pms\\intake.py's and keeps its stance: a file
that is skipped is always shown on screen with "include anyway", because
dropping a real manual is worse than reading a useless one.

Checked on 2026-11-12 against every finished vessel on the office PC: 1,666 file names
the spreadsheets took spares pages from, and 97,767 in the same vessels' folders they
took none from. A rule stays only if it skips (almost) none of the first. The rules and
words added that day skip none of the first and about 1,950 of the second; the navigation
rules, which skipped 33 of the first, were dropped. Still skipped with spares: 14 drawings
named only "drawings" or "plan" (of 5,229 such files), 5 lighting manuals (lighting is not
taken), 3 certificates and NOx technical files: 22, where the rules before skipped 77.
Of those 22, the four that were chapters of a maker's manual ("14) Drawings.pdf" among its
other numbered chapters) are read now, at the cost of 79 such chapters without spares
(about 1,300 pages on the whole PC). Every other rule tried for the rest read 30 to 200
files without spares for each one with: management plans, door plans, approval drawings,
NOx technical files.
"""
import re
from collections import defaultdict
from pathlib import PurePosixPath

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
    # Not "certifugal": a yard's misspelling of a centrifugal pump.
    (r"eiapp|nox tech|technical file|\bcert(ificate)?s?\b|certif(?!ug)\w*|certf\w*", "certificate"),
    # Not "... with test result": a manual with its test results bound in.
    (r"test record|shop (trial|test)|sea trial|(?<!with )test results?|test reports?|record of inspection"
     r"|inspection (and )?test|commissioning (reports?|records?)", "test record"),
    (r"\bdeclarations?\b|\bconformity\b|hazardous materials?|\bihm\b|green passport|asbestos", "declaration"),
    (r"msds|safety data sheets?", "safety data sheet"),
    (r"\bcheck ?lists?\b|\bforms?\b|\bletters?\b|\bsigned\b|\banalysis\b|maintenance schedules?",
     "form, letter or report"),
    # The ship's own stability papers. On BERGE DAISEN these 8 files were 55 of the
    # 167 minutes of reading and 777 pages to check, with no spares page among them.
    # Checked against the 138,184 PDF names under D:\KIRAN: 348 match, all of them
    # stability or tank documents. "sounding" alone would also catch sounding-pipe
    # drawings and valve manuals, hence the full phrases.
    (r"(sounding|ullage|gauge|tank|capacity|calibration) tables?|sounding (table )?(manual|book)"
     r"|loading (manual|guidance)|stability (booklet|book|manual|information)", "stability document"),
    # Wiring diagrams count as drawings, so _BOOKLET still saves "Catalogue, Instruction
    # Manual & Wiring Diagram of Gallery Equipment" and the like.
    (r"\bdrawings?\b|\bdwgs?\b|\bga\b|general arrangement|\bplans?\b|(connection|wiring) diagrams?"
     r"|\blayouts?\b|\bfoundations?\b|\bmimic\b|\bp ?id\b|\bp and id\b", "drawing or plan"),
    (r"packing list|invoice|purchase order", "commercial document"),
    # Navigation manuals (radar, gyro, autopilot, echo sounder, GMDSS...) and their makers'
    # names were skipped until 2026-11-12, but six clients' spreadsheets took spares from 33
    # of them (all 14 on each of one client's vessels): they are read.
]
# Yards call a maker's whole booklet "FINAL DRAWINGS" (D:\KIRAN\SOECHI LINES has a
# 122-page "01 FINAL DRAWINGS MAIN DIESEL GENERATOR ENGINE"), and those carry parts lists.
# So do their "working", "finish" and "completion" drawings (17 of them had spares pages),
# and a maker's chapters of "drawings & illustrations", "specs & dwgs", "attached drawings",
# outline and modification drawings (5 had spares pages, 77 on one vessel; 43 files without).
_BOOKLET = (r"final|finish|complet(e|ion)|working|manual|instruction|spare|parts|part lists?"
            r"|illustrations?|attached|specs?\b|specifications?|outline|modification")
# Papers bound with a manual are read too: "instruction manual with declaration of conformity".
_PAPERS = {"declaration", "safety data sheet", "form, letter or report"}
_BOUND = r"manual|instruction|spare|parts|part lists?|catalogue"


# A maker's manual split into numbered chapters, "13) Spare parts.pdf", "14) Drawings.pdf":
# its drawings chapter is read like the rest of the manual.
_CHAPTER = re.compile(r"^\s*\d{1,3}\s*\)")


def skip_reason(filename: str, beside: list[str] = ()) -> str:
    """Why this file is not read, or "" to read it. `beside`: the names of the other files
    dropped in the same folder."""
    name = re.sub(r"[\W_]+", " ", filename.lower().removesuffix(".pdf"))
    reason = next((reason for pattern, reason in _SKIP if re.search(pattern, name)), "")
    if reason == "drawing or plan" and re.search(_BOOKLET, name) or reason in _PAPERS and re.search(_BOUND, name):
        return ""
    if reason == "drawing or plan" and _CHAPTER.match(filename):
        chapters = [other for other in beside if other != filename and _CHAPTER.match(other)]
        if len(chapters) >= 3 and any(not skip_reason(other) for other in chapters):
            return ""
    return reason


def skip_reasons(rels: list[str]) -> list[str]:
    """skip_reason for each file of a drop, given as paths in the folder dropped."""
    folders = defaultdict(list)
    for rel in rels:
        folders[PurePosixPath(rel).parent].append(PurePosixPath(rel).name)
    return [skip_reason(PurePosixPath(rel).name, folders[PurePosixPath(rel).parent]) for rel in rels]
