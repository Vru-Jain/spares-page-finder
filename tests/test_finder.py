"""Run with the tool's own Python:  .venv\\Scripts\\python.exe -m pytest"""
import hashlib
import io
import json
import os
import tempfile
import time
import zipfile

os.environ["FINDER_DATA"] = tempfile.mkdtemp()   # before `import app`: it opens its database on import
os.environ["FINDER_HOSTS"] = "testserver"         # the host name the test client uses
os.environ["FINDER_MODEL"] = os.path.join(os.environ["FINDER_DATA"], "none.joblib")   # the fixed rules, unless a test trains one
os.environ["FINDER_RETRAIN_HOUR"] = "-1"   # no retraining at night while testing

import numpy as np
import openpyxl
import pandas as pd
import pymupdf as fitz
from fastapi.testclient import TestClient

import app
from finder import archive, classify, learn, looks, outputs, pictures, retrain, store, tables, text, triage


# ---- deciding pages ----------------------------------------------------------

def test_triage_reads_manuals_and_skips_the_rest():
    for name in ["6EY22(A)LW 0AE22-G00103 Jun.2022-10.pdf",
                 "TURBOCHARGER MANUAL TPS48-61(NV50FPB-2.R10).pdf",
                 "47.MAIN AIR COMPRESSOR_Done-listed.pdf",
                 "Planned maintenance of purifier.pdf",
                 "01 CO2 SYSTEM FINAL DRAWINGS C3.02.54.55-01.pdf",   # a maker's booklet
                 "M32-1 FLOAT TYPE LEVEL GAUGE FINAL DRAWING (Inc. INSTRUCTION BOOK).pdf",
                 "Hammar H20 Hydrostatic release unit Manual (HT-0019  11.03).PDF",
                 "28 - Catalogue, Footing Details, Instruction Manual & Wiring Diagram of Gallery Equipment.pdf",
                 # navigation manuals are read: clients take their spares
                 "FURUNO FAR-2117 Radar Operator's Manual.PDF",
                 "03 250W MFHF RADIO EQUIPMENT MODEL JSS-296 JLZ50508 C3-02-54-64-03.pdf",
                 "JRC JLR-21 instruction.pdf", "AUTO PILOT & GYROCOMPASS -1.pdf"]:
        assert triage.skip_reason(name) == "", name
    for name, reason in {
        "S1934_GE_EIAPP Cert. & NOx Tech. File (#3663).pdf": "certificate",
        "S1934_Generator Engine_Fianl (with Test record).pdf": "test record",
        "HG-03 SARITA GENERAL ARRANGEMENT.pdf": "drawing or plan",
        "MISCELLANEOUS CERTIFCATES (H-PART).pdf": "certificate",   # the yard's spelling
        "TEST REPORT FOR APPROVAL.pdf": "test record",
        "RECORD OF INSPECTION & TEST DURING CONSTRUCTION.pdf": "test record",
        "E-7 CONNECTION DIAGRAM OF ELECTRIC EQUIPMENT.pdf": "drawing or plan",
        "B-05 SOUNDING TABLE (W.B.T. , F.W.T.).pdf": "stability document",
        "B-15 STABILITY BOOKLET WITH LOADING GUIDANCE.pdf": "stability document",
        "CARGO TANK CALIBRATION TABLES M.T.  JO PROVEL  - FINAL DRAWING.pdf": "stability document",
        "4) Lighting system spare part list.pdf": "lighting",   # not taken, even as a spare part list
        "EM-58 Light signal column.pdf": "lighting",
        "AY-04 Radar & signal mast (incl. side light rec.).pdf": "lighting",
    }.items():
        assert triage.skip_reason(name) == reason, name
    for name in ["Light fuel oil purifier.pdf", "H1-6-1_0,5t Suez Canal Lamp Crane&0,3t Provision Crane.pdf",
                 "AB A Datasheets Manuals UV Unit & UV Lamps.pdf", "E-30 PORTABLE DAYLIGHT SIGNALLING LAMP.pdf"]:
        assert triage.skip_reason(name) == "", name


def test_skip_rules_never_skip_files_their_names_show_have_spares():
    """Papers that never carry spares are skipped; manuals that only share a word with them
    are read (all of these had spares pages in a finished vessel's spreadsheet)."""
    for name, reason in {
        "01 Declaration of conformity.pdf": "declaration",
        "Inventory of Hazardous Materials (IHM).pdf": "declaration",
        "2481 Grease safety data sheet.pdf": "safety data sheet",
        "065 MSDS en.pdf": "safety data sheet",
        "01 Chiller unit checklist.pdf": "form, letter or report",
        "05-24 Hot layup service letter.pdf": "form, letter or report",
        "Electrical load analysis.pdf": "form, letter or report",
        "04 Maintenance schedule.pdf": "form, letter or report",
        "03 Commissioning report gen1.pdf": "test record",
        "0002 BWTS equipment layout.pdf": "drawing or plan",
        "Main genset foundation.pdf": "drawing or plan",
        "000-100-020 P&ID fuel oil system.pdf": "drawing or plan",
    }.items():
        assert triage.skip_reason(name) == reason, name
    for name in ["08 Close coupled waste handling certifugal pump.pdf",   # the yard's spelling of centrifugal
                 "M-3-02 Main air compressor (with test result).pdf",
                 "Emergency generator working drawing.pdf",   # yards' names for a maker's booklet
                 "Oil cooler tank cleaning heater finish drawing.pdf",
                 "81 Break chain stopper completion drawing.pdf",
                 "10-1 Technic spec, instal, start-up, drawings, part list.pdf",
                 "07) Drawing & illustrations.pdf", "3) Engine specs & dwgs.pdf", "7) Attached drawings.pdf",
                 "Instruction manual with declaration of conformity.pdf",
                 "442-01 Spare part and inventory list of electric equipment.pdf",
                 "PID controller manual.pdf"]:
        assert triage.skip_reason(name) == "", name


def test_a_drawings_chapter_of_a_manual_is_read():
    """A maker's manual split into numbered chapters: its drawings chapter is read with the
    rest (four such chapters had spares pages). A drawings file among other drawings, or
    among chapters that are all skipped, is still skipped."""
    manual = ["THRUSTER/12) Trouble shooting.pdf", "THRUSTER/13) Spare parts.pdf",
              "THRUSTER/11) Service & maintenance.pdf", "THRUSTER/14) Drawings - Electronic control.pdf"]
    assert triage.skip_reasons(manual) == ["", "", "", ""]
    assert triage.skip_reasons(["ELSEWHERE/14) Drawings - Electronic control.pdf"] + manual[:3]) == ["drawing or plan", "", "", ""]
    plans = ["DRAWINGS/01) Cable drawings.pdf", "DRAWINGS/02) GA plan.pdf", "DRAWINGS/03) Layout.pdf",
             "DRAWINGS/04) Foundation.pdf"]
    assert triage.skip_reasons(plans) == ["drawing or plan"] * 4
    assert triage.skip_reasons(["SEAL/10) Drawings.pdf", "SEAL/01) General description.pdf"]) == ["drawing or plan", ""]
    assert triage.skip_reason("10) Drawings.pdf", ["00) Cover.pdf", "01) General description.pdf",
                                                   "07) Repair of the split seals.pdf", "10) Drawings.pdf"]) == ""


def turned_and_cropped(path):
    """Pages as manuals have them: upright, turned each way, cropped, and both."""
    doc = fitz.open()
    for turn, crop in [(0, None), (90, None), (180, None), (270, None), (0, (40, 60, 400, 500)),
                       (90, (20, 20, 280, 400)), (180, (20, 30, 260, 410)), (270, (20, 30, 260, 410))]:
        page = doc.new_page(width=300 if crop != (40, 60, 400, 500) else 600, height=500 if crop != (40, 60, 400, 500) else 800)
        page.insert_text((50, 100), f"turned {turn}", fontsize=14)
        page.draw_rect(fitz.Rect(30, 30, 80, 80), color=(1, 0, 0), fill=(1, 0, 0))
        if crop:
            page.set_cropbox(fitz.Rect(crop))
        page.set_rotation(turn)
    doc.save(path)
    return path


def test_a_tagged_page_looks_as_before_under_its_band(tmp_path):
    """Every page of the tagged manual is the page as it was, with a band on top saying
    what became of it, whichever way it was turned or cropped."""
    src = turned_and_cropped(tmp_path / "turned.pdf")
    tagged = outputs.tagged_pdf(src, {1: 1, 2: None, 3: 0})
    with fitz.open(src) as before, fitz.open(tagged) as after:
        assert after.page_count == before.page_count
        for a, b in zip(before, after):
            band = b.rect.height - a.rect.height
            assert b.rect.width == a.rect.width and band >= 18 and b.rotation == 0
            shown = a.get_pixmap(dpi=72).samples
            assert b.get_pixmap(dpi=72, clip=fitz.Rect(0, band, b.rect.width, b.rect.height)).samples == shown, a.number
            words = " ".join(w[4] for w in b.get_text("words") if w[3] <= band + 1)
            assert words == {0: "INCLUDED", 1: "LEFT OUT - NOT CHECKED"}.get(a.number, "LEFT OUT") + f" PAGE {a.number + 1} OF 8"
    tagged.unlink()


def test_the_title_page_of_a_long_manual_is_never_left_to_a_person(tmp_path):
    """Page 1 of a manual of TITLE_PAGE_FROM pages or more is its title page: left out,
    unless a parts table on it marks it spares (a parts catalogue's chapter). In a short
    file it is the drawing or the parts list itself."""
    long = classify.classify_pdf(make_book(tmp_path / "long.pdf", {1, 5}, pages=12))
    assert long[1] == tables.NO and long[5] == tables.CHECK   # the same table: page 5 is still asked about
    assert classify.title_page({**long, 1: tables.SPARES})[1] == tables.SPARES
    assert classify.classify_pdf(make_book(tmp_path / "one.pdf", {1}, pages=1))[1] == tables.CHECK


def test_bucket_needs_a_part_number_to_be_sure():
    table = lambda *cols: pd.DataFrame([["1"] * len(cols)] * 4, columns=cols)
    assert tables.bucket(table("Item", "Part No.", "Description", "Qty")) == tables.SPARES
    assert tables.bucket(table("No", "Level", "Order", "Part No.")) == tables.SPARES  # Yanmar catalogue
    assert tables.bucket(table("No.", "Name", "Material")) == tables.CHECK   # a drawing's own list
    assert tables.bucket(table("ITEM", "NAME OF", "QUANTITY")) == tables.CHECK  # header split in two
    assert tables.bucket(table("Interval", "Work", "Remarks")) == tables.NO
    assert tables.bucket(table("Interval", "Work", "Remarks"), confidence=25) == tables.CHECK


def make_manual(path):
    """Page 1 a parts table, page 2 prose, both with a text layer so no OCR runs."""
    doc = fitz.open()
    page = doc.new_page()
    rows = [("Item", "Part No.", "Description", "Qty")]
    rows += [(str(i), f"146673-0{i}0", f"Gasket type {i}", "2") for i in range(1, 9)]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((60 + 120 * c, 100 + 22 * r), cell, fontsize=10)
    doc.new_page().insert_text(
        (60, 100), "Start the engine and check the lubricating oil pressure before loading.")
    doc.save(path)
    return path


def as_images(path):
    """Each page of a PDF as a 200-dpi picture, the way the picture reader sees it."""
    with fitz.open(path) as doc:
        return [np.ascontiguousarray(np.frombuffer(
            (pix := page.get_pixmap(dpi=200, colorspace=fitz.csRGB)).samples, np.uint8
        ).reshape(pix.height, pix.width, 3)) for page in doc]


def test_text_pdf_end_to_end(tmp_path):
    """The contract with Parsel's engine: a table page is flagged, a prose page is not.
    Parsel reads this hand-drawn table as item + description only, hence "not NO"
    rather than SPARES. Accuracy is measured on real manuals, not here."""
    got = classify.classify_pdf(make_manual(tmp_path / "manual.pdf"))
    assert got[1] != tables.NO and got[2] == tables.NO and len(got) == 2


def test_a_scan_with_a_stamp_is_still_read_as_a_picture(tmp_path):
    """Page 13 of S1934_Alternator_Final.pdf: a scanned spare parts list that carried
    55 characters of text, enough for Parsel to treat it as a text page and never OCR it."""
    doc = fitz.open()
    scan = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 280), False)
    scan.clear_with(255)
    stamped = doc.new_page()
    stamped.insert_image(stamped.rect, pixmap=scan)
    stamped.insert_text((60, 100), "K SHIPBUILDING CO., LTD.   P.13/E")
    bare = doc.new_page()                     # a plain scan
    bare.insert_image(bare.rect, pixmap=scan)
    doc.new_page().insert_text((60, 100), "A page of ordinary text with no picture on it.")
    doc.save(tmp_path / "scan.pdf")
    assert classify.census(tmp_path / "scan.pdf") == ({1, 3}, {1, 2}, 3)   # page 1 is read both ways


def test_a_big_manual_is_read_by_several_processes_with_the_same_answers(tmp_path):
    """Text layers of a manual with PARALLEL_FROM pages or more are shared out over the
    text workers; every page must come back, read exactly as one process reads it."""
    doc = fitz.open()
    for number in range(text.PARALLEL_FROM + 5):
        with fitz.open(make_manual(tmp_path / "one.pdf")) as one:
            doc.insert_pdf(one, from_page=number % 2, to_page=number % 2)   # table, prose, table...
    doc.save(tmp_path / "big.pdf")
    pages = list(range(1, doc.page_count + 1))
    seen = []
    shared = text.read_pages(str(tmp_path / "big.pdf"), pages, lambda done, total: seen.append((done, total)))
    with fitz.open(tmp_path / "big.pdf") as again:
        alone = {n: text.reading(again, n) for n in pages}
    assert shared == alone and seen[-1] == (len(pages), len(pages))


def test_a_scanned_table_is_found_and_scanned_text_is_not(tmp_path):
    """The picture reader end to end, on pictures of make_manual's two pages."""
    pictures.start_engine(2)
    table, prose = as_images(make_manual(tmp_path / "manual.pdf"))
    assert pictures.judge(table) == tables.SPARES
    assert pictures.judge(prose) == tables.NO
    assert pictures.judge(np.ascontiguousarray(np.rot90(table))) == tables.SPARES   # scanned sideways


def test_a_heading_split_over_two_lines_is_read(tmp_path):
    """As on a Yara scrubber manual: "Part" over "No.", and "Part Description" in one line,
    in white on a dark band."""
    pictures.start_engine(2)
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(50, 80, 450, 125), color=None, fill=(0.05, 0.2, 0.45))
    page.insert_text((60, 98), "Part", fontsize=12, color=(1, 1, 1))
    page.insert_text((60, 116), "No.", fontsize=12, color=(1, 1, 1))
    page.insert_text((140, 107), "Part Description", fontsize=12, color=(1, 1, 1))
    for r in range(8):
        page.insert_text((60, 160 + 40 * r), f"0{2300 + 13 * r}", fontsize=10)
        page.insert_text((140, 160 + 40 * r), f"Gasket for sensor {r + 1}", fontsize=10)
    doc.save(tmp_path / "split.pdf")
    assert pictures.judge(as_images(tmp_path / "split.pdf")[0]) == tables.SPARES


def test_a_scan_with_no_table_found_goes_to_a_person_unless_it_is_nearly_blank(tmp_path):
    """Safety net: the picture reader misses tables (small print, upside down, German
    headings), so finding none rules a scan out only when the page is nearly empty."""
    pictures.start_engine(2)
    doc = fitz.open()
    drawing = doc.new_page()
    for i in range(30):   # a drawing's labels: short words all over the sheet, no table
        drawing.insert_text((40 + (i * 97) % 480, 60 + (i * 53) % 720), f"DETAIL {chr(65 + i % 26)}{i}", fontsize=9)
    doc.new_page()        # an empty sheet
    doc.save(tmp_path / "scans.pdf")
    drawing, blank = as_images(tmp_path / "scans.pdf")
    assert pictures.judge(drawing) == tables.CHECK
    assert pictures.judge(blank) == tables.NO


def picture_rows(manuals=4, pages=30, look=2.0, lines=False):
    """{(sha, page): (label, picture verdict, numbers[, lines])} for the picture model: every
    fifth page is a spares page; spares pages look alike (their numbers lean one way, by
    `look`) and, with `lines`, say alike."""
    rng = np.random.default_rng(0)
    said = lambda spares: "\n".join(rng.choice(
        ["PART NO", "DESCRIPTION", "QTY", "ITEM", "DWG NO", "BOLT", "NUT", "GASKET"] if spares else
        ["WARNING", "Operation", "Check the valve", "SECTION", "Fig.", "Pressure", "Motor", "Start"], 6))
    return {(f"manual{m}", n): (int(n % 5 == 0), tables.CHECK if n % 5 == 0 or n % 2 else tables.NO,
                                (rng.normal(size=768) + look * (n % 5 == 0) * (np.arange(768) < 50)).astype(np.float16),
                                *([said(n % 5 == 0)] if lines else []))
            for m in range(manuals) for n in range(1, pages + 1)}


def test_the_picture_model_keeps_nearly_every_spares_page_it_is_asked_about():
    """Its cut-off keeps LOOKS_KEEP of the spares pages the picture reader left to a person,
    scored by models that never saw their manuals; its test sets each manual's cut-off
    without that manual."""
    head = learn.fit_looks(picture_rows())
    tested = head["report"]["tested on manuals left out"]
    assert tested["spares check pages sent to unlikely"] <= 0.05 and tested["other check pages sent to unlikely"] > 0.5
    one = {k: v for k, v in picture_rows().items() if k[0] == "manual0"}
    assert learn.fit_looks(one) is None   # one manual: nothing to test a cut-off on


def test_pages_the_picture_model_doubts_go_apart_and_their_numbers_are_kept(tmp_path, monkeypatch):
    rows = picture_rows()
    head = learn.fit_looks(rows)
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()   # no text layer: judged by their picture
    doc.save(tmp_path / "scans.pdf")
    monkeypatch.setattr(learn, "load", lambda: {"unsure_from": 0.5, "spares_from": 0.9, "looks": head})
    looked = {1: rows[("manual0", 5)][2], 2: rows[("manual0", 1)][2], 3: rows[("manual0", 2)][2]}
    monkeypatch.setattr(pictures, "read_pictures", lambda path, pages, progress, quick=(): (
        {1: tables.CHECK, 2: tables.CHECK, 3: tables.NO}, looked, {1: "PART NO", 2: "", 3: ""}))
    got = classify.classify_pdf(tmp_path / "scans.pdf")
    assert got == {1: tables.CHECK, 2: tables.UNLIKELY, 3: tables.NO}   # only a page left to a person can move
    kept = looks.load(hashlib.sha256((tmp_path / "scans.pdf").read_bytes()).hexdigest())
    assert {n: (v, t) for n, (v, _, t) in kept.items()} == {
        1: (tables.CHECK, "PART NO"), 2: (tables.CHECK, ""), 3: (tables.NO, "")}   # and the lines read on them


def test_a_page_next_to_a_spares_page_stays_with_a_person(tmp_path, monkeypatch):
    """An exploded drawing beside its parts list looks like no parts page; it is never moved."""
    rows = picture_rows()
    head = learn.fit_looks(rows)
    doc = fitz.open()
    for _ in range(4):
        doc.new_page()
    doc.save(tmp_path / "scans.pdf")
    monkeypatch.setattr(learn, "load", lambda: {"unsure_from": 0.5, "spares_from": 0.9, "looks": head})
    other = rows[("manual0", 1)][2]   # a page the picture model sends to "unlikely"
    monkeypatch.setattr(pictures, "read_pictures", lambda path, pages, progress, quick=(): (
        {1: tables.SPARES, 2: tables.CHECK, 3: tables.NO, 4: tables.CHECK}, {n: other for n in range(1, 5)}, {}))
    assert classify.classify_pdf(tmp_path / "scans.pdf") == {
        1: tables.SPARES, 2: tables.CHECK, 3: tables.NO, 4: tables.UNLIKELY}


def test_what_a_scan_says_moves_pages_its_look_cannot():
    """Spares scans that look like any other page but say "PART NO", "QTY"...: the picture
    model alone moves few other check pages, with the lines model most, and each keeps the
    spares pages there. Pages without lines, and a model from before the lines model, are
    judged by the picture alone."""
    rows = picture_rows(manuals=8, pages=60, look=0.0, lines=True)
    head = learn.fit_looks(rows)
    with_lines = head["report"]["tested on manuals left out"]["check pages with lines"]
    assert with_lines["spares sent to unlikely"] <= 0.05
    assert with_lines["others sent to unlikely"] > 0.8 > 0.4 > with_lines["others sent to unlikely by the picture alone"]
    spares, other = rows[("manual0", 5)], rows[("manual0", 3)]
    vectors, verdicts = [spares[2], other[2]], [tables.CHECK, tables.CHECK]
    assert learn.looks_unlikely(head, vectors, verdicts, [spares[3], other[3]]).tolist() == [False, True]
    alone = learn.looks_unlikely(head, vectors, verdicts)
    assert learn.looks_unlikely(head, vectors, verdicts, ["", ""]).tolist() == alone.tolist()
    old = {k: v for k, v in head.items() if k != "lines"}
    assert learn.looks_unlikely(old, vectors, verdicts, [spares[3], other[3]]).tolist() == alone.tolist()
    learned = {("manual0", 3): other}
    learn._keep_lines(learned, {("manual0", 3): (0, tables.CHECK, other[2], "")})   # read again, lines not kept
    assert learned[("manual0", 3)][3] == other[3]


# ---- the server: drops, decisions, downloads ------------------------------------

def listed(client, manuals, **kw):
    """Step 1 of a drop, as the screen makes it: {path in the folder: bytes} ->
    (job, [{"id", "skip"}]). The server skips some files by their name."""
    made = client.post("/api/jobs", json={"files": [{"rel": rel, "size": len(b)} for rel, b in manuals.items()]},
                       **kw).json()
    return made["id"], made["files"]


def send(client, fid, content, piece=1 << 20):
    """Step 2: one manual's bytes in pieces, carrying on from what the server says has
    arrived, as the screen does. Returns the last answer."""
    at = 0
    while True:
        r = client.put(f"/api/files/{fid}/content?offset={at}", content=content[at:at + piece])
        if r.status_code == 409 and "received" in r.json():
            at = r.json()["received"]
            continue
        if r.status_code != 200 or r.json()["received"] >= len(content):
            return r
        at = r.json()["received"]


def drop(client, manuals):
    """A whole drop, as the screen sends it: files skipped by their name are not sent."""
    job, made = listed(client, manuals)
    for f, content in zip(made, manuals.values()):
        if not f["skip"]:
            assert send(client, f["id"], content).status_code == 200
    return job


def finished(client, job):
    """{path in the folder: file} once nothing is waiting or being read."""
    for _ in range(240):
        files = {f["rel"]: f for f in client.get(f"/api/jobs/{job}").json()["files"]}
        if not any(f["status"] in ("queued", "running") for f in files.values()):
            return files
        time.sleep(0.5)
    raise AssertionError("the worker never finished")


def sheet_rows(sheet):
    return [[cell.value for cell in row] for row in sheet.rows]


def tags(content):
    """What the band on top of each page of a tagged manual says, before its page number."""
    with fitz.open(stream=content) as doc:
        said = []
        for page in doc:
            band = page.get_drawings()[-1]["rect"]   # drawn last, over the page
            words = [w[4] for w in page.get_text("words", sort=True) if w[3] <= band.y1]
            said.append(" ".join(words[:words.index("PAGE")]))
        return said


def test_drop_correct_download(tmp_path):
    """The whole round trip a colleague makes, through the real worker."""
    client = TestClient(app.app)
    pdf = make_manual(tmp_path / "pump.pdf").read_bytes()
    job = drop(client, {"pump.pdf": pdf, "class certificate.pdf": pdf})
    files = finished(client, job)
    assert files["class certificate.pdf"]["status"] == "skipped"
    pump = files["pump.pdf"]
    assert pump["status"] == "done" and pump["pages_total"] == 2, pump

    # a person rules on both pages; the downloads follow the person
    mark = lambda page, spares: client.put(f"/api/files/{pump['id']}/pages/{page}", json={"spares": spares})
    assert mark(1, True).status_code == 200 and mark(2, True).status_code == 200
    book = openpyxl.load_workbook(io.BytesIO(client.get(f"/api/jobs/{job}/spares.xlsx").content))
    assert sheet_rows(book["Spares pages"]) == [["Manual", "Spares pages"], ["pump.pdf", "1, 2"]]
    assert ["class certificate.pdf", "Not read: certificate"] in sheet_rows(book["Needs attention"])

    assert mark(2, False).status_code == 200
    out = client.get(f"/api/files/{pump['id']}/tagged.pdf")   # the whole manual, each page tagged
    assert out.status_code == 200 and out.headers["content-disposition"].endswith("pump%20-%20tagged.pdf")
    assert tags(out.content) == ["INCLUDED", "LEFT OUT"]
    bundle = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job}/spares.zip").content))
    assert sorted(bundle.namelist()) == [outputs.allocation_name(job), "Spares pages.xlsx", "pump - tagged.pdf"]
    assert fitz.open(stream=bundle.read("pump - tagged.pdf")).page_count == 2

    # "include anyway": skipped by its name, so never sent, and sent now. The same bytes
    # as pump.pdf: kept once on disk, and its pages and corrections are reused
    skipped = files["class certificate.pdf"]["id"]
    assert client.post(f"/api/files/{skipped}/include").json()["send"] is True
    assert send(client, skipped, pdf).status_code == 200
    skipped = finished(client, job)["class certificate.pdf"]
    assert skipped["status"] == "done" and [p["label"] for p in skipped["pages"]] == [1, 0]
    sha = hashlib.sha256(pdf).hexdigest()
    assert [p.name for p in (store.DATA / "uploads").glob(f"{sha}*")] == [f"{sha}.pdf"]

    # the person's last word on each page is what the model will learn
    assert {key: label for key, (label, _) in learn.from_corrections()[0].items()} == {(sha, 1): 1, (sha, 2): 0}


def test_a_skipped_manual_can_be_looked_at_before_anyone_reads_it(tmp_path):
    """A file skipped by its name is still sent, after the others, and kept unread: its
    pages can be looked at, and "Read it" then reads it without sending it again."""
    client = TestClient(app.app)
    pdf = make_book(tmp_path / "cert.pdf", {1}, pages=3).read_bytes()
    job, made = listed(client, {"VESSEL/class certificate.pdf": pdf})
    fid = made[0]["id"]
    assert made[0]["skip"] == "certificate"
    assert send(client, fid, pdf).status_code == 200
    f = client.get(f"/api/jobs/{job}").json()["files"][0]
    assert (f["status"], f["kept"], f["pages_total"], f["note"]) == ("skipped", 1, 3, "certificate")
    assert client.get(f"/api/files/{fid}/page/3.png").status_code == 200
    assert client.put(f"/api/files/{fid}/content?offset=0", content=pdf).status_code == 409   # sent once
    assert client.post(f"/api/files/{fid}/include").json()["send"] is False
    assert finished(client, job)["VESSEL/class certificate.pdf"]["status"] == "done"


def test_the_allocation_sheet_is_the_typists_own_layout(tmp_path):
    """The allocation sheet lists every manual of a drop in D:\\AllocationTool's columns:
    its folder, its pages, and in DATA the pages included, as the typists write them, or
    "No Data". Learning never takes it back: its pages are the tool's own answers."""
    client = TestClient(app.app)
    job = drop(client, {"VESSEL/ENGINE/pump.pdf": make_book(tmp_path / "pump.pdf", {1}, pages=6).read_bytes(),
                        "VESSEL/DECK/winch.pdf": make_book(tmp_path / "winch.pdf", set(), pages=2).read_bytes(),
                        "VESSEL/DECK/class certificate.pdf": make_book(tmp_path / "c.pdf", set(), pages=1).read_bytes()})
    files = finished(client, job)
    pump, winch = files["VESSEL/ENGINE/pump.pdf"]["id"], files["VESSEL/DECK/winch.pdf"]["id"]
    with store.db() as con:   # as a person and the reader leave them: 1-3 and 5 in, 6 still to check
        con.execute("DELETE FROM pages WHERE file_id IN (?, ?)", (pump, winch))
        con.executemany("INSERT INTO pages(file_id, page, bucket, label) VALUES(?,?,?,?)",
                        [(pump, 1, "spares", None), (pump, 2, "check", 1), (pump, 3, "spares", None),
                         (pump, 5, "check", 1), (pump, 6, "check", None)])
    got = client.get(f"/api/jobs/{job}/allocation.xlsx")
    assert got.status_code == 200 and "Allocation%20Sheet%20-%20VESSEL%20-%20" in got.headers["content-disposition"]
    path = tmp_path / "allocation.xlsx"
    path.write_bytes(got.content)
    sheet = openpyxl.load_workbook(path).active
    rows = sheet_rows(sheet)
    assert rows[0][0] == "Allocation Sheet" and rows[1][0].startswith("VESSEL")
    assert rows[2] == outputs.ALLOCATION
    assert [r[:5] for r in rows[3:]] == [["VESSEL/DECK", "class certificate.pdf", None, "Not read: certificate", "pdf"],
                                          ["VESSEL/DECK", "winch.pdf", 2, "No Data", "pdf"],
                                          ["VESSEL/ENGINE", "pump.pdf", 6, "1-3,5", "pdf"]]
    assert sheet["K6"].value == "Still to check: 6" and sheet["D4"].fill.fgColor.rgb.endswith(outputs.FLAG_FILL)
    assert not any(cell.comment for row in sheet.iter_rows() for cell in row)   # cell notes stop some viewers opening it
    assert not any(n.startswith("xl/comments") for n in zipfile.ZipFile(path).namelist())
    assert outputs.parse_pages(outputs.page_ranges([1, 2, 3, 5, 6, 9, 40])) == {1, 2, 3, 5, 6, 9, 40}
    try:
        learn.spreadsheet_pages(path)
        raise AssertionError("learned from the tool's own answers")
    except ValueError as exc:
        assert "Spares Page Finder" in str(exc)


def test_earlier_drops_are_listed_and_pages_zoom_sharp(tmp_path):
    """The list on the left: newest first, a folder named as it was dropped. A page drawn
    sharp enough to zoom into comes as JPEG; the others as PNG."""
    client = TestClient(app.app)
    first = drop(client, {"VESSEL A/pump.pdf": make_book(tmp_path / "pump.pdf", {1}, pages=2).read_bytes(),
                          "VESSEL A/gear.pdf": make_book(tmp_path / "gear.pdf", {2}, pages=3).read_bytes()})
    second = drop(client, {"loose.pdf": make_book(tmp_path / "loose.pdf", {1}, pages=4).read_bytes()})
    finished(client, first)
    loose = finished(client, second)["loose.pdf"]
    newest = client.get("/api/jobs").json()
    assert [d["id"] for d in newest[:2]] == [second, first]
    assert (newest[1]["name"], newest[1]["manuals"], newest[1]["done"]) == ("VESSEL A", 2, 2)
    assert newest[0]["name"] == client.get(f"/api/jobs/{second}").json()["name"] == "loose.pdf"
    sharp = client.get(f"/api/files/{loose['id']}/page/1.png?big=2")
    assert sharp.status_code == 200 and sharp.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/files/{loose['id']}/page/1.png").headers["content-type"] == "image/png"


def test_a_manual_arrives_in_pieces_and_carries_on_after_a_cut(tmp_path):
    """Pieces keep each request under Cloudflare's 100 MB; a piece that does not follow
    on is refused with how much has arrived, so a cut connection costs one piece."""
    client = TestClient(app.app)
    content = make_book(tmp_path / "long.pdf", {1}, pages=4).read_bytes()
    job, [f] = listed(client, {"long.pdf": content})
    third = len(content) // 3
    assert client.put(f"/api/files/{f['id']}/content?offset=0", content=content[:third]).json() == {"received": third}
    again = client.put(f"/api/files/{f['id']}/content?offset=0", content=content[:third])   # sent twice
    assert again.status_code == 409 and again.json()["received"] == third
    assert send(client, f["id"], content, piece=third).status_code == 200   # carries on from the third
    assert finished(client, job)["long.pdf"]["status"] == "done"


def test_other_sites_and_other_files_are_refused(tmp_path):
    """No changes from another site's page, no answering to an unknown host name (DNS
    rebinding), and a file that only claims to be a PDF is listed but never opened."""
    client = TestClient(app.app)
    evil = {"Origin": "http://evil.example"}
    assert client.post("/api/jobs", headers=evil, json={"files": [{"rel": "pump.pdf"}]}).status_code == 403
    assert TestClient(app.app, base_url="http://evil.example").get("/").status_code == 400
    assert client.get("/docs").status_code == 404
    job, [f] = listed(client, {r"..\..\VESSEL\..\not really.pdf": b"MZ this is a program"},
                      headers={"Origin": "http://testserver"})
    assert send(client, f["id"], b"MZ this is a program").status_code == 200
    assert send(client, f["id"], b"%PDF- sent twice").status_code == 409
    [f] = client.get(f"/api/jobs/{job}").json()["files"]
    assert (f["rel"], f["status"], f["note"]) == ("VESSEL/not really.pdf", "skipped", "not a PDF file")


def test_the_logo_is_where_browsers_look_for_it():
    """In the tab, in bookmarks and on a phone's home screen; no other file is served."""
    client = TestClient(app.app)
    for name, kind in app.ICONS.items():
        got = client.get("/" + name)
        assert (got.status_code, got.headers["content-type"].split(";")[0]) == (200, kind), name
    page = client.get("/").text
    assert all(f'href="/{name}"' in page for name in ("favicon.ico", "favicon.svg", "apple-touch-icon.png", "manifest.json"))
    assert all(client.get(icon["src"]).status_code == 200 for icon in client.get("/manifest.json").json()["icons"])
    assert client.get("/app.py").status_code == client.get("/index.html").status_code == 404


def test_requests_that_could_harm_the_server_are_refused(tmp_path):
    """The security audit's findings, each one a request anyone on the network could make."""
    client = TestClient(app.app)
    # a page list with a huge range (a tracking sheet's "Marked Pages") must not fill the
    # memory: a range is filled in up to 100,000 pages at most, and a ten-digit number is no page
    assert len(outputs.parse_pages("1-999999999")) == 100_001
    assert outputs.parse_pages("1-2000000000") == set()
    # more bytes than the manual's size: writing stops and what arrived is thrown away
    job, [small] = listed(client, {"small.pdf": b"%PDF-" + b"x" * 5})
    assert client.put(f"/api/files/{small['id']}/content?offset=0", content=b"%PDF-" + b"x" * 50).status_code == 400
    assert not store.partial(small["id"]).exists()
    # a body sent in chunks, with no length to check
    chunks = (piece for piece in [b'{"files": [', b'{"rel": "a.pdf", "size": 1}]}'])
    assert client.post("/api/jobs", content=chunks, headers={"Content-Type": "application/json"}).status_code == 411
    # a manual larger than 4 GB is not taken, and names lose what Windows refuses
    job, [huge] = listed(client, {r"C:\VESSEL\pump<1>.pdf": b""})
    client.post("/api/jobs", json={"files": [{"rel": "huge.pdf", "size": 5 * 1024 ** 3}]})
    assert client.get("/api/jobs").json()[0]["name"] == "huge.pdf"
    assert [f["status"] for f in client.get(f"/api/jobs/{client.get('/api/jobs').json()[0]['id']}").json()["files"]] == ["skipped"]
    assert client.get(f"/api/jobs/{job}").json()["files"][0]["rel"] == "C_/VESSEL/pump_1_.pdf"


def test_the_data_folder_can_be_moved(tmp_path):
    """Manuals are kept by a path relative to the data folder, and paths kept in full by
    older versions are cut back to that on start-up, so a moved or copied folder works."""
    client = TestClient(app.app)
    content = make_manual(tmp_path / "pump.pdf").read_bytes()
    job = drop(client, {"pump.pdf": content})
    with store.db() as con:
        [kept] = con.execute("SELECT path FROM files WHERE job_id=?", (job,)).fetchone()
        assert kept == f"uploads/{hashlib.sha256(content).hexdigest()}.pdf" and store.pdf(kept).exists()
        old = con.execute("INSERT INTO files(job_id, path, rel, status) VALUES(?,?,?, 'done')",
                          (job, r"D:\spares-finder\data\uploads\job9\1.pdf", "old.pdf")).lastrowid
    store.setup()
    with store.db() as con:
        assert con.execute("SELECT path FROM files WHERE id=?", (old,)).fetchone()[0] == "uploads/job9/1.pdf"
        con.execute("DELETE FROM files WHERE id=?", (old,))
    copy = store.backup()
    assert copy.parent == store.DATA / "backups" and copy.stat().st_size > 0


def test_old_drops_are_archived_and_still_teach(tmp_path):
    """After KEEP_DAYS a drop's original manuals go. Its zip, its Excel list and what the
    pages people decided say are kept, so downloads and learning still work."""
    client = TestClient(app.app)
    content = make_book(tmp_path / "old.pdf", {2}, pages=4).read_bytes()
    job = drop(client, {"old.pdf": content})
    f = finished(client, job)["old.pdf"]
    client.put(f"/api/files/{f['id']}/pages/2", json={"spares": True})
    sha = hashlib.sha256(content).hexdigest()
    looks.save(sha, {2: (tables.CHECK, np.zeros(768, np.float16))})
    with store.db() as con:
        con.execute("UPDATE jobs SET created=datetime('now', '-30 days') WHERE id=?", (job,))
        [kept] = con.execute("SELECT path FROM files WHERE id=?", (f["id"],)).fetchone()
    assert job in archive.sweep()
    assert not store.pdf(kept).exists()
    assert looks.file_of(sha).exists()   # a person decided one of its pages: its numbers still teach
    assert client.get(f"/api/jobs/{job}").json()["archived"]
    assert client.get(f"/api/files/{f['id']}/page/1.png").status_code == 404
    kept_zip = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job}/spares.zip").content))
    assert sorted(kept_zip.namelist()) == [outputs.allocation_name(job), "Spares pages.xlsx",
                                           "old - spares.pdf"]   # the pages included, not the whole manual
    assert fitz.open(stream=kept_zip.read("old - spares.pdf")).page_count == 1
    assert {page: label for (s, page), (label, _) in learn.from_corrections()[0].items() if s == sha} == {2: 1}
    assert {page: label for (s, page), (label, *_) in learn.from_corrections()[1].items() if s == sha} == {2: 1}


def test_pages_the_picture_model_doubts_are_left_out(tmp_path):
    """Pages the picture model thinks are probably not spares are left out, without waiting
    for a person: not downloaded, not outstanding, not counted to check. A person can
    still include one. Pages still to check are left out of the downloads too, and say so."""
    client = TestClient(app.app)
    job = drop(client, {"doubts.pdf": make_book(tmp_path / "doubts.pdf", {1}, pages=4).read_bytes()})
    f = finished(client, job)["doubts.pdf"]
    with store.db() as con:   # as the picture model leaves them, and one page it sends to a person
        con.execute("UPDATE pages SET bucket='unlikely', label=NULL WHERE file_id=? AND page IN (2, 3)", (f["id"],))
        con.execute("INSERT OR REPLACE INTO pages(file_id, page, bucket) VALUES(?, 1, 'spares')", (f["id"],))
        con.execute("INSERT OR REPLACE INTO pages(file_id, page, bucket) VALUES(?, 4, 'check')", (f["id"],))
    client.put(f"/api/files/{f['id']}/pages/3", json={"spares": True})   # a person Includes one
    book = openpyxl.load_workbook(io.BytesIO(client.get(f"/api/jobs/{job}/spares.xlsx").content))
    assert sheet_rows(book["Spares pages"])[1] == ["doubts.pdf", "1, 3"]
    assert sheet_rows(book["Needs attention"])[1:] == [["doubts.pdf", "Pages still to check: 4"]]
    assert next(d for d in client.get("/api/jobs").json() if d["id"] == job)["to check"] == 1
    assert tags(client.get(f"/api/files/{f['id']}/tagged.pdf").content) == ["INCLUDED", "LEFT OUT", "INCLUDED",
                                                                            "LEFT OUT - NOT CHECKED"]


def test_a_drops_zip_is_made_in_the_background(tmp_path):
    """A whole vessel's zip takes minutes, longer than Cloudflare waits for one request:
    the screen asks for it to be made, waits showing how many manuals are in it so far, then
    fetches it."""
    client = TestClient(app.app)
    job = drop(client, {"bg.pdf": make_book(tmp_path / "bg.pdf", {1}, pages=2).read_bytes()})
    f = finished(client, job)["bg.pdf"]
    client.put(f"/api/files/{f['id']}/pages/1", json={"spares": True})
    state = client.post(f"/api/jobs/{job}/bundle").json()
    for _ in range(100):
        if state["state"] == "ready":
            break
        time.sleep(0.1)
        state = client.get(f"/api/jobs/{job}/bundle").json()
    assert state["state"] == "ready" and (state["done"], state["total"]) == (1, 1)
    assert "bg - tagged.pdf" in zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job}/spares.zip").content)).namelist()


def test_a_pdf_cut_off_is_unreadable_not_empty():
    """H1868_Volume 4.pdf: a download that stopped two thirds of the way, before the PDF's
    page list. It opens with no pages, and was once reported as read with nothing found."""
    client = TestClient(app.app)
    cut = (b"%PDF-1.4\n1 0 obj <</Type/Catalog/Pages 2 0 R>> endobj\n"
           b"3 0 obj <</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]>> endobj\n")
    job = drop(client, {"H1868_Volume 4.pdf": cut})
    f = finished(client, job)["H1868_Volume 4.pdf"]
    assert (f["status"], f["note"]) == ("failed", "No pages could be found in it: the PDF is damaged or incomplete")


def make_scan(path, pages=3):
    """make_manual's parts table as pictures only, so that reading it needs the OCR workers."""
    with fitz.open(make_manual(path.with_name("table.pdf"))) as source:
        pix = source[0].get_pixmap(dpi=150)
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    return path


def test_a_vessel_folder_cut_short_and_stopped(tmp_path):
    """A folder is listed first and sent manual by manual. A drop cut short keeps what
    arrived; Stop leaves the rest unread until Continue; downloads keep the folders."""
    client = TestClient(app.app)
    pump, scan = make_manual(tmp_path / "pump.pdf").read_bytes(), make_scan(tmp_path / "scan.pdf").read_bytes()
    job, made = listed(client, {"VESSEL/ENGINE/pump.pdf": pump, "VESSEL/ENGINE/scan.pdf": scan,
                                "VESSEL/DECK/crane.pdf": pump})
    ids = [f["id"] for f in made]
    send(client, ids[0], pump)
    send(client, ids[1], scan)   # and the page is closed before crane.pdf is sent
    assert client.post(f"/api/jobs/{job}/stop").status_code == 200   # long before the OCR workers are up
    files = finished(client, job)
    assert files["VESSEL/ENGINE/scan.pdf"]["status"] == "stopped"
    assert files["VESSEL/ENGINE/pump.pdf"]["status"] in ("stopped", "done")
    assert files["VESSEL/DECK/crane.pdf"]["status"] == "unsent"

    assert client.post(f"/api/jobs/{job}/continue").status_code == 200
    files = finished(client, job)
    assert [files[rel]["status"] for rel in sorted(files)] == ["unsent", "done", "done"]
    assert files["VESSEL/ENGINE/scan.pdf"]["pages_total"] == 3
    book = openpyxl.load_workbook(io.BytesIO(client.get(f"/api/jobs/{job}/spares.xlsx").content))
    assert ["crane.pdf", "Not sent: drop the folder again to finish"] in sheet_rows(book["Needs attention"])

    send(client, ids[2], pump)   # the folder dropped again: only the missing manual goes
    files = finished(client, job)
    assert all(f["status"] == "done" for f in files.values())
    client.put(f"/api/files/{ids[2]}/pages/1", json={"spares": True})
    bundle = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job}/spares.zip").content))
    assert {"VESSEL/DECK/crane - tagged.pdf", "VESSEL/ENGINE/pump - tagged.pdf", "VESSEL/ENGINE/scan - tagged.pdf"} <= set(bundle.namelist())


# ---- learning ----------------------------------------------------------------

PROSE = ("Before starting the pump, open the suction and delivery valves fully and check "
         "that the casing is filled with liquid. Never run the pump dry, as the mechanical "
         "seal will be damaged within seconds. Check the bearing temperature every watch.")


def make_book(path, parts, pages=6):
    """A manual whose pages in `parts` carry a parts table and the rest are prose, each
    with enough text for the model to judge it (text.USABLE_TEXT)."""
    doc = fitz.open()
    for number in range(1, pages + 1):
        page = doc.new_page()
        if number in parts:
            rows = [("Item", "Part No.", "Description", "Qty")]
            rows += [(str(i), f"{number}4667-0{i:02}0", f"Gasket ring {i}", "2") for i in range(1, 24)]
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    page.insert_text((60 + 120 * c, 60 + 16 * r), cell, fontsize=9)
        else:
            page.insert_textbox(fitz.Rect(60, 80, 540, 700), " ".join([PROSE] * 3), fontsize=10)
    doc.save(path)
    return path


def test_include_and_skip_are_learned(tmp_path):
    """People's clicks are what the model learns from; the last click on a page counts."""
    client = TestClient(app.app)
    content = make_book(tmp_path / "three.pdf", {2}, pages=3).read_bytes()
    job = drop(client, {"three.pdf": content})
    f = finished(client, job)["three.pdf"]
    mark = lambda page, spares: client.put(f"/api/files/{f['id']}/pages/{page}", json={"spares": spares})
    before = retrain.decisions()
    assert mark(1, True).status_code == mark(1, False).status_code == mark(2, True).status_code == 200
    assert retrain.decisions() != before   # so the night's retraining runs
    assert mark(7, True).status_code == 404
    sha = hashlib.sha256(content).hexdigest()
    assert {page: label for (s, page), (label, _) in learn.from_corrections()[0].items() if s == sha} == {1: 0, 2: 1}


def test_skipped_parts_pages_are_not_taught_as_not_spares_and_runs_are_learned(tmp_path, monkeypatch):
    """A merge spreadsheet lists pages 4 and 6 of a run of parts pages but not page 5, whose
    parts were typed from another page: page 5 is left out of learning, not taught as "not
    spares". With enough manuals, a second step learns from runs of pages, and the reader
    uses its cut-offs."""
    monkeypatch.setattr(learn, "SECOND_STEP", 3)
    rows = {}
    for m in range(6):
        path = make_book(tmp_path / f"m{m}.pdf", {4, 5, 6}, pages=9)
        listed = {4, 6} if m == 0 else {4, 5, 6}
        for n, reading in text.read_pages(str(path), list(range(1, 10)), lambda *_: None).items():
            rows[(f"manual{m}", n)] = (int(n in listed), reading)
    model = learn.fit(rows)
    assert model["report"]["doubtful pages left out"] == 1 and model["report"]["second step"]
    new = text.read_pages(str(make_book(tmp_path / "new.pdf", {2, 3}, pages=6)), list(range(1, 7)), lambda *_: None)
    chance, unsure_from, spares_from = learn.judge(model, new, 6)
    assert set(chance) == set(range(1, 7))
    assert (unsure_from, spares_from) == (model["second"]["unsure_from"], model["second"]["spares_from"])


def test_it_learns_again_each_morning_and_keeps_only_models_that_are_no_worse(tmp_path, monkeypatch):
    """Nothing new decided: nothing learned. New models that would miss more spares
    pages, or send more of them to "unlikely" than agreed, wait aside; the others are used."""
    monkeypatch.setattr(learn, "MODEL", tmp_path / "model" / "pages.joblib")
    monkeypatch.setattr(retrain, "state_file", lambda: tmp_path / "retrain.json")
    report = lambda missed, unlikely=0: {"pages": 10, "tested on manuals left out": {
        "spares pages missed": missed, "other pages sent to check": 0.4},
        "picture model": {"tested on manuals left out": {"spares check pages sent to unlikely": unlikely}}}
    learn.MODEL.parent.mkdir()
    learn.MODEL.write_bytes(b"in use")
    learn.MODEL.with_suffix(".json").write_text(json.dumps(report(0.010)))
    runs = []

    def learner(result):
        def run(pairs=()):
            runs.append(result)
            (retrain.candidate()).mkdir(parents=True, exist_ok=True)
            (retrain.candidate() / "pages.joblib").write_bytes(b"new")
            (retrain.candidate() / "training.joblib").write_bytes(b"pages learned from")
            (retrain.candidate() / "pages.json").write_text(json.dumps(result))
            return 0
        return run

    for decided, result, used in [("a", report(0.030), b"in use"),      # would miss more than KEEP allows
                                  ("a", report(0.001), b"in use"),      # nothing new decided
                                  ("b", report(0.011, 0.05), b"in use"),   # 5% of spares scans to "unlikely"
                                  ("c", report(0.011), b"new")]:        # no worse, within SLACK
        monkeypatch.setattr(retrain, "decisions", lambda: decided)
        monkeypatch.setattr(retrain, "_learn", learner(result))
        outcome = retrain.daily()
        assert learn.MODEL.read_bytes() == used, outcome
    assert len(runs) == 3 and outcome.startswith("now using")
    assert (learn.MODEL.with_name("pages.previous.joblib").read_bytes(), learn.training_file().read_bytes()) == (
        b"in use", b"pages learned from")
    assert TestClient(app.app).get("/api/health").json()["last retraining"]["outcome"] == outcome
    (tmp_path / "retrain.json").write_text("")   # found empty once: the server answers, and learns again
    assert TestClient(app.app).get("/api/health").json()["last retraining"] == {} and retrain.last() == {}
    # the office PC is on from 08:30 to 18:00: at 09:00, or a minute after it is switched on later
    from datetime import datetime
    monkeypatch.setattr(retrain, "HOUR", 9)
    assert retrain.wait(datetime(2026, 11, 4, 8, 0)) == 3600
    assert retrain.wait(datetime(2026, 11, 4, 9, 40)) == 60
    retrain.remember({"at": "2026-11-04 09:41", "outcome": "", "decisions": ""})
    assert retrain.wait(datetime(2026, 11, 4, 10, 0)) == 23 * 3600


def test_older_merge_spreadsheets_name_each_manual_in_a_heading_row(tmp_path):
    """Older merge files ("with group heading") have no Internal Remark column: a one-cell
    row names the manual, often as a path, sometimes typed a little differently."""
    folder = tmp_path / "vessel"
    (folder / "a").mkdir(parents=True)
    (folder / "b").mkdir()
    make_book(folder / "a" / "PUMP A.pdf", {2})
    (folder / "b" / "PUMP A.pdf").write_bytes((folder / "a" / "PUMP A.pdf").read_bytes())   # a copy: still one manual
    make_book(folder / "a" / "Operators manual.pdf", {1}, pages=3)
    make_book(folder / "b" / "Operators manual.pdf", {1}, pages=5)   # same name, another manual: not guessed
    make_book(folder / "a" / "(M-2-64) MAIN AIR COMPRESSOR (D441400000).pdf", {3})
    sheet = openpyxl.Workbook()
    for row in [["Name", "Spare Part No", "Ref Page "],
                ["Ship\\6 Machinery\\PUMP A.pdf"], ["Gasket", "X-1", 2], ["Ring", "X-2", "2/3"],
                ["Operators manual"], ["Bolt", "X-3", 1],
                ["(M-2-64) MAIN AIR COMPRESOR (D441400000)"], ["Valve", "X-4", 3]]:
        sheet.active.append(row)
    sheet.save(tmp_path / "merge.xlsx")
    listed = learn.spreadsheet_pages(tmp_path / "merge.xlsx")
    assert listed == {"Ship\\6 Machinery\\PUMP A.pdf": {2, 3}, "Operators manual": {1},
                      "(M-2-64) MAIN AIR COMPRESOR (D441400000)": {3}}
    found = learn.match(listed, folder)
    assert {m: p.name for m, p in found.items()} == {"Ship\\6 Machinery\\PUMP A.pdf": "PUMP A.pdf",
        "(M-2-64) MAIN AIR COMPRESOR (D441400000)": "(M-2-64) MAIN AIR COMPRESSOR (D441400000).pdf"}


def test_tracking_sheets_list_each_files_marked_pages(tmp_path):
    """TORM's tracking sheets: one sheet per vessel, a row per file with its marked pages,
    "NA" for a file with none (left out: it teaches nothing about its pages)."""
    book = openpyxl.Workbook()
    first = book.active
    first.title = "Torm Innovation"
    first.append(["Sr.No.", "Parent Folder", "Filename", "Equipment Name", "Marked Pages"])
    first.append([1, "ACCOMMODATION\\", "ACCOMMODATION PLAN.pdf", None, "NA"])
    first.append([2, None, "PUMP A.pdf", "Pump", "6,8,11-13"])
    second = book.create_sheet("Torm Emilie")
    second.append(["Sr.No.", "Parent Folder", "Filename", "Marked Pages"])
    second.append([1, None, "Davit.pdf", "2-3 (see note 4)"])
    book.save(tmp_path / "Allocation Sheet.xlsx")
    assert learn.spreadsheet_pages(tmp_path / "Allocation Sheet.xlsx") == {"PUMP A.pdf": {6, 8, 11, 12, 13},
                                                                          "Davit.pdf": {2, 3}}


def test_typing_allocation_sheets_name_each_manual_by_folder_and_file(tmp_path, monkeypatch):
    """A typing allocation sheet: Path, File and DATA. Two folders hold different
    manuals of one name, so the folder decides. "No Data" is a manual a typist found no
    spares page in: every page of it is not spares. A decision about the manual ("we don't
    take this Spares") and a file that is not a PDF teach nothing, and a file whose folder
    is not here is not here, whatever other folders hold."""
    monkeypatch.setattr(store, "DATA", tmp_path)   # no picture numbers
    folder = tmp_path / "VESSEL MANUALS"
    (folder / "MM-01 Pump").mkdir(parents=True)
    (folder / "MM-02 Boiler").mkdir()
    make_book(folder / "MM-01 Pump" / "Instruction manual.pdf", {2})
    make_book(folder / "MM-02 Boiler" / "Instruction manual.pdf", {1, 3}, pages=5)
    make_book(folder / "MM-02 Boiler" / "Arrangement.pdf", set(), pages=3)
    sheet = openpyxl.Workbook()
    where = "D:\\New folder\\VESSEL MANUALS\\"
    for row in [["Path", "File", "Pages", "DATA", "Type"],
                [where + "MM-01 Pump", "Instruction manual.pdf", 6, "2", "pdf"],
                [where + "MM-02 Boiler", "Instruction manual.pdf", 5, "1, 3", "pdf"],
                [where + "MM-02 Boiler", "Arrangement.pdf", 3, "No Data", "pdf"],
                [where + "MM-02 Boiler", "Valves.pdf", 9, "we don't take this Spares", "pdf"],
                [where + "MM-02 Boiler", "Notes.docx", 2, "1-2", "docx"],
                [where + "AM-05 Toilet", "Instruction manual.pdf", 4, "1", "pdf"]]:   # a folder not on this PC
        sheet.active.append(row)
    sheet.save(tmp_path / "Allocation.xlsx")
    listed = learn.spreadsheet_pages(tmp_path / "Allocation.xlsx")
    assert len(listed) == 4 and sorted(map(len, listed.values())) == [0, 1, 1, 2]
    found = learn.match(listed, folder)
    assert {(manual.split("\\")[-2], path.parent.name, path.stem) for manual, path in found.items()} == {
        ("MM-01 Pump", "MM-01 Pump", "Instruction manual"), ("MM-02 Boiler", "MM-02 Boiler", "Instruction manual"),
        ("MM-02 Boiler", "MM-02 Boiler", "Arrangement")}
    (folder / "MM-01 Pump" / "Broken.pdf").write_bytes(b"%PDF- cut off")   # a damaged file in the folder
    sheet.active.append([where + "MM-01 Pump", "Broken.pdf", 4, "1", "pdf"])
    sheet.save(tmp_path / "Allocation.xlsx")
    rows, _, missing = learn.from_spreadsheet(tmp_path / "Allocation.xlsx", folder)   # left out, not a stop
    assert sorted(label for label, _ in rows.values()) == [0] * 11 + [1] * 3
    assert missing == [where + r"AM-05 Toilet\Instruction manual.pdf"]   # not matched to a manual of that name elsewhere


def test_it_learns_from_a_merge_spreadsheet(tmp_path, monkeypatch):
    """Train from a spreadsheet the way finished merge files are laid out, then read a
    manual it has never seen with what it learned."""
    monkeypatch.setattr(learn, "MODEL", tmp_path / "model" / "pages.joblib")
    monkeypatch.setattr(store, "DATA", tmp_path)   # no corrections from the screen
    folder = tmp_path / "manuals"
    folder.mkdir()
    sheet = openpyxl.Workbook()
    sheet.active.append(["Name", "Spare Part No", "Internal Remark", "Ref Page "])
    for book, parts in {"M6-1 PUMP": {2, 5}, "M7-1 Centrifugal pump": {1, 4},
                        "F-23 STEERING GEAR": {3, 6}, "M10-1 OIL PURIFIER": {2, 3}}.items():
        make_book(folder / f"{book.lower()}.pdf", parts)
        for page in parts:   # pages as people type them: one at a time, or several with "/"
            sheet.active.append(["Gasket", "X-1", book, page if page % 2 else f"{page}/{page}"])
    sheet.active.append(["Bolt", "X-2", "M99 NOT IN THE FOLDER", "4"])
    sheet.save(tmp_path / "merge.xlsx")

    learn.main([str(tmp_path / "merge.xlsx"), str(folder)])
    report = learn.load()["report"]
    assert (report["manuals"], report["pages"], report["spares pages"]) == (4, 24, 8)

    # learning again needs nothing outside the model folder: not the spreadsheet, not the manuals
    (tmp_path / "merge.xlsx").unlink()
    for pdf in folder.iterdir():
        pdf.unlink()
    learn.main(["--again"])
    again = learn.load()["report"]
    assert (again["manuals"], again["pages"], again["spares pages"], again["sources"]) == (4, 24, 8, ["merge.xlsx"])

    got = classify.classify_pdf(make_book(tmp_path / "new.pdf", {1, 4}))
    assert got[1] == got[4] == tables.SPARES
    assert all(got[n] == tables.NO for n in (2, 3, 5, 6)), got
