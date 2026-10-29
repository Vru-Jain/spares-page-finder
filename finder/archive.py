"""Keeping storage in check: after KEEP_DAYS a drop is archived.

Whole vessels pass through the tool, so original manuals cannot be kept for ever. When
a drop is KEEP_DAYS old (and nothing in it is still being read):
  1. its downloads are made once more and kept: archive/job<n>.zip, every manual's
     spares PDF and the Excel list (the Excel list also stays available from the database)
  2. what the text layer of every page a person decided says is kept in the database
     (table readings), so the page model can still learn from those decisions
  3. its original manuals are deleted, except any another, newer drop still uses, and
     with them their picture numbers (looks.py), unless a person decided one of their
     pages: those the picture model still learns from
The screen then shows the drop as archived, with its zip to download.

    FINDER_KEEP_DAYS     days to keep originals (default 14)
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pymupdf as fitz

from finder import looks, outputs, store

KEEP_DAYS = int(os.environ.get("FINDER_KEEP_DAYS", 14))


def zip_of(job: int) -> Path:
    return store.DATA / "archive" / f"job{job}.zip"


def usage() -> dict:
    """Megabytes on disk, by kind."""
    size = lambda folder: round(sum(f.stat().st_size for f in (store.DATA / folder).rglob("*") if f.is_file()) / 1e6)
    there = lambda folder: size(folder) if (store.DATA / folder).exists() else 0
    return {"manuals": size("uploads"), "archived drops": size("archive"), "picture numbers": there("looks"),
            "backups": there("backups")}


def due() -> list[int]:
    """Drops old enough to archive, with nothing in them still waiting to be read."""
    with store.db() as con:
        return [r["id"] for r in con.execute(
            "SELECT id FROM jobs WHERE archived IS NULL AND created <= datetime('now', ?) "
            "AND NOT EXISTS (SELECT 1 FROM files WHERE job_id=jobs.id AND status IN ('queued', 'running'))",
            (f"-{KEEP_DAYS} days",))]


def archive(job: int) -> None:
    from finder import text   # Parsel's table passes: loaded only when a drop is archived
    made = outputs.bundle(job)
    made.replace(zip_of(job))
    with store.db() as con:
        files = con.execute("SELECT id, path, sha FROM files WHERE job_id=? AND path IS NOT NULL", (job,)).fetchall()
        decided = {f["id"]: [r["page"] for r in con.execute(
            "SELECT page FROM pages WHERE file_id=? AND label IS NOT NULL", (f["id"],))] for f in files}
    keep = []
    for f in files:
        if decided[f["id"]] and store.pdf(f["path"]).exists():
            with outputs._PDF, fitz.open(store.pdf(f["path"])) as doc:
                keep += [(f["id"], n, json.dumps(r)) for n in decided[f["id"]] if (r := text.reading(doc, n))]
    with store.db() as con:
        con.executemany("INSERT OR REPLACE INTO readings(file_id, page, reading) VALUES(?,?,?)", keep)
        con.execute("UPDATE jobs SET archived=CURRENT_TIMESTAMP WHERE id=?", (job,))
        still_used = {r["path"] for r in con.execute(
            "SELECT DISTINCT f.path FROM files f JOIN jobs j ON j.id = f.job_id "
            "WHERE j.archived IS NULL AND f.path IS NOT NULL")}
        ids = [r["id"] for r in con.execute("SELECT id FROM files WHERE job_id=?", (job,))]
        taught = {r["sha"] for r in con.execute(
            "SELECT DISTINCT f.sha FROM files f JOIN pages p ON p.file_id = f.id WHERE p.label IS NOT NULL")}
    for f in files:
        if f["path"] not in still_used:
            store.pdf(f["path"]).unlink(missing_ok=True)
            if f["sha"] and f["sha"] not in taught:
                looks.file_of(f["sha"]).unlink(missing_ok=True)
    for fid in ids:   # pieces of manuals that never finished arriving
        store.partial(fid).unlink(missing_ok=True)


def sweep() -> list[int]:
    """Archive every drop that is due. Returns the drops archived."""
    done = []
    for job in due():
        try:
            archive(job)
            done.append(job)
        except Exception as exc:   # one drop must not stop the rest; tried again next hour
            print(f"archiving drop {job} failed: {exc}", flush=True)
    return done


def start() -> None:
    """Look for drops to archive now, then every hour."""
    def run():
        while True:
            sweep()
            time.sleep(3600)
    threading.Thread(target=run, daemon=True).start()
