"""Keeping storage in check: after KEEP_DAYS a drop is archived.

Whole vessels pass through the tool, so original manuals cannot be kept for ever. When
a drop is KEEP_DAYS old (and nothing in it is still being read):
  1. its downloads are made once more and kept: archive/job<n>.zip, every manual's
     included pages as a PDF (not the whole tagged manual, which is what archiving frees)
     and the allocation sheet (which also stays available from the database)
  2. what the text layer of every page a person decided says is kept in the database
     (table readings), so the page model can still learn from those decisions
  3. its original manuals are deleted, except any another, newer drop still uses, and
     with them their picture numbers (looks.py), unless a person decided one of their
     pages: those the picture model still learns from
The screen then shows the drop as archived, with its zip to download.

A drop can also be deleted from the screen, at any age (delete): its manuals, picture numbers
and zip go the same way, at once; only what people decided on its pages is kept, unnamed,
for learning.

    FINDER_KEEP_DAYS     days to keep originals (default 14)
"""
from __future__ import annotations

import json
import os
import shutil
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
            "page pictures": there("pictures"), "backups": there("backups")}


def due() -> list[int]:
    """Drops old enough to archive, with nothing in them still waiting to be read."""
    with store.db() as con:
        return [r["id"] for r in con.execute(
            "SELECT id FROM jobs WHERE archived IS NULL AND deleted IS NULL AND created <= datetime('now', ?) "
            "AND NOT EXISTS (SELECT 1 FROM files WHERE job_id=jobs.id AND status IN ('queued', 'running'))",
            (f"-{KEEP_DAYS} days",))]


def _manuals(job: int) -> list:
    with store.db() as con:
        return con.execute("SELECT id, path, sha FROM files WHERE job_id=? AND path IS NOT NULL", (job,)).fetchall()


def _keep_readings(files) -> None:
    """Keep in the database what the text layer of every page a person decided says, so the
    page model can still learn from those decisions once the manual is gone."""
    from finder import text   # Parsel's table passes: loaded only when a drop's manuals go
    with store.db() as con:
        decided = {f["id"]: [r["page"] for r in con.execute(
            "SELECT page FROM pages WHERE file_id=? AND label IS NOT NULL", (f["id"],))] for f in files}
    keep = []
    for f in files:
        if decided[f["id"]] and store.pdf(f["path"]).exists():
            with outputs._PDF, fitz.open(store.pdf(f["path"])) as doc:
                keep += [(f["id"], n, json.dumps(r)) for n in decided[f["id"]] if (r := text.reading(doc, n))]
    with store.db() as con:
        con.executemany("INSERT OR REPLACE INTO readings(file_id, page, reading) VALUES(?,?,?)", keep)


def _unlink(path: Path, wait: float = 10) -> None:
    """Delete a file; if Windows says it is still open (a manual the reader has just been
    stopped on lets go of it a moment later), try again for up to `wait` seconds."""
    until = time.monotonic() + wait
    while True:
        try:
            return path.unlink(missing_ok=True)
        except PermissionError:
            if time.monotonic() > until:
                raise
            time.sleep(0.25)


def _remove_manuals(job: int, files) -> None:
    """Delete a drop's manuals, except any a drop still open uses, and with them their picture
    numbers, unless a person decided one of their pages: those the picture model still learns from."""
    with store.db() as con:
        still_used = {r["path"] for r in con.execute(
            "SELECT DISTINCT f.path FROM files f JOIN jobs j ON j.id = f.job_id "
            "WHERE j.archived IS NULL AND j.deleted IS NULL AND f.path IS NOT NULL")}
        ids = [r["id"] for r in con.execute("SELECT id FROM files WHERE job_id=?", (job,))]
        taught = {r["sha"] for r in con.execute(
            "SELECT DISTINCT f.sha FROM files f JOIN pages p ON p.file_id = f.id WHERE p.label IS NOT NULL")}
    for f in files:
        if f["path"] not in still_used:
            _unlink(store.pdf(f["path"]))
            shutil.rmtree(outputs.pictures_of(f["path"]), ignore_errors=True)
            if f["sha"] and f["sha"] not in taught:
                looks.file_of(f["sha"]).unlink(missing_ok=True)
    for fid in ids:   # pieces of manuals that never finished arriving
        _unlink(store.partial(fid))


def archive(job: int) -> None:
    made = outputs.bundle(job, whole=False)   # whole manuals would keep what archiving frees
    made.replace(zip_of(job))
    files = _manuals(job)
    _keep_readings(files)
    with store.db() as con:
        con.execute("UPDATE jobs SET archived=CURRENT_TIMESTAMP WHERE id=?", (job,))
    _remove_manuals(job, files)


def delete(job: int) -> None:
    """A drop someone deleted (the server has already taken it out of the list and stopped its
    reading): its manuals, their picture numbers and its zip go, as archiving removes them. The
    manuals with a page a person decided stay in the database, under no drop and with no name,
    with what those pages say, so the models go on learning from them; the rest of the drop is
    forgotten. sweep() finishes it if the server stopped part-way."""
    files = _manuals(job)
    _keep_readings(files)
    _remove_manuals(job, files)
    zip_of(job).unlink(missing_ok=True)
    with store.db() as con:
        forget = [(r["id"],) for r in con.execute(
            "SELECT id FROM files WHERE job_id=? AND NOT (status='done' AND id IN "
            "(SELECT file_id FROM pages WHERE label IS NOT NULL))", (job,))]
        for table, column in (("pages", "file_id"), ("readings", "file_id"), ("files", "id")):
            con.executemany(f"DELETE FROM {table} WHERE {column}=?", forget)
        con.execute("UPDATE files SET job_id=NULL, path=NULL, rel='' WHERE job_id=?", (job,))


def sweep() -> list[int]:
    """Finish deleting any drop a stop of the server cut short, then archive every drop that
    is due. Returns the drops archived."""
    with store.db() as con:
        unfinished = [r["id"] for r in con.execute(
            "SELECT id FROM jobs WHERE deleted IS NOT NULL AND EXISTS (SELECT 1 FROM files WHERE job_id=jobs.id)")]
    for job in unfinished:
        try:
            delete(job)
        except Exception as exc:   # a manual still open, say: tried again next hour
            print(f"deleting drop {job} failed: {exc}", flush=True)
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
