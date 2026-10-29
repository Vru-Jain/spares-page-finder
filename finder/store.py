"""The database: one SQLite file, data/finder.db (FINDER_DATA moves the data folder).

    jobs      one per drop; archived once its manuals have been removed (archive.py)
    files     one per manual in a drop: where it is kept, its path in the drop, its state
    pages     one per page: what the tool decided (bucket) and what a person decided
              (label). The labels are what finder/learn.py learns from.
    readings  what the text layer of each page a person decided says, kept when the
              manual itself is removed, so the tool can still learn from it

Files kept beside it: uploads/<sha256>.pdf (each manual once), uploads/partial/ (manuals
still arriving), looks/<sha256>.npz (each manual's picture numbers, looks.py),
archive/job<n>.zip (what is left of an archived drop), out/ (downloads being made),
backups/.

    python -m finder.store      a dated copy in data/backups (backup.bat does this)
"""
from __future__ import annotations

import datetime
import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

DATA = Path(os.environ.get("FINDER_DATA", Path(__file__).parent.parent / "data"))


def pdf(stored: str) -> Path:
    """Where a dropped manual is, from the path kept in the files table: relative to the
    data folder, so the tool keeps working when its folder is moved or copied."""
    return DATA / stored


def partial(fid: int) -> Path:
    """A manual still arriving, piece by piece."""
    return DATA / "uploads" / "partial" / f"{fid}.part"


@contextmanager
def db():
    """A connection that commits when the block ends, or rolls back on an error."""
    con = sqlite3.connect(DATA / "finder.db", timeout=30)
    con.row_factory = sqlite3.Row
    try:
        with con:
            yield con
    finally:
        con.close()


def setup() -> None:
    """Create the folders and tables, and bring an older database up to date."""
    for folder in ("uploads/partial", "out", "archive"):
        (DATA / folder).mkdir(parents=True, exist_ok=True)
    with db() as con:
        con.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs(
                id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS files(
                id INTEGER PRIMARY KEY, job_id INTEGER, path TEXT, rel TEXT, sha TEXT,
                status TEXT,              -- unsent | skipped | queued | running | done | failed | stopped
                note TEXT DEFAULT '',     -- why it was skipped, or why it failed
                pages_total INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS pages(
                file_id INTEGER, page INTEGER,
                bucket TEXT,              -- spares | check | no: what the tool decided
                label INTEGER,            -- 1 | 0: what a person decided
                PRIMARY KEY(file_id, page));
            CREATE TABLE IF NOT EXISTS readings(
                file_id INTEGER, page INTEGER,
                reading TEXT,             -- text.page_reading, as JSON
                PRIMARY KEY(file_id, page));
        """)
        if "archived" not in [c["name"] for c in con.execute("PRAGMA table_info(jobs)")]:
            con.execute("ALTER TABLE jobs ADD COLUMN archived TEXT")   # when its manuals were removed
        columns = [c["name"] for c in con.execute("PRAGMA table_info(files)")]
        if "reader" not in columns:   # which version of the reader decided the pages (jobs.READER)
            con.execute("ALTER TABLE files ADD COLUMN reader TEXT")
        if "size" not in columns:     # bytes, to recognise a folder dropped again
            con.execute("ALTER TABLE files ADD COLUMN size INTEGER")
        # Before 2026-10-29 the path was kept in full (D:\spares-finder\data\uploads\...): keep
        # only the part from "uploads", so a moved or copied folder still finds its manuals.
        for row in con.execute("SELECT id, path FROM files WHERE path LIKE '%uploads%'").fetchall():
            parts = PurePosixPath(row["path"].replace("\\", "/")).parts
            if parts[0] != "uploads" and "uploads" in parts:
                con.execute("UPDATE files SET path=? WHERE id=?",
                            ("/".join(parts[parts.index("uploads"):]), row["id"]))
        con.execute("UPDATE files SET status='queued' WHERE status='running'")   # cut off by a restart
        # Before 2026-10-26 a PDF with no readable pages was wrongly reported as read, with nothing found.
        con.execute("UPDATE files SET status='failed', note='No pages could be found in it: the PDF is "
                    "damaged or incomplete' WHERE status='done' AND pages_total=0")


def backup(keep: int = 14) -> Path:
    """A copy of the database in data/backups, named by date and time, made with SQLite's
    own backup so it is consistent even while the server is running. The newest `keep`
    copies are kept. People's marks and checked lists are the training data: worth keeping."""
    folder = DATA / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"finder-{datetime.datetime.now():%Y-%m-%d-%H%M}.db"
    source, copy = sqlite3.connect(DATA / "finder.db"), sqlite3.connect(dest)
    try:
        source.backup(copy)
    finally:
        copy.close()
        source.close()
    for old in sorted(folder.glob("finder-*.db"))[:-keep]:
        old.unlink()
    return dest


if __name__ == "__main__":
    if not (DATA / "finder.db").exists():
        sys.exit(f"No database at {DATA / 'finder.db'}")
    print(f"Backed up to {backup()}")
