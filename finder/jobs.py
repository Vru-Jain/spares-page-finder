"""The reading queue: manuals are read one at a time, by a separate reader process.

The reader (python -m finder.classify --serve) runs on its own so that a PDF that
crashes or hangs it cannot take the server down, and it is kept from manual to manual
so its worker processes start only once. It is sent one manual's path at a time and
answers with @progress lines, then @pages or @failed.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path

from finder import store
from finder.store import db

HERE = Path(__file__).parent.parent
# Bump this whenever classify.py, text.py, pictures.py, looks.py or tables.py change how
# pages are read or judged: an identical file read before is then read again, not reused.
READER = "8"
# ponytail: one flat limit per manual. Scale it by page count if big scans hit it.
TIME_LIMIT = 2 * 60 * 60
PROGRESS: dict[int, str] = {}   # file id -> "done total", while it is being read

_todo: queue.Queue[int] = queue.Queue()
_reader: subprocess.Popen | None = None
_now = {"file": None}           # the manual the reader is on, so that Stop can end it
_now_lock = threading.Lock()


def add(fid: int) -> None:
    """Queue a manual for reading."""
    _todo.put(fid)


def start() -> None:
    """Queue what was waiting when the server last stopped, and start the worker thread."""
    with db() as con:
        for row in con.execute("SELECT id FROM files WHERE status='queued' ORDER BY id"):
            _todo.put(row["id"])
    threading.Thread(target=_work, daemon=True).start()


def abandon(file_ids: list[int]) -> None:
    """Stop: if the reader is on one of these manuals, end it there. The caller has
    already marked them stopped, so their pages are not saved."""
    with _now_lock:
        if _now["file"] in file_ids and _reader is not None:
            _end(_reader)   # the next manual starts a new reader


def _work() -> None:
    while True:
        fid = _todo.get()
        try:
            _process(fid)
        except Exception as exc:   # one bad PDF must not stop the queue
            with db() as con:
                con.execute("UPDATE files SET status='failed', note=? WHERE id=? AND status='running'",
                            (str(exc)[:300], fid))


def _process(fid: int) -> None:
    # "AND status='running'" in the updates below: a manual someone stopped stays stopped
    with db() as con:
        row = con.execute("SELECT path FROM files WHERE id=? AND status='queued'", (fid,)).fetchone()
        if not row:
            return
        con.execute("UPDATE files SET status='running' WHERE id=?", (fid,))
    path = store.pdf(row["path"])
    with open(path, "rb") as fh:
        sha = hashlib.file_digest(fh, "sha256").hexdigest()
    with db() as con:
        twin = con.execute("SELECT id, pages_total FROM files WHERE sha=? AND reader=? "
                           "AND status='done' AND pages_total > 0", (sha, READER)).fetchone()
        if twin:   # the same manual again (sister ship, re-run): reuse its pages and people's marks
            if con.execute("UPDATE files SET status='done', sha=?, reader=?, pages_total=? "
                           "WHERE id=? AND status='running'", (sha, READER, twin["pages_total"], fid)).rowcount:
                con.execute("INSERT OR REPLACE INTO pages SELECT ?, page, bucket, label FROM pages "
                            "WHERE file_id=?", (fid, twin["id"]))
            return
    pages = _read(fid, str(path))
    with db() as con:
        if con.execute("UPDATE files SET status='done', sha=?, reader=?, pages_total=? "
                       "WHERE id=? AND status='running'", (sha, READER, len(pages), fid)).rowcount:
            con.executemany("INSERT OR REPLACE INTO pages(file_id, page, bucket) VALUES(?,?,?)",
                            [(fid, page, b) for page, b in pages.items()])


def _read(fid: int, path: str) -> dict[int, str]:
    """{page: bucket} from the reader, for one manual."""
    with _now_lock:   # Stop marks the file first and looks here second: one of the two sees the other
        with db() as con:
            if not con.execute("SELECT 1 FROM files WHERE id=? AND status='running'", (fid,)).fetchone():
                raise RuntimeError("stopped")
        _now["file"] = fid
        proc = _start_reader()
    watchdog = threading.Timer(TIME_LIMIT, _end, (proc,))
    watchdog.start()
    pages, failed, tail = None, None, []
    try:
        proc.stdin.write(path + "\n")
        proc.stdin.flush()
        for line in proc.stdout:
            if line.startswith("@progress "):
                PROGRESS[fid] = line[10:].strip()
            elif line.startswith("@pages "):
                pages = json.loads(line[7:])
                break
            elif line.startswith("@failed "):
                failed = line[8:].strip()
                break
            else:
                tail = (tail + [line.strip()])[-5:]
    except OSError:   # the reader had died before this manual: the next one starts afresh
        pass
    finally:
        watchdog.cancel()
        PROGRESS.pop(fid, None)
        with _now_lock:
            _now["file"] = None
    if failed:
        raise RuntimeError(failed)
    if pages is None:   # ended by the watchdog or by Stop, or crashed: the next manual starts a new one
        _end(proc)
        proc.wait()
        raise RuntimeError(" ".join(tail) or "the reader stopped unexpectedly")
    return {int(page): b for page, b in pages.items()}


def _start_reader() -> subprocess.Popen:
    """The running reader, or a new one if there is none."""
    global _reader
    if _reader is None or _reader.poll() is not None:
        _reader = subprocess.Popen(
            [sys.executable, "-m", "finder.classify", "--serve"], cwd=HERE,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            start_new_session=os.name != "nt")   # Linux, macOS: its own group, ended as one (see _end)
    return _reader


def _end(proc: subprocess.Popen) -> None:
    """End the reader and the worker processes it started."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.kill()
