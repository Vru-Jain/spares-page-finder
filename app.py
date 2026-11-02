"""Spares Page Finder's web server: the screen (web/index.html) and the requests it makes.
The work itself is in finder/; finder/__init__.py lists its parts.

    run.bat        (or: python -m uvicorn app:app --port 8600)

It can be published to the internet through Cloudflare Tunnel (see README.md). Two of
Cloudflare's limits shape it: a request may carry at most 100 MB, so manuals are sent
in 16 MB pieces, and a request must be answered within 100 seconds, so a whole drop's
zip is made in the background before it is fetched.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import threading
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.middleware.trustedhost import TrustedHostMiddleware

from finder import archive, jobs, outputs, retrain, store, triage

HERE = Path(__file__).parent
MAX_UPLOAD = 256 * 1024 ** 2    # per request: the screen sends 16 MB pieces
MAX_MANUAL = 4 * 1024 ** 3      # one manual; the largest seen so far was 316 MB
MIN_FREE = 5 * 1024 ** 3        # disk left free for the database, backups and the PC itself

store.setup()
jobs.start()
archive.start()
retrain.start()
app = FastAPI(title="Spares Page Finder", docs_url=None, redoc_url=None, openapi_url=None)


# ---- who may use it --------------------------------------------------------

def this_pc() -> list[str]:
    """The names a browser can reach this PC by. Any other Host header is refused, so a
    web page elsewhere cannot point its own domain name at this PC and read the manuals
    (DNS rebinding). FINDER_HOSTS adds names, comma-separated: a DNS alias, or the public
    address Cloudflare Tunnel publishes it at."""
    name = socket.gethostname()
    names = {"localhost", "127.0.0.1", name, socket.getfqdn()}
    try:
        names.update(socket.gethostbyname_ex(name)[2])   # its own addresses
    except OSError:
        pass
    names.update(n.strip() for n in os.environ.get("FINDER_HOSTS", "").split(",") if n.strip())
    return sorted(names | {n.lower() for n in names})   # browsers send the name in lower case


app.add_middleware(TrustedHostMiddleware, allowed_hosts=this_pc())


@app.middleware("http")
async def same_site_only(request: Request, call_next):
    """A page on another site may not make this one change anything (upload, re-read,
    decide): browsers send Origin with such requests, so it must name this server."""
    origin = request.headers.get("origin")
    if (request.method not in ("GET", "HEAD") and origin is not None
            and origin.split("://", 1)[-1] != request.headers.get("host")):
        return Response("Refused: the request came from another site", status_code=403)
    # A body sent in chunks has no length to check against MAX_UPLOAD: browsers always
    # send the length, so such a request is not the screen's.
    if "content-length" not in request.headers and request.headers.get("transfer-encoding"):
        return Response("Refused: the request did not say how long it is", status_code=411)
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD:
        return Response("Refused: too much in one request", status_code=413)
    return await call_next(request)


# ---- adding manuals, and stopping or restarting their reading ---------------
#
# A drop is sent in two steps: first the list of manuals, which the server checks by
# name (skipped files are never sent), then each manual in 16 MB pieces. Reading starts
# with the first manual to arrive. A drop cut short keeps what arrived: the rest show
# as "Not sent", and dropping the same folder again sends only those, each from the
# last piece that arrived. Identical manuals are kept once (data/uploads/<sha256>.pdf).

class Listed(BaseModel):
    rel: str        # the path inside the dropped folder, or just the file name
    size: int = 0


class Drop(BaseModel):
    files: list[Listed]


def clean_rel(rel: str) -> str:
    """The browser's path for a file is only ever shown and used to name downloads,
    never to touch the disk; even so, nothing in it may climb out of its folder, and no
    part may hold what Windows refuses in a name (a drive's ":", control characters),
    since it names the files in the downloaded zip."""
    parts = [re.sub(r'[\x00-\x1f<>:"|?*]', "_", part).strip(" .")
             for part in re.split(r"[\\/]+", rel) if part not in ("", ".", "..")]
    return "/".join(part for part in parts if part)[-400:] or "manual.pdf"


@app.post("/api/jobs")
def new_job(drop: Drop):
    if not drop.files:
        raise HTTPException(400, "Please drop PDF files")
    if len(drop.files) > 5000:
        raise HTTPException(400, "Too many files at once. Drop one vessel at a time.")
    made = []
    with store.db() as con:
        job = con.execute("INSERT INTO jobs DEFAULT VALUES").lastrowid
        for f in drop.files:
            rel = clean_rel(f.rel)
            reason = triage.skip_reason(PurePosixPath(rel).name) or ("larger than 4 GB" if f.size > MAX_MANUAL else "")
            fid = con.execute("INSERT INTO files(job_id, rel, size, status, note) VALUES(?,?,?,?,?)",
                              (job, rel, max(f.size, 0), "skipped" if reason else "unsent", reason)).lastrowid
            made.append({"id": fid, "skip": reason})
    return {"id": job, "files": made}


_SENDING: dict[int, asyncio.Lock] = {}   # one piece of a manual at a time, so two cannot interleave


@app.put("/api/files/{fid}/content")
async def send(fid: int, request: Request, offset: int = 0):
    """One piece of a manual, starting at byte `offset`. Answers how much has arrived;
    a piece that does not start where the last one ended is refused (409) with that
    number, so the screen carries on from there. Writing stops the moment more arrives
    than the manual's size, and nothing is taken when the disk is nearly full."""
    with store.db() as con:
        row = con.execute("SELECT size FROM files WHERE id=? AND status='unsent'", (fid,)).fetchone()
    if not row:
        raise HTTPException(409, "That manual has already been sent")
    async with _SENDING.setdefault(fid, asyncio.Lock()):
        part = store.partial(fid)
        have = part.stat().st_size if part.exists() else 0
        if offset != have:
            return JSONResponse({"detail": "Out of step", "received": have}, status_code=409)
        if shutil.disk_usage(store.DATA).free < MIN_FREE + (row["size"] - have):
            raise HTTPException(507, "The server's disk is nearly full. Ask whoever looks after it to make room.")
        with part.open("ab") as out:
            async for chunk in request.stream():
                if have + len(chunk) > row["size"]:
                    out.close()
                    part.unlink()
                    raise HTTPException(400, "More arrived than the file's size. Please drop it again.")
                out.write(chunk)
                have += len(chunk)
        if have == row["size"]:
            _finish(fid, part)
            _SENDING.pop(fid, None)
    return {"received": have}


def _finish(fid: int, part: Path) -> None:
    """A whole manual has arrived: keep it once under its fingerprint, and queue it."""
    with part.open("rb") as fh:
        sha = hashlib.file_digest(fh, "sha256").hexdigest()
    with part.open("rb") as fh:
        is_pdf = b"%PDF-" in fh.read(1024)   # where the PDF format puts its marker
    kept = f"uploads/{sha}.pdf"
    if not is_pdf or store.pdf(kept).exists():   # not a PDF, or the same manual was sent before
        part.unlink()
    else:
        part.replace(store.pdf(kept))
    with store.db() as con:
        changed = con.execute("UPDATE files SET path=?, sha=?, status=?, note=? WHERE id=? AND status='unsent'",
                              (kept if is_pdf else None, sha, "queued" if is_pdf else "skipped",
                               "" if is_pdf else "not a PDF file", fid)).rowcount
    if changed and is_pdf:
        jobs.add(fid)


@app.post("/api/jobs/{job}/stop")
def stop_job(job: int):
    """Stop reading a drop: manuals still waiting are not started, and the one being
    read is abandoned part-way. Manuals already read keep their pages."""
    with store.db() as con:
        halted = [r["id"] for r in con.execute(
            "SELECT id FROM files WHERE job_id=? AND status IN ('queued', 'running')", (job,))]
        con.execute("UPDATE files SET status='stopped' WHERE job_id=? AND status IN ('queued', 'running')", (job,))
    jobs.abandon(halted)
    return {"stopped": len(halted)}


@app.post("/api/jobs/{job}/continue")
def continue_job(job: int):
    """Read the manuals a Stop left unread, from the start of each."""
    with store.db() as con:
        ids = [r["id"] for r in con.execute(
            "SELECT id FROM files WHERE job_id=? AND status='stopped' ORDER BY rel", (job,))]
        con.execute("UPDATE files SET status='queued' WHERE job_id=? AND status='stopped'", (job,))
    for fid in ids:
        jobs.add(fid)
    return {"queued": len(ids)}


@app.post("/api/files/{fid}/include")
def include(fid: int):
    """Read a skipped document after all, a stopped one, or try a failed one again. A
    file skipped by its name was never sent: it is then waiting to be sent."""
    with store.db() as con:
        row = con.execute("SELECT path FROM files WHERE id=? AND status IN ('skipped', 'failed', 'stopped')",
                          (fid,)).fetchone()
        if not row:
            raise HTTPException(409, "That file is already being read")
        con.execute("UPDATE files SET status=?, note='' WHERE id=?", ("queued" if row["path"] else "unsent", fid))
    if row["path"]:
        jobs.add(fid)
    return {"ok": True, "send": not row["path"]}


# ---- results, and people's decisions ----------------------------------------

def drop_name(rels: list[str]) -> str:
    """What a drop is called on the screen: the folder it was dropped as, its one
    manual's name, or the first manual's name and how many more."""
    tops = {rel.split("/")[0] for rel in rels}
    if rels and len(tops) == 1 and all("/" in rel for rel in rels):
        return tops.pop()
    names = sorted(PurePosixPath(rel).name for rel in rels)
    return names[0] if len(names) == 1 else f"{names[0]} and {len(names) - 1} more"


@app.get("/api/jobs")
def recent_jobs(limit: int = 50):
    """The newest drops, for the list on the screen, so people can go back to one: what
    each is called, when it came, how far its reading is, how many pages wait for a person."""
    out = []
    with store.db() as con:
        for j in con.execute("SELECT id, created, archived FROM jobs ORDER BY id DESC LIMIT ?",
                             (min(max(limit, 1), 200),)).fetchall():
            files = con.execute("SELECT rel, status FROM files WHERE job_id=?", (j["id"],)).fetchall()
            if not files:
                continue
            count = lambda *states: sum(f["status"] in states for f in files)
            to_check = con.execute(
                "SELECT count(*) FROM pages p JOIN files f ON f.id = p.file_id WHERE f.job_id=? AND f.status='done' "
                "AND p.label IS NULL AND p.bucket IN ('check', 'unlikely')", (j["id"],)).fetchone()[0]
            out.append({"id": j["id"], "name": drop_name([f["rel"] for f in files]), "created": j["created"],
                        "archived": j["archived"], "manuals": len(files), "done": count("done"),
                        "reading": count("queued", "running"), "waiting": count("unsent", "stopped"),
                        "failed": count("failed"), "to check": to_check})
    return out


@app.get("/api/jobs/{job}")
def job_detail(job: int):
    with store.db() as con:
        found = con.execute("SELECT created, archived FROM jobs WHERE id=?", (job,)).fetchone()
        if not found:
            raise HTTPException(404, "No such job")
        files = [dict(r) for r in con.execute(
            "SELECT id, rel, size, status, note, pages_total FROM files WHERE job_id=? ORDER BY rel", (job,))]
        # ponytail: one query per file on every poll. Fine for hundreds of files.
        for f in files:
            f["progress"] = jobs.PROGRESS.get(f["id"], "")
            f["pages"] = [dict(r) for r in con.execute(
                "SELECT page, bucket, label FROM pages WHERE file_id=? "
                "AND (bucket != 'no' OR label IS NOT NULL) ORDER BY page", (f["id"],))]
    return {"id": job, "name": drop_name([f["rel"] for f in files]) if files else "Empty drop",
            "created": found["created"], "archived": found["archived"], "keep_days": archive.KEEP_DAYS, "files": files}


def the_file(fid: int) -> dict:
    """{"path": where a manual is kept, "rel": the path it was dropped with}."""
    with store.db() as con:
        row = con.execute("SELECT path, rel FROM files WHERE id=?", (fid,)).fetchone()
    if not row or not row["path"]:
        raise HTTPException(404, "No such file")
    if not store.pdf(row["path"]).exists():
        raise HTTPException(404, f"The manual was removed after {archive.KEEP_DAYS} days to save space")
    return {"path": store.pdf(row["path"]), "rel": row["rel"]}


@app.get("/api/files/{fid}/page/{page}.png")
def page_image(fid: int, page: int, big: int = 0, turn: int = 0):
    """A page's picture: big=0 for the list, 1 for the enlarged view, 2 sharp enough to
    zoom into a large drawing (sent as JPEG, which keeps an A0 sheet to a few MB)."""
    got = outputs.page_png(the_file(fid)["path"], page, big, turn)
    if got is None:
        raise HTTPException(404, "No such page")
    picture, kind = got
    return Response(picture, media_type=kind, headers={"Cache-Control": "max-age=86400"})


class Mark(BaseModel):
    spares: bool


@app.put("/api/files/{fid}/pages/{page}")
def mark(fid: int, page: int, body: Mark):
    """Include or Skip on one page."""
    with store.db() as con:
        changed = con.execute("UPDATE pages SET label=? WHERE file_id=? AND page=?",
                              (body.spares, fid, page)).rowcount
    if not changed:
        raise HTTPException(404, "No such page")
    return {"ok": True}


@app.post("/api/files/{fid}/leave-out-unlikely")
def leave_out_unlikely(fid: int):
    """Leave all out: every page still under "Probably not spares" is marked not spares,
    as if Skip had been pressed on each. Pages someone already decided keep their mark."""
    with store.db() as con:   # the pages it marked, so the screen shows what happened, not what it expected
        left = [r["page"] for r in con.execute(
            "UPDATE pages SET label=0 WHERE file_id=? AND bucket='unlikely' AND label IS NULL RETURNING page", (fid,))]
    return {"left out": sorted(left)}


# ---- downloads ----------------------------------------------------------------

def sent(path: Path, filename: str) -> FileResponse:
    """Send a download made in data/out, and delete it once it has gone."""
    return FileResponse(path, filename=filename, background=BackgroundTask(path.unlink, missing_ok=True))


@app.get("/api/files/{fid}/spares.pdf")
def download(fid: int):
    manual = the_file(fid)
    with store.db() as con:
        pages = outputs.chosen(con, fid)
    if not pages:
        raise HTTPException(404, "No spares pages were found in this manual")
    return sent(outputs.spares_pdf(manual["path"], pages), f"{PurePosixPath(manual['rel']).stem} - spares.pdf")


@app.get("/api/jobs/{job}/spares.xlsx")
def download_excel(job: int):
    return sent(outputs.excel(job), "Spares pages.xlsx")


_BUNDLES: dict[int, dict] = {}   # job -> {"state": making | ready | failed, "path", "note"}


def _make(job: int) -> None:
    try:
        _BUNDLES[job].update(state="ready", path=outputs.bundle(job))
    except Exception as exc:
        _BUNDLES[job].update(state="failed", note=str(exc)[:300])


@app.post("/api/jobs/{job}/bundle")
def make_bundle(job: int):
    """Start making a drop's zip, which can take minutes for a whole vessel."""
    if archive.zip_of(job).exists():
        return {"state": "ready"}
    if _BUNDLES.get(job, {}).get("state") != "making":
        _BUNDLES[job] = {"state": "making"}
        threading.Thread(target=_make, args=(job,), daemon=True).start()
    return {k: v for k, v in _BUNDLES[job].items() if k != "path"}


@app.get("/api/jobs/{job}/bundle")
def bundle_state(job: int):
    if archive.zip_of(job).exists():
        return {"state": "ready"}
    return {k: v for k, v in _BUNDLES.get(job, {"state": "none"}).items() if k != "path"}


@app.get("/api/jobs/{job}/spares.zip")
def download_all(job: int):
    """Every manual's spares pages as a PDF, in the folders it was dropped in, plus the
    Excel list: the one made in the background if it is ready, else made now."""
    if archive.zip_of(job).exists():   # an archived drop: its zip is all that is left
        return FileResponse(archive.zip_of(job), filename="Spares pages.zip")
    made = _BUNDLES.pop(job, {}) if _BUNDLES.get(job, {}).get("state") == "ready" else {}
    return sent(made.get("path") or outputs.bundle(job), "Spares pages.zip")


@app.get("/api/health")
def health():
    """For the service installer and for checking from outside: is the server up, which
    reader and model it uses, how much is waiting, how much disk it takes, and how the
    last night's retraining went."""
    report = HERE / "model" / "pages.json"
    with store.db() as con:
        waiting = con.execute("SELECT count(*) FROM files WHERE status IN ('queued', 'running')").fetchone()[0]
    return {"ok": True, "reader": jobs.READER, "waiting": waiting, "keep_days": archive.KEEP_DAYS,
            "storage MB": archive.usage(),
            "model trained": json.loads(report.read_text(encoding="utf-8"))["trained"] if report.exists() else None,
            "last retraining": {k: v for k, v in retrain.last().items() if k != "decisions"}}


@app.get("/")
def screen():
    # no-cache: browsers check for a newer screen each visit instead of showing a stale one
    return FileResponse(HERE / "web" / "index.html", headers={"Cache-Control": "no-cache"})


# The logo (web/icons), at the addresses browsers look for it: the tab, bookmarks, a
# phone's home screen. favicon.ico sits at the top because browsers ask for it there
# whatever the page says.
ICONS = {"favicon.ico": "image/x-icon", "favicon.svg": "image/svg+xml", "logo.svg": "image/svg+xml",
         "logo.png": "image/png", "apple-touch-icon.png": "image/png", "icon-192.png": "image/png",
         "icon-512.png": "image/png", "icon-maskable-512.png": "image/png",
         "manifest.json": "application/manifest+json"}


@app.get("/{name}")
def icon(name: str):
    if name not in ICONS:
        raise HTTPException(404, "No such file")
    return FileResponse(HERE / "web" / "icons" / name, media_type=ICONS[name],
                        headers={"Cache-Control": "max-age=86400"})
