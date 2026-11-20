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
import ipaddress
import json
import os
import re
import shutil
import socket
import ssl
import threading
import time
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.middleware.gzip import DEFAULT_EXCLUDED_CONTENT_TYPES, GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from finder import archive, jobs, outputs, retrain, signin, store, triage

HERE = Path(__file__).parent
MAX_UPLOAD = 256 * 1024 ** 2    # per request: the screen sends 16 MB pieces
MAX_MANUAL = 4 * 1024 ** 3      # one manual; the largest seen so far was 316 MB
MIN_FREE = 5 * 1024 ** 3        # disk left free for the database, backups and the PC itself

store.setup()
jobs.start()
archive.start()
retrain.start()
outputs.start_drawing()
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


# The office's own networks: a PC there opens the tool as before, without signing in.
OFFICE = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fe80::/10")]
# What a browser needs before signing in.
OPEN = {"/signin", "/api/signin"}
# What only the office's PCs and this one need, without signing in: the service installer's check,
# and a PC's one-time setup. From the internet they are behind the sign-in like everything else.
LOCAL = {"/api/health", "/SparesFinder-CA.crt", "/setup-pc.ps1"}


def _direct(request: Request):
    """The address a request came straight from, or None if a tunnel passed it on (Tailscale
    Funnel, Cloudflare Tunnel: they arrive from this PC itself and say who they are for in
    X-Forwarded-For, which is never trusted)."""
    if any(h in request.headers for h in ("x-forwarded-for", "forwarded", "x-real-ip", "cf-connecting-ip")):
        return None
    try:
        ip = ipaddress.ip_address((request.client.host if request.client else "").split("%")[0])
    except ValueError:
        return None
    return getattr(ip, "ipv4_mapped", None) or ip


def from_office(request: Request) -> bool:
    """Straight from a PC on the office network; never through a tunnel, whatever it claims,
    and never this PC's own localhost, which is the way tunnels come in."""
    ip = _direct(request)
    return ip is not None and any(ip.version == net.version and ip in net for net in OFFICE)


@app.middleware("http")
async def signed_in_only(request: Request, call_next):
    """From outside the office, nothing but the sign-in page until someone signs in
    (finder/signin.py)."""
    path = request.url.path
    if (path in OPEN or path.lstrip("/") in ICONS or from_office(request)
            or (path in LOCAL and getattr(_direct(request), "is_loopback", False))
            or signin.who(request.cookies.get(signin.COOKIE))):
        answer = await call_next(request)
    elif path.startswith("/api/"):
        answer = JSONResponse({"detail": "Please sign in again"}, status_code=401)
    else:
        answer = RedirectResponse("/signin", status_code=303)   # a link's #drop=… comes along
    # Another site may not show the tool inside its own page (to trick a click on Include or
    # Skip), nor be told which drop was open; a download is never taken for a web page.
    answer.headers.update({"X-Frame-Options": "DENY", "X-Content-Type-Options": "nosniff",
                           "Referrer-Policy": "same-origin"})
    return answer


# The screen and its data compressed (a whole vessel's drop: 322 KB a look, 25 KB compressed,
# every 2 s while it is read), which counts from home or a phone. Pictures, PDFs, sheets and
# zips are compressed already, and a zip of a vessel would only be slowed down.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5, exclude_content_types=(
    *DEFAULT_EXCLUDED_CONTENT_TYPES, "image/*", "application/pdf", "application/x-x509-ca-cert",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))


# Added last, so it is checked first: a request for another host name gets no further.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=this_pc())


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
        rels = [clean_rel(f.rel) for f in drop.files]
        for f, rel, by_name in zip(drop.files, rels, triage.skip_reasons(rels)):
            reason = by_name or ("larger than 4 GB" if f.size > MAX_MANUAL else "")
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
    with store.db() as con:   # a manual to read, or one skipped by its name, sent so its pages can be looked at
        row = con.execute("SELECT size FROM files WHERE id=? AND (status='unsent' OR (status='skipped' AND path IS NULL "
                          "AND note != 'not a PDF file' AND size <= ?))", (fid, MAX_MANUAL)).fetchone()
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
    """A whole manual has arrived: keep it once under its fingerprint, and queue it. One
    skipped by its name is kept unread, its pages there to be looked at."""
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
        if not changed:   # skipped by its name: it stays skipped
            con.execute("UPDATE files SET path=?, sha=?, pages_total=?, note=CASE WHEN ? THEN note ELSE 'not a PDF file' END "
                        "WHERE id=? AND status='skipped' AND path IS NULL",
                        (kept if is_pdf else None, sha, outputs.page_count(store.pdf(kept)) if is_pdf else 0, is_pdf, fid))
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


@app.delete("/api/jobs/{job}")
def delete_job(job: int):
    """Delete a drop, when someone asks to: it leaves the list at once, its reading and sending
    stop, and its manuals, page pictures and downloads are removed in the background
    (archive.delete). What people decided on its pages stays, unnamed, for the models to learn from."""
    with store.db() as con:
        if not con.execute("UPDATE jobs SET deleted=CURRENT_TIMESTAMP WHERE id=? AND deleted IS NULL", (job,)).rowcount:
            raise HTTPException(404, "No such drop")
        halted = [r["id"] for r in con.execute(
            "SELECT id FROM files WHERE job_id=? AND status IN ('queued', 'running')", (job,))]
        con.execute("UPDATE files SET status='stopped' WHERE job_id=? AND status IN ('unsent', 'queued', 'running')", (job,))
    jobs.abandon(halted)
    if made := _BUNDLES.pop(job, {}).get("path"):   # a zip made and not yet fetched
        made.unlink(missing_ok=True)
    threading.Thread(target=archive.delete, args=(job,), daemon=True).start()
    return {"ok": True}


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

drop_name = outputs.drop_name


@app.get("/api/jobs")
def recent_jobs(limit: int = 50):
    """The newest drops, for the list on the screen, so people can go back to one: what
    each is called, when it came, how far its reading is, how many pages wait for a person.
    Archived drops are not listed: their manuals are gone (a link to one still opens it).
    Deleted drops are gone altogether."""
    out = []
    with store.db() as con:
        for j in con.execute("SELECT id, created, archived FROM jobs WHERE archived IS NULL AND deleted IS NULL ORDER BY id DESC LIMIT ?",
                             (min(max(limit, 1), 200),)).fetchall():
            files = con.execute("SELECT rel, status FROM files WHERE job_id=?", (j["id"],)).fetchall()
            if not files:
                continue
            count = lambda *states: sum(f["status"] in states for f in files)
            to_check = con.execute(
                "SELECT count(*) FROM pages p JOIN files f ON f.id = p.file_id WHERE f.job_id=? AND f.status='done' "
                "AND p.label IS NULL AND p.bucket = 'check'", (j["id"],)).fetchone()[0]
            out.append({"id": j["id"], "name": drop_name([f["rel"] for f in files]), "created": j["created"],
                        "archived": j["archived"], "manuals": len(files), "done": count("done"),
                        "reading": count("queued", "running"), "waiting": count("unsent", "stopped"),
                        "failed": count("failed"), "to check": to_check})
    return out


@app.get("/api/jobs/{job}")
def job_detail(job: int):
    with store.db() as con:
        found = con.execute("SELECT created, archived, deleted FROM jobs WHERE id=?", (job,)).fetchone()
        if not found or found["deleted"]:
            raise HTTPException(404, "This drop was deleted" if found else "No such job")
        files = [dict(r) for r in con.execute(
            "SELECT id, rel, size, status, note, pages_total, path IS NOT NULL AS kept FROM files "
            "WHERE job_id=? ORDER BY rel", (job,))]
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
    zoom into a large drawing (sent as JPEG, which keeps an A0 sheet to a few MB). The first two
    are WebP, drawn once and kept."""
    got = outputs.page_picture(the_file(fid)["path"], page, big, turn)
    if got is None:
        raise HTTPException(404, "No such page")
    picture, kind = got
    # A page's picture never changes (a manual is kept under its fingerprint): the browser keeps it a month.
    return Response(picture, media_type=kind, headers={"Cache-Control": "max-age=2592000, immutable"})


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


# ---- downloads ----------------------------------------------------------------

def sent(path: Path, filename: str) -> FileResponse:
    """Send a download made in data/out, and delete it once it has gone."""
    return FileResponse(path, filename=filename, background=BackgroundTask(path.unlink, missing_ok=True))


@app.get("/api/files/{fid}/tagged.pdf")
def download(fid: int):
    """The whole manual, each page under a band saying whether it was included."""
    manual = the_file(fid)
    with store.db() as con:
        pages = outputs.standing(con, fid)
    return sent(outputs.tagged_pdf(manual["path"], pages), f"{PurePosixPath(manual['rel']).stem} - tagged.pdf")


@app.get("/api/jobs/{job}/allocation.xlsx")
def download_allocation(job: int):
    """The drop's allocation sheet, as the typists work from it, with its spares pages filled in."""
    return sent(outputs.allocation(job), outputs.allocation_name(job))


_BUNDLES: dict[int, dict] = {}   # job -> {"state": making | ready | failed, "path", "note", "done", "total"}


def _make(job: int) -> None:
    try:   # done / total: manuals in the zip so far, for the screen to show
        made = outputs.bundle(job, progress=lambda done, total: _BUNDLES[job].update(done=done, total=total))
        _BUNDLES[job].update(state="ready", path=made)
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
    """Every manual read, whole and tagged, in the folders it was dropped in, plus the
    allocation sheet: the one made in the background if it is ready, else made now. An archived
    drop's zip has only each manual's included pages."""
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


def unchanged_or(request: Request, page: Path) -> Response:
    """A page of the screen, which browsers check for a newer one each visit (no-cache) instead of
    showing a stale one: when theirs is still the one there is, it is not sent again (304)."""
    found = page.stat()
    tag = f'"{found.st_mtime_ns:x}-{found.st_size:x}"'
    headers = {"Cache-Control": "no-cache", "ETag": tag}
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304, headers=headers)
    return FileResponse(page, headers=headers)


@app.get("/")
def screen(request: Request):
    return unchanged_or(request, HERE / "web" / "index.html")


# ---- signing in, from outside the office ----------------------------------

@app.get("/signin")
def signin_page(request: Request):
    return unchanged_or(request, HERE / "web" / "signin.html")


class SignIn(BaseModel):
    name: str
    password: str


_checking = threading.Semaphore(1)


def _signin_log(request: Request, name: str, outcome: str) -> None:
    """data/logs/signin.log: every try, with the address it came from (as the tunnel tells it),
    so guessing shows. The name as typed, quoted, so nothing typed can forge a line."""
    try:
        (store.DATA / "logs").mkdir(parents=True, exist_ok=True)
        with (store.DATA / "logs" / "signin.log").open("a", encoding="utf-8") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {outcome}: {name.strip()[:40]!r} "
                      f"from {request.client.host if request.client else '?'}\n")
    except OSError:
        pass


@app.post("/api/signin")
def sign_in(body: SignIn, request: Request):
    # A password check takes ~40 ms of one core, on purpose (scrypt). One at a time, and a try
    # that comes while one runs is turned away at once: a flood of guesses from the internet
    # uses at most one of this PC's cores, and never holds the threads the screen needs.
    if not _checking.acquire(blocking=False):
        return JSONResponse({"detail": "Busy: please try again"}, status_code=503)
    try:
        name, locked = signin.check(body.name, body.password)
    finally:
        _checking.release()
    _signin_log(request, body.name, "signed in" if name else f"locked for {locked} min" if locked else "wrong password")
    if locked:
        return JSONResponse({"detail": "Too many wrong tries", "minutes": locked}, status_code=429)
    if not name:
        return JSONResponse({"detail": "Wrong name or password"}, status_code=401)
    answer = JSONResponse({"name": name})
    answer.set_cookie(signin.COOKIE, signin.cookie_for(name), max_age=signin.DAYS * 86400, httponly=True,
                      secure=request.url.scheme == "https", samesite="lax")
    return answer


@app.get("/api/me")
def me(request: Request):
    """Who is signed in on this browser: nobody (None) on the office network."""
    return {"name": signin.who(request.cookies.get(signin.COOKIE))}


@app.get("/signout")
def sign_out():
    answer = RedirectResponse("/signin", status_code=303)
    answer.delete_cookie(signin.COOKIE)
    return answer


# The logo (web/icons), at the addresses browsers look for it: the tab, bookmarks, a
# phone's home screen. favicon.ico sits at the top because browsers ask for it there
# whatever the page says.
ICONS = {"favicon.ico": "image/x-icon", "favicon.svg": "image/svg+xml", "logo.svg": "image/svg+xml",
         "logo.png": "image/png", "apple-touch-icon.png": "image/png", "icon-192.png": "image/png",
         "icon-512.png": "image/png", "icon-maskable-512.png": "image/png",
         "manifest.json": "application/manifest+json"}


@app.get("/SparesFinder-CA.crt")
def office_ca():
    """The office's certificate authority (scripts/make_certs.py), for a PC to trust once so
    that the tool opens over HTTPS without warnings. Only its public certificate."""
    ca = HERE / "certs" / "SparesFinder-CA.crt"
    if not ca.exists():
        raise HTTPException(404, "This server has no HTTPS certificate yet")
    return FileResponse(ca, media_type="application/x-x509-ca-cert", filename="SparesFinder-CA.crt")


@app.get("/setup-pc.ps1")
def setup_pc(request: Request):
    """The one-time setup for a PC that opens the tool (scripts/setup-pc.ps1), with this server's
    name and its office authority's fingerprint filled in."""
    ca = HERE / "certs" / "SparesFinder-CA.crt"
    if not ca.exists():
        raise HTTPException(404, "This server has no HTTPS certificate yet")
    der = ssl.PEM_cert_to_DER_cert(ca.read_text(encoding="ascii"))
    script = ((HERE / "scripts" / "setup-pc.ps1").read_text(encoding="utf-8")
              .replace("__SERVER__", request.url.hostname or socket.gethostname())
              .replace("__FINGERPRINT__", hashlib.sha256(der).hexdigest()))
    return Response(script.replace("\r\n", "\n").replace("\n", "\r\n"), media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="setup-pc.ps1"'})


@app.get("/{name}")
def icon(name: str):
    if name not in ICONS:
        raise HTTPException(404, "No such file")
    return FileResponse(HERE / "web" / "icons" / name, media_type=ICONS[name],
                        headers={"Cache-Control": "max-age=86400"})
