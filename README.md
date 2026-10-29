# Spares Page Finder

Finds the pages of ship manuals that carry spare-parts tables (parts lists, drawings
with a parts table, recommended spares, special tools), so colleagues only check those
pages instead of reading whole manuals. It runs on this PC; colleagues use it from
their browser. There is no AI chat model in it: two small OCR networks read pictures,
and a word-count model learns from pages people have checked.

## Using it

| To | Do |
|---|---|
| Start it for yourself | `run.bat`, then open http://localhost:8600 |
| Start it for colleagues too | `run.bat office`, then http://<this PC's name>:8600 |
| Read manuals | Drop PDFs or a whole vessel folder on the page, or use *Choose manuals* / *Choose a folder* |
| Check pages | Open a page, press Include or Skip; ← → step through the group |
| Stop a long drop | *Stop*; *Continue* or *Read it* picks up again |
| Get the results | *Excel list* (manual, spares pages) or *Download* (a zip of spares PDFs and the list) |
| Teach it a checked manual | *Paste checked list* under the manual, then run `retrain.bat` (about 7 minutes) |
| Undo a retraining | copy `model\pages.previous.joblib` over `model\pages.joblib` |

First time on a PC: `install.bat` (needs Python 3.12). Tests: `.venv\Scripts\python.exe -m pytest`.

## How a manual is read

```
 browser ──drop──▶ app.py ──name check (triage)──▶ queue (jobs) ──one manual──▶ reader process
                     ▲                                                         │
                     │                                       classify.py, for every page:
                     │                                         1. census: text layer? picture?
                     │                                         2. text layer read (text.py)
                     │                                         3. page model scores the text (learn.py)
                     │                                            high → spares   low → not spares
                     │                                            between → unsure
                     │                                         4. picture read with OCR (pictures.py)
                     │                                            for unsure pages: part-number table → spares, else check
                     │                                            for pages with little or no text: table → spares/check,
                     │                                            nothing found → check, unless the page is nearly blank
                     │                                                         │
                     └──── screen, Excel, zip (outputs.py) ◀── database (store.py) ◀── answers per page
                                                                    │
                     Include / Skip / checked lists ──▶ database ──▶ retrain.bat (learn.py) ──▶ model\pages.joblib
```

A page is only ever left out when the tool is sure: a page that cannot be read, a scan
the picture reader cannot make sense of, or a page the model is unsure about goes to a
person under "check".

## Where things are

| Path | What it is |
|---|---|
| `app.py` | The web server: the requests the screen makes |
| `web/index.html` | The screen |
| `finder/store.py` | The database (`data/finder.db`): drops, manuals, each page's answer, people's marks |
| `finder/triage.py` | Which files are skipped by name: certificates, test records, drawings and plans, stability papers, navigation manuals |
| `finder/jobs.py` | The reading queue: one manual at a time, handed to the reader process; Stop |
| `finder/classify.py` | The reader: the four steps above, for every page |
| `finder/text.py` | What a page's text layer says, and which parts-table columns it has |
| `finder/pictures.py` | What a page looks like, read with OCR, looking for a parts table |
| `finder/tables.py` | What counts as a parts table: the rule both readers share |
| `finder/pool.py` | The worker processes the readers use, kept on the PC's fast cores |
| `finder/learn.py` | The page model: learns from merge spreadsheets and checked pages; `retrain.bat` runs it |
| `finder/outputs.py` | Page pictures for the screen, spares PDFs, the Excel list, the zip |
| `engine/` | Parsel's table-reading code, copied unchanged so the tool runs on its own |
| `models/` | The OCR networks (PP-OCRv6 tiny) |
| `model/` | The trained page model, and its test report (`pages.json`) |
| `data/` | The database, dropped manuals and downloads being made (not part of the code) |
| `tests/` | The checks run by pytest |

## Settings that change answers

| Setting | Where | What it does |
|---|---|---|
| `USABLE_TEXT` = 500 | `finder/text.py` | Pages with less text than this are judged by their picture, not the model |
| `BLANK` = 5 | `finder/pictures.py` | A picture with no table found is left out only if it shows fewer short lines than this |
| `KEEP` = 0.99 | `finder/learn.py` | Share of checked spares pages the model's cut-off must keep (set at retraining) |
| `SIZES`, `QUICK` | `finder/pictures.py` | The sizes pictures are read at; unsure pages use the first only |
| `READER` | `finder/jobs.py` | Raise it after changing how pages are read, so earlier results are not reused |

Settings that do not change answers are read from the environment: `FINDER_DATA` (where
the data folder is), `FINDER_KEEP_DAYS` (default 14), `FINDER_HOSTS` (extra names the
tool answers to), `FINDER_WORKERS`, `FINDER_THREADS`, `FINDER_TEXT_WORKERS`, `FINDER_SIZES`.

## Running it for everyone

The tool runs on the office PC, which has the CPU it was measured on and room for the
manuals. Colleagues reach it in the office directly, and from anywhere through
Cloudflare Tunnel: Cloudflare carries the traffic to the PC, so no router port is
opened, and Cloudflare Access asks for a company email address before anyone sees
the page.

```
 colleague's browser ──https──▶ Cloudflare (Access: company email) ──tunnel──▶ cloudflared on the office PC ──▶ the tool, 127.0.0.1:8600
 office browser ─────────────────────────────── http://<office PC>:8600 (only with -Office) ──────────────────▶
```

Why not a hosting service such as Vercel or Cloudflare's own servers: Vercel takes
4.5 MB per request and stops a request after a few minutes, with no lasting disk, and
a manual is read for minutes and kept for days. Cloudflare Containers lose their disk
when they stop and have at most 4 CPUs; reading pictures needs the PC's 8 fast cores.

The tool is built for Cloudflare's limits: a manual goes up in 16 MB pieces (Cloudflare
takes 100 MB per request) and carries on after a dropped connection; a drop's zip is
made in the background (Cloudflare gives up on a request after 100 seconds).

**Once, by whoever owns the Cloudflare account** (free plan is enough):

1. Add the company's domain to Cloudflare (or buy one there).
2. Zero Trust > Networks > Tunnels > Create a tunnel > Cloudflared. Name it
   `spares-finder`, choose Windows, and copy the token it shows (`eyJh...`).
3. In the tunnel, add a public hostname, e.g. `spares.<domain>`, service
   `http://localhost:8600`.
4. Zero Trust > Access > Applications > Add > Self-hosted, for `spares.<domain>`, with
   a policy that allows emails ending in the company's domain (or a list of people).
   Do this before installing on the PC, so the page is never open without it.

**On the office PC**, as Administrator:

```
winget install NSSM.NSSM
winget install --id Cloudflare.cloudflared
powershell -ExecutionPolicy Bypass -File scripts\install-service.ps1 -PublicHost spares.<domain> -TunnelToken eyJh... -Office
```

It installs the tool as a Windows service (starts with the PC, restarts after a
crash, logs in `data\logs`), the Cloudflare connector as another, and the nightly
backup. Leave out `-Office` to reach it only through Cloudflare; leave out the
Cloudflare parts to use it only in the office. The PC must stay on and not sleep.

To remove it: `nssm remove SparesFinder confirm`, `cloudflared service uninstall`, and
delete the scheduled task "Spares Page Finder backup".

## Storage

Vessel folders are big, so the tool keeps as little as it can:

- Files skipped by name (certificates, navigation manuals...) are never sent. *Include
  anyway* asks for the file then.
- A manual is stored once under its fingerprint (`data/uploads/<sha256>.pdf`), however
  many drops it is in.
- After 14 days (`FINDER_KEEP_DAYS`) a drop is archived: its zip is made once and kept
  (`data/archive/`), the text of every checked page is kept so the model can still
  learn from it, and the original PDFs are deleted. The drop's answers and Excel list
  stay; the pages can no longer be opened.
- `/api/health` shows how much space manuals, archives and backups take.

## Backups and updates

- `backup.bat` copies the database to `data\backups` (the newest 14 are kept); the
  service installer runs it every night at 02:00. The database holds every answer
  and every mark; the manuals themselves are not backed up.
- The code is in the private GitHub repository `Vru-Jain/spares-page-finder`. To
  update the office PC: `git pull`, then `nssm restart SparesFinder`. If
  `requirements.txt` changed, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` first.
- Not in git, by design: `data/` (manuals and the database) and the trained model in
  `model/` (it holds text from client manuals). To move to a new PC: clone, run
  `install.bat`, copy `model\` and `data\` across, then install the service.
