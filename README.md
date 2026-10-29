# Spares Page Finder

Finds the pages of ship manuals that carry spare-parts tables (parts lists, drawings
with a parts table, recommended spares, special tools), so colleagues only check those
pages instead of reading whole manuals. It runs on this PC; colleagues use it from
their browser. There is no AI chat model in it: two small OCR networks read pictures,
a small image network (DINOv2-small) describes how scanned pages look, and two simple
models learn from pages people have checked.

## Using it

| To | Do |
|---|---|
| Start it for yourself | `run.bat`, then open http://localhost:8600 |
| Start it for colleagues too | `run.bat office`, then http://<this PC's name>:8600 |
| Read manuals | Drop PDFs or a whole vessel folder on the page, or use *Choose manuals* / *Choose a folder* |
| Check pages | Open a page, press Include or Skip; ← → step through the group |
| Clear "Probably not spares" | Look the group over, Include any spares page, then *Leave all out* |
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
                     │                                         5. picture model (looks.py + learn.py), on those
                     │                                            "check" pages: probably not spares → unlikely
                     │                                                         │
                     └──── screen, Excel, zip (outputs.py) ◀── database (store.py) ◀── answers per page
                                                                    │
                     Include / Skip / Leave all out / checked lists ──▶ database ──▶ retrain.bat (learn.py)
                                                                                     ──▶ model\pages.joblib
```

A page is only ever left out when the tool is sure: a page that cannot be read, a scan
the picture reader cannot make sense of, or a page the model is unsure about goes to a
person under "check". Pages the picture model thinks are not spares are shown apart,
under "Probably not spares", and are left out only when a person says so.

The picture model was tested on 6,198 scanned pages of H1868 and BERGE DAISEN (74
manuals), each manual predicted by a model that had not seen it, with its cut-off set
from the other manuals alone. It moved 23% of the other "check" pages to "Probably not
spares" (34% on H1868, 13% on BERGE DAISEN), and with them 2 of the 490 spares pages,
both in H1868 while it had learned from BERGE DAISEN alone. That is why the group is
looked over, not left out unseen. Its test is in `model/pages.json` after each
retraining. It costs about 0.1 s per scanned page.

## Where things are

| Path | What it is |
|---|---|
| `app.py` | The web server: the requests the screen makes |
| `web/index.html` | The screen |
| `finder/store.py` | The database (`data/finder.db`): drops, manuals, each page's answer, people's marks |
| `finder/triage.py` | Which files are skipped by name: certificates, test records, drawings and plans, stability papers, navigation manuals |
| `finder/jobs.py` | The reading queue: one manual at a time, handed to the reader process; Stop |
| `finder/classify.py` | The reader: the five steps above, for every page |
| `finder/text.py` | What a page's text layer says, and which parts-table columns it has |
| `finder/pictures.py` | What a page looks like, read with OCR, looking for a parts table |
| `finder/looks.py` | How a scanned page looks as a whole: 768 numbers from DINOv2-small, kept per manual |
| `finder/tables.py` | What counts as a parts table: the rule both readers share |
| `finder/pool.py` | The worker processes the readers use, kept on the PC's fast cores |
| `finder/learn.py` | The page model and the picture model: learn from merge spreadsheets and checked pages; `retrain.bat` runs it |
| `finder/outputs.py` | Page pictures for the screen, spares PDFs, the Excel list, the zip |
| `engine/` | Parsel's table-reading code, copied unchanged so the tool runs on its own |
| `models/` | The OCR networks (PP-OCRv6 tiny) and the image network (DINOv2-small, made by `scripts/convert_dinov2.py`) |
| `model/` | The trained models, and their test report (`pages.json`) |
| `data/` | The database, dropped manuals, their picture numbers and downloads being made (not part of the code) |
| `tests/` | The checks run by pytest |

## Settings that change answers

| Setting | Where | What it does |
|---|---|---|
| `USABLE_TEXT` = 500 | `finder/text.py` | Pages with less text than this are judged by their picture, not the model |
| `BLANK` = 5 | `finder/pictures.py` | A picture with no table found is left out only if it shows fewer short lines than this |
| `KEEP` = 0.99 | `finder/learn.py` | Share of checked spares pages the model's cut-off must keep (set at retraining) |
| picture model cut-off | `finder/learn.py`, `fit_looks` | The lowest score any checked spares page got from a model that never saw its manual (set at retraining) |
| `SIZES`, `QUICK` | `finder/pictures.py` | The sizes pictures are read at; unsure pages use the first only |
| `READER` | `finder/jobs.py` | Raise it after changing how pages are read, so earlier results are not reused |

Settings that do not change answers are read from the environment: `FINDER_DATA` (where
the data folder is), `FINDER_KEEP_DAYS` (default 14), `FINDER_HOSTS` (extra names the
tool answers to), `FINDER_WORKERS`, `FINDER_THREADS`, `FINDER_TEXT_WORKERS`, `FINDER_SIZES`.

## Running it for everyone

Everything stays in the office: the office PC reads the manuals, keeps them and the
database on its own disks, and colleagues open the tool from their browsers on the
office network. Nothing goes to a cloud service. The PC stays on whenever its plug is
on; Windows must not put it to sleep.

```
 colleague's browser (office network) ──▶ http://<office PC>:8600 ──▶ the tool on the office PC ──▶ data\ on its disks
```

**Once, on the office PC**, from PowerShell opened as Administrator:

```
winget install NSSM.NSSM
powershell -ExecutionPolicy Bypass -File D:\spares-finder\scripts\install-service.ps1 -Office
```

It installs:

- the tool as a Windows service: it starts with the PC, restarts after a crash, and logs
  to `data\logs` (rotated at 10 MB);
- a firewall rule that lets the office network reach port 8600 (Private and Domain
  networks only, never a public one);
- the nightly backup at 02:00.

Colleagues then use `http://<office PC's name>:8600`. Re-running the installer is safe.
To remove it: `nssm remove SparesFinder confirm`, then delete the firewall rule
"Spares Page Finder (TCP 8600)" and the scheduled task "Spares Page Finder backup".

Why not a hosting service: the reading needs minutes of CPU per manual on the PC's 8
fast cores. Vercel takes 4.5 MB per request, and Cloudflare's and Supabase's free plans
allow seconds of CPU at most (Supabase also caps files at 50 MB).

### From outside the office, later (optional)

The tool can't be reached from outside the office network, except over a company VPN.
If that is ever needed, Cloudflare's free tunnel and email sign-in can add it without
changing the code. The tool already sends manuals in 16 MB pieces and makes zips in the
background, to fit Cloudflare's 100 MB and 100-second limits.

1. In a Cloudflare account with the company's domain:
   - Zero Trust > Networks > Tunnels > Create a tunnel > Cloudflared (Windows); copy its
     token.
   - Add a public hostname such as `spares.<domain>` pointing at `http://localhost:8600`.
   - Zero Trust > Access > Applications > Self-hosted, for that hostname, allowing the
     company's email addresses. Do this before step 2, so the page is never open
     without sign-in.
2. On the office PC, as Administrator: `winget install --id Cloudflare.cloudflared`, then
   run the installer again with `-PublicHost spares.<domain> -TunnelToken <token> -Office`.
   To remove it later: `cloudflared service uninstall`.

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
