<img src="web/icons/logo.svg" width="96" alt="Spares Page Finder logo">

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
| Start it for yourself | `run.bat`, then open https://localhost:8600 (http:// until the certificates are made, see HTTPS) |
| Start it for colleagues too | `run.bat office`, then https://<this PC's name>:8600 |
| Read manuals | Drop PDFs or a whole vessel folder on the page, or use *Choose manuals* / *Choose a folder* |
| Check pages | Open a page, press Include or Skip; ← → step through the group. Groups fold away |
| Look closely at a big drawing | In the enlarged page: scroll to zoom, drag to move, *Fit width* / *Fit page*; a sharper picture loads as you zoom in |
| Go back to an earlier drop | Click it in the list on the left, or send its link (the address ends in `#drop=` and its number) |
| Look over the pages it was unsure of | Open *Left out*: they come first, marked *Unsure*; *Add* any spares page |
| Make a column wider or narrower | Drag the border between two columns, as in Claude; double-click it to put it back (or focus it and use ← →). Each browser keeps the widths |
| Find a manual in a big vessel folder | The middle column shows the folder as it was dropped, folders first and sorted as Explorer sorts them; click a folder (or →, ←) to open or fold it. Each folder says how many manuals it holds and how many pages in it are still to check. Only the list changes: manuals are still sent and read as dropped |
| Check a manual it skipped by its name | Open it: its pages show once it has been sent (skipped files go last); *Read it* if it has spares pages |
| Use the screen in Hindi or Marathi | Pick हिन्दी or मराठी at the top right; each browser keeps its choice. The downloads stay in English |
| Stop a long drop | *Stop*; *Continue* or *Read it* picks up again |
| Get the results | *Excel list* (manual, spares pages), *Allocation sheet* (the typists' sheet, in D:\AllocationTool's layout: each manual's folder and pages, and its spares pages in DATA as `1-3,5-11` or "No Data"), *Tagged PDF* (one manual whole, each page marked INCLUDED, LEFT OUT or LEFT OUT - NOT CHECKED across its top) or *Download all* (every manual tagged, in its folders, with both sheets, in one zip). Each download shows at the bottom right what it is doing (for the zip, how many manuals are in so far) until its file is saved |
| Teach it | *Include* or *Skip* the pages it is unsure of. Every morning at 09:00 (or soon after the PC is switched on, if later) it learns again from them by itself, and keeps the new models only if they miss no more spares pages (`finder/retrain.py`, `data/logs/retrain.log`); `retrain.bat` does the same at once |
| Teach it a finished vessel | `.venv\Scripts\python.exe -m finder.learn "merge file.xlsx" "vessel folder"` (of older merge files, the one *with group heading*; TORM's tracking sheets with *Marked Pages* work too); several pairs at once are fine |
| Undo a retraining | copy `model\pages.previous.joblib` and `.json` over `model\pages.joblib` and `.json` |

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
                     │                                            "check" pages, from how the page looks and the
                     │                                            lines OCR read on it: probably not spares → unlikely
                     │                                                         │
                     └──── screen, Excel, zip (outputs.py) ◀── database (store.py) ◀── answers per page
                                                                    │
                           Include / Skip / Add / Remove ──▶ database ──▶ every morning (retrain.py, learn.py)
                                                                                     ──▶ model\pages.joblib
```

Every page ends up in one of three groups: *To check* (a person decides), *Included*
and *Left out*. A page that cannot be read, a scan the picture reader cannot make sense
of, or a page the model is unsure about goes to a person under To check. Pages the
picture model thinks are probably not spares ("unlikely") are left out without waiting
for a person, but come first under Left out, marked unsure, so a glance can bring one
back (until 12 Nov they had a group of their own; on DANAPUTRI, tested held out, 0.4%
of its listed spares pages were among them).

## How accurate it is

**On vessels it had never seen** (blind test, 2026-10-29): four finished vessels from
`D:\KIRAN`, of two other clients (ATTICA: Nissos Samos, HS4 Highspeed; Royal Arctic:
Siuanan Royal, Nanoq), read whole by the tool trained on BERGE DAISEN and H1868 only,
then compared with their merge files:

| | Pages | Marked spares | To a person | Probably not spares | Left out |
|---|---|---|---|---|---|
| Spares pages (in the merge files) | 3,059 | 41% | 58% | 2 pages | **49 pages (1.6%)** |
| Other pages | 11,024 | 1% | 47% | 5% | 47% |

- Every scanned spares page was found: the 49 missed were text pages the page model
  scored too low, most of them maintenance-procedure pages the typists took parts
  from (24 in Nissos Samos, 19 of them in its Pielstick maintenance manual).
- Most of the work is the scans: the picture reader sends 75-90% of scanned pages
  that are not spares to a person, so as not to miss any.
- More vessels help. Tested one vessel at a time, a page model that had also learned
  from the other three missed half as many (25 of 1,141 text spares pages, against
  47), and at the same miss rate sent fewer pages to a person on all four (e.g. Nanoq:
  48% → 38% to keep 99%). The picture model ranked pages better on three of the four.

**Then on two more clients** (2026-10-29), the page model trained on those six vessels
was tested on TORM (Torm Innovation, Torm Emilie: tracking sheets) and SVITZER (10
vessels: merge files), text pages only:

| Unseen client | Text spares pages | Missed | Missed, after learning the other client | To check to keep 97%, before → after |
|---|---|---|---|---|
| TORM | 2,379 | 100 (4.2%) | 58 (2.4%) | 38% → 31% |
| SVITZER | 1,770 | 44 (2.5%) | 28 (1.6%) | 30% → 27% |

**Now** the page model has learned from all 18 vessels (542 manuals, 61,578 text pages,
6,287 of them spares pages); the picture model from 160 manuals (2,406 scanned spares
pages). Their own test, each manual held out, is in `model/pages.json`: 1.0% of text
spares pages missed and 40% of other text pages sent to a person; no scanned spares
page sent to "Probably not spares". The 12 TORM and SVITZER vessels still have to be
read by the picture reader before the picture model can learn from their scans.

The picture model's cut-off keeps 98% of the spares pages it is asked about, scored by
models that never saw their manuals; it costs about 0.1 s per scanned page. From 12 Nov
2026 it also reads the lines the picture reader read on a scan. On nine vessels it never
saw (44,356 scans left to a person), it moved about half of the other scans to "Probably
not spares" instead of 42%, and under 1% of the spares scans instead of 2.8%; a scan next
to a page marked spares is never moved.

## Where things are

| Path | What it is |
|---|---|
| `app.py` | The web server: the requests the screen makes |
| `web/index.html` | The screen |
| `finder/store.py` | The database (`data/finder.db`): drops, manuals, each page's answer, people's marks |
| `finder/triage.py` | Which files are skipped by name: certificates, test records, declarations, forms and letters, drawings and plans (but not a numbered drawings chapter of a maker's manual), stability papers, lighting |
| `finder/jobs.py` | The reading queue: one manual at a time, handed to the reader process; Stop |
| `finder/classify.py` | The reader: the five steps above, for every page |
| `finder/text.py` | What a page's text layer says, and which parts-table columns it has |
| `finder/pictures.py` | What a page looks like, read with OCR, looking for a parts table |
| `finder/looks.py` | How a scanned page looks as a whole: 768 numbers from DINOv2-small, kept per manual with the lines OCR read on it |
| `finder/tables.py` | What counts as a parts table: the rule both readers share |
| `finder/pool.py` | The worker processes the readers use, kept on the PC's fast cores |
| `finder/learn.py` | The page model and the picture model: learn from merge spreadsheets and checked pages |
| `finder/retrain.py` | Each morning at 09:00 copies the database and learns again when people have decided pages, and keeps the new models only if they are no worse |
| `finder/outputs.py` | Page pictures for the screen, tagged manuals, the Excel list, the allocation sheet, the zip |
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
| `KEEP` = 0.98 | `finder/learn.py` | Share of checked spares pages the model's cut-off must keep (set at retraining); 0.99 until 2 Nov 2026 |
| `SURE` = 0.85 | `finder/learn.py` | Share of pages marked spares without a person that the spreadsheets list, aiming at about 97% really spares pages (the spreadsheets leave out untyped parts lists); 0.90 until 6 Nov 2026 |
| `DOUBTFUL`, `NEAR` = 0.5, 2 | `finder/learn.py` | A "not spares" page scoring this high this near a listed spares page is left out of learning |
| `SECOND_STEP` = 50 | `finder/learn.py` | Manuals needed before the second step (runs of pages) is trained |
| `LOOKS_KEEP` = 0.98 | `finder/learn.py` | Share of the spares pages left to a person that the picture model must leave there, for pages without lines read on them (set at retraining, on scores from models that never saw the manual) |
| `LINES_KEEP` = 0.99 | `finder/learn.py` | The same for pages with lines read on them, judged by how they look and what they say: 0.99 within training keeps about 98.5% on vessels never seen |
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
 colleague's browser (office network) ──▶ https://<office PC>:8600 ──▶ the tool on the office PC ──▶ data\ on its disks
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
- a backup at 02:00, which runs only if the PC is on then (the service itself backs up every morning).

Colleagues then use `https://<office PC's name>:8600` (see HTTPS below). Re-running the installer is safe.
To remove it: `nssm remove SparesFinder confirm`, then delete the firewall rule
"Spares Page Finder (TCP 8600)" and the scheduled task "Spares Page Finder backup".

Why not a hosting service: the reading needs minutes of CPU per manual on the PC's 8
fast cores. Vercel takes 4.5 MB per request, and Cloudflare's and Supabase's free plans
allow seconds of CPU at most (Supabase also caps files at 50 MB).

### HTTPS

Chrome warns about, or blocks, every download from a plain `http://` address, and a file it
has blocked half-way will not open in Excel. So the tool is served over HTTPS, with
certificates from the office's own certificate authority (no outside service involved):

1. **Once, on the office PC** (made already on 2026-11-13; again only if the PC's name or
   address changes, or after about two years):
   `.venv\Scripts\python.exe scripts\make_certs.py <its office IP address>`, then re-run the
   service installer (as Administrator, as above). The installer switches the service to
   HTTPS and lets only administrators and SYSTEM read `certs\`.
2. **Once on each PC that opens the tool**, as that colleague (no administrator needed):
   open `https://<office PC>:8600/SparesFinder-CA.crt` (Chrome warns this first time:
   Advanced > Proceed) or copy `certs\SparesFinder-CA.crt` from the office PC, then run
   `certutil -user -addstore Root SparesFinder-CA.crt` and answer Yes (or double-click the
   file > Install Certificate > Current User > "Trusted Root Certification Authorities").
   Restart Chrome. The address now shows a padlock and downloads open normally.

The authority may only vouch for the office PC's names and the office's private addresses
(10.x, 172.16-31.x, 192.168.x, localhost), so even its key could not pass off another
website. `certs\` holds the private keys: never copy it off the office PC, never into git.
Old `http://` links stop working once the service is on HTTPS: use `https://`.

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

- Files skipped by name (certificates, drawings...) are sent after all the others and kept
  unread, so their pages can be looked at; *Read it* then reads one without sending it again.
- A manual is stored once under its fingerprint (`data/uploads/<sha256>.pdf`), however
  many drops it is in.
- After 14 days (`FINDER_KEEP_DAYS`) a drop is archived: its zip is made once and kept
  (`data/archive/`), with only each manual's included pages rather than the whole tagged manual, the text of every checked page is kept so the model can still
  learn from it, and the original PDFs are deleted. The drop's answers and Excel list
  stay; the pages can no longer be opened.
- `/api/health` shows how much space manuals, archives and backups take.

## Backups and updates

- The service copies the database to `data\backups` every morning before learning again
  (the newest 14 are kept); `backup.bat` does the same at any time. The office PC is off
  at night, so the installer's 02:00 task seldom runs. The database holds every answer
  and every mark; the manuals themselves are not backed up.
- The code is in the private GitHub repository `Vru-Jain/spares-page-finder`. To
  update the office PC: `git pull`, then `nssm restart SparesFinder`. If
  `requirements.txt` changed, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` first.
- Not in git, by design: `data/` (manuals and the database) and the trained model in
  `model/` (it holds text from client manuals). To move to a new PC: clone, run
  `install.bat`, copy `model\` and `data\` across, then install the service.
