# Working on Spares Page Finder

The office PC runs the `main` branch. Nothing reaches it until the owner pulls it there,
and `main` changes only through a reviewed pull request.

1. Make a branch from `main` and do the work there.
2. Run the tests: `.venv\Scripts\python.exe -m pytest`.
3. Open a pull request. The tests run by themselves; the owner reviews it.
4. Once the tests pass and the owner approves it, it is squash-merged into `main`.
5. The owner updates the office PC: `git pull`, then `nssm restart SparesFinder`.

## Never in the repository

- Client manuals, merge spreadsheets, tracking sheets, or anything from `data/`
- Trained models or what they learned from (`model/*.joblib`, `model/*.json`)
- Client, vessel or colleague names, in code, tests or commit messages: use made-up ones

## Changes that need numbers first

How pages are judged (`finder/classify.py`, `text.py`, `pictures.py`, `looks.py`,
`tables.py`, `engine/`) or learned (`finder/learn.py`, `retrain.py`) is agreed with the
owner before it is built. It is measured on manuals the model never saw, and the
numbers go in the pull request. A change to how pages are judged also bumps
`jobs.READER`, so manuals read before are read again.
