# The trained page model

This folder holds the page model the reader uses, and everything it learned from. They
are not in git: they are made on the PC that runs the tool, and contain text from
client manuals.

| File | What it is |
|---|---|
| `pages.joblib` | The models the reader uses (`finder/learn.py`): the page model, and the picture model behind "Probably not spares" |
| `pages.json` | Their test report: how they did on manuals they were not trained on |
| `training.joblib` | Every page they learned from: the answer, what the page's text said, and for scanned pages the picture verdict and numbers (`finder/looks.py`) |
| `pages.previous.*` | The models before the last retraining, to go back to |
| `pages.before-picture-model.*` | The page model alone, as it was before the picture model (2026-10-29): copy it over `pages.joblib` to turn "Probably not spares" off |

**On a new PC**, copy `pages.joblib`, `pages.json` and `training.joblib` here from the
PC that ran the tool. Without them the tool still works, using its fixed rules, until
someone adds a merge spreadsheet:

    .venv\Scripts\python.exe -m finder.learn "MERGE ALL FILES.xlsx" "manuals folder"

**To learn again** from everything so far plus what people have decided since, run
`retrain.bat`. **To go back** a retraining, copy `pages.previous.joblib` over
`pages.joblib` (and `pages.previous.json` over `pages.json`).
