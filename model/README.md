# The trained page model

This folder holds the page model the reader uses, and everything it learned from. They
are not in git: they are made on the PC that runs the tool, and contain text from
client manuals.

| File | What it is |
|---|---|
| `pages.joblib` | The model the reader uses (`finder/learn.py`) |
| `pages.json` | Its test report: how it did on manuals it was not trained on |
| `training.joblib` | Every page it learned from: the answer and what the page's text said |
| `pages.previous.*` | The model before the last retraining, to go back to |

**On a new PC**, copy `pages.joblib`, `pages.json` and `training.joblib` here from the
PC that ran the tool. Without them the tool still works, using its fixed rules, until
someone adds a merge spreadsheet:

    .venv\Scripts\python.exe -m finder.learn "MERGE ALL FILES.xlsx" "manuals folder"

**To learn again** from everything so far plus what people have decided since, run
`retrain.bat`. **To go back** a retraining, copy `pages.previous.joblib` over
`pages.joblib` (and `pages.previous.json` over `pages.json`).
