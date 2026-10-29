"""The Spares Page Finder's parts, in the order a manual meets them.

    store.py     the database: drops, manuals, every page's decision, people's marks
    triage.py    which files are skipped, by their name
    jobs.py      the reading queue: one manual at a time, handed to the reader process
    classify.py  the reader: decides every page of a manual
      text.py      what a page's text layer says
      pictures.py  what a page looks like, read with OCR
      tables.py    what counts as a parts table (the rule both readers share)
      pool.py      the worker processes both readers use
    learn.py     the page model: learns from checked pages, scores text pages
    outputs.py   page pictures for the screen, spares PDFs, the Excel list, the zip

app.py, next to this folder, is the web server; web/index.html is the screen.
"""
