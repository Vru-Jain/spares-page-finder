@echo off
rem Teach the page model again: the merge spreadsheets it learned from, plus every
rem Include, Skip and checked page list saved on the screen since. Takes several minutes.
rem The tool uses the new model from its next manual; the old one is kept as
rem model\pages.previous.joblib.
cd /d "%~dp0"
.venv\Scripts\python.exe -m finder.learn --again
pause
