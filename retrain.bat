@echo off
rem Learn again now, without waiting for the night: the merge spreadsheets it learned
rem from, plus every Include and Skip saved on the screen since. Takes several minutes.
rem The new models are used only if they miss no more spares pages than the ones in use
rem (see finder\retrain.py); the old ones are kept as model\pages.previous.*.
cd /d "%~dp0"
.venv\Scripts\python.exe -m finder.retrain
pause
