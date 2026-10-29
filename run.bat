@echo off
rem Spares Page Finder. Runs on its own Python (.venv, from requirements.txt) and reads
rem pages with its own copy of Parsel's engine (engine\).
rem
rem   run.bat            this PC only     http://localhost:8600
rem   run.bat office     colleagues too   http://<this PC's name>:8600   (Windows asks once to allow it)
set "HOST=127.0.0.1"
if /i "%~1"=="office" set "HOST=0.0.0.0"
cd /d "%~dp0"
".venv\Scripts\python.exe" -m uvicorn app:app --host %HOST% --port 8600
