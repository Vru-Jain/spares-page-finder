@echo off
rem Spares Page Finder. Runs on its own Python (.venv, from requirements.txt) and reads
rem pages with its own copy of Parsel's engine (engine\).
rem
rem   run.bat            this PC only     https://localhost:8600
rem   run.bat office     colleagues too   https://<this PC's name>:8600   (Windows asks once to allow it)
rem
rem Over HTTPS once certs\server.crt exists (scripts\make_certs.py, README.md "HTTPS"), else
rem over plain HTTP, whose downloads Chrome warns about.
set "HOST=127.0.0.1"
if /i "%~1"=="office" set "HOST=0.0.0.0"
cd /d "%~dp0"
set "TLS="
if exist "certs\server.crt" set "TLS=--ssl-keyfile certs\server.key --ssl-certfile certs\server.crt"
".venv\Scripts\python.exe" -m uvicorn app:app --host %HOST% --port 8600 %TLS%
