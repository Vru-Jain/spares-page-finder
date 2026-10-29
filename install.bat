@echo off
rem First-time setup on a PC: the tool's own Python environment (.venv) with the
rem packages in requirements.txt, then the tests. Needs Python 3.12 from python.org.
cd /d "%~dp0"
py -3.12 -m venv .venv || (echo Python 3.12 is needed: https://www.python.org/downloads/ & exit /b 1)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || exit /b 1
.venv\Scripts\python.exe -m pytest -q
echo.
echo Ready. Start it with run.bat, or install it as a service: scripts\install-service.ps1
pause
