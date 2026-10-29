@echo off
rem Copy the database (every page's answer, people's marks and checked lists) to
rem data\backups\finder-<date>.db. The newest 14 copies are kept. Safe while the tool
rem is running. The service installer schedules this every night.
cd /d "%~dp0"
.venv\Scripts\python.exe -m finder.store
