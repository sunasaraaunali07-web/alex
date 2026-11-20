@echo off
title AI English-Speaking Friend
cd /d "%~dp0"
set PYTHONPATH=backend
echo Starting AI English-Speaking Friend on http://localhost:8000 ...
start http://localhost:8000
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
