@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 web_app.py --open %*
) else (
  python web_app.py --open %*
)
if errorlevel 1 pause
