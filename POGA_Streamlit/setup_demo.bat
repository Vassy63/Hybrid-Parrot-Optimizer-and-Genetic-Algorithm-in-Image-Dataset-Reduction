@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_demo.ps1"
if errorlevel 1 echo Cai dat chua hoan tat.
pause
