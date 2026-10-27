@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Hay chay setup_demo.bat truoc.
    pause
    exit /b 1
)
set PYTHONUTF8=1
set TORCH_HOME=%~dp0cache\torch
set MPLCONFIGDIR=%~dp0cache\matplotlib
".venv\Scripts\python.exe" -m streamlit run src\app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
pause
