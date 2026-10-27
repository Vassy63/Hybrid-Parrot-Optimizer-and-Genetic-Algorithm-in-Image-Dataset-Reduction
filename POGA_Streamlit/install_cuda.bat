@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Hay chay setup_demo.bat truoc.
    pause
    exit /b 1
)
echo Hay dong cua so start_demo.bat truoc khi thay PyTorch.
echo Cai PyTorch CUDA 12.6 cho NVIDIA RTX 3060. Lan cai nay can Internet va tai nhieu GB.
".venv\Scripts\python.exe" -m pip install --force-reinstall torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo Cai CUDA chua thanh cong. Giu lai thong bao loi o tren.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Chua nhan GPU'); assert torch.cuda.is_available(), 'CUDA chua san sang'"
if errorlevel 1 (
    echo Can kiem tra lai PyTorch va driver NVIDIA truoc khi train GPU.
) else (
    echo GPU da san sang. Mo lai start_demo.bat.
)
pause
