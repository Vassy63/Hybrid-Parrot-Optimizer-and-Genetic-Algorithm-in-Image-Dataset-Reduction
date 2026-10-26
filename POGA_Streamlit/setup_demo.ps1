$ErrorActionPreference = 'Stop'
$pogaRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $pogaRoot
$pogaPython = $env:POGA_PYTHON
if (-not $pogaPython) {
    $pogaCandidate = Get-Command python -ErrorAction SilentlyContinue
    if ($pogaCandidate -and $pogaCandidate.Source -notmatch 'WindowsApps') { $pogaPython = $pogaCandidate.Source }
}
if (-not $pogaPython) {
    $pogaBundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $pogaBundled) { $pogaPython = $pogaBundled }
}
if (-not $pogaPython) { throw 'Can Python 3.10 tro len. Cai Python hoac dat bien POGA_PYTHON tro den python.exe.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $pogaPython -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Khong tao duoc moi truong .venv.' }
}
$pogaVenv = Join-Path $pogaRoot '.venv\Scripts\python.exe'
& $pogaVenv -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Khong cap nhat duoc pip. Kiem tra ket noi Internet.' }
& $pogaVenv -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Cai thu vien chua thanh cong. Xem thong bao pip o tren.' }
Write-Host 'Cai xong. Chay start_demo.bat de mo web.' -ForegroundColor Green
Write-Host 'Neu PyTorch chua nhan RTX 3060, cai ban CUDA theo README.md.'
