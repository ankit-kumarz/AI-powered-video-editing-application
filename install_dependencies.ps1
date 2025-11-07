Write-Host "Installing AI Video Editor Dependencies..." -ForegroundColor Green
Write-Host ""

Write-Host "[1/4] Installing Python backend dependencies..." -ForegroundColor Yellow
Set-Location backend

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

& "venv\Scripts\Activate.ps1"
Write-Host "Installing production dependencies..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Installing development dependencies..." -ForegroundColor Cyan
pip install -r requirements_dev.txt
deactivate

Set-Location ..

Write-Host ""
Write-Host "[2/4] Installing Node.js frontend dependencies..." -ForegroundColor Yellow
Set-Location frontend

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing npm packages..." -ForegroundColor Cyan
    npm install
} else {
    Write-Host "Updating npm packages..." -ForegroundColor Cyan
    npm update
}

Set-Location ..

Write-Host ""
Write-Host "[3/4] Installing FFmpeg (if not already installed)..." -ForegroundColor Yellow
Write-Host "Please ensure FFmpeg is installed and available in your PATH" -ForegroundColor Cyan
Write-Host "Download from: https://ffmpeg.org/download.html" -ForegroundColor Cyan
Write-Host ""

Write-Host "[4/4] Setting up environment..." -ForegroundColor Yellow
if (-not (Test-Path "temp")) { New-Item -ItemType Directory -Name "temp" }
if (-not (Test-Path "temp\uploads")) { New-Item -ItemType Directory -Name "uploads" -Path "temp" }
if (-not (Test-Path "temp\processed")) { New-Item -ItemType Directory -Name "processed" -Path "temp" }

Write-Host ""
Write-Host "Dependencies installation completed!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "1. Backend: cd backend && venv\Scripts\Activate.ps1 && python main.py" -ForegroundColor White
Write-Host "2. Frontend: cd frontend && npm run dev" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to continue"
