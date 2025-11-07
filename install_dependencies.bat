@echo off
echo Installing AI Video Editor Dependencies...
echo.

echo [1/4] Installing Python backend dependencies...
cd backend
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo Installing production dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo Installing development dependencies...
pip install -r requirements_dev.txt
deactivate
cd ..

echo.
echo [2/4] Installing Node.js frontend dependencies...
cd frontend
if not exist "node_modules" (
    echo Installing npm packages...
    npm install
) else (
    echo Updating npm packages...
    npm update
)
cd ..

echo.
echo [3/4] Installing FFmpeg (if not already installed)...
echo Please ensure FFmpeg is installed and available in your PATH
echo Download from: https://ffmpeg.org/download.html
echo.

echo [4/4] Setting up environment...
if not exist "temp" mkdir temp
if not exist "temp\uploads" mkdir temp\uploads
if not exist "temp\processed" mkdir temp\processed

echo.
echo Dependencies installation completed!
echo.
echo To start the application:
echo 1. Backend: cd backend && venv\Scripts\activate.bat && python main.py
echo 2. Frontend: cd frontend && npm run dev
echo.
pause
