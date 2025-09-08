@echo off
echo Stopping all Python processes...
taskkill /IM python.exe /F 2>nul
echo.
echo Starting new API server...
cd /d "C:\Users\tooic\OneDrive - UTS\term2\advanced-ml\at2\api"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000