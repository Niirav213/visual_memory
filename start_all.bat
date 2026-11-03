@echo off
echo ====================================================================
echo   Starting Visual Memory AI — Ultra-Fast React + Express + FastAPI Stack
echo ====================================================================
echo.

start "FastAPI Core ML Engine (Port 8000)" cmd /k "python -m apps.api.main"
timeout /t 3 /nobreak > nul

start "Express.js Streaming Gateway (Port 5000)" cmd /k "node server.js"
timeout /t 2 /nobreak > nul

start "Next.js / React 18 Dashboard (Port 3000)" cmd /k "npm run dev"
timeout /t 4 /nobreak > nul

start http://localhost:3000/live

echo.
echo All services started! Opening http://localhost:3000/live in your browser.
echo ====================================================================
pause
