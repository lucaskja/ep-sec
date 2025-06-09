@echo off
echo ========================================
echo Hill Cipher Overnight Breaking Session
echo ========================================
echo.
echo This will run all Hill cipher tests in sequence:
echo - 3x3 known and unknown (15-45 min each)
echo - 4x4 known and unknown (2-8 hours each)  
echo - 5x5 known and unknown (8-24+ hours each)
echo.
echo Results will be saved in overnight_results/
echo Logs will be saved in overnight_results/logs/
echo.
echo Press Ctrl+C to stop at any time
echo.
pause

echo Starting overnight session...
python run_all_hill_ciphers_overnight.py

echo.
echo Session completed! Check overnight_results/ folder for results.
pause
