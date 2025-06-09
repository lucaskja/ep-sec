@echo off
echo =============================================
echo Fully Optimized Hill Cipher Breaking Session
echo =============================================
echo.
echo This will run all Hill cipher tests with maximum GPU optimization:
echo - 3x3 known and unknown (5-15 min each)
echo - 4x4 known and unknown (15-60 min each)  
echo - 5x5 known and unknown (1-4 hours each)
echo.
echo Total estimated time: 3-10 hours (vs 18-50+ hours before)
echo Expected GPU utilization: 80-95%%
echo.
echo Results will be saved in overnight_results/
echo Logs will be saved in overnight_results/logs/
echo.
echo Press Ctrl+C to stop at any time
echo.
pause

echo Starting fully optimized overnight session...
python run_all_hill_ciphers_overnight.py

echo.
echo Session completed! Check overnight_results/ folder for results.
pause
