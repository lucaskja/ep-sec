# Hill Cipher Overnight Breaking Session
# PowerShell script for Windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Hill Cipher Overnight Breaking Session" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This will run all Hill cipher tests in sequence:" -ForegroundColor Yellow
Write-Host "- 3x3 known and unknown (15-45 min each)" -ForegroundColor White
Write-Host "- 4x4 known and unknown (2-8 hours each)" -ForegroundColor White
Write-Host "- 5x5 known and unknown (8-24+ hours each)" -ForegroundColor White
Write-Host ""

Write-Host "Results will be saved in overnight_results/" -ForegroundColor Green
Write-Host "Logs will be saved in overnight_results/logs/" -ForegroundColor Green
Write-Host ""

Write-Host "Press Ctrl+C to stop at any time" -ForegroundColor Red
Write-Host ""

$confirmation = Read-Host "Press Enter to start or Ctrl+C to cancel"

Write-Host "Starting overnight session..." -ForegroundColor Green
Write-Host "Timestamp: $(Get-Date)" -ForegroundColor Gray

try {
    python run_all_hill_ciphers_overnight.py
    Write-Host ""
    Write-Host "Session completed successfully!" -ForegroundColor Green
    Write-Host "Check overnight_results/ folder for results." -ForegroundColor Yellow
}
catch {
    Write-Host ""
    Write-Host "Session encountered an error: $_" -ForegroundColor Red
    Write-Host "Check overnight_results/logs/ for details." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
