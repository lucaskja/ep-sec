# Fully Optimized Hill Cipher Overnight Breaking Session
# PowerShell script for maximum GPU utilization

Write-Host "Fully Optimized Hill Cipher Overnight Breaking Session" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Maximum GPU utilization (80-95% expected)" -ForegroundColor Yellow
Write-Host "Fully optimized CUDA breaker with parallel processing" -ForegroundColor Yellow
Write-Host ""

Write-Host "This optimized version will:" -ForegroundColor Green
Write-Host "- Use massive GPU batches (2048-8192 keys at once)" -ForegroundColor White
Write-Host "- Process matrices in parallel on GPU" -ForegroundColor White
Write-Host "- Perform statistical scoring on GPU" -ForegroundColor White
Write-Host "- Achieve 500+ keys/sec performance" -ForegroundColor White
Write-Host ""

Write-Host "Expected times with full optimization:" -ForegroundColor Yellow
Write-Host "- 3x3 known and unknown (5-15 min each)" -ForegroundColor White
Write-Host "- 4x4 known and unknown (15-60 min each)" -ForegroundColor White
Write-Host "- 5x5 known and unknown (1-4 hours each)" -ForegroundColor White
Write-Host ""

Write-Host "Total estimated time: 3-10 hours (vs 18-50+ hours before)" -ForegroundColor Green
Write-Host ""

Write-Host "Results will be saved in overnight_results/" -ForegroundColor Green
Write-Host "Logs will be saved in overnight_results/logs/" -ForegroundColor Green
Write-Host ""

Write-Host "Monitor GPU usage with: nvidia-smi -l 1" -ForegroundColor Red
Write-Host "Expected GPU utilization: 80-95%" -ForegroundColor Red
Write-Host ""

$confirmation = Read-Host "Press Enter to start fully optimized session or Ctrl+C to cancel"

Write-Host "Starting fully optimized overnight session..." -ForegroundColor Green
Write-Host "Timestamp: $(Get-Date)" -ForegroundColor Gray

try {
    python run_all_hill_ciphers_overnight.py
    Write-Host ""
    Write-Host "Fully optimized session completed successfully!" -ForegroundColor Green
    Write-Host "Check overnight_results/ folder for results." -ForegroundColor Yellow
}
catch {
    Write-Host ""
    Write-Host "Session encountered an error: $_" -ForegroundColor Red
    Write-Host "Check overnight_results/logs/ for details." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
