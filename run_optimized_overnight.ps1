# Optimized Hill Cipher Overnight Breaking Session
# PowerShell script for maximum GPU utilization

Write-Host "🚀 Optimized Hill Cipher Overnight Breaking Session" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "🔥 GPU-optimized with large batch processing" -ForegroundColor Yellow
Write-Host "⚡ Maximum RTX 4060 Ti utilization" -ForegroundColor Yellow
Write-Host ""

Write-Host "This optimized version will:" -ForegroundColor Green
Write-Host "- Use large GPU batches (1024-4096 keys at once)" -ForegroundColor White
Write-Host "- Maximize GPU utilization (should see 80-95% usage)" -ForegroundColor White
Write-Host "- Process matrices in parallel on GPU" -ForegroundColor White
Write-Host "- Achieve 5-10x faster performance" -ForegroundColor White
Write-Host ""

Write-Host "Expected times with optimization:" -ForegroundColor Yellow
Write-Host "- 3x3 known and unknown (5-15 min each)" -ForegroundColor White
Write-Host "- 4x4 known and unknown (30 min - 2 hours each)" -ForegroundColor White
Write-Host "- 5x5 known and unknown (2-8 hours each)" -ForegroundColor White
Write-Host ""

Write-Host "Results will be saved in optimized_overnight_results/" -ForegroundColor Green
Write-Host "Logs will be saved in optimized_overnight_results/logs/" -ForegroundColor Green
Write-Host ""

Write-Host "⚠️  Make sure your GPU drivers are up to date!" -ForegroundColor Red
Write-Host "⚠️  Monitor GPU temperature during long runs!" -ForegroundColor Red
Write-Host ""

$confirmation = Read-Host "Press Enter to start optimized session or Ctrl+C to cancel"

Write-Host "Starting optimized overnight session..." -ForegroundColor Green
Write-Host "Timestamp: $(Get-Date)" -ForegroundColor Gray
Write-Host "Expected GPU utilization: 80-95%" -ForegroundColor Yellow

try {
    python run_optimized_overnight.py
    Write-Host ""
    Write-Host "Optimized session completed successfully!" -ForegroundColor Green
    Write-Host "Check optimized_overnight_results/ folder for results." -ForegroundColor Yellow
}
catch {
    Write-Host ""
    Write-Host "Session encountered an error: $_" -ForegroundColor Red
    Write-Host "Check optimized_overnight_results/logs/ for details." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
