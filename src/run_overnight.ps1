# PowerShell script to run the Hill Cipher Breaker overnight on Windows
# Author: Lucas Kledeglau Jahchan Alves

# Create necessary directories
New-Item -Path "relatorios\hibrido\conhecidos" -ItemType Directory -Force | Out-Null
New-Item -Path "relatorios\hibrido\desconhecidos" -ItemType Directory -Force | Out-Null

# Record start time
$startTime = Get-Date
"Starting processing at $startTime" | Out-File -FilePath "relatorios\hibrido\log.txt"

# Display start message
Write-Host "Starting Hill Cipher Breaker at $startTime" -ForegroundColor Green
Write-Host "Running hybrid breaker..." -ForegroundColor Cyan

# Run the hybrid breaker
try {
    python src/hill_cipher_hybrid_fixed.py
    $success = $true
} catch {
    Write-Host "Error running Hill Cipher Breaker: $_" -ForegroundColor Red
    "Error running Hill Cipher Breaker: $_" | Out-File -FilePath "relatorios\hibrido\log.txt" -Append
    $success = $false
}

# Record end time
$endTime = Get-Date
"Processing completed at $endTime" | Out-File -FilePath "relatorios\hibrido\log.txt" -Append

# Calculate and record duration
$duration = $endTime - $startTime
"Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" | Out-File -FilePath "relatorios\hibrido\log.txt" -Append

# Results summary
"Results summary:" | Out-File -FilePath "relatorios\hibrido\log.txt" -Append
if (Test-Path "relatorios\hibrido\resumo.txt") {
    Get-Content "relatorios\hibrido\resumo.txt" | Out-File -FilePath "relatorios\hibrido\log.txt" -Append
} else {
    "No summary file found." | Out-File -FilePath "relatorios\hibrido\log.txt" -Append
}

# Display completion message
if ($success) {
    Write-Host "`nProcessing completed at $endTime" -ForegroundColor Green
    Write-Host "Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" -ForegroundColor Green
    Write-Host "Check results in relatorios\hibrido\" -ForegroundColor Cyan
} else {
    Write-Host "`nProcessing failed. Check log file for details." -ForegroundColor Red
}

# Keep console window open if script was double-clicked
if ($Host.Name -eq "ConsoleHost") {
    Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
