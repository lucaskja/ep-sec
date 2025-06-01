# PowerShell script to run the Hill Cipher Breaker overnight on Windows
# Author: Lucas Kledeglau Jahchan Alves

# Get the directory where the PowerShell script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Create necessary directories
New-Item -Path "$projectRoot\relatorios\hibrido\conhecidos" -ItemType Directory -Force | Out-Null
New-Item -Path "$projectRoot\relatorios\hibrido\desconhecidos" -ItemType Directory -Force | Out-Null

# Record start time
$startTime = Get-Date
"Starting processing at $startTime" | Out-File -FilePath "$projectRoot\relatorios\hibrido\log.txt"

# Display start message
Write-Host "Starting Hill Cipher Breaker at $startTime" -ForegroundColor Green
Write-Host "Running hybrid breaker..." -ForegroundColor Cyan

# Run the hybrid breaker
try {
    # Use the correct path to the Python script
    python "$scriptDir\hill_cipher_hybrid_fixed.py"
    $success = $true
} catch {
    Write-Host "Error running Hill Cipher Breaker: $_" -ForegroundColor Red
    "Error running Hill Cipher Breaker: $_" | Out-File -FilePath "$projectRoot\relatorios\hibrido\log.txt" -Append
    $success = $false
}

# Record end time
$endTime = Get-Date
"Processing completed at $endTime" | Out-File -FilePath "$projectRoot\relatorios\hibrido\log.txt" -Append

# Calculate and record duration
$duration = $endTime - $startTime
"Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" | Out-File -FilePath "$projectRoot\relatorios\hibrido\log.txt" -Append

# Results summary
"Results summary:" | Out-File -FilePath "$projectRoot\relatorios\hibrido\log.txt" -Append
if (Test-Path "$projectRoot\relatorios\hibrido\resumo.txt") {
    Get-Content "$projectRoot\relatorios\hibrido\resumo.txt" | Out-File -FilePath "$projectRoot\relatorios\hibrido\log.txt" -Append
} else {
    "No summary file found." | Out-File -FilePath "$projectRoot\relatorios\hibrido\log.txt" -Append
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
