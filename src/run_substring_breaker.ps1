# PowerShell script to run the Substring-based Hill Cipher Breaker on Windows
# Author: Lucas Kledeglau Jahchan Alves

# Get the directory where the PowerShell script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Create necessary directories
New-Item -Path "$projectRoot\logs" -ItemType Directory -Force | Out-Null
New-Item -Path "$projectRoot\relatorios\substring\known" -ItemType Directory -Force | Out-Null
New-Item -Path "$projectRoot\relatorios\substring\unknown" -ItemType Directory -Force | Out-Null

# Record start time
$startTime = Get-Date
"Starting substring-based Hill cipher breaker at $startTime" | Out-File -FilePath "$projectRoot\logs\substring_breaker.log"

# Display start message
Write-Host "Starting Substring-based Hill Cipher Breaker at $startTime" -ForegroundColor Green
Write-Host "This will break Hill ciphers by checking if decrypted text is a substring of the normalized text" -ForegroundColor Cyan
Write-Host "- 2x2: Efficient for small matrices" -ForegroundColor Cyan
Write-Host "- 3x3: May take longer but more accurate" -ForegroundColor Cyan

# Ask user which matrix sizes to process
Write-Host "`nWhich matrix sizes would you like to process?" -ForegroundColor Yellow
Write-Host "1. All sizes (2x2, 3x3)" -ForegroundColor White
Write-Host "2. Only 2x2 matrices" -ForegroundColor White
Write-Host "3. Only 3x3 matrices" -ForegroundColor White
Write-Host "4. Custom selection" -ForegroundColor White

$choice = Read-Host "Enter your choice (1-4)"

$sizeParam = ""
switch ($choice) {
    "1" { $sizeParam = "--sizes 2 3" }
    "2" { $sizeParam = "--sizes 2" }
    "3" { $sizeParam = "--sizes 3" }
    "4" {
        $customSizes = Read-Host "Enter matrix sizes separated by spaces (e.g., '2 3')"
        $sizeParam = "--sizes $customSizes"
    }
    default { $sizeParam = "--sizes 2 3" }
}

# Run the substring-based Hill cipher breaker
try {
    Write-Host "`nRunning substring-based Hill cipher breaker..." -ForegroundColor Cyan
    
    # Build the command
    $command = "python `"$scriptDir\run_substring_breaker.py`" $sizeParam"
    Write-Host "Executing: $command" -ForegroundColor Gray
    
    # Execute the command
    Invoke-Expression $command
    $success = $true
} catch {
    $errorMessage = "Error running substring-based Hill cipher breaker: $_"
    Write-Host $errorMessage -ForegroundColor Red
    $errorMessage | Out-File -FilePath "$projectRoot\logs\substring_breaker.log" -Append
    $success = $false
}

# Record end time
$endTime = Get-Date
"Processing completed at $endTime" | Out-File -FilePath "$projectRoot\logs\substring_breaker.log" -Append

# Calculate and record duration
$duration = $endTime - $startTime
"Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" | Out-File -FilePath "$projectRoot\logs\substring_breaker.log" -Append

# Display completion message
if ($success) {
    Write-Host "`nProcessing completed at $endTime" -ForegroundColor Green
    Write-Host "Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" -ForegroundColor Green
    Write-Host "Check results in relatorios\substring\" -ForegroundColor Cyan
    
    # Show summary of results if available
    $knownDir = "$projectRoot\relatorios\substring\known"
    $unknownDir = "$projectRoot\relatorios\substring\unknown"
    
    if ((Test-Path $knownDir) -or (Test-Path $unknownDir)) {
        Write-Host "`nResults Summary:" -ForegroundColor Yellow
        
        if (Test-Path $knownDir) {
            Write-Host "Known texts:" -ForegroundColor White
            Get-ChildItem -Path $knownDir -Directory | ForEach-Object {
                $size = $_.Name -replace "hill_(\d+)x\d+", '$1'
                $reportPath = Join-Path $_.FullName "relatorio.txt"
                if (Test-Path $reportPath) {
                    $content = Get-Content $reportPath -Raw
                    if ($content -match "Key matrix:") {
                        Write-Host "  Matrix ${size}x${size}: Key found" -ForegroundColor Cyan
                        
                        # Show decrypted text
                        if ($content -match "Decrypted text \(first 200 chars\): ([A-Z]+)\.\.\.") {
                            $decryptedText = $Matches[1]
                            Write-Host "    Decrypted text: $($decryptedText.Substring(0, [Math]::Min(50, $decryptedText.Length)))..." -ForegroundColor Gray
                        }
                    }
                }
            }
        }
        
        if (Test-Path $unknownDir) {
            Write-Host "Unknown texts:" -ForegroundColor White
            Get-ChildItem -Path $unknownDir -Directory | ForEach-Object {
                $size = $_.Name -replace "hill_(\d+)x\d+", '$1'
                $reportPath = Join-Path $_.FullName "relatorio.txt"
                if (Test-Path $reportPath) {
                    $content = Get-Content $reportPath -Raw
                    if ($content -match "Key matrix:") {
                        Write-Host "  Matrix ${size}x${size}: Key found" -ForegroundColor Cyan
                        
                        # Show decrypted text
                        if ($content -match "Decrypted text \(first 200 chars\): ([A-Z]+)\.\.\.") {
                            $decryptedText = $Matches[1]
                            Write-Host "    Decrypted text: $($decryptedText.Substring(0, [Math]::Min(50, $decryptedText.Length)))..." -ForegroundColor Gray
                        }
                    }
                }
            }
        }
    }
} else {
    Write-Host "`nProcessing failed. Check log file for details." -ForegroundColor Red
}

# Keep console window open if script was double-clicked
if ($Host.Name -eq "ConsoleHost") {
    Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
