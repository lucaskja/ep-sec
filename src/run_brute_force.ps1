# PowerShell script to run the Basic Hill Cipher Breaker (Brute Force) on Windows
# Author: Lucas Kledeglau Jahchan Alves

# Get the directory where the PowerShell script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Create necessary directories
New-Item -Path "$projectRoot\logs" -ItemType Directory -Force | Out-Null
New-Item -Path "$projectRoot\relatorios\brute_force\known" -ItemType Directory -Force | Out-Null
New-Item -Path "$projectRoot\relatorios\brute_force\unknown" -ItemType Directory -Force | Out-Null

# Record start time
$startTime = Get-Date
"Starting brute force Hill cipher breaker at $startTime" | Out-File -FilePath "$projectRoot\logs\brute_force.log"

# Display start message
Write-Host "Starting Basic Hill Cipher Breaker (Brute Force) at $startTime" -ForegroundColor Green
Write-Host "This will use brute force approach to break Hill cipher matrices" -ForegroundColor Cyan
Write-Host "- 2x2: Exhaustive search of all possible matrices" -ForegroundColor Cyan
Write-Host "- 3x3: Exhaustive search (may take a long time)" -ForegroundColor Cyan

# Ask user which matrix sizes to process
Write-Host "`nWhich matrix sizes would you like to process?" -ForegroundColor Yellow
Write-Host "1. All sizes (2x2, 3x3)" -ForegroundColor White
Write-Host "2. Only 2x2 matrices" -ForegroundColor White
Write-Host "3. Only 3x3 matrices (warning: may take a very long time)" -ForegroundColor White
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
    default { $sizeParam = "--sizes 2" }
}

# Run the brute force breaker
try {
    Write-Host "`nRunning brute force Hill cipher breaker..." -ForegroundColor Cyan
    
    # Build the command
    $command = "python `"$scriptDir\run_brute_force.py`" $sizeParam"
    Write-Host "Executing: $command" -ForegroundColor Gray
    
    # Execute the command
    Invoke-Expression $command
    $success = $true
} catch {
    $errorMessage = "Error running brute force Hill cipher breaker: $_"
    Write-Host $errorMessage -ForegroundColor Red
    $errorMessage | Out-File -FilePath "$projectRoot\logs\brute_force.log" -Append
    $success = $false
}

# Record end time
$endTime = Get-Date
"Processing completed at $endTime" | Out-File -FilePath "$projectRoot\logs\brute_force.log" -Append

# Calculate and record duration
$duration = $endTime - $startTime
"Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" | Out-File -FilePath "$projectRoot\logs\brute_force.log" -Append

# Display completion message
if ($success) {
    Write-Host "`nProcessing completed at $endTime" -ForegroundColor Green
    Write-Host "Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" -ForegroundColor Green
    Write-Host "Check results in relatorios\brute_force\" -ForegroundColor Cyan
    
    # Show summary of results if available
    $knownDir = "$projectRoot\relatorios\brute_force\known"
    $unknownDir = "$projectRoot\relatorios\brute_force\unknown"
    
    if ((Test-Path $knownDir) -or (Test-Path $unknownDir)) {
        Write-Host "`nResults Summary:" -ForegroundColor Yellow
        
        if (Test-Path $knownDir) {
            Write-Host "Known texts:" -ForegroundColor White
            Get-ChildItem -Path $knownDir -Directory | ForEach-Object {
                $size = $_.Name -replace "hill_(\d+)x\d+", '$1'
                $reportPath = Join-Path $_.FullName "relatorio.txt"
                if (Test-Path $reportPath) {
                    $content = Get-Content $reportPath -Raw
                    if ($content -match "Top (\d+) results:") {
                        Write-Host "  Matrix ${size}x${size}: $($Matches[1]) potential matrices found" -ForegroundColor Cyan
                        
                        # Show best decrypted text
                        if ($content -match "Decrypted text \(first 100 chars\): ([A-Z]+)\.\.\.") {
                            $decryptedText = $Matches[1]
                            Write-Host "    Best text: $($decryptedText.Substring(0, [Math]::Min(50, $decryptedText.Length)))..." -ForegroundColor Gray
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
                    if ($content -match "Top (\d+) results:") {
                        Write-Host "  Matrix ${size}x${size}: $($Matches[1]) potential matrices found" -ForegroundColor Cyan
                        
                        # Show best decrypted text
                        if ($content -match "Decrypted text \(first 100 chars\): ([A-Z]+)\.\.\.") {
                            $decryptedText = $Matches[1]
                            Write-Host "    Best text: $($decryptedText.Substring(0, [Math]::Min(50, $decryptedText.Length)))..." -ForegroundColor Gray
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
