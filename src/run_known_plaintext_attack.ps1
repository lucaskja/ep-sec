# PowerShell script to run the Known Plaintext Attack on Hill Cipher on Windows
# Author: Lucas Kledeglau Jahchan Alves

# Get the directory where the PowerShell script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Create necessary directories
New-Item -Path "$projectRoot\logs" -ItemType Directory -Force | Out-Null
New-Item -Path "$projectRoot\relatorios\known_plaintext\known" -ItemType Directory -Force | Out-Null
New-Item -Path "$projectRoot\relatorios\known_plaintext\unknown" -ItemType Directory -Force | Out-Null

# Record start time
$startTime = Get-Date
"Starting known plaintext attack at $startTime" | Out-File -FilePath "$projectRoot\logs\known_plaintext_attack.log"

# Display start message
Write-Host "Starting Known Plaintext Attack on Hill Cipher at $startTime" -ForegroundColor Green
Write-Host "This will use common substrings from avesso_da_pele.txt to break Hill cipher matrices" -ForegroundColor Cyan
Write-Host "- 2x2: Known plaintext attack with common substrings" -ForegroundColor Cyan
Write-Host "- 3x3: Known plaintext attack with common substrings" -ForegroundColor Cyan
Write-Host "- 4x4: Known plaintext attack with common substrings" -ForegroundColor Cyan
Write-Host "- 5x5: Known plaintext attack with common substrings" -ForegroundColor Cyan

# Ask user which matrix sizes to process
Write-Host "`nWhich matrix sizes would you like to process?" -ForegroundColor Yellow
Write-Host "1. All sizes (2x2, 3x3, 4x4, 5x5)" -ForegroundColor White
Write-Host "2. Only 2x2 matrices" -ForegroundColor White
Write-Host "3. Only 3x3 matrices" -ForegroundColor White
Write-Host "4. Only 4x4 matrices" -ForegroundColor White
Write-Host "5. Only 5x5 matrices" -ForegroundColor White
Write-Host "6. Custom selection" -ForegroundColor White

$choice = Read-Host "Enter your choice (1-6)"

$sizeParam = ""
switch ($choice) {
    "1" { $sizeParam = "--sizes 2 3 4 5" }
    "2" { $sizeParam = "--sizes 2" }
    "3" { $sizeParam = "--sizes 3" }
    "4" { $sizeParam = "--sizes 4" }
    "5" { $sizeParam = "--sizes 5" }
    "6" {
        $customSizes = Read-Host "Enter matrix sizes separated by spaces (e.g., '2 5')"
        $sizeParam = "--sizes $customSizes"
    }
    default { $sizeParam = "--sizes 2 3 4 5" }
}

# Ask user how many top results to show
$topResults = Read-Host "How many top results would you like to see? (Press Enter for default: 5)"
$topParam = ""
if ($topResults -match "^\d+$") {
    $topParam = "--top-n $topResults"
    Write-Host "Showing top $topResults results" -ForegroundColor Cyan
} else {
    Write-Host "Showing top 5 results (default)" -ForegroundColor Cyan
}

# Run the known plaintext attack
try {
    Write-Host "`nRunning known plaintext attack..." -ForegroundColor Cyan
    
    # Build the command
    $command = "python `"$scriptDir\run_known_plaintext_attack.py`" $sizeParam $topParam"
    Write-Host "Executing: $command" -ForegroundColor Gray
    
    # Execute the command
    Invoke-Expression $command
    $success = $true
} catch {
    $errorMessage = "Error running known plaintext attack: $_"
    Write-Host $errorMessage -ForegroundColor Red
    $errorMessage | Out-File -FilePath "$projectRoot\logs\known_plaintext_attack.log" -Append
    $success = $false
}

# Record end time
$endTime = Get-Date
"Processing completed at $endTime" | Out-File -FilePath "$projectRoot\logs\known_plaintext_attack.log" -Append

# Calculate and record duration
$duration = $endTime - $startTime
"Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" | Out-File -FilePath "$projectRoot\logs\known_plaintext_attack.log" -Append

# Display completion message
if ($success) {
    Write-Host "`nProcessing completed at $endTime" -ForegroundColor Green
    Write-Host "Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" -ForegroundColor Green
    Write-Host "Check results in relatorios\known_plaintext\" -ForegroundColor Cyan
    
    # Show summary of results if available
    $knownDir = "$projectRoot\relatorios\known_plaintext\known"
    $unknownDir = "$projectRoot\relatorios\known_plaintext\unknown"
    
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
                        if ($content -match "Decrypted text \(first 200 chars\): ([A-Z]+)\.\.\.") {
                            $decryptedText = $Matches[1]
                            Write-Host "    Best text: $($decryptedText.Substring(0, [Math]::Min(50, $decryptedText.Length)))..." -ForegroundColor Gray
                        }
                        
                        # Show common words found
                        if ($content -match "Common words found: ([A-Z, ]+)") {
                            $commonWords = $Matches[1]
                            Write-Host "    Common words: $commonWords" -ForegroundColor Gray
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
                        if ($content -match "Decrypted text \(first 200 chars\): ([A-Z]+)\.\.\.") {
                            $decryptedText = $Matches[1]
                            Write-Host "    Best text: $($decryptedText.Substring(0, [Math]::Min(50, $decryptedText.Length)))..." -ForegroundColor Gray
                        }
                        
                        # Show common words found
                        if ($content -match "Common words found: ([A-Z, ]+)") {
                            $commonWords = $Matches[1]
                            Write-Host "    Common words: $commonWords" -ForegroundColor Gray
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
