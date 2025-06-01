# PowerShell script to run the Enhanced Hill Cipher Breaker on Windows
# Author: Lucas Kledeglau Jahchan Alves

# Get the directory where the PowerShell script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Create necessary directories
New-Item -Path "$projectRoot\logs" -ItemType Directory -Force | Out-Null
New-Item -Path "$projectRoot\data" -ItemType Directory -Force | Out-Null
New-Item -Path "$projectRoot\relatorios\enhanced\known" -ItemType Directory -Force | Out-Null
New-Item -Path "$projectRoot\relatorios\enhanced\unknown" -ItemType Directory -Force | Out-Null

# Record start time
$startTime = Get-Date
"Starting enhanced Hill cipher breaker at $startTime" | Out-File -FilePath "$projectRoot\logs\enhanced_breaker.log"

# Display start message
Write-Host "Starting Enhanced Hill Cipher Breaker at $startTime" -ForegroundColor Green
Write-Host "This will use improved techniques for breaking Hill cipher matrices" -ForegroundColor Cyan
Write-Host "- 2x2: Exhaustive search with enhanced scoring" -ForegroundColor Cyan
Write-Host "- 3x3, 4x4, 5x5: Advanced techniques with Portuguese language model" -ForegroundColor Cyan

# Check if dictionary exists
$dictPath = "$projectRoot\data\portuguese_dict.txt"
if (-not (Test-Path $dictPath)) {
    Write-Host "`nPortuguese dictionary not found. It will be downloaded or created automatically." -ForegroundColor Yellow
} else {
    $dictSize = (Get-Item $dictPath).Length
    $dictLines = (Get-Content $dictPath).Count
    Write-Host "`nFound Portuguese dictionary with $dictLines words ($($dictSize/1KB) KB)" -ForegroundColor Cyan
}

# Ask user which matrix sizes to process
Write-Host "`nWhich matrix sizes would you like to process?" -ForegroundColor Yellow
Write-Host "1. All sizes (2x2, 3x3, 4x4, 5x5)" -ForegroundColor White
Write-Host "2. Only problematic sizes (3x3, 4x4, 5x5)" -ForegroundColor White
Write-Host "3. Only 2x2 matrices (exhaustive search)" -ForegroundColor White
Write-Host "4. Custom selection" -ForegroundColor White

$choice = Read-Host "Enter your choice (1-4)"

$sizeParam = ""
switch ($choice) {
    "1" { $sizeParam = "--sizes 2 3 4 5" }
    "2" { $sizeParam = "--sizes 3 4 5" }
    "3" { $sizeParam = "--sizes 2" }
    "4" {
        $customSizes = Read-Host "Enter matrix sizes separated by spaces (e.g., '3 5')"
        $sizeParam = "--sizes $customSizes"
    }
    default { $sizeParam = "--sizes 2 3 4 5" }
}

# Ask user how many threads to use
$threadCount = Read-Host "How many threads would you like to use? (Press Enter for default)"
$threadParam = ""
if ($threadCount -match "^\d+$") {
    $threadParam = "--threads $threadCount"
    Write-Host "Using $threadCount threads for processing" -ForegroundColor Cyan
} else {
    Write-Host "Using default thread count (based on CPU cores)" -ForegroundColor Cyan
}

# Ask if user wants to use known plaintext
$knownTextParam = ""
$useKnownText = Read-Host "Use known plaintext for attack? (y/n, default: y)"
if ($useKnownText -ne "n") {
    $knownTextParam = "--use-known-text"
    Write-Host "Using known plaintext for attack" -ForegroundColor Cyan
} else {
    Write-Host "Not using known plaintext" -ForegroundColor Cyan
}

# Run the enhanced breaker
try {
    Write-Host "`nRunning enhanced Hill cipher breaker..." -ForegroundColor Cyan
    
    # Build the command
    $command = "python `"$scriptDir\run_enhanced.py`" $sizeParam $threadParam $knownTextParam"
    Write-Host "Executing: $command" -ForegroundColor Gray
    
    # Execute the command
    Invoke-Expression $command
    $success = $true
} catch {
    $errorMessage = "Error running enhanced Hill cipher breaker: $_"
    Write-Host $errorMessage -ForegroundColor Red
    $errorMessage | Out-File -FilePath "$projectRoot\logs\enhanced_breaker.log" -Append
    $success = $false
}

# Record end time
$endTime = Get-Date
"Processing completed at $endTime" | Out-File -FilePath "$projectRoot\logs\enhanced_breaker.log" -Append

# Calculate and record duration
$duration = $endTime - $startTime
"Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" | Out-File -FilePath "$projectRoot\logs\enhanced_breaker.log" -Append

# Display completion message
if ($success) {
    Write-Host "`nProcessing completed at $endTime" -ForegroundColor Green
    Write-Host "Total duration: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds" -ForegroundColor Green
    Write-Host "Check results in relatorios\enhanced\" -ForegroundColor Cyan
    
    # Show summary of results if available
    $knownDir = "$projectRoot\relatorios\enhanced\known"
    $unknownDir = "$projectRoot\relatorios\enhanced\unknown"
    
    if ((Test-Path $knownDir) -or (Test-Path $unknownDir)) {
        Write-Host "`nResults Summary:" -ForegroundColor Yellow
        
        if (Test-Path $knownDir) {
            Write-Host "Known texts:" -ForegroundColor White
            Get-ChildItem -Path $knownDir -Directory | ForEach-Object {
                $size = $_.Name -replace "hill_(\d+)x\d+", '$1'
                $reportPath = Join-Path $_.FullName "relatorio.txt"
                if (Test-Path $reportPath) {
                    $content = Get-Content $reportPath -Raw
                    if ($content -match "Palavras válidas: (\d+)/(\d+) \((\d+\.\d+)%\)") {
                        Write-Host "  Matrix ${size}x${size}: $($Matches[1])/$($Matches[2]) valid words ($($Matches[3])%)" -ForegroundColor Cyan
                        
                        # Show best decrypted text
                        if ($content -match "Texto decifrado \(bruto\): ([A-Z]+)\.\.\.") {
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
                    if ($content -match "Palavras válidas: (\d+)/(\d+) \((\d+\.\d+)%\)") {
                        Write-Host "  Matrix ${size}x${size}: $($Matches[1])/$($Matches[2]) valid words ($($Matches[3])%)" -ForegroundColor Cyan
                        
                        # Show best decrypted text
                        if ($content -match "Texto decifrado \(bruto\): ([A-Z]+)\.\.\.") {
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
