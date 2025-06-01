# PowerShell script to run Hill cipher breaker on all encrypted texts
# Author: Lucas Kledeglau Jahchan Alves

# Set error action preference
$ErrorActionPreference = "Stop"

# Get the directory of this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# Create results directory if it doesn't exist
if (-not (Test-Path -Path "$ProjectDir\results")) {
    New-Item -ItemType Directory -Path "$ProjectDir\results"
}

# Function to run Hill cipher breaker on a file
function Run-HillBreaker {
    param (
        [string]$FilePath,
        [int]$KeySize,
        [string]$Method = "auto",
        [string]$OutputPrefix = ""
    )
    
    $FileName = [System.IO.Path]::GetFileName($FilePath)
    $OutputFile = "$ProjectDir\results\${OutputPrefix}_${FileName}_${KeySize}x${KeySize}.txt"
    
    Write-Host "Processing $FilePath with key size ${KeySize}x${KeySize}..."
    
    # Run the Hill cipher breaker
    python "$ProjectDir\breakers\hill_breaker.py" --ciphertext-file $FilePath --key-size $KeySize --method $Method --output-dir "$ProjectDir\results"
    
    # Check if the decryption was successful
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully processed $FilePath with key size ${KeySize}x${KeySize}" -ForegroundColor Green
    } else {
        Write-Host "Failed to process $FilePath with key size ${KeySize}x${KeySize}" -ForegroundColor Red
    }
}

# Process known texts
Write-Host "Processing known texts..." -ForegroundColor Cyan
$knownTexts = Get-ChildItem -Path "$ProjectDir\..\textos_conhecidos\Cifrado\Hill" -Filter "*.txt"
foreach ($file in $knownTexts) {
    # Try different key sizes
    foreach ($keySize in 2, 3, 4, 5) {
        Run-HillBreaker -FilePath $file.FullName -KeySize $keySize -OutputPrefix "known"
    }
}

# Process unknown texts
Write-Host "Processing unknown texts..." -ForegroundColor Cyan
$unknownTexts = Get-ChildItem -Path "$ProjectDir\..\textos_desconhecidos\Cifrado\Hill" -Filter "*.txt"
foreach ($file in $unknownTexts) {
    # Try different key sizes
    foreach ($keySize in 2, 3, 4, 5) {
        Run-HillBreaker -FilePath $file.FullName -KeySize $keySize -OutputPrefix "unknown"
    }
}

Write-Host "All processing complete!" -ForegroundColor Green
