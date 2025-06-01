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

# Process known texts with correct key sizes
Write-Host "Processing known texts with correct key sizes..." -ForegroundColor Cyan

# Grupo02_2 files use 2x2 matrices
$knownTexts2x2 = Get-ChildItem -Path "$ProjectDir\..\textos_conhecidos\Cifrado\Hill" -Filter "Grupo02_2*.txt"
foreach ($file in $knownTexts2x2) {
    Run-HillBreaker -FilePath $file.FullName -KeySize 2 -OutputPrefix "known"
}

# Grupo02_3 files use 3x3 matrices
$knownTexts3x3 = Get-ChildItem -Path "$ProjectDir\..\textos_conhecidos\Cifrado\Hill" -Filter "Grupo02_3*.txt"
foreach ($file in $knownTexts3x3) {
    Run-HillBreaker -FilePath $file.FullName -KeySize 3 -OutputPrefix "known"
}

# Grupo02_4 files use 4x4 matrices
$knownTexts4x4 = Get-ChildItem -Path "$ProjectDir\..\textos_conhecidos\Cifrado\Hill" -Filter "Grupo02_4*.txt"
foreach ($file in $knownTexts4x4) {
    Run-HillBreaker -FilePath $file.FullName -KeySize 4 -OutputPrefix "known"
}

# Grupo02_5 files use 5x5 matrices
$knownTexts5x5 = Get-ChildItem -Path "$ProjectDir\..\textos_conhecidos\Cifrado\Hill" -Filter "Grupo02_5*.txt"
foreach ($file in $knownTexts5x5) {
    Run-HillBreaker -FilePath $file.FullName -KeySize 5 -OutputPrefix "known"
}

# Process unknown texts with correct key sizes
Write-Host "Processing unknown texts with correct key sizes..." -ForegroundColor Cyan

# Grupo02_2 files use 2x2 matrices
$unknownTexts2x2 = Get-ChildItem -Path "$ProjectDir\..\textos_desconhecidos\Cifrado\Hill" -Filter "Grupo02_2*.txt"
foreach ($file in $unknownTexts2x2) {
    Run-HillBreaker -FilePath $file.FullName -KeySize 2 -OutputPrefix "unknown"
}

# Grupo02_3 files use 3x3 matrices
$unknownTexts3x3 = Get-ChildItem -Path "$ProjectDir\..\textos_desconhecidos\Cifrado\Hill" -Filter "Grupo02_3*.txt"
foreach ($file in $unknownTexts3x3) {
    Run-HillBreaker -FilePath $file.FullName -KeySize 3 -OutputPrefix "unknown"
}

# Grupo02_4 files use 4x4 matrices
$unknownTexts4x4 = Get-ChildItem -Path "$ProjectDir\..\textos_desconhecidos\Cifrado\Hill" -Filter "Grupo02_4*.txt"
foreach ($file in $unknownTexts4x4) {
    Run-HillBreaker -FilePath $file.FullName -KeySize 4 -OutputPrefix "unknown"
}

# Grupo02_5 files use 5x5 matrices
$unknownTexts5x5 = Get-ChildItem -Path "$ProjectDir\..\textos_desconhecidos\Cifrado\Hill" -Filter "Grupo02_5*.txt"
foreach ($file in $unknownTexts5x5) {
    Run-HillBreaker -FilePath $file.FullName -KeySize 5 -OutputPrefix "unknown"
}

Write-Host "All processing complete!" -ForegroundColor Green
