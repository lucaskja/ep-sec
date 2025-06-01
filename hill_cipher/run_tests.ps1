# PowerShell script to run all unit tests

# Get the directory of this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Run the tests
Write-Host "Running Hill Cipher core tests..." -ForegroundColor Cyan
python -m unittest tests/test_hill_cipher.py

Write-Host "Running Known-Plaintext Attack tests..." -ForegroundColor Cyan
python -m unittest tests/test_kpa.py

Write-Host "Running Genetic Algorithm tests..." -ForegroundColor Cyan
python -m unittest tests/test_genetic.py

Write-Host "All tests completed!" -ForegroundColor Green
