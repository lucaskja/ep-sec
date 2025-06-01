#!/bin/bash
# Run all unit tests

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the tests
echo "Running Hill Cipher core tests..."
python3 -m unittest tests/test_hill_cipher.py

echo "Running Known-Plaintext Attack tests..."
python3 -m unittest tests/test_kpa.py

echo "Running Genetic Algorithm tests..."
python3 -m unittest tests/test_genetic.py

echo "All tests completed!"
