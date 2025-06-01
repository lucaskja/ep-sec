#!/bin/bash
# Bash script to run Hill cipher breaker on all encrypted texts
# Author: Lucas Kledeglau Jahchan Alves

# Exit on error
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Create results directory if it doesn't exist
mkdir -p "$PROJECT_DIR/results"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Function to run Hill cipher breaker on a file
run_hill_breaker() {
    local file_path=$1
    local key_size=$2
    local method=${3:-"auto"}
    local output_prefix=${4:-""}
    local plaintext_file=${5:-""}
    
    local file_name=$(basename "$file_path")
    local output_file="$PROJECT_DIR/results/${output_prefix}_${file_name}_${key_size}x${key_size}.txt"
    local log_file="$PROJECT_DIR/logs/${output_prefix}_${file_name}_${key_size}x${key_size}.log"
    
    echo "Processing $file_path with key size ${key_size}x${key_size}..."
    
    # Build command arguments
    local args=("$PROJECT_DIR/breakers/hill_breaker.py" 
                "--ciphertext-file" "$file_path" 
                "--key-size" "$key_size" 
                "--method" "$method" 
                "--output-dir" "$PROJECT_DIR/results" 
                "--log-file" "$log_file")
    
    if [ -n "$plaintext_file" ] && [ -f "$plaintext_file" ]; then
        echo -e "\033[0;36mUsing plaintext file: $plaintext_file\033[0m"
        args+=("--plaintext-file" "$plaintext_file")
    fi
    
    # Run the Hill cipher breaker
    python3 "${args[@]}"
    
    # Check if the decryption was successful
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32mSuccessfully processed $file_path with key size ${key_size}x${key_size}\033[0m"
    else
        echo -e "\033[0;31mFailed to process $file_path with key size ${key_size}x${key_size}\033[0m"
    fi
}

# Get the path to normalized text
NORMALIZED_TEXT_PATH="$PROJECT_DIR/data/normalized_text.txt"
if [ ! -f "$NORMALIZED_TEXT_PATH" ]; then
    echo -e "\033[0;33mWarning: Normalized text file not found at $NORMALIZED_TEXT_PATH\033[0m"
    NORMALIZED_TEXT_PATH=""
fi

# Process known texts with correct key sizes
echo -e "\033[0;36mProcessing known texts with correct key sizes...\033[0m"

# Grupo02_2 files use 2x2 matrices
for file in "$PROJECT_DIR/../textos_conhecidos/Cifrado/Hill/Grupo02_2"*.txt; do
    if [ -f "$file" ]; then
        run_hill_breaker "$file" 2 "auto" "known" "$NORMALIZED_TEXT_PATH"
    fi
done

# Grupo02_3 files use 3x3 matrices
for file in "$PROJECT_DIR/../textos_conhecidos/Cifrado/Hill/Grupo02_3"*.txt; do
    if [ -f "$file" ]; then
        run_hill_breaker "$file" 3 "auto" "known" "$NORMALIZED_TEXT_PATH"
    fi
done

# Grupo02_4 files use 4x4 matrices
for file in "$PROJECT_DIR/../textos_conhecidos/Cifrado/Hill/Grupo02_4"*.txt; do
    if [ -f "$file" ]; then
        run_hill_breaker "$file" 4 "auto" "known" "$NORMALIZED_TEXT_PATH"
    fi
done

# Grupo02_5 files use 5x5 matrices
for file in "$PROJECT_DIR/../textos_conhecidos/Cifrado/Hill/Grupo02_5"*.txt; do
    if [ -f "$file" ]; then
        run_hill_breaker "$file" 5 "auto" "known" "$NORMALIZED_TEXT_PATH"
    fi
done

# Process unknown texts with correct key sizes
echo -e "\033[0;36mProcessing unknown texts with correct key sizes...\033[0m"

# Grupo02_2 files use 2x2 matrices
for file in "$PROJECT_DIR/../textos_desconhecidos/Cifrado/Hill/Grupo02_2"*.txt; do
    if [ -f "$file" ]; then
        run_hill_breaker "$file" 2 "auto" "unknown"
    fi
done

# Grupo02_3 files use 3x3 matrices
for file in "$PROJECT_DIR/../textos_desconhecidos/Cifrado/Hill/Grupo02_3"*.txt; do
    if [ -f "$file" ]; then
        run_hill_breaker "$file" 3 "auto" "unknown"
    fi
done

# Grupo02_4 files use 4x4 matrices
for file in "$PROJECT_DIR/../textos_desconhecidos/Cifrado/Hill/Grupo02_4"*.txt; do
    if [ -f "$file" ]; then
        run_hill_breaker "$file" 4 "auto" "unknown"
    fi
done

# Grupo02_5 files use 5x5 matrices
for file in "$PROJECT_DIR/../textos_desconhecidos/Cifrado/Hill/Grupo02_5"*.txt; do
    if [ -f "$file" ]; then
        run_hill_breaker "$file" 5 "auto" "unknown"
    fi
done

echo -e "\033[0;32mAll processing complete!\033[0m"
