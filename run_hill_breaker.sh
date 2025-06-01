#!/bin/bash
# Bash script to run Hill cipher breaker on all encrypted texts
# Author: Lucas Kledeglau Jahchan Alves

# Exit on error
set -e

# Create results directory if it doesn't exist
mkdir -p results

# Function to run Hill cipher breaker on a file
run_hill_breaker() {
    local file_path=$1
    local key_size=$2
    local method=${3:-"auto"}
    local output_prefix=${4:-""}
    
    local file_name=$(basename "$file_path")
    local output_file="results/${output_prefix}_${file_name}_${key_size}x${key_size}.txt"
    
    echo "Processing $file_path with key size ${key_size}x${key_size}..."
    
    # Run the Hill cipher breaker
    python3 src/hill_cipher_breaker.py --ciphertext-file "$file_path" --key-size "$key_size" --method "$method" --output-dir "results"
    
    # Check if the decryption was successful
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32mSuccessfully processed $file_path with key size ${key_size}x${key_size}\033[0m"
    else
        echo -e "\033[0;31mFailed to process $file_path with key size ${key_size}x${key_size}\033[0m"
    fi
}

# Process known texts
echo -e "\033[0;36mProcessing known texts...\033[0m"
for file in textos_conhecidos/Cifrado/Hill/*.txt; do
    # Try different key sizes
    for key_size in 2 3 4 5; do
        run_hill_breaker "$file" "$key_size" "auto" "known"
    done
done

# Process unknown texts
echo -e "\033[0;36mProcessing unknown texts...\033[0m"
for file in textos_desconhecidos/Cifrado/Hill/*.txt; do
    # Try different key sizes
    for key_size in 2 3 4 5; do
        run_hill_breaker "$file" "$key_size" "auto" "unknown"
    done
done

echo -e "\033[0;32mAll processing complete!\033[0m"
