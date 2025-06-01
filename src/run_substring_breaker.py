#!/usr/bin/env python3
"""
Script to run the substring-based Hill cipher breaker on all available ciphertexts.
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Tuple, Optional
import numpy as np

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.substring_hill_breaker import break_hill_cipher, load_normalized_text, preprocess_text
from src.hill_cipher import decrypt_hill

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Substring-based Hill Cipher Breaker")
    parser.add_argument("--known-dir", default="textos_conhecidos", help="Directory with known texts")
    parser.add_argument("--unknown-dir", default="textos_desconhecidos", help="Directory with unknown texts")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3], help="Matrix sizes to process")
    parser.add_argument("--normalized-text", default="data/normalized_text.txt", help="Path to normalized text file")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        filename="substring_breaker.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load normalized text
    normalized_text = load_normalized_text(args.normalized_text)
    if not normalized_text:
        print("Error: Normalized text is empty.")
        return
    
    print(f"Loaded normalized text: {len(normalized_text)} characters")
    
    # Process known texts
    process_texts(args.known_dir, args.sizes, normalized_text, known=True)
    
    # Process unknown texts
    process_texts(args.unknown_dir, args.sizes, normalized_text, known=False)

def process_texts(base_dir: str, sizes: List[int], normalized_text: str, known: bool = False):
    """
    Process texts in the given directory.
    
    Args:
        base_dir: Base directory containing texts
        sizes: Matrix sizes to process
        normalized_text: Normalized text to search in
        known: Whether these are known texts
    """
    text_type = "known" if known else "unknown"
    print(f"\n=== Processing {text_type} texts ===")
    
    for size in sizes:
        # Path to ciphertext
        cipher_path = os.path.join(base_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
        
        if not os.path.exists(cipher_path):
            print(f"File {cipher_path} not found. Skipping...")
            continue
        
        # Read ciphertext
        with open(cipher_path, 'r') as f:
            ciphertext = f.read().strip()
        
        print(f"\n--- Breaking {size}x{size} cipher ({text_type} text) ---")
        print(f"Ciphertext length: {len(ciphertext)} characters")
        
        # Break the cipher
        start_time = time.time()
        key_matrix = break_hill_cipher(ciphertext, size, normalized_text)
        elapsed_time = time.time() - start_time
        
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        
        # Generate report
        if key_matrix is not None:
            report = f"=== Substring-based Hill Cipher Breaker Report ===\n"
            report += f"Matrix size: {size}x{size}\n"
            report += f"Text type: {text_type.capitalize()}\n"
            report += f"Execution time: {elapsed_time:.2f} seconds\n\n"
            
            # Add key matrix
            report += f"Key matrix:\n{key_matrix}\n\n"
            
            # Add decrypted text
            decrypted = decrypt_hill(preprocess_text(ciphertext), key_matrix)
            report += f"Decrypted text (first 200 chars): {decrypted[:200]}...\n"
            
            print(report)
            
            # Save report
            report_dir = f"relatorios/substring/{text_type}/hill_{size}x{size}"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "relatorio.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {report_path}")
            
            # Save decrypted text
            decrypted_path = os.path.join(report_dir, "decrypted.txt")
            with open(decrypted_path, 'w') as f:
                f.write(decrypted)
            print(f"Decrypted text saved to {decrypted_path}")
            
            # Save matrix
            matrix_path = os.path.join(report_dir, "matrix.txt")
            with open(matrix_path, 'w') as f:
                f.write(str(key_matrix))
            print(f"Matrix saved to {matrix_path}")
        else:
            print("No key matrix found.")

if __name__ == "__main__":
    main()
