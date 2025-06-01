#!/usr/bin/env python3
"""
Script to run the basic Hill Cipher Breaker (brute force approach).

This script runs the basic Hill cipher breaker on all available ciphertexts.
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Tuple
import numpy as np

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import our modules
from src.hill_cipher_breaker import HillCipherBreaker

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Basic Hill Cipher Breaker (Brute Force)")
    parser.add_argument("--known-dir", default="textos_conhecidos", help="Directory with known texts")
    parser.add_argument("--unknown-dir", default="textos_desconhecidos", help="Directory with unknown texts")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3], help="Matrix sizes to process (2 or 3)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        filename="brute_force.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Process known texts
    process_texts(args.known_dir, args.sizes, known=True)
    
    # Process unknown texts
    process_texts(args.unknown_dir, args.sizes, known=False)

def process_texts(base_dir: str, sizes: List[int], known: bool = False):
    """
    Process texts in the given directory.
    
    Args:
        base_dir: Base directory containing texts
        sizes: Matrix sizes to process
        known: Whether these are known texts
    """
    text_type = "known" if known else "unknown"
    print(f"\n=== Processing {text_type} texts ===")
    
    for size in sizes:
        if size > 3:
            print(f"Size {size}x{size} is too large for brute force approach. Skipping...")
            continue
            
        # Path to ciphertext
        cipher_path = os.path.join(base_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
        
        if not os.path.exists(cipher_path):
            print(f"File {cipher_path} not found. Skipping...")
            continue
        
        # Read ciphertext
        with open(cipher_path, 'r') as f:
            ciphertext = f.read().strip()
        
        print(f"\n--- Breaking {size}x{size} cipher ({text_type} text) ---")
        
        # Run brute force breaker
        start_time = time.time()
        breaker = HillCipherBreaker()
        results = breaker.break_cipher(ciphertext, size)
        elapsed_time = time.time() - start_time
        
        # Generate report
        if results:
            report = f"=== Brute Force Hill Cipher Breaker Report ===\n"
            report += f"Matrix size: {size}x{size}\n"
            report += f"Text type: {text_type.capitalize()}\n"
            report += f"Execution time: {elapsed_time:.2f} seconds\n\n"
            
            # Add top results
            report += f"Top {len(results)} results:\n"
            for i, (matrix, decrypted, score) in enumerate(results):
                report += f"\n--- Result {i+1} (Score: {score:.2f}) ---\n"
                report += f"Matrix:\n{matrix}\n"
                report += f"Decrypted text (first 100 chars): {decrypted[:100]}...\n"
            
            print(report)
            
            # Save report
            report_dir = f"relatorios/brute_force/{text_type}/hill_{size}x{size}"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "relatorio.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {report_path}")
            
            # Save decrypted text for the best result
            if results:
                best_matrix, best_decrypted, best_score = results[0]
                decrypted_path = os.path.join(report_dir, "decrypted.txt")
                with open(decrypted_path, 'w') as f:
                    f.write(best_decrypted)
                print(f"Best decryption saved to {decrypted_path}")
        else:
            print("No results found.")

if __name__ == "__main__":
    main()
