#!/usr/bin/env python3
"""
Script to run the Enhanced Hill Cipher Breaker.

This script runs the enhanced Hill cipher breaker on all available ciphertexts.
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
from src.enhanced_hill_breaker import EnhancedHillBreaker
from src.hill_cipher_analyzer import HillCipherAnalyzer
from src.hill_cipher import decrypt_hill

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Hill Cipher Breaker")
    parser.add_argument("--known-dir", default="textos_conhecidos", help="Directory with known texts")
    parser.add_argument("--unknown-dir", default="textos_desconhecidos", help="Directory with unknown texts")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5], help="Matrix sizes to process")
    parser.add_argument("--dict-path", default="data/portuguese_dict.txt", help="Path to Portuguese dictionary")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads to use")
    parser.add_argument("--use-known-text", action="store_true", help="Use known plaintext for attack")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        filename="enhanced_hill_breaker.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Process known texts
    process_texts(args.known_dir, args.sizes, args.dict_path, args.threads, known=True, use_known_text=args.use_known_text)
    
    # Process unknown texts
    process_texts(args.unknown_dir, args.sizes, args.dict_path, args.threads, known=False, use_known_text=args.use_known_text)

def process_texts(base_dir: str, sizes: List[int], dict_path: str, num_threads: int, known: bool = False, use_known_text: bool = True):
    """
    Process texts in the given directory.
    
    Args:
        base_dir: Base directory containing texts
        sizes: Matrix sizes to process
        dict_path: Path to Portuguese dictionary
        num_threads: Number of threads to use
        known: Whether these are known texts
        use_known_text: Whether to use known plaintext for attack
    """
    text_type = "known" if known else "unknown"
    print(f"\n=== Processing {text_type} texts ===")
    
    for size in sizes:
        # Path to ciphertext
        cipher_path = os.path.join(base_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
        
        if not os.path.exists(cipher_path):
            print(f"File {cipher_path} not found. Skipping...")
            continue
        
        # Path to original text (for known texts)
        original_text_path = None
        if known and use_known_text:
            for text_file in os.listdir(os.path.join(base_dir, "textos")):
                if text_file.endswith(".txt"):
                    original_text_path = os.path.join(base_dir, "textos", text_file)
                    print(f"Using original text: {original_text_path}")
                    break
        
        # Read ciphertext
        with open(cipher_path, 'r') as f:
            ciphertext = f.read().strip()
        
        print(f"\n--- Breaking {size}x{size} cipher ({text_type} text) ---")
        
        # First, try the statistical analyzer to get potential matrices
        start_time = time.time()
        analyzer = HillCipherAnalyzer(size)
        potential_matrices = []
        
        try:
            # Get potential matrices from analyzer
            analyzer_results = analyzer.analyze_ciphertext(ciphertext)
            potential_matrices = [matrix for matrix, _, _ in analyzer_results]
            print(f"Statistical analyzer found {len(potential_matrices)} potential matrices")
        except Exception as e:
            print(f"Error in statistical analyzer: {e}")
        
        # Create breaker
        breaker = EnhancedHillBreaker(size, dict_path, num_threads)
        
        # Break cipher
        results = []
        
        # First, try the potential matrices from the analyzer
        if potential_matrices:
            print("Testing matrices from statistical analyzer...")
            for matrix in potential_matrices:
                try:
                    decrypted = decrypt_hill(ciphertext, matrix)
                    score = breaker.score_text(decrypted)
                    results.append((matrix, decrypted, score))
                except Exception as e:
                    print(f"Error testing matrix: {e}")
        
        # Then run the regular breaker
        print("Running enhanced breaker...")
        breaker_results = breaker.break_cipher(ciphertext, original_text_path)
        results.extend(breaker_results)
        
        # Sort results by score
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:100]  # Keep top 100
        
        elapsed_time = time.time() - start_time
        
        # Generate report
        if results:
            report = breaker.generate_report(results, ciphertext)
            print(report)
            print(f"Execution time: {elapsed_time:.2f} seconds")
            
            # Save report
            report_dir = f"relatorios/enhanced/{text_type}/hill_{size}x{size}"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "relatorio.txt")
            with open(report_path, 'w') as f:
                f.write(report)
                f.write(f"\n\nExecution time: {elapsed_time:.2f} seconds")
            print(f"Report saved to {report_path}")
        else:
            print("No results found.")

if __name__ == "__main__":
    main()
