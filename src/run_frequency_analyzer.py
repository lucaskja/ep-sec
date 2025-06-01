#!/usr/bin/env python3
"""
Script to run the Frequency Analyzer for Hill Cipher.

This script runs the frequency analyzer on all available ciphertexts.
"""

import os
import sys
import time
import argparse
import logging
import re
from typing import List, Tuple
import numpy as np

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import our modules
from src.frequency_analyzer import FrequencyAnalyzer
from src.hill_cipher import decrypt_hill

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Frequency Analyzer for Hill Cipher")
    parser.add_argument("--known-dir", default="textos_conhecidos", help="Directory with known texts")
    parser.add_argument("--unknown-dir", default="textos_desconhecidos", help="Directory with unknown texts")
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5], help="Matrix sizes to process")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to show")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        filename="frequency_analyzer.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Process known texts
    process_texts(args.known_dir, args.sizes, args.top, known=True)
    
    # Process unknown texts
    process_texts(args.unknown_dir, args.sizes, args.top, known=False)

def process_texts(base_dir: str, sizes: List[int], top_n: int, known: bool = False):
    """
    Process texts in the given directory.
    
    Args:
        base_dir: Base directory containing texts
        sizes: Matrix sizes to process
        top_n: Number of top results to show
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
        
        # Run frequency analyzer
        start_time = time.time()
        analyzer = FrequencyAnalyzer(size)
        results = analyzer.analyze_ciphertext(ciphertext, top_n)
        elapsed_time = time.time() - start_time
        
        # Generate report
        if results:
            report = generate_report(results, ciphertext, size, text_type, elapsed_time)
            print(report)
            
            # Save report
            report_dir = f"relatorios/frequency/{text_type}/hill_{size}x{size}"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "relatorio.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {report_path}")
        else:
            print("No results found.")

def generate_report(results: List[Tuple[np.ndarray, str, float]], ciphertext: str, 
                   size: int, text_type: str, elapsed_time: float) -> str:
    """
    Generate a report from the results.
    
    Args:
        results: List of (matrix, decrypted_text, score) tuples
        ciphertext: Original ciphertext
        size: Matrix size
        text_type: Type of text (known or unknown)
        elapsed_time: Execution time in seconds
        
    Returns:
        Report as a string
    """
    report = f"=== Frequency Analyzer Report ===\n"
    report += f"Matrix size: {size}x{size}\n"
    report += f"Text type: {text_type.capitalize()}\n"
    report += f"Execution time: {elapsed_time:.2f} seconds\n\n"
    
    # Add top results
    report += f"Top {len(results)} results:\n"
    for i, (matrix, decrypted, score) in enumerate(results):
        report += f"\n--- Result {i+1} (Score: {score:.2f}) ---\n"
        report += f"Matrix:\n{matrix}\n"
        report += f"Decrypted text (first 100 chars): {decrypted[:100]}...\n"
        
        # Add word analysis
        words = re.findall(r'[A-Z]{2,}', decrypted)
        common_words = ['DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
                       'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS']
        found_common = [word for word in words if word in common_words]
        report += f"Common words found: {', '.join(found_common[:10])}\n"
    
    return report

if __name__ == "__main__":
    main()
