#!/usr/bin/env python3
"""
Hill Cipher Hybrid Breaker - Combines basic and optimized approaches

This script implements a hybrid approach to break the Hill cipher:
- Uses the basic breaker (fast) for 2x2 matrices
- Uses the optimized breaker (more accurate) for 3x3, 4x4, and 5x5 matrices

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import time
import sys
import argparse
from typing import List, Tuple, Dict, Optional

# Import different breakers
try:
    from hill_cipher_breaker import HillCipherBreaker as BasicBreaker
    from hill_cipher_breaker_optimized import HillCipherBreaker as OptimizedBreaker
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.hill_cipher_breaker import HillCipherBreaker as BasicBreaker
    from src.hill_cipher_breaker_optimized import HillCipherBreaker as OptimizedBreaker

def process_all_ciphers(known_dir: str = "textos_conhecidos", 
                        unknown_dir: str = "textos_desconhecidos",
                        sizes: List[int] = [2, 3, 4, 5],
                        save_reports: bool = True) -> Dict:
    """
    Process all ciphers using the hybrid approach.
    
    Args:
        known_dir: Directory with known texts
        unknown_dir: Directory with unknown texts
        sizes: Matrix sizes to process
        save_reports: If True, save reports to files
        
    Returns:
        Dictionary with results and statistics
    """
    results = {}
    
    # Check if directories exist
    if not os.path.exists(known_dir) or not os.path.exists(unknown_dir):
        print(f"Error: Directories {known_dir} or {unknown_dir} not found.")
        return results
    
    # Create breaker instances
    basic_breaker = BasicBreaker()
    optimized_breaker = OptimizedBreaker()
    
    print("=== Hill Cipher Hybrid Breaker ===")
    print("Using hybrid approach:")
    print("- Basic breaker for 2x2 matrices")
    print("- Optimized breaker for 3x3, 4x4, and 5x5 matrices")
    
    # Process known texts
    print("\n=== Processing known texts ===")
    for size in sizes:
        cipher_path = os.path.join(known_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
        
        if not os.path.exists(cipher_path):
            print(f"File {cipher_path} not found. Skipping...")
            continue
        
        # Find corresponding original text
        original_text_path = None
        for text_file in os.listdir(os.path.join(known_dir, "textos")):
            if text_file.endswith(".txt"):
                original_text_path = os.path.join(known_dir, "textos", text_file)
                print(f"Using original text: {original_text_path}")
                break
        
        # Read ciphertext
        with open(cipher_path, 'r') as f:
            ciphertext = f.read().strip()
        
        print(f"\n--- Breaking {size}x{size} cipher (known text) ---")
        start_time = time.time()
        
        # Choose appropriate breaker
        if size == 2:
            print(f"Using basic breaker for {size}x{size} matrix...")
            results_list = basic_breaker.break_cipher(ciphertext, size)
        else:  # For 3x3, 4x4, and 5x5, use optimized breaker
            print(f"Using optimized breaker for {size}x{size} matrix...")
            try:
                # Use optimized breaker directly
                results_list = optimized_breaker.break_cipher(ciphertext, size, known_text_path=original_text_path)
            except Exception as e:
                print(f"Error in advanced techniques: {e}")
                results_list = []
        
        elapsed_time = time.time() - start_time
        
        # Generate report
        if results_list:
            if size == 2:
                report = basic_breaker.generate_report(results_list, ciphertext, size)
            else:
                report = optimized_breaker.generate_report(results_list, ciphertext, size, known_text_path=original_text_path)
            
            print(report)
            print(f"Execution time: {elapsed_time:.2f} seconds")
            
            # Save report
            if save_reports:
                report_dir = f"relatorios/hibrido/conhecidos/hill_{size}x{size}"
                os.makedirs(report_dir, exist_ok=True)
                report_path = os.path.join(report_dir, "relatorio.txt")
                with open(report_path, 'w') as f:
                    f.write(report)
                    f.write(f"\n\nExecution time: {elapsed_time:.2f} seconds")
                print(f"Report saved to {report_path}")
            
            # Store results
            results[f"known_{size}x{size}"] = {
                "best_matrix": results_list[0][0].tolist() if results_list else None,
                "best_text": results_list[0][1][:100] if results_list else None,
                "best_score": results_list[0][2] if results_list else None,
                "time": elapsed_time
            }
        else:
            print("No results found.")
            results[f"known_{size}x{size}"] = {"error": "No results found"}
    
    # Process unknown texts
    print("\n=== Processing unknown texts ===")
    for size in sizes:
        cipher_path = os.path.join(unknown_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
        
        if not os.path.exists(cipher_path):
            print(f"File {cipher_path} not found. Skipping...")
            continue
        
        # Read ciphertext
        with open(cipher_path, 'r') as f:
            ciphertext = f.read().strip()
        
        print(f"\n--- Breaking {size}x{size} cipher (unknown text) ---")
        start_time = time.time()
        
        # Choose appropriate breaker
        if size == 2:
            print(f"Using basic breaker for {size}x{size} matrix...")
            results_list = basic_breaker.break_cipher(ciphertext, size)
        else:  # For 3x3, 4x4, and 5x5, use optimized breaker
            print(f"Using optimized breaker for {size}x{size} matrix...")
            try:
                # Use optimized breaker directly
                results_list = optimized_breaker.break_cipher(ciphertext, size)
            except Exception as e:
                print(f"Error in advanced techniques: {e}")
                results_list = []
        
        elapsed_time = time.time() - start_time
        
        # Generate report
        if results_list:
            if size == 2:
                report = basic_breaker.generate_report(results_list, ciphertext, size)
            else:
                report = optimized_breaker.generate_report(results_list, ciphertext, size)
            
            print(report)
            print(f"Execution time: {elapsed_time:.2f} seconds")
            
            # Save report
            if save_reports:
                report_dir = f"relatorios/hibrido/desconhecidos/hill_{size}x{size}"
                os.makedirs(report_dir, exist_ok=True)
                report_path = os.path.join(report_dir, "relatorio.txt")
                with open(report_path, 'w') as f:
                    f.write(report)
                    f.write(f"\n\nExecution time: {elapsed_time:.2f} seconds")
                print(f"Report saved to {report_path}")
            
            # Store results
            results[f"unknown_{size}x{size}"] = {
                "best_matrix": results_list[0][0].tolist() if results_list else None,
                "best_text": results_list[0][1][:100] if results_list else None,
                "best_score": results_list[0][2] if results_list else None,
                "time": elapsed_time
            }
        else:
            print("No results found.")
            results[f"unknown_{size}x{size}"] = {"error": "No results found"}
    
    # Save results summary
    if save_reports:
        summary_path = "relatorios/hibrido/resumo.txt"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write("=== RESULTS SUMMARY ===\n\n")
            
            f.write("Known Texts:\n")
            for size in sizes:
                key = f"known_{size}x{size}"
                if key in results:
                    if "error" in results[key]:
                        f.write(f"- Matrix {size}x{size}: {results[key]['error']}\n")
                    else:
                        f.write(f"- Matrix {size}x{size}: Score {results[key]['best_score']:.4f}, Time {results[key]['time']:.2f}s\n")
            
            f.write("\nUnknown Texts:\n")
            for size in sizes:
                key = f"unknown_{size}x{size}"
                if key in results:
                    if "error" in results[key]:
                        f.write(f"- Matrix {size}x{size}: {results[key]['error']}\n")
                    else:
                        f.write(f"- Matrix {size}x{size}: Score {results[key]['best_score']:.4f}, Time {results[key]['time']:.2f}s\n")
        
        print(f"\nSummary saved to {summary_path}")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Hill Cipher Hybrid Breaker")
    parser.add_argument("--known-dir", default="textos_conhecidos", help="Directory with known texts")
    parser.add_argument("--unknown-dir", default="textos_desconhecidos", help="Directory with unknown texts")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5], help="Matrix sizes to process")
    parser.add_argument("--no-save", action="store_true", help="Don't save reports to files")
    
    args = parser.parse_args()
    
    # Process all ciphers
    process_all_ciphers(
        known_dir=args.known_dir,
        unknown_dir=args.unknown_dir,
        sizes=args.sizes,
        save_reports=not args.no_save
    )

if __name__ == "__main__":
    main()
