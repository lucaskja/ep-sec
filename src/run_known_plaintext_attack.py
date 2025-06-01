#!/usr/bin/env python3
"""
Script to run the known plaintext attack on all available ciphertexts.
"""

import os
import sys
import time
import argparse
import logging
from typing import List

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import our modules
from src.known_plaintext_attack import find_potential_keys, score_decryption, preprocess_text
from src.hill_cipher import decrypt_hill
from data.common_substrings import KNOWN_PLAINTEXT_SEGMENTS

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Known Plaintext Attack on Hill Cipher")
    parser.add_argument("--known-dir", default="textos_conhecidos", help="Directory with known texts")
    parser.add_argument("--unknown-dir", default="textos_desconhecidos", help="Directory with unknown texts")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5], help="Matrix sizes to process")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top results to show")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        filename="known_plaintext_attack.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Process known texts
    process_texts(args.known_dir, args.sizes, args.top_n, known=True)
    
    # Process unknown texts
    process_texts(args.unknown_dir, args.sizes, args.top_n, known=False)

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
        print(f"Ciphertext length: {len(ciphertext)} characters")
        print(f"Using {len(KNOWN_PLAINTEXT_SEGMENTS)} known plaintext segments")
        
        # Try each known plaintext segment
        all_results = []
        
        start_time = time.time()
        
        for i, plaintext in enumerate(KNOWN_PLAINTEXT_SEGMENTS):
            if len(plaintext) >= size * size:
                print(f"Trying plaintext segment {i+1}/{len(KNOWN_PLAINTEXT_SEGMENTS)}: '{plaintext[:20]}...'")
                
                segment_start_time = time.time()
                results = find_potential_keys(ciphertext, plaintext, size)
                segment_elapsed_time = time.time() - segment_start_time
                
                print(f"Found {len(results)} potential keys in {segment_elapsed_time:.2f} seconds")
                
                all_results.extend(results)
        
        # Remove duplicates and sort by score
        unique_results = []
        seen_matrices = set()
        
        for matrix, score in all_results:
            matrix_tuple = tuple(matrix.flatten())
            if matrix_tuple not in seen_matrices:
                seen_matrices.add(matrix_tuple)
                unique_results.append((matrix, score))
        
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        elapsed_time = time.time() - start_time
        
        # Generate report
        if unique_results:
            report = f"=== Known Plaintext Attack Report ===\n"
            report += f"Matrix size: {size}x{size}\n"
            report += f"Text type: {text_type.capitalize()}\n"
            report += f"Execution time: {elapsed_time:.2f} seconds\n\n"
            
            # Add top results
            report += f"Top {min(top_n, len(unique_results))} results:\n"
            for i, (matrix, score) in enumerate(unique_results[:top_n]):
                report += f"\n--- Result {i+1} (Score: {score:.2f}) ---\n"
                report += f"Matrix:\n{matrix}\n"
                
                # Decrypt and show sample
                decrypted = decrypt_hill(preprocess_text(ciphertext), matrix)
                report += f"Decrypted text (first 200 chars): {decrypted[:200]}...\n"
                
                # Check for common words
                found_words = []
                for word in ["QUE", "PARA", "COM", "UMA", "ELA", "ERA", "MINHA", "MAS", "POR", "MAIS", 
                             "SUA", "QUANDO", "PORQUE", "TINHA", "ESTAVA", "ELE", "DISSE", "COMO", "FOI"]:
                    if word in decrypted:
                        found_words.append(word)
                
                report += f"Common words found: {', '.join(found_words[:10])}\n"
            
            print(report)
            
            # Save report
            report_dir = f"relatorios/known_plaintext/{text_type}/hill_{size}x{size}"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "relatorio.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {report_path}")
            
            # Save decrypted text for the best result
            if unique_results:
                best_matrix, best_score = unique_results[0]
                decrypted = decrypt_hill(preprocess_text(ciphertext), best_matrix)
                decrypted_path = os.path.join(report_dir, "decrypted.txt")
                with open(decrypted_path, 'w') as f:
                    f.write(decrypted)
                print(f"Best decryption saved to {decrypted_path}")
        else:
            print("No results found.")

if __name__ == "__main__":
    main()
