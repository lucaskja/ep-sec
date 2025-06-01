#!/usr/bin/env python3
"""
Script to perform a known plaintext attack on Hill cipher using common substrings.
This script uses the common substrings found in avesso_da_pele.txt to try to break
Hill ciphers of various sizes.
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import List, Tuple, Optional

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
try:
    from src.utils import (
        text_to_numbers, numbers_to_text, is_invertible_matrix,
        mod_inverse, matrix_mod_inverse, ALPHABET_SIZE
    )
    from src.hill_cipher import decrypt_hill
    from data.common_substrings import COMMON_WORDS, KNOWN_PLAINTEXT_SEGMENTS
except ImportError:
    # If that fails, try relative import
    try:
        from utils import (
            text_to_numbers, numbers_to_text, is_invertible_matrix,
            mod_inverse, matrix_mod_inverse, ALPHABET_SIZE
        )
        from hill_cipher import decrypt_hill
        from data.common_substrings import COMMON_WORDS, KNOWN_PLAINTEXT_SEGMENTS
    except ImportError:
        print("Error: Could not import required modules.")
        print("Make sure you have run find_common_substrings.py first.")
        sys.exit(1)

def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing non-alphabetic characters and converting to uppercase.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Remove non-alphabetic characters and convert to uppercase
    return ''.join(c for c in text.upper() if 'A' <= c <= 'Z')

def find_potential_keys(ciphertext: str, plaintext: str, matrix_size: int) -> List[Tuple[np.ndarray, float]]:
    """
    Find potential keys using known plaintext.
    
    Args:
        ciphertext: Encrypted text
        plaintext: Known plaintext
        matrix_size: Size of the Hill cipher matrix
        
    Returns:
        List of (key_matrix, score) tuples
    """
    # Preprocess texts
    clean_ciphertext = preprocess_text(ciphertext)
    clean_plaintext = preprocess_text(plaintext)
    
    # Convert to numbers
    cipher_nums = text_to_numbers(clean_ciphertext)
    plain_nums = text_to_numbers(clean_plaintext)
    
    # Ensure plaintext is at least matrix_size characters long
    if len(plain_nums) < matrix_size:
        print(f"Warning: Plaintext '{plaintext}' is too short for {matrix_size}x{matrix_size} matrix.")
        return []
    
    # Try different starting positions in the ciphertext
    results = []
    max_start = len(cipher_nums) - matrix_size * matrix_size
    
    for start_pos in range(0, max_start, matrix_size):
        # Extract matrix_size blocks from ciphertext and plaintext
        cipher_blocks = []
        plain_blocks = []
        
        for i in range(matrix_size):
            block_start = start_pos + i * matrix_size
            if block_start + matrix_size <= len(cipher_nums) and i * matrix_size + matrix_size <= len(plain_nums):
                cipher_block = cipher_nums[block_start:block_start + matrix_size]
                plain_block = plain_nums[i * matrix_size:i * matrix_size + matrix_size]
                
                cipher_blocks.append(cipher_block)
                plain_blocks.append(plain_block)
        
        # If we don't have enough blocks, skip this position
        if len(cipher_blocks) < matrix_size or len(plain_blocks) < matrix_size:
            continue
        
        # Create matrices
        C = np.array(cipher_blocks).T
        P = np.array(plain_blocks).T
        
        # Check if P is invertible
        try:
            if not is_invertible_matrix(P):
                continue
                
            # Calculate P^-1
            P_inv = matrix_mod_inverse(P)
            
            # Calculate K = C * P^-1
            K = (C @ P_inv) % ALPHABET_SIZE
            
            # Check if K is invertible
            if not is_invertible_matrix(K):
                continue
                
            # Try to decrypt the entire ciphertext with this key
            decrypted = decrypt_hill(clean_ciphertext, K)
            
            # Score the decryption
            score = score_decryption(decrypted)
            
            # Add to results
            results.append((K, score))
            
        except Exception as e:
            # Skip this position if there's an error
            continue
    
    # Sort results by score
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def score_decryption(decrypted_text: str) -> float:
    """
    Score decrypted text based on presence of common words.
    
    Args:
        decrypted_text: Text to score
        
    Returns:
        Score (higher is better)
    """
    score = 0
    
    # Check for common words
    for word in COMMON_WORDS:
        if word in decrypted_text:
            score += 10  # Higher weight for common words
    
    # Check vowel ratio (Portuguese has ~46% vowels)
    vowels = sum(1 for c in decrypted_text if c in 'AEIOU')
    vowel_ratio = vowels / len(decrypted_text) if decrypted_text else 0
    if 0.4 <= vowel_ratio <= 0.5:
        score += 5
    
    # Check for common letter patterns
    common_patterns = ['QUE', 'DE', 'DO', 'DA', 'OS', 'AS', 'EM', 'COM', 'POR']
    for pattern in common_patterns:
        score += decrypted_text.count(pattern) * 2
    
    return score

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Known plaintext attack on Hill cipher")
    parser.add_argument("--cipher-file", required=True, help="Path to ciphertext file")
    parser.add_argument("--size", type=int, required=True, choices=[2, 3, 4, 5], help="Matrix size")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top results to show")
    
    args = parser.parse_args()
    
    # Read ciphertext
    try:
        with open(args.cipher_file, 'r') as f:
            ciphertext = f.read().strip()
    except Exception as e:
        print(f"Error reading ciphertext file: {e}")
        return
    
    print(f"Performing known plaintext attack on {args.size}x{args.size} Hill cipher...")
    print(f"Ciphertext length: {len(ciphertext)} characters")
    print(f"Using {len(KNOWN_PLAINTEXT_SEGMENTS)} known plaintext segments")
    
    # Try each known plaintext segment
    all_results = []
    
    for i, plaintext in enumerate(KNOWN_PLAINTEXT_SEGMENTS):
        if len(plaintext) >= args.size * args.size:
            print(f"\nTrying plaintext segment {i+1}/{len(KNOWN_PLAINTEXT_SEGMENTS)}: '{plaintext[:20]}...'")
            
            start_time = time.time()
            results = find_potential_keys(ciphertext, plaintext, args.size)
            elapsed_time = time.time() - start_time
            
            print(f"Found {len(results)} potential keys in {elapsed_time:.2f} seconds")
            
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
    
    # Show top results
    print(f"\nTop {min(args.top_n, len(unique_results))} results:")
    
    for i, (matrix, score) in enumerate(unique_results[:args.top_n]):
        print(f"\n--- Result {i+1} (Score: {score:.2f}) ---")
        print(f"Matrix:\n{matrix}")
        
        # Decrypt and show sample
        decrypted = decrypt_hill(preprocess_text(ciphertext), matrix)
        print(f"Decrypted text (first 100 chars): {decrypted[:100]}...")
        
        # Check for common words
        found_words = []
        for word in COMMON_WORDS[:20]:  # Check top 20 common words
            if word in decrypted:
                found_words.append(word)
        
        print(f"Common words found: {', '.join(found_words[:10])}")
    
    # Save best result to file
    if unique_results:
        best_matrix, best_score = unique_results[0]
        
        # Create output directory
        os.makedirs("results", exist_ok=True)
        
        # Save matrix
        matrix_path = f"results/best_matrix_{args.size}x{args.size}.txt"
        with open(matrix_path, 'w') as f:
            f.write(f"# Best matrix (Score: {best_score:.2f})\n")
            f.write(str(best_matrix))
        
        # Save decrypted text
        decrypted = decrypt_hill(preprocess_text(ciphertext), best_matrix)
        decrypted_path = f"results/decrypted_{args.size}x{args.size}.txt"
        with open(decrypted_path, 'w') as f:
            f.write(decrypted)
        
        print(f"\nBest matrix saved to {matrix_path}")
        print(f"Decrypted text saved to {decrypted_path}")

if __name__ == "__main__":
    main()
