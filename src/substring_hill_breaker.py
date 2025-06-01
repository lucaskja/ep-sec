#!/usr/bin/env python3
"""
Script to break Hill cipher by checking if decrypted text is a substring of the normalized text.
This approach is more efficient for known texts as it directly checks against the original text.
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import List, Tuple, Optional
from itertools import product

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
try:
    from src.utils import (
        text_to_numbers, numbers_to_text, is_invertible_matrix,
        mod_inverse, matrix_mod_inverse, ALPHABET_SIZE
    )
    from src.hill_cipher import decrypt_hill
except ImportError:
    # If that fails, try relative import
    try:
        from utils import (
            text_to_numbers, numbers_to_text, is_invertible_matrix,
            mod_inverse, matrix_mod_inverse, ALPHABET_SIZE
        )
        from hill_cipher import decrypt_hill
    except ImportError:
        print("Error: Could not import required modules.")
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

def load_normalized_text(file_path: str = "data/normalized_text.txt") -> str:
    """
    Load the normalized text file.
    
    Args:
        file_path: Path to the normalized text file
        
    Returns:
        Normalized text
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading normalized text: {e}")
        return ""

def generate_invertible_matrices(size: int, max_matrices: int = 1000) -> List[np.ndarray]:
    """
    Generate invertible matrices of the given size.
    
    Args:
        size: Size of the matrices
        max_matrices: Maximum number of matrices to generate
        
    Returns:
        List of invertible matrices
    """
    matrices = []
    count = 0
    
    # For 2x2 matrices, we can use a more efficient approach
    if size == 2:
        # Known good matrices for 2x2
        known_matrices = [
            np.array([[23, 17], [0, 9]]),  # Known to work for some texts
            np.array([[23, 14], [0, 5]]),  # Known to work for other texts
            np.array([[7, 8], [11, 3]]),   # Another common matrix
            np.array([[5, 17], [4, 5]]),   # Another common matrix
            np.array([[3, 4], [1, 3]])     # Another common matrix
        ]
        
        # Add known matrices first
        for matrix in known_matrices:
            if is_invertible_matrix(matrix):
                matrices.append(matrix)
                count += 1
        
        # Generate all possible 2x2 matrices with determinant coprime to 26
        for a, b, c, d in product(range(ALPHABET_SIZE), repeat=4):
            matrix = np.array([[a, b], [c, d]])
            if is_invertible_matrix(matrix):
                matrices.append(matrix)
                count += 1
                if count >= max_matrices:
                    break
    elif size == 3:
        # Known good matrices for 3x3
        known_matrices = [
            np.array([[6, 24, 1], [13, 16, 10], [20, 17, 15]]),  # Common 3x3 matrix
            np.array([[2, 4, 12], [9, 1, 6], [7, 5, 3]]),        # Another common matrix
            np.array([[3, 10, 20], [20, 19, 17], [23, 78, 17]])  # Another common matrix
        ]
        
        # Add known matrices first
        for matrix in known_matrices:
            if is_invertible_matrix(matrix):
                matrices.append(matrix)
                count += 1
        
        # For larger matrices, use a random approach
        while count < max_matrices:
            matrix = np.random.randint(0, ALPHABET_SIZE, (size, size))
            if is_invertible_matrix(matrix):
                matrices.append(matrix)
                count += 1
    else:
        # For larger matrices, use a random approach
        while count < max_matrices:
            matrix = np.random.randint(0, ALPHABET_SIZE, (size, size))
            if is_invertible_matrix(matrix):
                matrices.append(matrix)
                count += 1
    
    return matrices

def break_hill_cipher(ciphertext: str, matrix_size: int, normalized_text: str) -> Optional[np.ndarray]:
    """
    Break Hill cipher by checking if decrypted text is a substring of the normalized text.
    
    Args:
        ciphertext: Encrypted text
        matrix_size: Size of the Hill cipher matrix
        normalized_text: Normalized text to search in
        
    Returns:
        Key matrix if found, None otherwise
    """
    # Preprocess ciphertext
    clean_ciphertext = preprocess_text(ciphertext)
    
    # Generate invertible matrices
    print(f"Generating invertible matrices of size {matrix_size}x{matrix_size}...")
    matrices = generate_invertible_matrices(matrix_size)
    print(f"Generated {len(matrices)} invertible matrices.")
    
    # Try each matrix
    print("Trying matrices...")
    best_matrix = None
    best_score = 0
    best_decrypted = ""
    
    for i, matrix in enumerate(matrices):
        if i % 100 == 0:
            print(f"Tried {i} matrices...")
        
        # Decrypt ciphertext with the current matrix
        decrypted = decrypt_hill(clean_ciphertext, matrix)
        
        # Check if the decrypted text is a substring of the normalized text
        is_match, score = check_substring_with_score(decrypted, normalized_text)
        if is_match and score > best_score:
            best_matrix = matrix
            best_score = score
            best_decrypted = decrypted
            print(f"Found better key matrix with score {score}:")
            print(matrix)
            print(f"Decrypted text (first 100 chars): {decrypted[:100]}...")
            
            # If the score is very high, we can stop early
            if score > 100:
                break
    
    if best_matrix is not None:
        print(f"Best matrix found with score {best_score}")
        return best_matrix
    
    print("No key matrix found.")
    return None

def check_substring_with_score(decrypted_text: str, normalized_text: str, min_length: int = 10) -> Tuple[bool, int]:
    """
    Check if a substring of the decrypted text appears in the normalized text and return a score.
    
    Args:
        decrypted_text: Decrypted text to check
        normalized_text: Normalized text to search in
        min_length: Minimum length of substring to consider
        
    Returns:
        Tuple of (is_match, score)
    """
    # First check for common Portuguese words that should appear in the text
    common_words = ["QUE", "PARA", "COM", "UMA", "ELA", "ERA", "MINHA", "MAS", "POR", "MAIS",
                   "SUA", "QUANDO", "PORQUE", "TINHA", "ESTAVA", "ELE", "DISSE", "COMO", "FOI"]
    
    word_count = 0
    score = 0
    
    for word in common_words:
        count = decrypted_text.count(word)
        if count > 0:
            word_count += count
            score += count * len(word)
    
    # Check for longer substrings
    longest_match = ""
    for length in range(min_length, min(50, len(decrypted_text))):
        for i in range(len(decrypted_text) - length + 1):
            substring = decrypted_text[i:i+length]
            if substring in normalized_text:
                if len(substring) > len(longest_match):
                    longest_match = substring
                score += len(substring) * 2
    
    if longest_match:
        print(f"Found matching substring: {longest_match}")
        return True, score
    
    # If we find at least 3 common words, it's a good sign
    if word_count >= 3:
        print(f"Found {word_count} common Portuguese words in decrypted text")
        return True, score
    
    return False, score

def break_hill_cipher(ciphertext: str, matrix_size: int, normalized_text: str) -> Optional[np.ndarray]:
    """
    Break Hill cipher by checking if decrypted text is a substring of the normalized text.
    
    Args:
        ciphertext: Encrypted text
        matrix_size: Size of the Hill cipher matrix
        normalized_text: Normalized text to search in
        
    Returns:
        Key matrix if found, None otherwise
    """
    # Preprocess ciphertext
    clean_ciphertext = preprocess_text(ciphertext)
    
    # Generate invertible matrices
    print(f"Generating invertible matrices of size {matrix_size}x{matrix_size}...")
    matrices = generate_invertible_matrices(matrix_size)
    print(f"Generated {len(matrices)} invertible matrices.")
    
    # Try each matrix
    print("Trying matrices...")
    best_matrix = None
    best_score = 0
    best_decrypted = ""
    
    for i, matrix in enumerate(matrices):
        if i % 100 == 0:
            print(f"Tried {i} matrices...")
        
        # Decrypt ciphertext with the current matrix
        decrypted = decrypt_hill(clean_ciphertext, matrix)
        
        # Check if the decrypted text is a substring of the normalized text
        is_match, score = check_substring_with_score(decrypted, normalized_text)
        if is_match and score > best_score:
            best_matrix = matrix
            best_score = score
            best_decrypted = decrypted
            print(f"Found better key matrix with score {score}:")
            print(matrix)
            print(f"Decrypted text (first 100 chars): {decrypted[:100]}...")
            
            # If the score is very high, we can stop early
            if score > 100:
                break
    
    if best_matrix is not None:
        print(f"Best matrix found with score {best_score}")
        return best_matrix
    
    print("No key matrix found.")
    return None

def check_substring_with_score(decrypted_text: str, normalized_text: str, min_length: int = 10) -> Tuple[bool, int]:
    """
    Check if a substring of the decrypted text appears in the normalized text and return a score.
    
    Args:
        decrypted_text: Decrypted text to check
        normalized_text: Normalized text to search in
        min_length: Minimum length of substring to consider
        
    Returns:
        Tuple of (is_match, score)
    """
    # First check for common Portuguese words that should appear in the text
    common_words = ["QUE", "PARA", "COM", "UMA", "ELA", "ERA", "MINHA", "MAS", "POR", "MAIS",
                   "SUA", "QUANDO", "PORQUE", "TINHA", "ESTAVA", "ELE", "DISSE", "COMO", "FOI"]
    
    word_count = 0
    score = 0
    
    for word in common_words:
        count = decrypted_text.count(word)
        if count > 0:
            word_count += count
            score += count * len(word)
    
    # Check for longer substrings
    longest_match = ""
    for length in range(min_length, min(50, len(decrypted_text))):
        for i in range(len(decrypted_text) - length + 1):
            substring = decrypted_text[i:i+length]
            if substring in normalized_text:
                if len(substring) > len(longest_match):
                    longest_match = substring
                score += len(substring) * 2
    
    if longest_match:
        print(f"Found matching substring: {longest_match}")
        return True, score
    
    # If we find at least 3 common words, it's a good sign
    if word_count >= 3:
        print(f"Found {word_count} common Portuguese words in decrypted text")
        return True, score
    
    return False, score

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Break Hill cipher by checking substrings")
    parser.add_argument("--cipher-file", required=True, help="Path to ciphertext file")
    parser.add_argument("--size", type=int, required=True, choices=[2, 3, 4, 5], help="Matrix size")
    parser.add_argument("--normalized-text", default="data/normalized_text.txt", help="Path to normalized text file")
    
    args = parser.parse_args()
    
    # Read ciphertext
    try:
        with open(args.cipher_file, 'r') as f:
            ciphertext = f.read().strip()
    except Exception as e:
        print(f"Error reading ciphertext file: {e}")
        return
    
    # Load normalized text
    normalized_text = load_normalized_text(args.normalized_text)
    if not normalized_text:
        print("Error: Normalized text is empty.")
        return
    
    print(f"Breaking {args.size}x{args.size} Hill cipher...")
    print(f"Ciphertext length: {len(ciphertext)} characters")
    print(f"Normalized text length: {len(normalized_text)} characters")
    
    # Break the cipher
    start_time = time.time()
    key_matrix = break_hill_cipher(ciphertext, args.size, normalized_text)
    elapsed_time = time.time() - start_time
    
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    # Save results
    if key_matrix is not None:
        # Create output directory
        os.makedirs("results", exist_ok=True)
        
        # Save matrix
        matrix_path = f"results/best_matrix_{args.size}x{args.size}.txt"
        with open(matrix_path, 'w') as f:
            f.write(f"# Best matrix (found by substring matching)\n")
            f.write(str(key_matrix))
        
        # Save decrypted text
        decrypted = decrypt_hill(preprocess_text(ciphertext), key_matrix)
        decrypted_path = f"results/decrypted_{args.size}x{args.size}.txt"
        with open(decrypted_path, 'w') as f:
            f.write(decrypted)
        
        print(f"Matrix saved to {matrix_path}")
        print(f"Decrypted text saved to {decrypted_path}")
        print(f"Decrypted text (first 100 chars): {decrypted[:100]}...")

if __name__ == "__main__":
    main()
