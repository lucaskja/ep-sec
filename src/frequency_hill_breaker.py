#!/usr/bin/env python3
"""
Frequency-based Hill cipher breaker for Portuguese text.
This implementation follows the approach described in frequency_analyzer.md.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import re
from itertools import permutations

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

def load_ngram_frequencies(n: int) -> Dict[str, float]:
    """
    Load n-gram frequencies from JSON file.
    
    Args:
        n: Size of n-grams (1 for letters, 2 for bigrams, etc.)
        
    Returns:
        Dictionary mapping n-grams to their frequencies
    """
    # Handle special case for n=1 (letters)
    if n == 1:
        file_path = "data/letter_frequencies.json"
    else:
        file_path = f"data/{n}gram_frequencies.json"
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading n-gram frequencies from {file_path}: {e}")
        # Fallback to hardcoded frequencies if file not found
        if n == 1:
            return {
                "A": 0.14986, "E": 0.13262, "O": 0.10220, "S": 0.07421, "R": 0.06145,
                "I": 0.05978, "N": 0.05240, "D": 0.04377, "M": 0.04986, "U": 0.04954,
                "T": 0.04001, "C": 0.03962, "L": 0.02617, "P": 0.02694, "V": 0.02352,
                "G": 0.01086, "Q": 0.01630, "H": 0.01234, "F": 0.01005, "B": 0.00933,
                "Z": 0.00452, "J": 0.00277, "X": 0.00156, "K": 0.00022, "Y": 0.00002,
                "W": 0.00004
            }
        return {}

def load_top_ngrams() -> Dict[str, List[List]]:
    """
    Load top n-grams from JSON file.
    
    Returns:
        Dictionary mapping n-gram sizes to lists of [n-gram, frequency] pairs
    """
    file_path = "data/top_ngrams.json"
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading top n-grams: {e}")
        return {}

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

def count_ngrams(text: str, n: int) -> List[Tuple[str, int]]:
    """
    Count n-grams in the text.
    
    Args:
        text: Input text
        n: Size of n-grams
        
    Returns:
        List of (n-gram, count) tuples, sorted by count in descending order
    """
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    counts = Counter(ngrams)
    return counts.most_common()

def get_top_plaintext_ngrams(n: int, k: int = 10) -> List[str]:
    """
    Get the top k plaintext n-grams based on Portuguese language statistics.
    
    Args:
        n: Size of n-grams
        k: Number of top n-grams to return
        
    Returns:
        List of top k plaintext n-grams
    """
    top_ngrams = load_top_ngrams()
    if str(n) in top_ngrams:
        return [item[0] for item in top_ngrams[str(n)][:k]]
    
    # Fallback to hardcoded common n-grams if file not found
    if n == 1:
        return ['A', 'E', 'O', 'S', 'R', 'I', 'N', 'D', 'M', 'U']
    elif n == 2:
        return ['DE', 'OS', 'ES', 'RA', 'EN', 'SE', 'ER', 'AN', 'AS', 'OC']
    elif n == 3:
        return ['QUE', 'ENT', 'COM', 'ROS', 'IST', 'ADO', 'ELA', 'PRA', 'INH', 'EST']
    elif n == 4:
        return ['DESE', 'OSSE', 'ROTA', 'ADOU', 'MENT', 'ENTE', 'PARA', 'ANDO', 'OQUE', 'ESTA']
    elif n == 5:
        return ['PORTE', 'LIGAR', 'QUESE', 'ENTRE', 'CONSE', 'MENTE', 'ESTES', 'ESTES', 'ESTES', 'ESTES']
    else:
        return []

def form_matrices(plaintext_ngrams: List[str], ciphertext_ngrams: List[str], n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Form plaintext and ciphertext matrices from n-grams.
    
    Args:
        plaintext_ngrams: List of plaintext n-grams
        ciphertext_ngrams: List of ciphertext n-grams
        n: Size of n-grams
        
    Returns:
        List of (P_stack, C_stack) tuples
    """
    matrices = []
    
    # Convert n-grams to numeric vectors
    plaintext_vectors = [text_to_numbers(ngram) for ngram in plaintext_ngrams]
    ciphertext_vectors = [text_to_numbers(ngram) for ngram in ciphertext_ngrams]
    
    # We need at least n distinct n-grams to form an n×n matrix
    if len(plaintext_vectors) < n or len(ciphertext_vectors) < n:
        return matrices
    
    # Try different permutations of plaintext n-grams
    for p_perm in permutations(plaintext_vectors, n):
        # Form P_stack matrix
        P_stack = np.array(p_perm).T
        
        # Check if P_stack is invertible mod 26
        if not is_invertible_matrix(P_stack):
            continue
        
        # Try different permutations of ciphertext n-grams
        for c_perm in permutations(ciphertext_vectors, n):
            # Form C_stack matrix
            C_stack = np.array(c_perm).T
            
            matrices.append((P_stack, C_stack))
    
    return matrices

def recover_key(P_stack: np.ndarray, C_stack: np.ndarray) -> Optional[np.ndarray]:
    """
    Recover the key matrix from plaintext and ciphertext matrices.
    
    Args:
        P_stack: Plaintext matrix
        C_stack: Ciphertext matrix
        
    Returns:
        Key matrix if invertible, None otherwise
    """
    try:
        # Compute P_stack^(-1) mod 26
        P_inv = matrix_mod_inverse(P_stack)
        
        # Compute K = C_stack * P_stack^(-1) mod 26
        K = (C_stack @ P_inv) % ALPHABET_SIZE
        
        # Check if K is invertible mod 26
        if is_invertible_matrix(K):
            return K
    except Exception as e:
        pass
    
    return None

def score_decryption(decrypted_text: str, normalized_text: str) -> float:
    """
    Score the decryption based on n-gram frequencies and substring matching.
    
    Args:
        decrypted_text: Decrypted text to score
        normalized_text: Normalized text for substring matching
        
    Returns:
        Score (higher is better)
    """
    score = 0
    
    # Check for common Portuguese words
    common_words = ["QUE", "PARA", "COM", "UMA", "ELA", "ERA", "MINHA", "MAS", "POR", "MAIS",
                   "SUA", "QUANDO", "PORQUE", "TINHA", "ESTAVA", "ELE", "DISSE", "COMO", "FOI"]
    
    for word in common_words:
        count = decrypted_text.count(word)
        if count > 0:
            score += count * len(word)
    
    # Check for substrings in normalized text
    for length in range(10, min(50, len(decrypted_text))):
        for i in range(len(decrypted_text) - length + 1):
            substring = decrypted_text[i:i+length]
            if substring in normalized_text:
                score += length * 2
                # If we find a long substring, that's a very good sign
                if length >= 20:
                    score += 100
                break
    
    # Check letter frequencies
    letter_freqs = load_ngram_frequencies(1)
    if letter_freqs:
        letter_counts = Counter(decrypted_text)
        total_letters = len(decrypted_text)
        
        for letter, count in letter_counts.items():
            observed_freq = count / total_letters
            expected_freq = letter_freqs.get(letter, 0)
            
            # Score based on how close the observed frequency is to the expected frequency
            similarity = 1 - min(abs(observed_freq - expected_freq) / max(expected_freq, 0.001), 1)
            score += similarity * 10
    
    # Check vowel ratio (Portuguese has ~46% vowels)
    vowels = sum(1 for c in decrypted_text if c in 'AEIOU')
    vowel_ratio = vowels / len(decrypted_text) if decrypted_text else 0
    if 0.4 <= vowel_ratio <= 0.5:
        score += 50
    elif 0.35 <= vowel_ratio <= 0.55:
        score += 25
    
    return score

def break_hill_cipher(ciphertext: str, matrix_size: int, normalized_text: str) -> Optional[np.ndarray]:
    """
    Break Hill cipher using frequency analysis.
    
    Args:
        ciphertext: Encrypted text
        matrix_size: Size of the Hill cipher matrix
        normalized_text: Normalized text for validation
        
    Returns:
        Key matrix if found, None otherwise
    """
    # Preprocess ciphertext
    clean_ciphertext = preprocess_text(ciphertext)
    
    # Count n-grams in ciphertext
    ciphertext_ngrams = [ngram for ngram, _ in count_ngrams(clean_ciphertext, matrix_size)]
    
    # Get top plaintext n-grams
    plaintext_ngrams = get_top_plaintext_ngrams(matrix_size, k=10)
    
    print(f"Top {len(ciphertext_ngrams)} ciphertext {matrix_size}-grams: {ciphertext_ngrams[:10]}")
    print(f"Using Portuguese {matrix_size}-grams: {plaintext_ngrams}")
    
    # Form matrices
    matrices = form_matrices(plaintext_ngrams, ciphertext_ngrams, matrix_size)
    
    # Try each matrix pair
    best_key = None
    best_score = 0
    
    for P_stack, C_stack in matrices:
        # Recover key
        key = recover_key(P_stack, C_stack)
        if key is None:
            continue
        
        # Decrypt ciphertext with the key
        decrypted = decrypt_hill(clean_ciphertext, key)
        
        # Score decryption
        score = score_decryption(decrypted, normalized_text)
        
        if score > best_score:
            best_score = score
            best_key = key
            print(f"Found potential key with score {score:.2f}:")
            print(f"Key matrix:\n{key}")
            print(f"Decryption sample: {decrypted[:50]}...")
    
    print(f"Found {len([m for m in matrices if recover_key(*m) is not None])} potential keys for {matrix_size}x{matrix_size} matrix")
    
    return best_key

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Break Hill cipher using frequency analysis")
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
    
    print(f"Analyzing ciphertext with matrix size {args.size}x{args.size}...")
    print(f"Breaking {args.size}x{args.size} Hill cipher...")
    
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
            f.write(f"# Best matrix (found by frequency analysis)\n")
            f.write(str(key_matrix))
        
        # Save decrypted text
        decrypted = decrypt_hill(preprocess_text(ciphertext), key_matrix)
        decrypted_path = f"results/decrypted_{args.size}x{args.size}.txt"
        with open(decrypted_path, 'w') as f:
            f.write(decrypted)
        
        print(f"Matrix saved to {matrix_path}")
        print(f"Decrypted text saved to {decrypted_path}")
        print(f"Decrypted text (first 100 chars): {decrypted[:100]}...")
    else:
        print("No key matrix found.")

if __name__ == "__main__":
    main()
