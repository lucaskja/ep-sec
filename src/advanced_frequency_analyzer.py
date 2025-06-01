#!/usr/bin/env python3
"""
Advanced Frequency Analyzer for Hill Cipher - Uses frequency analysis to break Hill ciphers
with matrix sizes 3x3, 4x4, and 5x5 for Portuguese text.

Based on the updated frequency_analyzer.md document.
"""

import os
import re
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
import argparse
import time
from collections import Counter
import itertools
from unidecode import unidecode
import sympy as sp

# Import Portuguese n-grams
try:
    from src.portuguese_ngrams import COMMON_NGRAMS, LETTER_FREQUENCIES, DIGRAMS, TRIGRAMS, QUADGRAMS, PENTAGRAMS
except ImportError:
    try:
        from portuguese_ngrams import COMMON_NGRAMS, LETTER_FREQUENCIES, DIGRAMS, TRIGRAMS, QUADGRAMS, PENTAGRAMS
    except ImportError:
        print("Warning: portuguese_ngrams.py not found. Using default n-grams.")


# Import utility functions if available
try:
    from src.utils import (
        text_to_numbers, numbers_to_text, is_invertible_matrix,
        mod_inverse, matrix_mod_inverse, ALPHABET_SIZE
    )
    from src.hill_cipher import decrypt_hill
    from src.portuguese_statistics import (
        LETTER_FREQUENCIES, DIGRAM_FREQUENCIES, TRIGRAM_FREQUENCIES,
        ONE_LETTER_WORDS, TWO_LETTER_WORDS, THREE_LETTER_WORDS,
        score_portuguese_text
    )
except ImportError:
    # If that fails, try relative import or define necessary functions
    try:
        from utils import (
            text_to_numbers, numbers_to_text, is_invertible_matrix,
            mod_inverse, matrix_mod_inverse, ALPHABET_SIZE
        )
        from hill_cipher import decrypt_hill
        from portuguese_statistics import (
            LETTER_FREQUENCIES, DIGRAM_FREQUENCIES, TRIGRAM_FREQUENCIES,
            ONE_LETTER_WORDS, TWO_LETTER_WORDS, THREE_LETTER_WORDS,
            score_portuguese_text
        )
    except ImportError:
        # Define necessary functions here if imports fail
        def text_to_numbers(text: str) -> List[int]:
            """Convert text to numbers (A=0, B=1, ..., Z=25)."""
            return [ord(c) - ord('A') for c in text]
        
        def numbers_to_text(numbers: List[int]) -> str:
            """Convert numbers to text (0=A, 1=B, ..., 25=Z)."""
            return ''.join(chr((n % 26) + ord('A')) for n in numbers)
        
        def is_invertible_matrix(matrix: np.ndarray, mod: int = 26) -> bool:
            """Check if matrix is invertible mod 26."""
            det = int(round(np.linalg.det(matrix))) % mod
            return math.gcd(det, mod) == 1
        
        def mod_inverse(a: int, m: int) -> int:
            """Compute modular inverse of a mod m."""
            g, x, y = extended_gcd(a, m)
            if g != 1:
                raise ValueError(f"{a} has no modular inverse mod {m}")
            else:
                return x % m
        
        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            """Extended Euclidean Algorithm."""
            if a == 0:
                return b, 0, 1
            else:
                g, x, y = extended_gcd(b % a, a)
                return g, y - (b // a) * x, x
        
        def matrix_mod_inverse(matrix: np.ndarray, mod: int = 26) -> np.ndarray:
            """Compute modular inverse of matrix mod 26."""
            # Convert to sympy Matrix for exact arithmetic
            sym_matrix = sp.Matrix(matrix.tolist())
            det = int(sym_matrix.det() % mod)
            if math.gcd(det, mod) != 1:
                raise ValueError(f"Matrix determinant {det} is not invertible mod {mod}")
            
            # Compute modular inverse of determinant
            det_inv = mod_inverse(det, mod)
            
            # Compute adjugate matrix
            adj = sym_matrix.adjugate()
            
            # Compute inverse: inv = det_inv * adj mod mod
            inv = (det_inv * adj) % mod
            
            # Convert back to numpy array
            return np.array(inv.tolist(), dtype=int)
        
        ALPHABET_SIZE = 26

# Define Portuguese language statistics if not imported
if 'LETTER_FREQUENCIES' not in globals():
    LETTER_FREQUENCIES = {
        'A': 14.63, 'B': 1.04, 'C': 3.88, 'D': 4.99, 'E': 12.57, 'F': 1.02,
        'G': 1.30, 'H': 1.28, 'I': 6.18, 'J': 0.40, 'K': 0.02, 'L': 2.78,
        'M': 4.74, 'N': 5.05, 'O': 10.73, 'P': 2.52, 'Q': 1.20, 'R': 6.53,
        'S': 7.81, 'T': 4.34, 'U': 4.63, 'V': 1.67, 'W': 0.01, 'X': 0.21,
        'Y': 0.01, 'Z': 0.47
    }

class AdvancedFrequencyAnalyzer:
    """
    Advanced frequency analyzer for Hill ciphers using Portuguese language statistics.
    """
    
    def __init__(self, matrix_size: int):
        """
        Initialize the frequency analyzer.
        
        Args:
            matrix_size: Size of the Hill cipher matrix (3, 4, or 5)
        """
        self.matrix_size = matrix_size
        self.setup_common_ngrams()
        
    def setup_common_ngrams(self):
        """Set up common n-grams for Portuguese language based on matrix size."""
        # Use extracted n-grams if available
        if 'COMMON_NGRAMS' in globals():
            self.common_ngrams = COMMON_NGRAMS
        else:
            # Fallback to default n-grams
            self.common_ngrams = {
                3: ["QUE", "ENT", "COM", "ROS", "IST", "ADO", 
                    "ELA", "PRA", "INH", "EST", "NTE", "ERA", "AND", "UMA", "STA", 
                    "RES", "MEN", "CON", "DOS", "ANT"],
                4: ["VOCE", "INHA", "PARA", "AQUE", "EVOC", "ANDO", "OQUE", "ESTA", 
                    "TAVA", "ENTE", "EQUE", "RQUE", "MINH", "OCES", "ENAO", "ENTA", 
                    "MENT", "QUEE", "STAV", "NHAM"],
                5: ["EVOCE", "MINHA", "VOCES", "STAVA", "INHAM", "ESTAV", "OVOCE", 
                    "ORQUE", "TINHA", "NHAMA", "PORQU", "HAMAE", "AQUEL", "UEVOC", 
                    "QUEVO", "UANDO", "QUAND", "AVOCE", "DISSE", "EPOIS"]
            }
        
        # Common Portuguese words for validation
        self.common_words = ["DE", "QUE", "E", "A", "O", "DA", "DO", "EM", "PARA", "COM",
                            "NAO", "UMA", "OS", "NO", "SE", "NA", "POR", "MAIS", "AS", "DOS"]
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing non-alphabetic characters, stripping accents,
        and converting to uppercase.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Strip accents
        no_accents = unidecode(text)
        # Remove non-alphabetic characters and convert to uppercase
        return re.sub(r'[^A-Za-z]', '', no_accents).upper()
    
    def split_into_blocks(self, text: str, block_size: int, overlapping: bool = False) -> List[str]:
        """
        Split text into blocks of specified size.
        
        Args:
            text: Input text
            block_size: Size of each block
            overlapping: Whether to use overlapping blocks (sliding window)
            
        Returns:
            List of blocks
        """
        if overlapping:
            return [text[i:i+block_size] for i in range(len(text) - block_size + 1)]
        else:
            return [text[i:i+block_size] for i in range(0, len(text), block_size)]
    
    def count_block_frequencies(self, blocks: List[str]) -> List[Tuple[str, int]]:
        """
        Count frequencies of blocks.
        
        Args:
            blocks: List of text blocks
            
        Returns:
            List of (block, frequency) tuples sorted by frequency
        """
        counter = Counter(blocks)
        return sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    def ngram_to_vector(self, ngram: str) -> np.ndarray:
        """
        Convert an n-letter string to an (n x 1) column vector of ints mod 26.
        
        Args:
            ngram: String of length n
            
        Returns:
            Column vector as numpy array
        """
        nums = text_to_numbers(ngram)
        return np.array(nums, dtype=int).reshape(-1, 1)
    
    def build_stack_matrix(self, ngrams: List[str]) -> np.ndarray:
        """
        Given a list of n distinct n-grams (all length n),
        return an (n x n) matrix whose columns are the numeric vectors of these n-grams.
        
        Args:
            ngrams: List of n-grams, each of length n
            
        Returns:
            Matrix with n-grams as columns
        """
        n = len(ngrams[0])
        if any(len(ng) != n for ng in ngrams):
            raise ValueError("All ngrams must have the same length n.")
        # Stack each column
        cols = [self.ngram_to_vector(ng) for ng in ngrams]
        return np.hstack(cols) % 26
    
    def matrix_mod_inv(self, M: np.ndarray, mod: int = 26) -> np.ndarray:
        """
        Compute the modular inverse of an n x n integer matrix M modulo `mod`.
        
        Args:
            M: Matrix to invert
            mod: Modulus (default: 26)
            
        Returns:
            Inverse matrix mod 26
            
        Raises:
            ValueError: If matrix is not invertible mod 26
        """
        # Convert to a sympy Matrix for exact arithmetic
        sym_M = sp.Matrix(M.tolist())
        det = int(sym_M.det() % mod)
        if math.gcd(det, mod) != 1:
            raise ValueError(f"Matrix determinant {det} is not invertible mod {mod}")
        
        # Compute modular inverse of determinant
        inv_det = mod_inverse(det, mod)
        
        # Compute adjugate: adj(M) = det * M^{-1} (over integers)
        adj = sym_M.adjugate()
        
        # M^{-1} = inv_det * adj(M) mod mod
        inv_M = (inv_det * adj) % mod
        
        # Convert back to numpy array of ints
        inv_np = np.array(inv_M.tolist(), dtype=int) % mod
        return inv_np
    
    def recover_key_matrix(self, P_stack: np.ndarray, C_stack: np.ndarray, mod: int = 26) -> np.ndarray:
        """
        Given P_stack (n x n) and C_stack (n x n), compute key K such that
        C_stack ≡ K * P_stack (mod mod).
        
        Args:
            P_stack: Matrix with plaintext n-grams as columns
            C_stack: Matrix with ciphertext n-grams as columns
            mod: Modulus (default: 26)
            
        Returns:
            Key matrix K
            
        Raises:
            ValueError: If P_stack is not invertible or K is not invertible
        """
        # 1. Compute inverse of P_stack mod 26
        P_inv = self.matrix_mod_inv(P_stack, mod)
        
        # 2. K = C_stack * P_inv mod 26
        K = (C_stack @ P_inv) % mod
        
        # Validate invertibility of K
        sym_K = sp.Matrix(K.tolist())
        det_K = int(sym_K.det() % mod)
        if math.gcd(det_K, mod) != 1:
            raise ValueError(f"Recovered K has determinant {det_K} not invertible mod {mod}.")
        
        return K
    
    def looks_like_portuguese(self, plaintext: str) -> bool:
        """
        Heuristic: check if decrypted snippet contains common Portuguese words or patterns.
        
        Args:
            plaintext: Text to check
            
        Returns:
            True if text likely contains Portuguese
        """
        # Check for common Portuguese words
        word_count = sum(1 for word in self.common_words if word in plaintext)
        
        # Check vowel ratio (Portuguese has ~46% vowels)
        vowels = sum(1 for c in plaintext if c in 'AEIOU')
        vowel_ratio = vowels / len(plaintext) if plaintext else 0
        vowel_check = 0.4 <= vowel_ratio <= 0.5
        
        # Check for common Portuguese patterns
        pattern_check = ('QUE' in plaintext or 'DE' in plaintext or 'DO' in plaintext or 
                         'DA' in plaintext or 'OS' in plaintext or 'AS' in plaintext)
        
        # Return True if at least 3 common words or good vowel ratio and pattern check
        return (word_count >= 3) or (vowel_check and pattern_check)
    
    def decrypt_with_key(self, ciphertext: str, key_matrix: np.ndarray) -> str:
        """
        Decrypt ciphertext using the given key matrix.
        
        Args:
            ciphertext: Text to decrypt
            key_matrix: Key matrix
            
        Returns:
            Decrypted text
        """
        # Compute inverse key
        K_inv = self.matrix_mod_inv(key_matrix)
        
        # Convert ciphertext to numbers and split into blocks
        cipher_nums = text_to_numbers(ciphertext)
        n = key_matrix.shape[0]
        
        # Pad if necessary
        if len(cipher_nums) % n != 0:
            cipher_nums += [23] * (n - (len(cipher_nums) % n))  # Pad with 'X'
        
        # Split into blocks
        cipher_blocks = [cipher_nums[i:i+n] for i in range(0, len(cipher_nums), n)]
        
        # Decrypt each block
        decrypted_blocks = []
        for block in cipher_blocks:
            c_vec = np.array(block, dtype=int).reshape(n, 1)
            p_vec = (K_inv @ c_vec) % 26
            decrypted_blocks.append(numbers_to_text(p_vec.flatten().tolist()))
        
        return ''.join(decrypted_blocks)
    
    def break_hill_3x3(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Attempt to break a 3x3 Hill cipher.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of (key_matrix, decrypted_text, score) tuples
        """
        print(f"Breaking 3x3 Hill cipher...")
        
        # Preprocess ciphertext
        clean_cipher = self.preprocess_text(ciphertext)
        
        # Count trigram frequencies
        tri_blocks = self.split_into_blocks(clean_cipher, 3)
        tri_counts = self.count_block_frequencies(tri_blocks)
        top_cipher_trigrams = [t for t, _ in tri_counts[:8]]
        
        print(f"Top 8 ciphertext trigrams: {top_cipher_trigrams}")
        
        # Candidate Portuguese trigrams
        top_plain_trigrams = self.common_ngrams[3][:8]
        
        print(f"Using Portuguese trigrams: {top_plain_trigrams}")
        
        # Generate candidate triplets
        results = []
        count = 0
        max_combinations = 1000  # Limit combinations to avoid excessive computation
        
        # Generate combinations of 3 distinct ciphertext trigrams and 3 distinct plaintext trigrams
        for c_triplet in itertools.permutations(top_cipher_trigrams, 3):
            if count >= max_combinations:
                break
                
            for p_triplet in itertools.permutations(top_plain_trigrams, 3):
                count += 1
                if count >= max_combinations:
                    break
                
                try:
                    # Build P_stack and C_stack
                    P_stack = self.build_stack_matrix(list(p_triplet))
                    C_stack = self.build_stack_matrix(list(c_triplet))
                    
                    # Recover key matrix
                    K_candidate = self.recover_key_matrix(P_stack, C_stack)
                    
                    # Decrypt a portion of the ciphertext
                    decrypted_snippet = self.decrypt_with_key(clean_cipher[:100], K_candidate)
                    
                    # Check if it looks like Portuguese
                    if self.looks_like_portuguese(decrypted_snippet):
                        # Decrypt the entire ciphertext
                        decrypted_full = self.decrypt_with_key(clean_cipher, K_candidate)
                        
                        # Score the decryption
                        score = self.score_decryption(decrypted_full)
                        
                        results.append((K_candidate, decrypted_full, score))
                        
                        print(f"Found potential key with score {score:.2f}:")
                        print(f"Key matrix:\n{K_candidate}")
                        print(f"Decryption sample: {decrypted_snippet[:50]}...")
                        
                        # If we have a very good match, we can stop early
                        if score > 15:
                            break
                except ValueError:
                    # Matrix not invertible or other error
                    continue
        
        # Sort results by score
        results.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(results)} potential keys for 3x3 matrix")
        
        return results[:10]  # Return top 10 results
    
    def break_hill_4x4(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Attempt to break a 4x4 Hill cipher.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of (key_matrix, decrypted_text, score) tuples
        """
        print(f"Breaking 4x4 Hill cipher...")
        
        # Preprocess ciphertext
        clean_cipher = self.preprocess_text(ciphertext)
        
        # Count 4-gram frequencies
        quad_blocks = self.split_into_blocks(clean_cipher, 4)
        quad_counts = self.count_block_frequencies(quad_blocks)
        top_cipher_quads = [q for q, _ in quad_counts[:6]]  # Limit to top 6 to reduce combinations
        
        print(f"Top 6 ciphertext 4-grams: {top_cipher_quads}")
        
        # Candidate Portuguese 4-grams
        top_plain_quads = self.common_ngrams[4][:6]  # Limit to top 6
        
        print(f"Using Portuguese 4-grams: {top_plain_quads}")
        
        # Generate candidate sets
        results = []
        count = 0
        max_combinations = 500  # Limit combinations to avoid excessive computation
        
        # Use combinations instead of permutations to reduce search space
        for c_quad_set in itertools.combinations(top_cipher_quads, 4):
            if count >= max_combinations:
                break
                
            for p_quad_set in itertools.combinations(top_plain_quads, 4):
                # For each combination, try a limited number of permutations
                for c_quad_perm in itertools.permutations(c_quad_set):
                    if count >= max_combinations:
                        break
                        
                    # Try direct mapping first
                    try:
                        count += 1
                        
                        # Build P_stack and C_stack
                        P_stack = self.build_stack_matrix(list(p_quad_set))
                        C_stack = self.build_stack_matrix(list(c_quad_perm))
                        
                        # Recover key matrix
                        K_candidate = self.recover_key_matrix(P_stack, C_stack)
                        
                        # Decrypt a portion of the ciphertext
                        decrypted_snippet = self.decrypt_with_key(clean_cipher[:100], K_candidate)
                        
                        # Check if it looks like Portuguese
                        if self.looks_like_portuguese(decrypted_snippet):
                            # Decrypt the entire ciphertext
                            decrypted_full = self.decrypt_with_key(clean_cipher, K_candidate)
                            
                            # Score the decryption
                            score = self.score_decryption(decrypted_full)
                            
                            results.append((K_candidate, decrypted_full, score))
                            
                            print(f"Found potential key with score {score:.2f}:")
                            print(f"Key matrix:\n{K_candidate}")
                            print(f"Decryption sample: {decrypted_snippet[:50]}...")
                            
                            # If we have a very good match, we can stop early
                            if score > 15:
                                break
                    except ValueError:
                        # Matrix not invertible or other error
                        continue
        
        # Sort results by score
        results.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(results)} potential keys for 4x4 matrix")
        
        return results[:10]  # Return top 10 results
    
    def break_hill_5x5(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Attempt to break a 5x5 Hill cipher.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of (key_matrix, decrypted_text, score) tuples
        """
        print(f"Breaking 5x5 Hill cipher...")
        
        # Preprocess ciphertext
        clean_cipher = self.preprocess_text(ciphertext)
        
        # Count 5-gram frequencies
        penta_blocks = self.split_into_blocks(clean_cipher, 5)
        penta_counts = self.count_block_frequencies(penta_blocks)
        top_cipher_pentas = [p for p, _ in penta_counts[:7]]
        
        print(f"Top 7 ciphertext 5-grams: {top_cipher_pentas}")
        
        # Candidate Portuguese 5-grams
        top_plain_pentas = self.common_ngrams[5]
        
        print(f"Using Portuguese 5-grams: {top_plain_pentas}")
        
        # Generate candidate sets
        results = []
        count = 0
        max_combinations = 200  # Limit combinations to avoid excessive computation
        
        # Reduce combinations: try combinations instead of permutations
        candidate_cipher_sets = list(itertools.combinations(top_cipher_pentas, 5))
        candidate_plain_sets = list(itertools.combinations(top_plain_pentas, 5))
        
        for c_penta_set in candidate_cipher_sets:
            if count >= max_combinations:
                break
                
            for p_penta_set in candidate_plain_sets:
                count += 1
                if count >= max_combinations:
                    break
                
                # Try direct alignment first
                try:
                    # Build P_stack and C_stack
                    P_stack = self.build_stack_matrix(list(p_penta_set))
                    C_stack = self.build_stack_matrix(list(c_penta_set))
                    
                    # Recover key matrix
                    K_candidate = self.recover_key_matrix(P_stack, C_stack)
                    
                    # Decrypt a portion of the ciphertext
                    decrypted_snippet = self.decrypt_with_key(clean_cipher[:100], K_candidate)
                    
                    # Check if it looks like Portuguese
                    if self.looks_like_portuguese(decrypted_snippet):
                        # Decrypt the entire ciphertext
                        decrypted_full = self.decrypt_with_key(clean_cipher, K_candidate)
                        
                        # Score the decryption
                        score = self.score_decryption(decrypted_full)
                        
                        results.append((K_candidate, decrypted_full, score))
                        
                        print(f"Found potential key with score {score:.2f}:")
                        print(f"Key matrix:\n{K_candidate}")
                        print(f"Decryption sample: {decrypted_snippet[:50]}...")
                        
                        # If we have a very good match, we can stop early
                        if score > 15:
                            break
                except ValueError:
                    # Matrix not invertible or other error
                    continue
        
        # Sort results by score
        results.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(results)} potential keys for 5x5 matrix")
        
        return results[:10]  # Return top 10 results
    
    def score_decryption(self, decrypted_text: str) -> float:
        """
        Score decrypted text based on Portuguese language statistics.
        
        Args:
            decrypted_text: Text to score
            
        Returns:
            Score (higher is better)
        """
        # Check for common Portuguese words
        word_count = sum(1 for word in self.common_words if word in decrypted_text)
        
        # Check vowel ratio
        vowels = sum(1 for c in decrypted_text if c in 'AEIOU')
        vowel_ratio = vowels / len(decrypted_text) if decrypted_text else 0
        vowel_score = 5 if 0.4 <= vowel_ratio <= 0.5 else 0
        
        # Check letter frequencies
        letter_counts = Counter(decrypted_text)
        letter_score = 0
        for letter, count in letter_counts.items():
            freq = count / len(decrypted_text)
            expected_freq = LETTER_FREQUENCIES.get(letter, 0) / 100 if 'LETTER_FREQUENCIES' in globals() else 0.001
            # Score based on how close the frequency is to expected
            similarity = 1 - min(abs(freq - expected_freq) / max(expected_freq, 0.001), 1)
            letter_score += similarity
        
        # Check for n-grams
        ngram_score = 0
        if len(decrypted_text) >= 3:
            for trigram in self.common_ngrams[3]:
                if trigram in decrypted_text:
                    ngram_score += 1
        
        # Combine scores
        total_score = word_count * 2 + vowel_score + letter_score + ngram_score
        
        return total_score
    
    def analyze_ciphertext(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Analyze ciphertext and return potential key matrices.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of (key_matrix, decrypted_text, score) tuples
        """
        print(f"Analyzing ciphertext with matrix size {self.matrix_size}x{self.matrix_size}...")
        
        # Choose the appropriate breaking method based on matrix size
        if self.matrix_size == 3:
            return self.break_hill_3x3(ciphertext)
        elif self.matrix_size == 4:
            return self.break_hill_4x4(ciphertext)
        elif self.matrix_size == 5:
            return self.break_hill_5x5(ciphertext)
        else:
            raise ValueError(f"Unsupported matrix size: {self.matrix_size}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Advanced Frequency Analyzer for Hill Cipher")
    parser.add_argument("--size", type=int, choices=[3, 4, 5], required=True, help="Matrix size (3, 4, or 5)")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext to analyze")
    parser.add_argument("--file", type=str, help="File containing ciphertext")
    
    args = parser.parse_args()
    
    # Get ciphertext from argument or file
    ciphertext = ""
    if args.ciphertext:
        ciphertext = args.ciphertext
    elif args.file:
        with open(args.file, 'r') as f:
            ciphertext = f.read()
    else:
        parser.error("Either --ciphertext or --file must be provided")
    
    # Create analyzer and analyze ciphertext
    analyzer = AdvancedFrequencyAnalyzer(args.size)
    start_time = time.time()
    results = analyzer.analyze_ciphertext(ciphertext)
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
    print(f"Top {len(results)} potential key matrices:")
    
    for i, (matrix, decrypted, score) in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {score:.2f}) ---")
        print(f"Matrix:\n{matrix}")
        print(f"Decrypted text (first 100 chars): {decrypted[:100]}...")

if __name__ == "__main__":
    main()
