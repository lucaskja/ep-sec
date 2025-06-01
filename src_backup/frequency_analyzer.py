#!/usr/bin/env python3
"""
Frequency Analyzer for Hill Cipher - Uses frequency analysis to break Hill ciphers
with matrix sizes 3x3, 4x4, and 5x5 for Portuguese text.

Based on the frequency_analyzer.md document.
"""

import os
import re
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
import argparse
import time
from collections import Counter

# Import utility functions
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
    # If that fails, try relative import
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

class FrequencyAnalyzer:
    """
    Frequency analyzer for Hill ciphers using Portuguese language statistics.
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
        # Common Portuguese n-grams by matrix size
        self.common_ngrams = {
            3: ["QUE", "ENT", "COM", "NTE", "EST", "AVA", "ARA", "ADO", "PAR", "NDO", 
                "NAO", "ERA", "AND", "UMA", "STA", "RES", "MEN", "CON", "DOS", "ANT"],
            4: ["DESE", "PORT", "PALA", "MENT", "ENTE", "ANDO", "PARA", "AVEL", "ESTA", "ACAO",
                "IDAD", "NTOS", "ENTE", "ANDO", "AVEL", "MENT", "IDAD", "NTOS", "ENTE", "ANDO"],
            5: ["BRASI", "PALAV", "MENTE", "DESEN", "PORTA", "AMENT", "ENCIA", "TAMEN", "ENCIA", "TAMEN",
                "ENCIA", "TAMEN", "ENCIA", "TAMEN", "ENCIA", "TAMEN", "ENCIA", "TAMEN", "ENCIA", "TAMEN"]
        }
        
        # Most common letters in Portuguese (in order of frequency)
        self.common_letters = "EAOSRINDMUTCLPVGHQBFZJXKWY"
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing non-alphabetic characters and converting to uppercase.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove non-alphabetic characters and convert to uppercase
        return re.sub(r'[^A-Z]', '', text.upper())
    
    def split_into_blocks(self, text: str, block_size: int) -> List[str]:
        """
        Split text into blocks of specified size.
        
        Args:
            text: Input text
            block_size: Size of each block
            
        Returns:
            List of blocks
        """
        return [text[i:i+block_size] for i in range(0, len(text) - block_size + 1)]
    
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
    
    def get_matrix_from_block(self, block: str) -> np.ndarray:
        """
        Convert a text block to a column vector.
        
        Args:
            block: Text block
            
        Returns:
            Column vector as numpy array
        """
        return np.array([ord(c) - ord('A') for c in block]).reshape(-1, 1)
    
    def get_matrix_from_blocks(self, blocks: List[str]) -> np.ndarray:
        """
        Convert multiple text blocks to a matrix (each block as a column).
        
        Args:
            blocks: List of text blocks
            
        Returns:
            Matrix with blocks as columns
        """
        if len(blocks) != self.matrix_size:
            raise ValueError(f"Need exactly {self.matrix_size} blocks for a {self.matrix_size}x{self.matrix_size} matrix")
        
        columns = [self.get_matrix_from_block(block) for block in blocks]
        return np.hstack(columns)
    
    def compute_key_matrix(self, plaintext_blocks: List[str], ciphertext_blocks: List[str]) -> Optional[np.ndarray]:
        """
        Compute the key matrix from plaintext and ciphertext blocks.
        
        Args:
            plaintext_blocks: List of plaintext blocks
            ciphertext_blocks: List of corresponding ciphertext blocks
            
        Returns:
            Key matrix if invertible, None otherwise
        """
        P = self.get_matrix_from_blocks(plaintext_blocks)
        C = self.get_matrix_from_blocks(ciphertext_blocks)
        
        # Check if P is invertible mod 26
        if not is_invertible_matrix(P):
            return None
        
        # Compute P^-1 mod 26
        P_inv = matrix_mod_inverse(P)
        if P_inv is None:
            return None
        
        # Compute K = C * P^-1 mod 26
        K = (C @ P_inv) % 26
        return K
    
    def score_decryption(self, decrypted_text: str) -> float:
        """
        Score decrypted text based on Portuguese language statistics.
        
        Args:
            decrypted_text: Decrypted text to score
            
        Returns:
            Score (higher is better)
        """
        return score_portuguese_text(decrypted_text)
    
    def analyze_ciphertext(self, ciphertext: str, top_n: int = 10) -> List[Tuple[np.ndarray, str, float]]:
        """
        Analyze ciphertext and return potential key matrices.
        
        Args:
            ciphertext: Encrypted text
            top_n: Number of top results to return
            
        Returns:
            List of (key_matrix, decrypted_text, score) tuples
        """
        print(f"Analyzing ciphertext with matrix size {self.matrix_size}x{self.matrix_size}...")
        
        # Preprocess ciphertext
        clean_ciphertext = self.preprocess_text(ciphertext)
        
        # Split into blocks and count frequencies
        blocks = self.split_into_blocks(clean_ciphertext, self.matrix_size)
        block_frequencies = self.count_block_frequencies(blocks)
        
        print(f"Found {len(block_frequencies)} unique blocks")
        print(f"Top 10 most frequent blocks: {[block for block, _ in block_frequencies[:10]]}")
        
        # Get the most frequent blocks
        most_frequent_blocks = [block for block, _ in block_frequencies[:20]]
        
        # Try different combinations of plaintext guesses
        results = []
        
        # Try common n-grams as plaintext guesses
        common_ngrams = self.common_ngrams.get(self.matrix_size, [])
        
        # Try combinations of common n-grams
        import itertools
        for plaintext_blocks in itertools.permutations(common_ngrams, self.matrix_size):
            for cipher_blocks in itertools.permutations(most_frequent_blocks[:5], self.matrix_size):
                try:
                    key_matrix = self.compute_key_matrix(plaintext_blocks, cipher_blocks)
                    if key_matrix is not None:
                        # Try decrypting with this key
                        decrypted = decrypt_hill(clean_ciphertext, key_matrix)
                        score = self.score_decryption(decrypted)
                        results.append((key_matrix, decrypted, score))
                except Exception as e:
                    continue
        
        # Sort results by score
        results.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(results)} potential key matrices")
        
        return results[:top_n]

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Frequency Analyzer for Hill Cipher")
    parser.add_argument("--size", type=int, choices=[3, 4, 5], required=True, help="Matrix size (3, 4, or 5)")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext to analyze")
    parser.add_argument("--file", type=str, help="File containing ciphertext")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to show")
    
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
    analyzer = FrequencyAnalyzer(args.size)
    start_time = time.time()
    results = analyzer.analyze_ciphertext(ciphertext, args.top)
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
