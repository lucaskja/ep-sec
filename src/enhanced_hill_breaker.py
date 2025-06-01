#!/usr/bin/env python3
"""
Enhanced Hill Cipher Breaker - Improved version for breaking Hill cipher

This module implements advanced techniques for breaking Hill cipher,
with special focus on 3x3, 4x4, and 5x5 matrices using known plaintext
and Portuguese language knowledge.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import re
import math
import numpy as np
import multiprocessing as mp
import concurrent.futures
import pickle
import time
import logging
import itertools
import sys
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions with fallback
try:
    from src.utils import (
        text_to_numbers, numbers_to_text, is_invertible_matrix,
        mod_inverse, matrix_mod_inverse, ALPHABET_SIZE
    )
    from src.hill_cipher import decrypt_hill, known_plaintext_attack
    from src.portuguese_language_model import PortugueseLanguageModel
except ImportError:
    # If that fails, try relative import
    from utils import (
        text_to_numbers, numbers_to_text, is_invertible_matrix,
        mod_inverse, matrix_mod_inverse, ALPHABET_SIZE
    )
    from hill_cipher import decrypt_hill, known_plaintext_attack
    from portuguese_language_model import PortugueseLanguageModel

class EnhancedHillBreaker:
    """Enhanced Hill Cipher Breaker with advanced techniques."""
    
    def __init__(self, matrix_size: int, dict_path: str = None, num_threads: int = None):
        """
        Initialize the enhanced Hill cipher breaker.
        
        Args:
            matrix_size: Size of the matrix (2, 3, 4, or 5)
            dict_path: Path to Portuguese dictionary file (optional)
            num_threads: Number of threads for parallel processing (optional)
        """
        self.matrix_size = matrix_size
        self.language_model = PortugueseLanguageModel(dict_path)
        self.num_threads = num_threads or min(8, mp.cpu_count())
        
        # Set up logging
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"enhanced_hill_{matrix_size}x{matrix_size}.log")
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        
        logging.info(f"Initialized Enhanced Hill Breaker for {matrix_size}x{matrix_size} matrices")
    
    def break_cipher(self, ciphertext: str, known_text_path: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Break Hill cipher using enhanced techniques.
        
        Args:
            ciphertext: Encrypted text
            known_text_path: Path to known plaintext file (optional)
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        logging.info(f"Starting to break {self.matrix_size}x{self.matrix_size} Hill cipher")
        
        # Clean ciphertext
        ciphertext = re.sub(r'[^A-Z]', '', ciphertext.upper())
        
        # Try different approaches in order of effectiveness
        results = []
        
        # 1. Try known plaintext attack if text is available
        if known_text_path and os.path.exists(known_text_path):
            logging.info("Attempting known plaintext attack")
            kp_results = self.enhanced_known_plaintext_attack(ciphertext, known_text_path)
            results.extend(kp_results)
            
            # If we got good results, return them
            if results and results[0][2] > 0.5:
                logging.info("Known plaintext attack successful")
                return self.finalize_results(results)
        
        # 2. Try pattern-based attack
        logging.info("Attempting pattern-based attack")
        pattern_results = self.pattern_based_attack(ciphertext)
        results.extend(pattern_results)
        
        # 3. Try statistical attack
        logging.info("Attempting statistical attack")
        stat_results = self.statistical_attack(ciphertext)
        results.extend(stat_results)
        
        # 4. Try brute force with optimizations
        logging.info("Attempting optimized brute force")
        bf_results = self.optimized_brute_force(ciphertext)
        results.extend(bf_results)
        
        # Combine, score, and sort results
        return self.finalize_results(results)
    
    def enhanced_known_plaintext_attack(self, ciphertext: str, known_text_path: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Enhanced known plaintext attack using fragments of known text.
        
        Args:
            ciphertext: Encrypted text
            known_text_path: Path to known plaintext file
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        try:
            # Load known text
            with open(known_text_path, 'r', encoding='latin-1') as f:
                known_text = f.read().upper()
            
            # Clean text
            known_text = re.sub(r'[^A-Z]', '', known_text)
            
            # Calculate required fragment size
            fragment_size = self.matrix_size * self.matrix_size
            
            # Try with different offsets in the known text
            for offset in range(0, min(5000, len(known_text) - fragment_size), self.matrix_size):
                plaintext_fragment = known_text[offset:offset+fragment_size]
                cipher_fragment = ciphertext[:fragment_size]
                
                try:
                    # Try to find key matrix
                    key_matrix = known_plaintext_attack(plaintext_fragment, cipher_fragment, self.matrix_size)
                    
                    # Test if this key produces readable text
                    decrypted = decrypt_hill(ciphertext, key_matrix)
                    score = self.language_model.score_text(decrypted)
                    
                    # Add to results if score is positive
                    if score > 0:
                        results.append((key_matrix, decrypted, score))
                        
                        # Log progress
                        logging.info(f"Found candidate with score {score:.4f}")
                        
                        # If we found a very good match, return early
                        if score > 0.8:
                            return [(key_matrix, decrypted, score)]
                except Exception as e:
                    logging.debug(f"Error in known plaintext attack at offset {offset}: {e}")
                    continue
            
            # Try with sliding window approach
            if not results:
                logging.info("Trying sliding window approach for known plaintext attack")
                
                # Try different window sizes
                for window_size in range(fragment_size, fragment_size * 3, fragment_size):
                    for offset in range(0, min(1000, len(known_text) - window_size), fragment_size):
                        plaintext_window = known_text[offset:offset+window_size]
                        cipher_window = ciphertext[:window_size]
                        
                        # Try to match different alignments
                        for align_offset in range(0, window_size - fragment_size + 1, self.matrix_size):
                            try:
                                p_fragment = plaintext_window[align_offset:align_offset+fragment_size]
                                c_fragment = cipher_window[align_offset:align_offset+fragment_size]
                                
                                key_matrix = known_plaintext_attack(p_fragment, c_fragment, self.matrix_size)
                                decrypted = decrypt_hill(ciphertext, key_matrix)
                                score = self.language_model.score_text(decrypted)
                                
                                if score > 0:
                                    results.append((key_matrix, decrypted, score))
                            except Exception:
                                continue
        except Exception as e:
            logging.error(f"Error in enhanced known plaintext attack: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:10]  # Return top 10 candidates
    
    def pattern_based_attack(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Pattern-based attack using common Portuguese patterns.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        # Common Portuguese patterns to try
        patterns = [
            "QUEAOSEU", "COMOPARA", "ESTAMOSNOS", "PORQUENAO",
            "TEMOSUMADOS", "MUITOBEMFEITO", "AGORAEVERDADE",
            "ESTAMOSJUNTOS", "PARAOSNOSSOS", "COMUMACERTA"
        ]
        
        # Adjust patterns based on matrix size
        size = self.matrix_size
        adjusted_patterns = []
        
        for pattern in patterns:
            if len(pattern) >= size * size:
                adjusted_patterns.append(pattern[:size * size])
        
        # Try each pattern
        for pattern in adjusted_patterns:
            try:
                # Use the pattern as potential plaintext
                cipher_fragment = ciphertext[:len(pattern)]
                
                key_matrix = known_plaintext_attack(pattern, cipher_fragment, size)
                decrypted = decrypt_hill(ciphertext, key_matrix)
                score = self.language_model.score_text(decrypted)
                
                if score > 0:
                    results.append((key_matrix, decrypted, score))
            except Exception:
                continue
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:5]  # Return top 5 candidates
    
    def statistical_attack(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Statistical attack using Portuguese language properties.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        # Generate matrices based on Portuguese letter frequencies
        matrices = self.generate_statistical_matrices()
        
        # Process matrices in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Split matrices into chunks
            chunk_size = max(1, len(matrices) // self.num_threads)
            chunks = [matrices[i:i+chunk_size] for i in range(0, len(matrices), chunk_size)]
            
            # Process each chunk
            futures = []
            for chunk in chunks:
                future = executor.submit(self.process_matrices, chunk, ciphertext)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                chunk_results = future.result()
                results.extend(chunk_results)
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:20]  # Return top 20 candidates
    
    def generate_statistical_matrices(self) -> List[np.ndarray]:
        """
        Generate matrices based on Portuguese letter frequencies.
        
        Returns:
            List of matrices
        """
        matrices = []
        size = self.matrix_size
        
        # Get most frequent letters in Portuguese
        freq_letters = sorted(self.language_model.letter_freq.items(), 
                             key=lambda x: x[1], reverse=True)
        freq_letters = [ord(letter) - ord('A') for letter, _ in freq_letters[:10]]
        
        # Generate matrices with high-frequency letters in strategic positions
        if size == 3:
            # For 3x3, focus on diagonal and first row/column
            for a in freq_letters[:5]:  # Top-left
                for e in freq_letters[:5]:  # Middle
                    for i in freq_letters[:5]:  # Bottom-right
                        # Ensure determinant is coprime with 26
                        det = (a * e * i) % 26
                        if math.gcd(det, 26) == 1:
                            # Create matrix with these elements on diagonal
                            matrix = np.zeros((3, 3), dtype=int)
                            matrix[0, 0] = a
                            matrix[1, 1] = e
                            matrix[2, 2] = i
                            
                            # Add some common values for other positions
                            for b in range(0, 26, 5):  # Top-middle
                                for c in range(0, 26, 5):  # Top-right
                                    matrix[0, 1] = b
                                    matrix[0, 2] = c
                                    
                                    # Check if matrix is invertible
                                    if is_invertible_matrix(matrix):
                                        matrices.append(matrix.copy())
        
        elif size == 4:
            # For 4x4, use block structure
            for a in freq_letters[:3]:
                for f in freq_letters[:3]:
                    for k in freq_letters[:3]:
                        for p in freq_letters[:3]:
                            # Create diagonal matrix
                            matrix = np.zeros((4, 4), dtype=int)
                            matrix[0, 0] = a
                            matrix[1, 1] = f
                            matrix[2, 2] = k
                            matrix[3, 3] = p
                            
                            # Add some common values
                            matrix[0, 1] = 1
                            matrix[1, 2] = 1
                            matrix[2, 3] = 1
                            
                            # Check if matrix is invertible
                            if is_invertible_matrix(matrix):
                                matrices.append(matrix.copy())
        
        elif size == 5:
            # For 5x5, use even more simplified approach
            for a in freq_letters[:2]:
                for g in freq_letters[:2]:
                    for m in freq_letters[:2]:
                        for s in freq_letters[:2]:
                            for y in freq_letters[:2]:
                                # Create diagonal matrix
                                matrix = np.zeros((5, 5), dtype=int)
                                matrix[0, 0] = a
                                matrix[1, 1] = g
                                matrix[2, 2] = m
                                matrix[3, 3] = s
                                matrix[4, 4] = y
                                
                                # Add some structure
                                matrix[0, 1] = 1
                                matrix[1, 2] = 1
                                matrix[2, 3] = 1
                                matrix[3, 4] = 1
                                
                                # Check if matrix is invertible
                                if is_invertible_matrix(matrix):
                                    matrices.append(matrix.copy())
        
        # If we don't have enough matrices, add some random ones
        while len(matrices) < 100:
            matrix = np.random.randint(0, 26, (size, size))
            if is_invertible_matrix(matrix):
                matrices.append(matrix)
        
        return matrices
    
    def process_matrices(self, matrices: List[np.ndarray], ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Process a list of matrices.
        
        Args:
            matrices: List of matrices to process
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        for matrix in matrices:
            try:
                decrypted = decrypt_hill(ciphertext, matrix)
                score = self.language_model.score_text(decrypted)
                
                if score > 0:
                    results.append((matrix, decrypted, score))
            except Exception:
                continue
        
        return results
    
    def optimized_brute_force(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Optimized brute force attack.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        size = self.matrix_size
        
        # For larger matrices, we need to be more selective
        if size >= 4:
            # Generate a limited set of matrices
            matrices = self.generate_selective_matrices(500)
        else:
            # For 3x3, we can try more matrices
            matrices = self.generate_selective_matrices(2000)
        
        # Process matrices in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Split matrices into chunks
            chunk_size = max(1, len(matrices) // self.num_threads)
            chunks = [matrices[i:i+chunk_size] for i in range(0, len(matrices), chunk_size)]
            
            # Process each chunk
            futures = []
            for chunk in chunks:
                future = executor.submit(self.process_matrices, chunk, ciphertext)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                chunk_results = future.result()
                results.extend(chunk_results)
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:20]  # Return top 20 candidates
    
    def generate_selective_matrices(self, limit: int) -> List[np.ndarray]:
        """
        Generate a selective set of matrices.
        
        Args:
            limit: Maximum number of matrices to generate
            
        Returns:
            List of matrices
        """
        matrices = []
        size = self.matrix_size
        
        # Generate matrices with specific structures
        
        # 1. Identity-based matrices
        identity = np.eye(size, dtype=int)
        matrices.append(identity)
        
        # Add variations of identity matrix
        for _ in range(min(50, limit // 10)):
            variation = identity.copy()
            
            # Modify some elements
            for _ in range(size):
                i, j = np.random.randint(0, size, 2)
                variation[i, j] = np.random.randint(1, 26)
            
            if is_invertible_matrix(variation):
                matrices.append(variation)
        
        # 2. Upper triangular matrices
        for _ in range(min(100, limit // 5)):
            matrix = np.zeros((size, size), dtype=int)
            
            # Set diagonal elements (must be coprime with 26)
            coprimes = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
            for i in range(size):
                matrix[i, i] = np.random.choice(coprimes)
            
            # Set upper triangular elements
            for i in range(size):
                for j in range(i+1, size):
                    matrix[i, j] = np.random.randint(0, 26)
            
            if is_invertible_matrix(matrix):
                matrices.append(matrix)
        
        # 3. Block matrices
        if size >= 4:
            # Create 2x2 blocks
            block_size = 2
            num_blocks = size // block_size
            
            for _ in range(min(100, limit // 5)):
                matrix = np.zeros((size, size), dtype=int)
                
                # Fill diagonal blocks with invertible 2x2 matrices
                for b in range(num_blocks):
                    while True:
                        block = np.random.randint(0, 26, (block_size, block_size))
                        if is_invertible_matrix(block):
                            matrix[b*block_size:(b+1)*block_size, b*block_size:(b+1)*block_size] = block
                            break
                
                # Fill some off-diagonal elements
                for _ in range(size):
                    i, j = np.random.randint(0, size, 2)
                    if i // block_size != j // block_size:  # Different blocks
                        matrix[i, j] = np.random.randint(0, 26)
                
                if is_invertible_matrix(matrix):
                    matrices.append(matrix)
        
        # 4. Random matrices with high probability of being invertible
        while len(matrices) < limit:
            # Create random matrix with bias towards invertible matrices
            matrix = np.zeros((size, size), dtype=int)
            
            # Set diagonal elements to values likely to be coprime with 26
            coprimes = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
            for i in range(size):
                matrix[i, i] = np.random.choice(coprimes)
            
            # Set other elements
            for i in range(size):
                for j in range(size):
                    if i != j:
                        matrix[i, j] = np.random.randint(0, 26)
            
            if is_invertible_matrix(matrix):
                matrices.append(matrix)
        
        return matrices[:limit]
    
    def finalize_results(self, results: List[Tuple[np.ndarray, str, float]]) -> List[Tuple[np.ndarray, str, float]]:
        """
        Finalize results by removing duplicates and sorting.
        
        Args:
            results: List of results
            
        Returns:
            Finalized list of results
        """
        # Remove duplicates
        unique_results = []
        seen_texts = set()
        
        for matrix, decrypted, score in results:
            # Use first 100 chars as key to identify duplicates
            text_key = decrypted[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append((matrix, decrypted, score))
        
        # Sort by score
        unique_results.sort(key=lambda x: x[2], reverse=True)
        
        # Return top 20 results
        return unique_results[:20]
    
    def generate_report(self, results: List[Tuple[np.ndarray, str, float]], ciphertext: str) -> str:
        """
        Generate a report of the results.
        
        Args:
            results: List of results
            ciphertext: Original ciphertext
            
        Returns:
            Report as string
        """
        if not results:
            return "No results found."
        
        report = []
        report.append(f"=== RELATÓRIO DE DECIFRAGEM DA CIFRA DE HILL (ENHANCED) ===")
        report.append(f"Tamanho da matriz: {self.matrix_size}x{self.matrix_size}")
        report.append(f"Texto cifrado: {ciphertext[:50]}...")
        report.append(f"Número de resultados: {len(results)}")
        report.append("")
        report.append("Melhores resultados:")
        report.append("")
        
        # Show top 5 results
        for i, (matrix, decrypted, score) in enumerate(results[:5], 1):
            report.append(f"--- Resultado #{i} (Score: {score:.4f}) ---")
            report.append(f"Matriz chave:")
            for row in matrix:
                report.append(str(row))
            
            # Show raw decrypted text
            report.append(f"Texto decifrado (bruto): {decrypted[:100]}...")
            
            # Show processed text with word segmentation
            processed_text = self.post_process_text(decrypted)
            report.append(f"Texto decifrado (processado): {processed_text[:100]}...")
            
            # Count valid words
            valid_count, total_count = self.language_model.count_valid_words(decrypted)
            valid_percent = valid_count / total_count * 100 if total_count > 0 else 0
            report.append(f"Palavras válidas: {valid_count}/{total_count} ({valid_percent:.2f}%)")
            
            report.append("")
        
        return "\n".join(report)
    
    def post_process_text(self, text: str) -> str:
        """
        Post-process decrypted text to improve readability.
        
        Args:
            text: Raw decrypted text
            
        Returns:
            Processed text
        """
        # Clean text
        text = re.sub(r'[^A-Z]', '', text.upper())
        
        # Split into words
        words = []
        i = 0
        while i < len(text):
            # Try different word lengths
            best_word = None
            best_score = -float('inf')
            
            for length in range(1, min(15, len(text) - i) + 1):
                word = text[i:i+length]
                
                # Score this word
                score = 0
                
                # Valid word
                if self.language_model.contains(word):
                    score = length ** 2
                
                # Common beginning
                elif any(word.startswith(beginning) for beginning in self.language_model.common_beginnings):
                    score = length
                
                # Common ending
                elif any(word.endswith(ending) for ending in self.language_model.common_endings):
                    score = length
                
                # Single letter (only A, E, O are valid)
                elif length == 1 and word in self.language_model.valid_single_letters:
                    score = 0.5
                
                if score > best_score:
                    best_score = score
                    best_word = word
            
            # If no good word found, use a single character
            if best_word is None:
                best_word = text[i]
                i += 1
            else:
                i += len(best_word)
            
            words.append(best_word)
        
        # Join words with spaces
        processed = " ".join(words)
        
        # Add periods after some words to improve readability
        processed = re.sub(r'([A-Z]{3,})\s([A-Z])', r'\1. \2', processed)
        
        return processed
