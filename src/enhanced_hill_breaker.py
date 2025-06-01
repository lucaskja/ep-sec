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
        
        # For 2x2 matrices, use exhaustive search
        if self.matrix_size == 2:
            logging.info("Using exhaustive search for 2x2 matrix")
            return self.exhaustive_search_2x2(ciphertext, known_text_path)
        
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
        Pattern-based attack using common Portuguese patterns with improved thread utilization.
        
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
            "ESTAMOSJUNTOS", "PARAOSNOSSOS", "COMUMACERTA",
            "DEUMAVEZ", "NAOPODIA", "ESTAVAMOS", "QUANDOELE",
            "APENASUMA", "TODOSOSDIAS", "MUITOBEM", "SEMPREQUEPOSSIVEL",
            "DEPOISDE", "ANTESDE", "DURANTEO", "ENTRETANTO",
            "CONTUDO", "POREM", "TODAVIA", "MASELE", "ELESAO"
        ]
        
        # Adjust patterns based on matrix size
        size = self.matrix_size
        adjusted_patterns = []
        
        for pattern in patterns:
            if len(pattern) >= size * size:
                adjusted_patterns.append(pattern[:size * size])
        
        logging.info(f"Pattern-based attack: Using {len(adjusted_patterns)} patterns")
        
        # Process patterns in parallel using a work queue approach
        if len(adjusted_patterns) > self.num_threads:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit each pattern as a separate task
                future_to_pattern = {executor.submit(self.try_pattern, pattern, ciphertext): pattern 
                                    for pattern in adjusted_patterns}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_pattern):
                    pattern = future_to_pattern[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logging.error(f"Error processing pattern {pattern}: {e}")
        else:
            # If we have fewer patterns than threads, process sequentially
            for pattern in adjusted_patterns:
                try:
                    result = self.try_pattern(pattern, ciphertext)
                    if result:
                        results.append(result)
                except Exception as e:
                    logging.error(f"Error processing pattern {pattern}: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:50]  # Return top 50 candidates
    
    def try_pattern(self, pattern: str, ciphertext: str) -> Optional[Tuple[np.ndarray, str, float]]:
        """
        Try a specific pattern for pattern-based attack.
        
        Args:
            pattern: Pattern to try
            ciphertext: Encrypted text
            
        Returns:
            Tuple (key_matrix, decrypted_text, score) if successful, None otherwise
        """
        try:
            # Use the pattern as potential plaintext
            cipher_fragment = ciphertext[:len(pattern)]
            
            key_matrix = known_plaintext_attack(pattern, cipher_fragment, self.matrix_size)
            decrypted = decrypt_hill(ciphertext, key_matrix)
            score = self.language_model.score_text(decrypted)
            
            if score > 0:
                return (key_matrix, decrypted, score)
        except Exception:
            pass
        
        return None
    
    def statistical_attack(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Statistical attack using Portuguese language properties with improved thread utilization.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        # Generate matrices based on Portuguese letter frequencies
        # Scale the number of matrices with thread count for better utilization
        matrices = self.generate_statistical_matrices(self.num_threads * 50)
        
        logging.info(f"Statistical attack: Generated {len(matrices)} matrices")
        
        # Use a queue-based approach for better thread utilization
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Create smaller chunks for better load balancing
            optimal_chunk_size = max(10, len(matrices) // (self.num_threads * 5))
            chunks = [matrices[i:i+optimal_chunk_size] for i in range(0, len(matrices), optimal_chunk_size)]
            
            logging.info(f"Split into {len(chunks)} chunks of size ~{optimal_chunk_size}")
            
            # Submit all chunks to the executor
            future_to_chunk = {executor.submit(self.process_matrices, chunk, ciphertext): i for i, chunk in enumerate(chunks)}
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    
                    # Log progress
                    completed += 1
                    if completed % 10 == 0 or completed == len(chunks):
                        logging.info(f"Statistical attack: Processed {completed}/{len(chunks)} chunks ({completed*100/len(chunks):.1f}%)")
                        
                except Exception as e:
                    logging.error(f"Error processing chunk {chunk_idx}: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:100]  # Return top 100 candidates
    
    def generate_statistical_matrices(self, limit: int = 1000) -> List[np.ndarray]:
        """
        Generate matrices based on Portuguese letter frequencies.
        
        Args:
            limit: Maximum number of matrices to generate
            
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
                                        
                                        # Add variations to increase diversity
                                        for _ in range(3):  # Add 3 variations per matrix
                                            variation = matrix.copy()
                                            # Modify some elements randomly
                                            for _ in range(2):
                                                i, j = np.random.randint(0, 3, 2)
                                                variation[i, j] = np.random.randint(0, 26)
                                            
                                            if is_invertible_matrix(variation):
                                                matrices.append(variation.copy())
                                        
                                        if len(matrices) >= limit:
                                            return matrices[:limit]
        
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
                                
                                # Add variations
                                for _ in range(5):  # More variations for 4x4
                                    variation = matrix.copy()
                                    # Modify some elements randomly
                                    for _ in range(3):
                                        i, j = np.random.randint(0, 4, 2)
                                        variation[i, j] = np.random.randint(0, 26)
                                    
                                    if is_invertible_matrix(variation):
                                        matrices.append(variation.copy())
                                
                                if len(matrices) >= limit:
                                    return matrices[:limit]
        
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
                                    
                                    # Add variations
                                    for _ in range(10):  # Even more variations for 5x5
                                        variation = matrix.copy()
                                        # Modify some elements randomly
                                        for _ in range(4):
                                            i, j = np.random.randint(0, 5, 2)
                                            variation[i, j] = np.random.randint(0, 26)
                                        
                                        if is_invertible_matrix(variation):
                                            matrices.append(variation.copy())
                                    
                                    if len(matrices) >= limit:
                                        return matrices[:limit]
        
        # If we don't have enough matrices, add some random ones
        while len(matrices) < limit:
            matrix = np.random.randint(0, 26, (size, size))
            if is_invertible_matrix(matrix):
                matrices.append(matrix)
        
        return matrices[:limit]
    
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
                
                # Calculate score using multiple methods
                score = 0
                
                # 1. Score using language model
                lang_score = self.language_model.score_text(decrypted)
                score += lang_score * 2  # Double weight for language model score
                
                # 2. Count valid words
                valid_count, total_count = self.language_model.count_valid_words(decrypted)
                if total_count > 0:
                    word_score = valid_count / total_count
                    score += word_score * 10  # Higher weight for valid words (increased from 5)
                
                # 3. Check for common Portuguese words
                common_words = ['DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
                               'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS']
                
                word_count = 0
                for word in common_words:
                    if word in decrypted:
                        word_count += 1
                
                # Bonus for common words
                score += word_count * 0.5  # Increased from 0.2
                
                # Add to results if score is positive
                if score > 0:
                    results.append((matrix, decrypted, score))
            except Exception:
                continue
        
        return results
    
    def optimized_brute_force(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Optimized brute force attack using work queue for better thread utilization.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        size = self.matrix_size
        
        # For larger matrices, generate more matrices to fully utilize threads
        if size >= 4:
            # Generate more matrices for 4x4 and 5x5
            matrices = self.generate_selective_matrices(self.num_threads * 100)  # Scale with thread count
        else:
            # For 3x3, we can try even more matrices
            matrices = self.generate_selective_matrices(self.num_threads * 500)  # Scale with thread count
        
        logging.info(f"Generated {len(matrices)} matrices for size {size}x{size}")
        
        # Use a queue-based approach for better thread utilization
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Create smaller chunks for better load balancing
            optimal_chunk_size = max(10, len(matrices) // (self.num_threads * 10))  # Much smaller chunks
            chunks = [matrices[i:i+optimal_chunk_size] for i in range(0, len(matrices), optimal_chunk_size)]
            
            logging.info(f"Split into {len(chunks)} chunks of size ~{optimal_chunk_size}")
            
            # Submit all chunks to the executor
            future_to_chunk = {executor.submit(self.process_matrices, chunk, ciphertext): i for i, chunk in enumerate(chunks)}
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    
                    # Log progress
                    completed += 1
                    if completed % 10 == 0 or completed == len(chunks):
                        logging.info(f"Processed {completed}/{len(chunks)} chunks ({completed*100/len(chunks):.1f}%)")
                        
                except Exception as e:
                    logging.error(f"Error processing chunk {chunk_idx}: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:100]  # Return top 100 candidates
    
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
        
        # Return top 100 results (increased from 20)
        return unique_results[:100]
    
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
            
            # Check if this might be the correct solution
            if valid_percent > 30 or score > 5:
                report.append("*** POSSÍVEL SOLUÇÃO CORRETA ***")
            
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
        
        # Try to identify words based on dictionary
        processed_text = ""
        i = 0
        
        while i < len(text):
            # Try to find the longest valid word starting at position i
            found_word = False
            for length in range(min(15, len(text) - i), 0, -1):
                word = text[i:i+length]
                
                # Check if it's a valid word
                if self.language_model.contains(word):
                    processed_text += word + " "
                    i += length
                    found_word = True
                    break
            
            # If no valid word found, add a single character
            if not found_word:
                processed_text += text[i]
                if i < len(text) - 1:
                    # Add space after single characters except for A, E, O
                    if text[i] not in self.language_model.valid_single_letters:
                        processed_text += " "
                i += 1
        
        # Add periods to improve readability
        processed_text = re.sub(r'([A-Z]{3,})\s([A-Z])', r'\1. \2', processed_text)
        
        return processed_text
    def exhaustive_search_2x2(self, ciphertext: str, known_text_path: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Perform exhaustive search for 2x2 matrices with improved thread utilization.
        
        Args:
            ciphertext: Encrypted text
            known_text_path: Path to known plaintext file (optional)
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        # Generate all possible 2x2 matrices
        logging.info("Generating all invertible 2x2 matrices")
        matrices = []
        
        # Use coprimes for faster generation of invertible matrices
        coprimes_with_26 = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
        
        # Generate matrices with determinant coprime with 26
        for a in range(26):
            for b in range(26):
                for c in range(26):
                    for d in coprimes_with_26:  # At least one element must be coprime with 26
                        matrix = np.array([[a, b], [c, d]])
                        det = (a * d - b * c) % 26
                        if math.gcd(det, 26) == 1:  # Check if determinant is coprime with 26
                            matrices.append(matrix)
        
        logging.info(f"Generated {len(matrices)} invertible 2x2 matrices")
        
        # Process matrices in parallel using a work queue approach
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Create smaller chunks for better load balancing
            optimal_chunk_size = max(100, len(matrices) // (self.num_threads * 10))
            chunks = [matrices[i:i+optimal_chunk_size] for i in range(0, len(matrices), optimal_chunk_size)]
            
            logging.info(f"Split into {len(chunks)} chunks of size ~{optimal_chunk_size}")
            
            # Submit all chunks to the executor
            future_to_chunk = {executor.submit(self.process_2x2_matrices, chunk, ciphertext, known_text_path, i, len(chunks)): i 
                              for i, chunk in enumerate(chunks)}
            
            # Process results as they complete and monitor thread utilization
            completed = 0
            start_time = time.time()
            active_threads = self.num_threads
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    
                    # Log progress
                    completed += 1
                    if completed % 10 == 0 or completed == len(chunks):
                        elapsed = time.time() - start_time
                        matrices_per_second = (completed * optimal_chunk_size) / elapsed if elapsed > 0 else 0
                        logging.info(f"2x2 search: Processed {completed}/{len(chunks)} chunks ({completed*100/len(chunks):.1f}%) - "
                                    f"~{matrices_per_second:.0f} matrices/sec")
                        
                except Exception as e:
                    logging.error(f"Error processing chunk {chunk_idx}: {e}")
                    active_threads -= 1
                    logging.warning(f"Thread utilization: {active_threads}/{self.num_threads} active threads")
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:100]  # Return top 100 candidates
    
    def process_2x2_matrices(self, matrices: List[np.ndarray], ciphertext: str, 
                            known_text_path: str = None, chunk_idx: int = 0, 
                            total_chunks: int = 1) -> List[Tuple[np.ndarray, str, float]]:
        """
        Process a list of 2x2 matrices with enhanced scoring.
        
        Args:
            matrices: List of matrices to process
            ciphertext: Encrypted text
            known_text_path: Path to known plaintext file (optional)
            chunk_idx: Index of current chunk (for logging)
            total_chunks: Total number of chunks (for logging)
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        # Load known text for comparison if available
        known_text = None
        if known_text_path and os.path.exists(known_text_path):
            try:
                with open(known_text_path, 'r', encoding='latin-1') as f:
                    known_text = f.read().upper()
                known_text = re.sub(r'[^A-Z]', '', known_text)
            except Exception as e:
                logging.error(f"Error loading known text: {e}")
        
        # Process each matrix
        for i, matrix in enumerate(matrices):
            try:
                # Log progress periodically
                if i % 1000 == 0:
                    logging.debug(f"Chunk {chunk_idx+1}/{total_chunks}: Processed {i}/{len(matrices)} matrices")
                
                # Decrypt ciphertext
                decrypted = decrypt_hill(ciphertext, matrix)
                
                # Calculate score using multiple methods
                score = 0
                
                # 1. Score using language model
                lang_score = self.language_model.score_text(decrypted)
                score += lang_score * 2  # Double weight for language model score
                
                # 2. Count valid words
                valid_count, total_count = self.language_model.count_valid_words(decrypted)
                if total_count > 0:
                    word_score = valid_count / total_count
                    score += word_score * 5  # High weight for valid words
                
                # 3. Check for common Portuguese words
                common_words = ['DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
                               'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS']
                
                for word in common_words:
                    if word in decrypted:
                        score += 0.2  # Bonus for each common word found
                
                # 4. Compare with known text if available
                if known_text:
                    # Calculate similarity with known text
                    min_len = min(len(decrypted), len(known_text))
                    matches = sum(1 for i in range(min_len) if decrypted[i] == known_text[i])
                    similarity = matches / min_len
                    score += similarity * 10  # Very high weight for similarity
                
                # Add to results if score is positive
                if score > 0:
                    results.append((matrix, decrypted, score))
            except Exception as e:
                logging.debug(f"Error processing matrix: {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:1000]  # Return top 1000 candidates from this chunk
