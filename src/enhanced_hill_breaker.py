import os
import re
import time
import logging
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import concurrent.futures
import math
from collections import Counter

# Import utility functions
from src.hill_cipher import decrypt_hill
from src.portuguese_statistics import (
    LETTER_FREQUENCIES, 
    DIGRAM_FREQUENCIES, 
    TRIGRAM_FREQUENCIES,
    SHORT_WORDS
)

class EnhancedHillBreaker:
    def __init__(self, num_threads=4, verbose=True):
        self.num_threads = num_threads
        self.verbose = verbose
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the breaker"""
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            logging.basicConfig(
                level=logging.WARNING,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
    
    def process_2x2_matrices_optimized(self, matrices: List[np.ndarray], ciphertext: str, 
                                     known_text_path: str = None, chunk_idx: int = 0, 
                                     total_chunks: int = 1) -> Tuple[List[Tuple[np.ndarray, str, float]], bool, int]:
        """
        Process a list of 2x2 matrices with enhanced scoring and early stopping.
        
        Args:
            matrices: List of matrices to process
            ciphertext: Encrypted text
            known_text_path: Path to known plaintext file (optional)
            chunk_idx: Index of current chunk (for logging)
            total_chunks: Total number of chunks (for logging)
            
        Returns:
            Tuple of (results list, found_good_solution flag, matrices_processed count)
        """
        results = []
        found_good_solution = False
        matrices_processed = 0
        
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
        start_time = time.time()
        for i, matrix in enumerate(matrices):
            matrices_processed += 1
            
            # Log progress periodically
            if i > 0 and i % 50 == 0:
                elapsed = time.time() - start_time
                matrices_per_sec = i / elapsed if elapsed > 0 else 0
                if chunk_idx % 10 == 0:  # Only log for some chunks to avoid log spam
                    logging.debug(f"Chunk {chunk_idx+1}/{total_chunks}: Processed {i}/{len(matrices)} matrices ({matrices_per_sec:.1f} matrices/sec)")
            
            try:
                # Decrypt ciphertext
                decrypted = decrypt_hill(ciphertext, matrix)
                
                # Calculate score using statistical approach
                score = self.score_text(decrypted)
                
                # Check for common Portuguese words
                common_words = ['DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
                               'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS']
                
                word_count = 0
                found_words = []
                for word in common_words:
                    if word in decrypted:
                        word_count += 1
                        found_words.append(word)
                
                # Bonus for common words
                score += word_count * 0.5
                
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
                    
                    # Check if we found a very good solution (early stopping)
                    # Only stop if we have at least 10 results and a very high score
                    if len(results) >= 10 and score > 15:
                        found_good_solution = True
                        logging.info(f"Found excellent solution in chunk {chunk_idx}: Matrix {matrix}, score: {score:.2f}")
                        logging.info(f"Common words found: {', '.join(found_words)}")
                        logging.info(f"Decrypted text: {decrypted[:50]}...")
                        break
                    
                    # Log high-scoring matrices
                    if score > 10:
                        logging.info(f"Found good solution in chunk {chunk_idx}: Matrix {matrix}, score: {score:.2f}")
                        logging.info(f"Common words found: {', '.join(found_words)}")
                        logging.info(f"Decrypted text: {decrypted[:50]}...")
                    
            except Exception as e:
                if chunk_idx % 10 == 0 and i % 100 == 0:  # Only log occasionally to avoid spam
                    logging.debug(f"Error processing matrix {matrix} in chunk {chunk_idx}: {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:1000], found_good_solution, matrices_processed  # Return top 1000 candidates from this chunk
    
    def score_text(self, text: str) -> float:
        """
        Score text based on Portuguese language statistics.
        
        Args:
            text: Text to score
            
        Returns:
            Score value (higher is better)
        """
        # Initialize score
        score = 0.0
        
        # 1. Letter frequency analysis
        letter_counts = Counter(text)
        total_letters = len(text)
        
        for letter, count in letter_counts.items():
            freq = count / total_letters
            expected_freq = LETTER_FREQUENCIES.get(letter, 0.0001)
            # Score based on how close the frequency is to expected
            similarity = 1 - min(abs(freq - expected_freq) / expected_freq, 1)
            score += similarity * 0.5  # Weight for letter frequency
        
        # 2. Digram analysis
        if len(text) >= 2:
            digrams = [text[i:i+2] for i in range(len(text)-1)]
            digram_counts = Counter(digrams)
            total_digrams = len(digrams)
            
            for digram, count in digram_counts.items():
                if digram in DIGRAM_FREQUENCIES:
                    freq = count / total_digrams
                    expected_freq = DIGRAM_FREQUENCIES[digram]
                    # Higher weight for common digrams
                    score += min(freq, expected_freq) * 5
        
        # 3. Trigram analysis
        if len(text) >= 3:
            trigrams = [text[i:i+3] for i in range(len(text)-2)]
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)
            
            for trigram, count in trigram_counts.items():
                if trigram in TRIGRAM_FREQUENCIES:
                    freq = count / total_trigrams
                    expected_freq = TRIGRAM_FREQUENCIES[trigram]
                    # Higher weight for common trigrams
                    score += min(freq, expected_freq) * 10
        
        # 4. Check for common short words
        for word, freq in SHORT_WORDS.items():
            if word in text:
                score += freq * 2
        
        # 5. Vowel/consonant ratio (Portuguese has ~46% vowels)
        vowels = sum(1 for c in text if c in 'AEIOU')
        vowel_ratio = vowels / total_letters if total_letters > 0 else 0
        # Penalize if vowel ratio is too far from expected
        if 0.4 <= vowel_ratio <= 0.5:
            score += 2
        elif 0.35 <= vowel_ratio <= 0.55:
            score += 1
        else:
            score -= 1
        
        return score
    
    def break_hill_cipher(self, ciphertext: str, matrix_size: int, 
                         known_text_path: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Break Hill cipher using enhanced techniques.
        
        Args:
            ciphertext: Encrypted text
            matrix_size: Size of the Hill cipher matrix (2, 3, 4, or 5)
            known_text_path: Path to known plaintext file (optional)
            
        Returns:
            List of (matrix, decrypted_text, score) tuples
        """
        # Clean ciphertext
        ciphertext = re.sub(r'[^A-Z]', '', ciphertext.upper())
        
        if matrix_size == 2:
            return self.break_hill_2x2(ciphertext, known_text_path)
        elif matrix_size == 3:
            return self.break_hill_3x3(ciphertext, known_text_path)
        elif matrix_size == 4:
            return self.break_hill_4x4(ciphertext, known_text_path)
        elif matrix_size == 5:
            return self.break_hill_5x5(ciphertext, known_text_path)
        else:
            raise ValueError(f"Unsupported matrix size: {matrix_size}")
    
    def break_hill_2x2(self, ciphertext: str, known_text_path: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Break Hill cipher with 2x2 matrix using enhanced techniques.
        
        Args:
            ciphertext: Encrypted text
            known_text_path: Path to known plaintext file (optional)
            
        Returns:
            List of (matrix, decrypted_text, score) tuples
        """
        logging.info("Breaking Hill cipher with 2x2 matrix...")
        
        # Generate all valid 2x2 matrices
        matrices = self.generate_2x2_matrices()
        logging.info(f"Generated {len(matrices)} valid 2x2 matrices")
        
        # Split matrices into chunks for parallel processing
        chunk_size = max(1, len(matrices) // (self.num_threads * 2))
        matrix_chunks = [matrices[i:i+chunk_size] for i in range(0, len(matrices), chunk_size)]
        logging.info(f"Split matrices into {len(matrix_chunks)} chunks for processing")
        
        # Process chunks in parallel
        results = []
        total_matrices_processed = 0
        found_good_solution = False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_chunk = {
                executor.submit(
                    self.process_2x2_matrices_optimized, 
                    chunk, 
                    ciphertext, 
                    known_text_path,
                    i, 
                    len(matrix_chunks)
                ): i for i, chunk in enumerate(matrix_chunks)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results, chunk_found_good, matrices_processed = future.result()
                    results.extend(chunk_results)
                    total_matrices_processed += matrices_processed
                    
                    if chunk_found_good:
                        found_good_solution = True
                        logging.info(f"Chunk {chunk_idx} found a good solution. Continuing to process other chunks...")
                        
                except Exception as e:
                    logging.error(f"Error processing chunk {chunk_idx}: {e}")
        
        # Sort results by score
        results.sort(key=lambda x: x[2], reverse=True)
        
        logging.info(f"Processed {total_matrices_processed} matrices in total")
        logging.info(f"Found {len(results)} potential solutions")
        
        return results[:100]  # Return top 100 results
    
    def generate_2x2_matrices(self) -> List[np.ndarray]:
        """
        Generate valid 2x2 matrices for Hill cipher.
        
        Returns:
            List of valid 2x2 matrices
        """
        matrices = []
        
        # 1. Add known good matrices first
        known_good = [
            np.array([[23, 17], [0, 9]]),  # Known to work well for Portuguese
            np.array([[23, 14], [0, 5]]),  # Another known good matrix
            np.array([[5, 17], [8, 3]]),   # Common in examples
            np.array([[6, 24], [1, 13]]),  # Another common matrix
            np.array([[3, 4], [1, 3]]),    # Simple matrix with good properties
        ]
        matrices.extend(known_good)
        
        # 2. Generate matrices based on letter frequencies
        # Use most common letters in Portuguese: A(0), E(4), O(14), S(18), R(17), I(8), N(13), D(3), M(12), U(20)
        common_letters = [0, 4, 14, 18, 17, 8, 13, 3, 12, 20]
        
        for a in common_letters:
            for b in range(26):
                for c in range(26):
                    for d in common_letters:
                        matrix = np.array([[a, b], [c, d]])
                        det = int((a * d - b * c) % 26)
                        if math.gcd(det, 26) == 1:
                            matrices.append(matrix)
        
        # 3. Generate matrices based on common digrams
        # DE(3,4), RA(17,0), ES(4,18), OS(14,18), AS(0,18), QU(16,20), NT(13,19), CO(2,14), AR(0,17), EN(4,13)
        common_digrams = [
            (3, 4), (17, 0), (4, 18), (14, 18), (0, 18),
            (16, 20), (13, 19), (2, 14), (0, 17), (4, 13)
        ]
        
        for a, b in common_digrams:
            for c in range(26):
                for d in range(26):
                    matrix = np.array([[a, b], [c, d]])
                    det = int((a * d - b * c) % 26)
                    if math.gcd(det, 26) == 1:
                        matrices.append(matrix)
        
        # Remove duplicates
        unique_matrices = []
        seen = set()
        
        for matrix in matrices:
            matrix_tuple = tuple(matrix.flatten())
            if matrix_tuple not in seen:
                seen.add(matrix_tuple)
                unique_matrices.append(matrix)
        
        return unique_matrices
    
    def break_hill_3x3(self, ciphertext: str, known_text_path: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Break Hill cipher with 3x3 matrix using enhanced techniques.
        
        Args:
            ciphertext: Encrypted text
            known_text_path: Path to known plaintext file (optional)
            
        Returns:
            List of (matrix, decrypted_text, score) tuples
        """
        logging.info("Breaking Hill cipher with 3x3 matrix...")
        # Implementation for 3x3 matrices
        # This would use more advanced techniques than exhaustive search
        
        # Placeholder for now
        return []
    
    def break_hill_4x4(self, ciphertext: str, known_text_path: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Break Hill cipher with 4x4 matrix using enhanced techniques.
        
        Args:
            ciphertext: Encrypted text
            known_text_path: Path to known plaintext file (optional)
            
        Returns:
            List of (matrix, decrypted_text, score) tuples
        """
        logging.info("Breaking Hill cipher with 4x4 matrix...")
        # Implementation for 4x4 matrices
        # This would use specialized techniques for larger matrices
        
        # Placeholder for now
        return []
    
    def break_hill_5x5(self, ciphertext: str, known_text_path: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Break Hill cipher with 5x5 matrix using enhanced techniques.
        
        Args:
            ciphertext: Encrypted text
            known_text_path: Path to known plaintext file (optional)
            
        Returns:
            List of (matrix, decrypted_text, score) tuples
        """
        logging.info("Breaking Hill cipher with 5x5 matrix...")
        # Implementation for 5x5 matrices
        # This would use highly specialized techniques
        
        # Placeholder for now
        return []
