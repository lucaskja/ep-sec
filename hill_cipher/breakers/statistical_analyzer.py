#!/usr/bin/env python3
"""
Statistical Analysis Module for Hill Cipher Breaking

This module implements comprehensive statistical analysis techniques
for breaking Hill Cipher encrypted texts using N-gram frequency analysis.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import json
import numpy as np
import logging
import math
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from itertools import product
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hill_cipher import HillCipher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('statistical_analyzer')

class StatisticalAnalyzer:
    """
    Statistical analyzer for Hill Cipher breaking using N-gram frequency analysis.
    """
    
    def __init__(self, key_size: int, data_dir: str = None):
        """
        Initialize the statistical analyzer.
        
        Args:
            key_size: Size of the Hill cipher key matrix
            data_dir: Directory containing frequency data files
        """
        self.key_size = key_size
        self.hill_cipher = HillCipher(key_size)
        
        # Set data directory
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        else:
            self.data_dir = data_dir
            
        # Load frequency data
        self.letter_frequencies = self._load_frequencies('letter_frequencies.json')
        self.bigram_frequencies = self._load_frequencies('2gram_frequencies.json')
        self.trigram_frequencies = self._load_frequencies('3gram_frequencies.json')
        
        # Try to load higher order n-grams if available
        try:
            self.fourgram_frequencies = self._load_frequencies('4gram_frequencies.json')
        except:
            self.fourgram_frequencies = {}
            
        try:
            self.fivegram_frequencies = self._load_frequencies('5gram_frequencies.json')
        except:
            self.fivegram_frequencies = {}
        
        logger.info(f"Initialized Statistical Analyzer for {key_size}x{key_size} Hill Cipher")
        logger.info(f"Loaded {len(self.letter_frequencies)} letter frequencies")
        logger.info(f"Loaded {len(self.bigram_frequencies)} bigram frequencies")
        logger.info(f"Loaded {len(self.trigram_frequencies)} trigram frequencies")
    
    def _load_frequencies(self, filename: str) -> Dict[str, float]:
        """Load frequency data from JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Frequency file {filename} not found")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error parsing {filename}")
            return {}
    
    def extract_ngrams(self, text: str, n: int) -> List[str]:
        """
        Extract n-grams from text.
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        text = text.upper()
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    def calculate_ngram_frequencies(self, text: str, n: int) -> Dict[str, float]:
        """
        Calculate n-gram frequencies in text.
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            Dictionary of n-gram frequencies
        """
        ngrams = self.extract_ngrams(text, n)
        if not ngrams:
            return {}
        
        counts = Counter(ngrams)
        total = len(ngrams)
        
        return {ngram: count / total for ngram, count in counts.items()}
    
    def chi_squared_test(self, observed_freq: Dict[str, float], 
                        expected_freq: Dict[str, float]) -> float:
        """
        Calculate chi-squared statistic for frequency comparison.
        
        Args:
            observed_freq: Observed frequencies
            expected_freq: Expected frequencies
            
        Returns:
            Chi-squared statistic (lower is better)
        """
        chi_squared = 0.0
        
        # Get all n-grams that appear in either distribution
        all_ngrams = set(observed_freq.keys()) | set(expected_freq.keys())
        
        for ngram in all_ngrams:
            observed = observed_freq.get(ngram, 0.0)
            expected = expected_freq.get(ngram, 1e-10)  # Small value to avoid division by zero
            
            if expected > 0:
                chi_squared += ((observed - expected) ** 2) / expected
        
        return chi_squared
    
    def bhattacharyya_distance(self, freq1: Dict[str, float], 
                              freq2: Dict[str, float]) -> float:
        """
        Calculate Bhattacharyya distance between two frequency distributions.
        
        Args:
            freq1: First frequency distribution
            freq2: Second frequency distribution
            
        Returns:
            Bhattacharyya distance (lower is better)
        """
        all_ngrams = set(freq1.keys()) | set(freq2.keys())
        
        bc = 0.0  # Bhattacharyya coefficient
        for ngram in all_ngrams:
            p1 = freq1.get(ngram, 1e-10)
            p2 = freq2.get(ngram, 1e-10)
            bc += math.sqrt(p1 * p2)
        
        # Bhattacharyya distance
        if bc <= 0:
            return float('inf')
        return -math.log(bc)
    
    def score_text(self, text: str, use_multiple_ngrams: bool = True) -> float:
        """
        Score text based on how well it matches Portuguese language patterns.
        
        Args:
            text: Text to score
            use_multiple_ngrams: Whether to use multiple n-gram sizes
            
        Returns:
            Score (higher is better for Portuguese text)
        """
        if len(text) < self.key_size:
            return float('-inf')
        
        total_score = 0.0
        weight_sum = 0.0
        
        # Letter frequency analysis (weight: 1.0)
        if self.letter_frequencies:
            observed_letters = self.calculate_ngram_frequencies(text, 1)
            letter_score = -self.chi_squared_test(observed_letters, self.letter_frequencies)
            total_score += letter_score * 1.0
            weight_sum += 1.0
        
        # Bigram frequency analysis (weight: 2.0)
        if self.bigram_frequencies and len(text) >= 2:
            observed_bigrams = self.calculate_ngram_frequencies(text, 2)
            bigram_score = -self.chi_squared_test(observed_bigrams, self.bigram_frequencies)
            total_score += bigram_score * 2.0
            weight_sum += 2.0
        
        # Trigram frequency analysis (weight: 3.0)
        if use_multiple_ngrams and self.trigram_frequencies and len(text) >= 3:
            observed_trigrams = self.calculate_ngram_frequencies(text, 3)
            trigram_score = -self.chi_squared_test(observed_trigrams, self.trigram_frequencies)
            total_score += trigram_score * 3.0
            weight_sum += 3.0
        
        # 4-gram analysis if available (weight: 2.0)
        if (use_multiple_ngrams and self.fourgram_frequencies and 
            len(text) >= 4 and len(self.fourgram_frequencies) > 100):
            observed_fourgrams = self.calculate_ngram_frequencies(text, 4)
            fourgram_score = -self.chi_squared_test(observed_fourgrams, self.fourgram_frequencies)
            total_score += fourgram_score * 2.0
            weight_sum += 2.0
        
        if weight_sum == 0:
            return float('-inf')
        
        return total_score / weight_sum
    
    def generate_key_candidates(self, max_candidates: int = 10000) -> List[np.ndarray]:
        """
        Generate potential key matrices for Hill cipher.
        
        Args:
            max_candidates: Maximum number of candidates to generate
            
        Returns:
            List of potential key matrices
        """
        candidates = []
        attempts = 0
        max_attempts = max_candidates * 10  # Safety limit
        
        logger.info(f"Generating key candidates for {self.key_size}x{self.key_size} matrix...")
        
        while len(candidates) < max_candidates and attempts < max_attempts:
            attempts += 1
            
            # Generate random matrix with values 0-25
            matrix = np.random.randint(0, 26, size=(self.key_size, self.key_size))
            
            # Check if matrix is invertible
            if self.hill_cipher.is_invertible(matrix):
                candidates.append(matrix)
                
                if len(candidates) % 1000 == 0:
                    logger.info(f"Generated {len(candidates)} valid candidates...")
        
        logger.info(f"Generated {len(candidates)} valid key candidates from {attempts} attempts")
        return candidates
    
    def smart_key_generation(self, ciphertext: str, max_candidates: int = 5000) -> List[np.ndarray]:
        """
        Generate key candidates using smart heuristics based on ciphertext analysis.
        
        Args:
            ciphertext: The encrypted text
            max_candidates: Maximum number of candidates to generate
            
        Returns:
            List of potential key matrices
        """
        candidates = []
        
        # Analyze ciphertext for patterns
        cipher_letter_freq = self.calculate_ngram_frequencies(ciphertext, 1)
        cipher_bigram_freq = self.calculate_ngram_frequencies(ciphertext, 2)
        
        # Get most common letters and bigrams in ciphertext
        common_cipher_letters = sorted(cipher_letter_freq.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]
        common_cipher_bigrams = sorted(cipher_bigram_freq.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
        
        # Get most common letters and bigrams in Portuguese
        common_pt_letters = sorted(self.letter_frequencies.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
        common_pt_bigrams = sorted(self.bigram_frequencies.items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
        
        logger.info("Using smart key generation based on frequency analysis...")
        
        # Generate candidates with bias towards mapping common patterns
        attempts = 0
        max_attempts = max_candidates * 20
        
        while len(candidates) < max_candidates and attempts < max_attempts:
            attempts += 1
            
            # Start with random matrix
            matrix = np.random.randint(0, 26, size=(self.key_size, self.key_size))
            
            # Apply some bias based on frequency analysis
            if attempts % 3 == 0:  # Every third attempt, use frequency-based bias
                # Slightly bias towards values that might map common cipher letters 
                # to common Portuguese letters
                for i in range(self.key_size):
                    for j in range(self.key_size):
                        if np.random.random() < 0.3:  # 30% chance to apply bias
                            # Use a value that might help map frequent patterns
                            matrix[i, j] = np.random.choice([1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25])
            
            # Check if matrix is invertible
            if self.hill_cipher.is_invertible(matrix):
                candidates.append(matrix)
                
                if len(candidates) % 500 == 0:
                    logger.info(f"Generated {len(candidates)} smart candidates...")
        
        logger.info(f"Generated {len(candidates)} smart key candidates")
        return candidates
    
    def break_cipher_statistical(self, ciphertext: str, 
                               max_candidates: int = 5000,
                               use_smart_generation: bool = True,
                               early_stopping_threshold: float = -50.0,
                               progress_callback=None) -> Tuple[Optional[np.ndarray], Optional[str], float]:
        """
        Break Hill cipher using statistical analysis.
        
        Args:
            ciphertext: Encrypted text
            max_candidates: Maximum number of key candidates to test
            use_smart_generation: Whether to use smart key generation
            early_stopping_threshold: Stop if score exceeds this threshold
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (best_key, decrypted_text, best_score)
        """
        logger.info(f"Starting statistical analysis attack on {self.key_size}x{self.key_size} Hill cipher")
        logger.info(f"Ciphertext length: {len(ciphertext)}")
        
        # Generate key candidates
        if use_smart_generation:
            candidates = self.smart_key_generation(ciphertext, max_candidates)
        else:
            candidates = self.generate_key_candidates(max_candidates)
        
        if not candidates:
            logger.error("No valid key candidates generated")
            return None, None, float('-inf')
        
        best_key = None
        best_decrypted = None
        best_score = float('-inf')
        
        start_time = time.time()
        
        for i, key in enumerate(candidates):
            try:
                # Decrypt with this key
                decrypted = self.hill_cipher.decrypt(ciphertext, key)
                
                # Score the decrypted text
                score = self.score_text(decrypted)
                
                # Update best if this is better
                if score > best_score:
                    best_score = score
                    best_key = key.copy()
                    best_decrypted = decrypted
                    
                    logger.info(f"New best score: {score:.2f} at candidate {i+1}")
                    logger.info(f"Key: {key.flatten()}")
                    logger.info(f"Decrypted: {decrypted[:50]}...")
                    
                    # Early stopping if score is very good
                    if score > early_stopping_threshold:
                        logger.info(f"Early stopping - excellent score achieved: {score:.2f}")
                        break
                
                # Progress callback
                if progress_callback and (i + 1) % 100 == 0:
                    progress_callback(i + 1, len(candidates), best_score)
                
                # Progress logging
                if (i + 1) % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    logger.info(f"Tested {i+1}/{len(candidates)} candidates "
                              f"({rate:.1f} keys/sec, best score: {best_score:.2f})")
                
            except Exception as e:
                logger.debug(f"Error testing key {i}: {e}")
                continue
        
        elapsed = time.time() - start_time
        logger.info(f"Statistical analysis completed in {elapsed:.2f} seconds")
        logger.info(f"Best score: {best_score:.2f}")
        
        if best_key is not None:
            logger.info(f"Best key found: {best_key.flatten()}")
            logger.info(f"Decrypted text: {best_decrypted}")
        
        return best_key, best_decrypted, best_score
    
    def validate_with_known_plaintext(self, ciphertext: str, known_plaintext: str) -> bool:
        """
        Validate the statistical approach using known plaintext.
        
        Args:
            ciphertext: Encrypted text
            known_plaintext: Known plaintext
            
        Returns:
            True if validation successful
        """
        logger.info("Validating statistical approach with known plaintext...")
        
        # Try to break the cipher
        key, decrypted, score = self.break_cipher_statistical(ciphertext, max_candidates=2000)
        
        if key is None:
            logger.error("Failed to find any key")
            return False
        
        # Clean both texts for comparison
        known_clean = known_plaintext.upper().replace(' ', '').replace('\n', '')
        decrypted_clean = decrypted.upper().replace(' ', '').replace('\n', '')
        
        # Remove padding from decrypted text
        decrypted_clean = decrypted_clean.rstrip('X')
        
        # Compare
        if known_clean == decrypted_clean:
            logger.info("✓ Validation successful - exact match!")
            return True
        elif known_clean in decrypted_clean or decrypted_clean in known_clean:
            logger.info("✓ Validation successful - partial match!")
            return True
        else:
            logger.warning("✗ Validation failed - no match")
            logger.info(f"Expected: {known_clean[:50]}...")
            logger.info(f"Got:      {decrypted_clean[:50]}...")
            return False

def main():
    """Main function for testing the statistical analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hill Cipher Statistical Analyzer")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext to analyze")
    parser.add_argument("--ciphertext-file", type=str, help="File containing ciphertext")
    parser.add_argument("--key-size", type=int, default=2, choices=[2, 3, 4, 5], 
                       help="Size of the key matrix")
    parser.add_argument("--max-candidates", type=int, default=5000, 
                       help="Maximum number of key candidates to test")
    parser.add_argument("--known-plaintext", type=str, help="Known plaintext for validation")
    parser.add_argument("--validate", action="store_true", help="Run validation test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get ciphertext
    ciphertext = args.ciphertext
    if args.ciphertext_file:
        with open(args.ciphertext_file, 'r') as f:
            ciphertext = f.read().strip()
    
    if not ciphertext:
        parser.error("Ciphertext must be provided")
    
    # Create analyzer
    analyzer = StatisticalAnalyzer(args.key_size)
    
    if args.validate and args.known_plaintext:
        # Run validation
        success = analyzer.validate_with_known_plaintext(ciphertext, args.known_plaintext)
        if success:
            print("Validation successful!")
        else:
            print("Validation failed!")
    else:
        # Run analysis
        key, decrypted, score = analyzer.break_cipher_statistical(
            ciphertext, max_candidates=args.max_candidates
        )
        
        if key is not None:
            print(f"Best key found: {key.flatten()}")
            print(f"Score: {score:.2f}")
            print(f"Decrypted text: {decrypted}")
        else:
            print("No suitable key found")

if __name__ == "__main__":
    main()
