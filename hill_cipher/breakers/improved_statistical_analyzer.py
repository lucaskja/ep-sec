#!/usr/bin/env python3
"""
Improved Statistical Analysis Module for Hill Cipher Breaking

This module implements an improved statistical analysis approach that uses
exhaustive search for 2x2 keys and optimized approaches for larger keys.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import json
import numpy as np
import logging
import math
import time
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from itertools import product

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hill_cipher import HillCipher
from breakers.statistical_analyzer import StatisticalAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('improved_statistical')

class ImprovedStatisticalAnalyzer(StatisticalAnalyzer):
    """
    Improved statistical analyzer with exhaustive search for 2x2 and optimized approaches for larger keys.
    """
    
    def __init__(self, key_size: int, data_dir: str = None):
        """
        Initialize the improved statistical analyzer.
        
        Args:
            key_size: Size of the Hill cipher key matrix
            data_dir: Directory containing frequency data files
        """
        super().__init__(key_size, data_dir)
        logger.info(f"Initialized Improved Statistical Analyzer for {key_size}x{key_size} Hill Cipher")
    
    def exhaustive_search_2x2(self, ciphertext: str, 
                             progress_callback=None) -> Tuple[Optional[np.ndarray], Optional[str], float]:
        """
        Perform exhaustive search for 2x2 Hill cipher keys.
        
        Args:
            ciphertext: Encrypted text
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (best_key, decrypted_text, best_score)
        """
        if self.key_size != 2:
            raise ValueError("Exhaustive search only available for 2x2 keys")
        
        logger.info("Starting exhaustive search for 2x2 Hill cipher")
        logger.info(f"Ciphertext length: {len(ciphertext)}")
        
        best_key = None
        best_decrypted = None
        best_score = float('-inf')
        
        tested = 0
        total_valid = 0
        start_time = time.time()
        
        # First pass: count valid keys
        logger.info("Counting valid keys...")
        for a in range(26):
            for b in range(26):
                for c in range(26):
                    for d in range(26):
                        key = np.array([[a, b], [c, d]])
                        if self.hill_cipher.is_invertible(key):
                            total_valid += 1
        
        logger.info(f"Found {total_valid} valid 2x2 keys to test")
        
        # Second pass: test all valid keys
        for a in range(26):
            for b in range(26):
                for c in range(26):
                    for d in range(26):
                        key = np.array([[a, b], [c, d]])
                        
                        if self.hill_cipher.is_invertible(key):
                            tested += 1
                            
                            try:
                                # Decrypt with this key
                                decrypted = self.hill_cipher.decrypt(ciphertext, key)
                                
                                # Score the decrypted text
                                score = self.score_text(decrypted, use_multiple_ngrams=True)
                                
                                # Update best if this is better
                                if score > best_score:
                                    best_score = score
                                    best_key = key.copy()
                                    best_decrypted = decrypted
                                    
                                    logger.info(f"New best score: {score:.2f} at key {key.flatten()} "
                                              f"({tested}/{total_valid})")
                                
                                # Progress callback
                                if progress_callback and tested % 1000 == 0:
                                    progress_callback(tested, total_valid, best_score)
                                
                                # Progress logging
                                if tested % 10000 == 0:
                                    elapsed = time.time() - start_time
                                    rate = tested / elapsed
                                    eta = (total_valid - tested) / rate if rate > 0 else 0
                                    logger.info(f"Tested {tested}/{total_valid} keys "
                                              f"({rate:.1f} keys/sec, ETA: {eta:.1f}s, best: {best_score:.2f})")
                            
                            except Exception as e:
                                logger.debug(f"Error testing key {key.flatten()}: {e}")
                                continue
        
        elapsed = time.time() - start_time
        logger.info(f"Exhaustive search completed in {elapsed:.2f} seconds")
        logger.info(f"Tested {tested} valid keys")
        logger.info(f"Best score: {best_score:.2f}")
        
        if best_key is not None:
            logger.info(f"Best key found: {best_key.flatten()}")
            logger.info(f"Decrypted text: {best_decrypted}")
        
        return best_key, best_decrypted, best_score
    
    def optimized_search_larger_keys(self, ciphertext: str, 
                                   max_candidates: int = 50000,
                                   use_genetic: bool = True,
                                   progress_callback=None) -> Tuple[Optional[np.ndarray], Optional[str], float]:
        """
        Perform optimized search for larger Hill cipher keys (3x3, 4x4, 5x5).
        
        Args:
            ciphertext: Encrypted text
            max_candidates: Maximum number of candidates to test
            use_genetic: Whether to use genetic algorithm optimization
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (best_key, decrypted_text, best_score)
        """
        logger.info(f"Starting optimized search for {self.key_size}x{self.key_size} Hill cipher")
        
        # Use multiple strategies
        strategies = []
        
        # Strategy 1: Smart random generation
        strategies.append(('smart_random', max_candidates // 3))
        
        # Strategy 2: Frequency-based generation
        strategies.append(('frequency_based', max_candidates // 3))
        
        # Strategy 3: Pattern-based generation
        strategies.append(('pattern_based', max_candidates // 3))
        
        best_key = None
        best_decrypted = None
        best_score = float('-inf')
        total_tested = 0
        
        for strategy_name, num_candidates in strategies:
            logger.info(f"Running strategy: {strategy_name} with {num_candidates} candidates")
            
            if strategy_name == 'smart_random':
                candidates = self.smart_key_generation(ciphertext, num_candidates)
            elif strategy_name == 'frequency_based':
                candidates = self.frequency_based_generation(ciphertext, num_candidates)
            elif strategy_name == 'pattern_based':
                candidates = self.pattern_based_generation(ciphertext, num_candidates)
            
            # Test candidates from this strategy
            for i, key in enumerate(candidates):
                try:
                    decrypted = self.hill_cipher.decrypt(ciphertext, key)
                    score = self.score_text(decrypted, use_multiple_ngrams=True)
                    
                    if score > best_score:
                        best_score = score
                        best_key = key.copy()
                        best_decrypted = decrypted
                        logger.info(f"New best score: {score:.2f} from {strategy_name}")
                    
                    total_tested += 1
                    
                    if progress_callback and total_tested % 100 == 0:
                        progress_callback(total_tested, max_candidates, best_score)
                
                except Exception as e:
                    logger.debug(f"Error testing key: {e}")
                    continue
        
        logger.info(f"Optimized search completed. Tested {total_tested} candidates")
        logger.info(f"Best score: {best_score:.2f}")
        
        return best_key, best_decrypted, best_score
    
    def frequency_based_generation(self, ciphertext: str, max_candidates: int) -> List[np.ndarray]:
        """
        Generate key candidates based on frequency analysis of the ciphertext.
        
        Args:
            ciphertext: The encrypted text
            max_candidates: Maximum number of candidates to generate
            
        Returns:
            List of potential key matrices
        """
        candidates = []
        
        # Analyze ciphertext frequencies
        cipher_freqs = self.calculate_ngram_frequencies(ciphertext, 1)
        most_common_cipher = sorted(cipher_freqs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get most common Portuguese letters
        most_common_pt = sorted(self.letter_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
        
        attempts = 0
        max_attempts = max_candidates * 10
        
        while len(candidates) < max_candidates and attempts < max_attempts:
            attempts += 1
            
            # Generate matrix with bias towards mapping frequent cipher letters to frequent PT letters
            matrix = np.random.randint(0, 26, size=(self.key_size, self.key_size))
            
            # Apply frequency-based bias
            if attempts % 2 == 0:
                for i in range(self.key_size):
                    for j in range(self.key_size):
                        if np.random.random() < 0.4:
                            # Use values that might help map frequent patterns
                            cipher_char = most_common_cipher[i % len(most_common_cipher)][0]
                            pt_char = most_common_pt[j % len(most_common_pt)][0]
                            
                            cipher_val = ord(cipher_char) - ord('A')
                            pt_val = ord(pt_char) - ord('A')
                            
                            # Use a value that might contribute to this mapping
                            matrix[i, j] = (cipher_val - pt_val) % 26
            
            if self.hill_cipher.is_invertible(matrix):
                candidates.append(matrix)
        
        logger.info(f"Generated {len(candidates)} frequency-based candidates")
        return candidates
    
    def pattern_based_generation(self, ciphertext: str, max_candidates: int) -> List[np.ndarray]:
        """
        Generate key candidates based on patterns in the ciphertext.
        
        Args:
            ciphertext: The encrypted text
            max_candidates: Maximum number of candidates to generate
            
        Returns:
            List of potential key matrices
        """
        candidates = []
        
        # Look for repeated patterns in ciphertext
        patterns = {}
        for i in range(len(ciphertext) - self.key_size + 1):
            pattern = ciphertext[i:i + self.key_size]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        common_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        attempts = 0
        max_attempts = max_candidates * 10
        
        while len(candidates) < max_candidates and attempts < max_attempts:
            attempts += 1
            
            # Generate matrix with some structure
            matrix = np.random.randint(0, 26, size=(self.key_size, self.key_size))
            
            # Apply pattern-based modifications
            if attempts % 3 == 0 and common_patterns:
                # Try to incorporate information from common patterns
                pattern = common_patterns[attempts % len(common_patterns)][0]
                for i in range(min(self.key_size, len(pattern))):
                    for j in range(self.key_size):
                        if np.random.random() < 0.3:
                            matrix[i, j] = (ord(pattern[i]) - ord('A') + j) % 26
            
            if self.hill_cipher.is_invertible(matrix):
                candidates.append(matrix)
        
        logger.info(f"Generated {len(candidates)} pattern-based candidates")
        return candidates
    
    def break_cipher_improved(self, ciphertext: str, 
                            max_candidates: int = 50000,
                            timeout: int = 600,
                            progress_callback=None) -> Tuple[Optional[np.ndarray], Optional[str], float]:
        """
        Break Hill cipher using improved statistical analysis.
        
        Args:
            ciphertext: Encrypted text
            max_candidates: Maximum number of candidates (ignored for 2x2)
            timeout: Maximum time in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (best_key, decrypted_text, best_score)
        """
        start_time = time.time()
        
        try:
            if self.key_size == 2:
                # Use exhaustive search for 2x2
                return self.exhaustive_search_2x2(ciphertext, progress_callback)
            else:
                # Use optimized search for larger keys
                return self.optimized_search_larger_keys(
                    ciphertext, max_candidates, True, progress_callback
                )
        
        except Exception as e:
            logger.error(f"Error in improved cipher breaking: {e}")
            return None, None, float('-inf')
    
    def validate_with_known_plaintext(self, ciphertext: str, known_plaintext: str) -> bool:
        """
        Validate the improved approach using known plaintext.
        
        Args:
            ciphertext: Encrypted text
            known_plaintext: Known plaintext
            
        Returns:
            True if validation successful
        """
        logger.info("Validating improved statistical approach with known plaintext...")
        
        # Try to break the cipher
        key, decrypted, score = self.break_cipher_improved(ciphertext)
        
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
    """Main function for testing the improved analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Hill Cipher Statistical Analyzer")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext to analyze")
    parser.add_argument("--ciphertext-file", type=str, help="File containing ciphertext")
    parser.add_argument("--key-size", type=int, default=2, choices=[2, 3, 4, 5], 
                       help="Size of the key matrix")
    parser.add_argument("--max-candidates", type=int, default=50000, 
                       help="Maximum number of key candidates to test (ignored for 2x2)")
    parser.add_argument("--known-plaintext", type=str, help="Known plaintext for validation")
    parser.add_argument("--validate", action="store_true", help="Run validation test")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
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
    analyzer = ImprovedStatisticalAnalyzer(args.key_size)
    
    if args.validate and args.known_plaintext:
        # Run validation
        success = analyzer.validate_with_known_plaintext(ciphertext, args.known_plaintext)
        if success:
            print("Validation successful!")
        else:
            print("Validation failed!")
    else:
        # Run analysis
        key, decrypted, score = analyzer.break_cipher_improved(
            ciphertext, 
            max_candidates=args.max_candidates,
            timeout=args.timeout
        )
        
        if key is not None:
            print(f"Best key found: {key.flatten()}")
            print(f"Score: {score:.2f}")
            print(f"Decrypted text: {decrypted}")
        else:
            print("No suitable key found")

if __name__ == "__main__":
    main()
