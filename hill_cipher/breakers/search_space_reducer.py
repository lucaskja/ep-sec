#!/usr/bin/env python3
"""
Search Space Reduction Techniques for Hill Cipher

This module implements various techniques to reduce the search space
for Hill cipher keys, making brute force attacks more feasible.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import numpy as np
import logging
import math
from typing import List, Tuple, Optional, Set, Iterator
from itertools import product, combinations
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hill_cipher import HillCipher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('search_space_reducer')

class SearchSpaceReducer:
    """
    Implements various techniques to reduce Hill cipher key search space.
    """
    
    def __init__(self, key_size: int):
        """
        Initialize the search space reducer.
        
        Args:
            key_size: Size of the Hill cipher key matrix
        """
        self.key_size = key_size
        self.hill_cipher = HillCipher(key_size)
        self.modulus = 26
        
        # Precompute useful values
        self.coprime_values = [i for i in range(26) if math.gcd(i, 26) == 1]
        self.non_coprime_values = [i for i in range(26) if math.gcd(i, 26) != 1]
        
        logger.info(f"Initialized SearchSpaceReducer for {key_size}x{key_size} matrices")
        logger.info(f"Coprime values (mod 26): {self.coprime_values}")
        logger.info(f"Non-coprime values (mod 26): {self.non_coprime_values}")
    
    def estimate_total_search_space(self) -> int:
        """
        Estimate the total search space without reductions.
        
        Returns:
            Estimated number of valid keys
        """
        total_matrices = 26 ** (self.key_size * self.key_size)
        
        # Rough estimate: about 1/3 of matrices are invertible
        # This is a very rough approximation
        estimated_invertible = total_matrices // 3
        
        logger.info(f"Total possible matrices: {total_matrices:,}")
        logger.info(f"Estimated invertible matrices: {estimated_invertible:,}")
        
        return estimated_invertible
    
    def generate_keys_by_determinant_constraint(self, target_determinants: List[int] = None) -> Iterator[np.ndarray]:
        """
        Generate keys with specific determinant values (must be coprime with 26).
        
        Args:
            target_determinants: List of target determinant values. If None, uses all coprime values.
            
        Yields:
            Valid key matrices
        """
        if target_determinants is None:
            target_determinants = self.coprime_values
        
        logger.info(f"Generating keys with determinants: {target_determinants}")
        
        if self.key_size == 2:
            # For 2x2: det = ad - bc
            for det_target in target_determinants:
                for a in range(26):
                    for b in range(26):
                        for c in range(26):
                            # Calculate d such that ad - bc ≡ det_target (mod 26)
                            if a == 0:
                                continue  # Skip to avoid division by zero issues
                            
                            try:
                                # d ≡ (det_target + bc) * a^(-1) (mod 26)
                                a_inv = self.hill_cipher.mod_inverse(a, 26)
                                d = ((det_target + b * c) * a_inv) % 26
                                
                                key = np.array([[a, b], [c, d]])
                                
                                # Verify the determinant
                                actual_det = (a * d - b * c) % 26
                                if actual_det == det_target and self.hill_cipher.is_invertible(key):
                                    yield key
                            except ValueError:
                                continue
        else:
            # For larger matrices, use general approach (less efficient)
            logger.warning(f"Determinant constraint for {self.key_size}x{self.key_size} matrices uses general generation")
            for key in self.generate_keys_smart_sampling():
                det = int(round(np.linalg.det(key))) % 26
                if det in target_determinants:
                    yield key
    
    def generate_keys_with_structure_constraints(self) -> Iterator[np.ndarray]:
        """
        Generate keys with specific structural constraints to reduce search space.
        
        Yields:
            Structured key matrices
        """
        logger.info("Generating keys with structural constraints")
        
        # Strategy 1: Upper triangular matrices
        logger.info("Generating upper triangular matrices...")
        for key in self._generate_upper_triangular():
            if self.hill_cipher.is_invertible(key):
                yield key
        
        # Strategy 2: Lower triangular matrices
        logger.info("Generating lower triangular matrices...")
        for key in self._generate_lower_triangular():
            if self.hill_cipher.is_invertible(key):
                yield key
        
        # Strategy 3: Diagonal matrices
        logger.info("Generating diagonal matrices...")
        for key in self._generate_diagonal():
            if self.hill_cipher.is_invertible(key):
                yield key
        
        # Strategy 4: Matrices with many zeros
        logger.info("Generating sparse matrices...")
        for key in self._generate_sparse_matrices():
            if self.hill_cipher.is_invertible(key):
                yield key
    
    def _generate_upper_triangular(self) -> Iterator[np.ndarray]:
        """Generate upper triangular matrices."""
        # Upper triangular: all elements below diagonal are 0
        n = self.key_size
        
        # Generate diagonal elements (must be non-zero for invertibility)
        for diag_values in product(range(1, 26), repeat=n):
            # Generate upper triangular elements
            upper_elements = n * (n - 1) // 2  # Number of elements above diagonal
            for upper_values in product(range(26), repeat=upper_elements):
                key = np.zeros((n, n), dtype=int)
                
                # Set diagonal
                np.fill_diagonal(key, diag_values)
                
                # Set upper triangular elements
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        key[i, j] = upper_values[idx]
                        idx += 1
                
                yield key
    
    def _generate_lower_triangular(self) -> Iterator[np.ndarray]:
        """Generate lower triangular matrices."""
        # Lower triangular: all elements above diagonal are 0
        n = self.key_size
        
        # Generate diagonal elements (must be non-zero for invertibility)
        for diag_values in product(range(1, 26), repeat=n):
            # Generate lower triangular elements
            lower_elements = n * (n - 1) // 2  # Number of elements below diagonal
            for lower_values in product(range(26), repeat=lower_elements):
                key = np.zeros((n, n), dtype=int)
                
                # Set diagonal
                np.fill_diagonal(key, diag_values)
                
                # Set lower triangular elements
                idx = 0
                for i in range(1, n):
                    for j in range(i):
                        key[i, j] = lower_values[idx]
                        idx += 1
                
                yield key
    
    def _generate_diagonal(self) -> Iterator[np.ndarray]:
        """Generate diagonal matrices."""
        n = self.key_size
        
        # All diagonal elements must be coprime with 26 for invertibility
        for diag_values in product(self.coprime_values, repeat=n):
            key = np.diag(diag_values)
            yield key
    
    def _generate_sparse_matrices(self, max_nonzero: int = None) -> Iterator[np.ndarray]:
        """Generate matrices with limited number of non-zero elements."""
        if max_nonzero is None:
            max_nonzero = self.key_size + 2  # Slightly more than minimal
        
        n = self.key_size
        total_elements = n * n
        
        # Choose positions for non-zero elements
        for num_nonzero in range(n, min(max_nonzero + 1, total_elements)):
            for positions in combinations(range(total_elements), num_nonzero):
                # Generate values for non-zero positions
                for values in product(range(1, 26), repeat=num_nonzero):
                    key = np.zeros((n, n), dtype=int)
                    
                    for pos, val in zip(positions, values):
                        row, col = divmod(pos, n)
                        key[row, col] = val
                    
                    if self.hill_cipher.is_invertible(key):
                        yield key
    
    def generate_keys_frequency_based(self, ciphertext: str, max_keys: int = 10000) -> Iterator[np.ndarray]:
        """
        Generate keys based on frequency analysis of the ciphertext.
        
        Args:
            ciphertext: The encrypted text to analyze
            max_keys: Maximum number of keys to generate
            
        Yields:
            Keys biased towards frequency patterns
        """
        logger.info("Generating keys based on frequency analysis")
        
        # Analyze ciphertext frequencies
        from collections import Counter
        char_freq = Counter(ciphertext.upper())
        most_common_cipher = [char for char, _ in char_freq.most_common(5)]
        
        # Portuguese most common letters
        portuguese_common = ['A', 'E', 'O', 'S', 'R']
        
        logger.info(f"Most common in cipher: {most_common_cipher}")
        logger.info(f"Most common in Portuguese: {portuguese_common}")
        
        generated = 0
        
        # Generate keys that might map common cipher letters to common Portuguese letters
        for attempt in range(max_keys * 10):  # Safety multiplier
            if generated >= max_keys:
                break
            
            key = np.random.randint(0, 26, size=(self.key_size, self.key_size))
            
            # Apply frequency-based bias
            if attempt % 3 == 0:  # Every third key gets frequency bias
                for i in range(self.key_size):
                    for j in range(self.key_size):
                        if np.random.random() < 0.4:  # 40% chance to apply bias
                            # Bias towards values that might help with frequency mapping
                            cipher_idx = i % len(most_common_cipher)
                            port_idx = j % len(portuguese_common)
                            
                            cipher_val = ord(most_common_cipher[cipher_idx]) - ord('A')
                            port_val = ord(portuguese_common[port_idx]) - ord('A')
                            
                            # Use difference as bias
                            bias_val = (cipher_val - port_val) % 26
                            key[i, j] = bias_val
            
            if self.hill_cipher.is_invertible(key):
                yield key
                generated += 1
    
    def generate_keys_smart_sampling(self, sample_size: int = 50000) -> Iterator[np.ndarray]:
        """
        Generate keys using smart sampling techniques.
        
        Args:
            sample_size: Number of keys to generate
            
        Yields:
            Smartly sampled keys
        """
        logger.info(f"Generating {sample_size} keys using smart sampling")
        
        generated = 0
        attempts = 0
        max_attempts = sample_size * 20  # Safety limit
        
        strategies = [
            self._sample_small_values,
            self._sample_coprime_heavy,
            self._sample_pattern_based,
            self._sample_random
        ]
        
        while generated < sample_size and attempts < max_attempts:
            attempts += 1
            
            # Choose strategy
            strategy = strategies[attempts % len(strategies)]
            key = strategy()
            
            if self.hill_cipher.is_invertible(key):
                yield key
                generated += 1
                
                if generated % 5000 == 0:
                    logger.info(f"Generated {generated}/{sample_size} smart keys...")
    
    def _sample_small_values(self) -> np.ndarray:
        """Sample keys with preference for smaller values."""
        # Bias towards smaller values (0-10)
        key = np.random.choice(range(11), size=(self.key_size, self.key_size))
        return key
    
    def _sample_coprime_heavy(self) -> np.ndarray:
        """Sample keys with preference for coprime values."""
        # Higher probability for coprime values
        values = self.coprime_values + self.coprime_values + list(range(26))  # Double weight for coprimes
        key = np.random.choice(values, size=(self.key_size, self.key_size))
        return key
    
    def _sample_pattern_based(self) -> np.ndarray:
        """Sample keys with some structural patterns."""
        key = np.random.randint(0, 26, size=(self.key_size, self.key_size))
        
        # Apply some patterns
        if np.random.random() < 0.3:  # 30% chance for diagonal emphasis
            for i in range(self.key_size):
                key[i, i] = np.random.choice(self.coprime_values)
        
        if np.random.random() < 0.2:  # 20% chance for some zeros
            num_zeros = np.random.randint(1, self.key_size)
            for _ in range(num_zeros):
                i, j = np.random.randint(0, self.key_size, 2)
                if i != j:  # Don't zero diagonal elements
                    key[i, j] = 0
        
        return key
    
    def _sample_random(self) -> np.ndarray:
        """Sample completely random keys."""
        return np.random.randint(0, 26, size=(self.key_size, self.key_size))
    
    def estimate_reduction_factor(self, technique: str) -> float:
        """
        Estimate how much a technique reduces the search space.
        
        Args:
            technique: Name of the reduction technique
            
        Returns:
            Reduction factor (original_space / reduced_space)
        """
        total_space = self.estimate_total_search_space()
        
        if technique == "determinant_constraint":
            # Only coprime determinants are valid
            reduction = 26 / len(self.coprime_values)
            return reduction
        
        elif technique == "upper_triangular":
            # Upper triangular matrices
            n = self.key_size
            diagonal_choices = 25  # Non-zero diagonal elements
            upper_choices = 26 ** (n * (n - 1) // 2)
            reduced_space = (diagonal_choices ** n) * upper_choices
            return total_space / reduced_space
        
        elif technique == "diagonal":
            # Diagonal matrices only
            reduced_space = len(self.coprime_values) ** self.key_size
            return total_space / reduced_space
        
        elif technique == "sparse":
            # Rough estimate for sparse matrices
            return 100  # Very rough estimate
        
        else:
            return 1.0  # No reduction
    
    def get_reduction_recommendations(self) -> List[Tuple[str, str, float]]:
        """
        Get recommendations for search space reduction techniques.
        
        Returns:
            List of (technique_name, description, estimated_reduction_factor)
        """
        recommendations = [
            (
                "determinant_constraint",
                "Focus on matrices with coprime determinants only",
                self.estimate_reduction_factor("determinant_constraint")
            ),
            (
                "structural_constraints",
                "Use triangular, diagonal, or sparse matrices",
                self.estimate_reduction_factor("upper_triangular")
            ),
            (
                "frequency_based",
                "Generate keys based on ciphertext frequency analysis",
                10.0  # Estimated
            ),
            (
                "smart_sampling",
                "Use biased sampling towards likely key patterns",
                5.0   # Estimated
            ),
            (
                "early_stopping",
                "Stop search when good enough solution is found",
                float('inf')  # Can provide infinite speedup
            )
        ]
        
        return recommendations

def main():
    """Main function for testing the search space reducer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hill Cipher Search Space Reducer")
    parser.add_argument("--key-size", type=int, default=3, choices=[2, 3, 4, 5], 
                       help="Size of the key matrix")
    parser.add_argument("--technique", type=str, 
                       choices=["determinant", "structural", "frequency", "smart"],
                       default="smart", help="Reduction technique to test")
    parser.add_argument("--sample-size", type=int, default=1000, 
                       help="Number of keys to generate for testing")
    parser.add_argument("--ciphertext", type=str, 
                       help="Ciphertext for frequency-based generation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create reducer
    reducer = SearchSpaceReducer(args.key_size)
    
    # Show recommendations
    print(f"Search Space Reduction Recommendations for {args.key_size}x{args.key_size} Hill Cipher")
    print("=" * 70)
    
    recommendations = reducer.get_reduction_recommendations()
    for technique, description, reduction in recommendations:
        print(f"{technique:20s}: {description}")
        print(f"{'':20s}  Estimated reduction factor: {reduction:.1f}x")
        print()
    
    # Test the specified technique
    print(f"Testing {args.technique} technique with {args.sample_size} samples...")
    
    start_time = time.time()
    count = 0
    
    if args.technique == "determinant":
        generator = reducer.generate_keys_by_determinant_constraint()
    elif args.technique == "structural":
        generator = reducer.generate_keys_with_structure_constraints()
    elif args.technique == "frequency" and args.ciphertext:
        generator = reducer.generate_keys_frequency_based(args.ciphertext, args.sample_size)
    elif args.technique == "smart":
        generator = reducer.generate_keys_smart_sampling(args.sample_size)
    else:
        print("Invalid technique or missing ciphertext for frequency-based generation")
        return
    
    for key in generator:
        count += 1
        if count <= 5:  # Show first 5 keys
            print(f"Key {count}: {key.flatten()}")
        if count >= args.sample_size:
            break
    
    elapsed = time.time() - start_time
    print(f"\nGenerated {count} valid keys in {elapsed:.2f} seconds")
    print(f"Rate: {count/elapsed:.1f} keys/second")

if __name__ == "__main__":
    main()
