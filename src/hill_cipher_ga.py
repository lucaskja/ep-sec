#!/usr/bin/env python3
"""
Hill Cipher Genetic Algorithm-based Frequency Analysis

This module implements a genetic algorithm approach to break the Hill cipher
using n-gram frequency analysis. It evolves a population of candidate keys
and evaluates them based on how well the decrypted text matches expected
language statistics.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import numpy as np
import random
import logging
import json
import time
import math
from typing import List, Tuple, Dict, Optional, Union, Callable
from collections import Counter
import multiprocessing as mp
from functools import partial

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hill_cipher_ga')

class HillCipherGA:
    """
    Hill Cipher Genetic Algorithm-based Frequency Analysis.
    
    This class implements a genetic algorithm to break the Hill cipher
    using n-gram frequency analysis.
    """
    
    def __init__(self, key_size: int, language_frequencies: Dict[str, Dict[str, float]]):
        """
        Initialize the Hill Cipher GA solver.
        
        Args:
            key_size: Size of the Hill cipher key matrix (NxN)
            language_frequencies: Dictionary of n-gram frequencies for the target language
        """
        self.key_size = key_size
        self.modulus = 26  # For English/Portuguese alphabet
        self.language_frequencies = language_frequencies
        self.best_key = None
        self.best_fitness = float('-inf')
        self.best_decryption = None
        
        # GA parameters
        self.population_size = 100
        self.elite_size = 10
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8
        self.tournament_size = 5
        
        logger.info(f"Initialized Hill Cipher GA solver with key size {key_size}x{key_size}")
    
    def text_to_numbers(self, text: str) -> List[int]:
        """
        Convert text to numerical values (A=0, B=1, ..., Z=25).
        
        Args:
            text: Input text (uppercase letters only)
            
        Returns:
            List of numerical values
        """
        text = text.upper()
        return [ord(char) - ord('A') for char in text if 'A' <= char <= 'Z']
    
    def numbers_to_text(self, numbers: List[int]) -> str:
        """
        Convert numerical values back to text.
        
        Args:
            numbers: List of numerical values
            
        Returns:
            Text representation
        """
        return ''.join([chr((n % self.modulus) + ord('A')) for n in numbers])
    
    def text_to_matrix(self, text: str) -> np.ndarray:
        """
        Convert text to numerical matrix suitable for Hill cipher operations.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array with shape (n, key_size) where n is the number of blocks
        """
        numbers = self.text_to_numbers(text)
        
        # Ensure the length is a multiple of key_size
        if len(numbers) % self.key_size != 0:
            # Pad with 'X' (23) if needed
            padding = self.key_size - (len(numbers) % self.key_size)
            numbers.extend([23] * padding)
            logger.debug(f"Padded input with {padding} 'X' characters")
        
        # Reshape into matrix with key_size columns
        return np.array(numbers).reshape(-1, self.key_size)
    
    def matrix_to_text(self, matrix: np.ndarray) -> str:
        """
        Convert numerical matrix back to text.
        
        Args:
            matrix: Numpy array with numerical values
            
        Returns:
            Text representation
        """
        numbers = matrix.flatten()
        return self.numbers_to_text(numbers)
    
    def mod_inverse(self, a: int, m: int = 26) -> int:
        """
        Calculate the modular multiplicative inverse of a number.
        
        Args:
            a: Number to find inverse for
            m: Modulus (default: 26)
            
        Returns:
            Modular multiplicative inverse
            
        Raises:
            ValueError: If the inverse doesn't exist
        """
        for i in range(1, m):
            if (a * i) % m == 1:
                return i
        raise ValueError(f"Modular inverse of {a} mod {m} does not exist")
    
    def matrix_mod_inverse(self, matrix: np.ndarray, mod: int = 26) -> np.ndarray:
        """
        Calculate the modular multiplicative inverse of a matrix.
        
        Args:
            matrix: Square matrix to invert
            mod: Modulus (default: 26)
            
        Returns:
            Inverted matrix in the given modulus
            
        Raises:
            ValueError: If the matrix is not invertible in the given modulus
        """
        n = matrix.shape[0]
        
        # Calculate determinant and ensure it's invertible
        det = round(np.linalg.det(matrix)) % mod
        try:
            det_inv = self.mod_inverse(det, mod)
        except ValueError:
            raise ValueError("Matrix is not invertible mod 26")
        
        # For 2x2 matrix, use the simple formula
        if n == 2:
            adj = np.array([
                [matrix[1, 1], -matrix[0, 1]],
                [-matrix[1, 0], matrix[0, 0]]
            ]) % mod
            return (det_inv * adj) % mod
        
        # For larger matrices, use the adjugate method
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                # Get the minor by removing row i and column j
                minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
                # Calculate the cofactor
                cofactor = round(np.linalg.det(minor)) * (-1) ** (i + j)
                adj[j, i] = cofactor % mod  # Note the transpose here (j,i)
        
        return (det_inv * adj) % mod
    
    def decrypt(self, ciphertext: str, key: np.ndarray) -> str:
        """
        Decrypt ciphertext using the Hill cipher.
        
        Args:
            ciphertext: Text to decrypt
            key: Key matrix
            
        Returns:
            Decrypted text
        """
        try:
            # Convert ciphertext to matrix
            C = self.text_to_matrix(ciphertext)
            
            # Calculate inverse key
            K_inv = self.matrix_mod_inverse(key)
            
            # Decrypt: P = C * K^(-1)
            P = (C @ K_inv) % self.modulus
            
            # Convert back to text
            return self.matrix_to_text(P)
        except Exception as e:
            logger.debug(f"Decryption error: {e}")
            return ""
    
    def is_invertible(self, matrix: np.ndarray) -> bool:
        """
        Check if a matrix is invertible in modulo 26.
        
        Args:
            matrix: Matrix to check
            
        Returns:
            True if invertible, False otherwise
        """
        try:
            det = int(round(np.linalg.det(matrix))) % self.modulus
            return math.gcd(det, self.modulus) == 1
        except:
            return False
    
    def generate_random_key(self) -> np.ndarray:
        """
        Generate a random invertible key matrix.
        
        Returns:
            Random invertible key matrix
        """
        while True:
            # Generate random matrix
            key = np.random.randint(0, self.modulus, (self.key_size, self.key_size))
            
            # Check if invertible
            if self.is_invertible(key):
                return key
    
    def extract_ngrams(self, text: str, n: int) -> Dict[str, float]:
        """
        Extract n-gram frequencies from text.
        
        Args:
            text: Input text
            n: Size of n-grams
            
        Returns:
            Dictionary mapping n-grams to their frequencies
        """
        # Extract n-grams
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        
        # Count n-grams
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        
        # Calculate frequencies
        return {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    
    def calculate_fitness(self, key: np.ndarray, ciphertext: str) -> float:
        """
        Calculate fitness of a key based on n-gram frequency match.
        
        Args:
            key: Key matrix to evaluate
            ciphertext: Encrypted text
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Decrypt ciphertext
            decrypted = self.decrypt(ciphertext, key)
            if not decrypted:
                return float('-inf')
            
            # Calculate fitness based on n-gram frequencies
            fitness = 0
            
            # Check 1-grams (letters)
            if '1' in self.language_frequencies:
                decrypted_1grams = self.extract_ngrams(decrypted, 1)
                for ngram, freq in decrypted_1grams.items():
                    expected_freq = self.language_frequencies['1'].get(ngram, 0.0001)
                    # Penalize less common n-grams more
                    fitness += min(freq, expected_freq) / max(freq, expected_freq)
            
            # Check 2-grams
            if '2' in self.language_frequencies:
                decrypted_2grams = self.extract_ngrams(decrypted, 2)
                for ngram, freq in decrypted_2grams.items():
                    expected_freq = self.language_frequencies['2'].get(ngram, 0.0001)
                    # 2-grams are more important than 1-grams
                    fitness += 2 * min(freq, expected_freq) / max(freq, expected_freq)
            
            # Check 3-grams
            if '3' in self.language_frequencies and len(decrypted) >= 3:
                decrypted_3grams = self.extract_ngrams(decrypted, 3)
                for ngram, freq in decrypted_3grams.items():
                    expected_freq = self.language_frequencies['3'].get(ngram, 0.0001)
                    # 3-grams are more important than 2-grams
                    fitness += 3 * min(freq, expected_freq) / max(freq, expected_freq)
            
            # Check vowel ratio (Portuguese has ~46% vowels)
            vowels = sum(1 for c in decrypted if c in 'AEIOU')
            vowel_ratio = vowels / len(decrypted) if decrypted else 0
            if 0.4 <= vowel_ratio <= 0.5:
                fitness += 10
            elif 0.35 <= vowel_ratio <= 0.55:
                fitness += 5
            
            # Update best key if this one is better
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_key = key.copy()
                self.best_decryption = decrypted
                logger.info(f"New best fitness: {fitness:.2f}")
                logger.info(f"Decryption sample: {decrypted[:50]}...")
            
            return fitness
        except Exception as e:
            logger.debug(f"Fitness calculation error: {e}")
            return float('-inf')
    
    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        """
        Select a parent using tournament selection.
        
        Args:
            population: List of candidate keys
            fitnesses: List of fitness scores
            
        Returns:
            Selected parent
        """
        # Select tournament_size individuals randomly
        tournament_indices = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        # Select the best individual from the tournament
        winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        return population[winner_idx]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child matrix
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        # Create child matrix
        child = np.zeros((self.key_size, self.key_size), dtype=int)
        
        # Choose crossover type
        crossover_type = random.choice(['row', 'column', 'element'])
        
        if crossover_type == 'row':
            # Row-wise crossover
            crossover_point = random.randint(1, self.key_size - 1)
            child[:crossover_point] = parent1[:crossover_point]
            child[crossover_point:] = parent2[crossover_point:]
        elif crossover_type == 'column':
            # Column-wise crossover
            crossover_point = random.randint(1, self.key_size - 1)
            child[:, :crossover_point] = parent1[:, :crossover_point]
            child[:, crossover_point:] = parent2[:, crossover_point:]
        else:
            # Element-wise crossover
            for i in range(self.key_size):
                for j in range(self.key_size):
                    child[i, j] = parent1[i, j] if random.random() < 0.5 else parent2[i, j]
        
        # Ensure the child is invertible
        if not self.is_invertible(child):
            # If not invertible, return one of the parents
            return parent1.copy() if random.random() < 0.5 else parent2.copy()
        
        return child
    
    def mutate(self, matrix: np.ndarray) -> np.ndarray:
        """
        Mutate a matrix.
        
        Args:
            matrix: Matrix to mutate
            
        Returns:
            Mutated matrix
        """
        # Create a copy of the matrix
        mutated = matrix.copy()
        
        # Determine mutation type
        mutation_type = random.choice(['single', 'row', 'column', 'swap'])
        
        if mutation_type == 'single':
            # Mutate a single element
            i = random.randint(0, self.key_size - 1)
            j = random.randint(0, self.key_size - 1)
            # Ensure the new value is different
            original_value = mutated[i, j]
            new_value = random.randint(0, self.modulus - 1)
            while new_value == original_value:
                new_value = random.randint(0, self.modulus - 1)
            mutated[i, j] = new_value
        elif mutation_type == 'row':
            # Mutate a row
            i = random.randint(0, self.key_size - 1)
            mutated[i] = np.random.randint(0, self.modulus, self.key_size)
        elif mutation_type == 'column':
            # Mutate a column
            j = random.randint(0, self.key_size - 1)
            mutated[:, j] = np.random.randint(0, self.modulus, self.key_size)
        else:
            # Swap two elements
            i1, j1 = random.randint(0, self.key_size - 1), random.randint(0, self.key_size - 1)
            i2, j2 = random.randint(0, self.key_size - 1), random.randint(0, self.key_size - 1)
            # Ensure we're swapping different elements
            while i1 == i2 and j1 == j2:
                i2, j2 = random.randint(0, self.key_size - 1), random.randint(0, self.key_size - 1)
            mutated[i1, j1], mutated[i2, j2] = mutated[i2, j2], mutated[i1, j1]
        
        # Ensure the mutated matrix is invertible
        if not self.is_invertible(mutated):
            return matrix.copy()
        
        return mutated
    
    def evolve_population(self, population: List[np.ndarray], fitnesses: List[float]) -> List[np.ndarray]:
        """
        Evolve the population to the next generation.
        
        Args:
            population: Current population
            fitnesses: Fitness scores for the population
            
        Returns:
            New population
        """
        # Sort population by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        
        # Keep elite individuals
        new_population = sorted_population[:self.elite_size]
        
        # Fill the rest of the population with offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.tournament_selection(population, fitnesses)
            parent2 = self.tournament_selection(population, fitnesses)
            
            # Create offspring
            child = self.crossover(parent1, parent2)
            
            # Mutate with probability mutation_rate
            if random.random() < self.mutation_rate:
                child = self.mutate(child)
            
            # Add to new population
            new_population.append(child)
        
        return new_population
    
    def evaluate_population(self, population: List[np.ndarray], ciphertext: str) -> List[float]:
        """
        Evaluate the fitness of each individual in the population.
        
        Args:
            population: List of candidate keys
            ciphertext: Encrypted text
            
        Returns:
            List of fitness scores
        """
        # Use multiprocessing for parallel evaluation
        with mp.Pool() as pool:
            fitnesses = pool.map(partial(self.calculate_fitness, ciphertext=ciphertext), population)
        
        return fitnesses
    
    def crack(self, ciphertext: str, generations: int = 100, early_stopping: int = 20) -> Tuple[Optional[np.ndarray], str]:
        """
        Attempt to crack the Hill cipher using genetic algorithm.
        
        Args:
            ciphertext: Encrypted text
            generations: Maximum number of generations
            early_stopping: Stop if no improvement after this many generations
            
        Returns:
            Tuple of (best key matrix, decrypted text)
        """
        logger.info(f"Starting GA-based attack on {self.key_size}x{self.key_size} Hill cipher")
        logger.info(f"Ciphertext length: {len(ciphertext)} characters")
        
        # Initialize population
        population = [self.generate_random_key() for _ in range(self.population_size)]
        
        # Reset best solution
        self.best_key = None
        self.best_fitness = float('-inf')
        self.best_decryption = None
        
        # Evolution loop
        start_time = time.time()
        no_improvement_count = 0
        prev_best_fitness = float('-inf')
        
        for generation in range(generations):
            # Evaluate population
            fitnesses = self.evaluate_population(population, ciphertext)
            
            # Log progress
            max_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            logger.info(f"Generation {generation+1}/{generations}: Max fitness = {max_fitness:.2f}, Avg fitness = {avg_fitness:.2f}")
            
            # Check for early stopping
            if self.best_fitness > prev_best_fitness:
                prev_best_fitness = self.best_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= early_stopping:
                logger.info(f"Early stopping after {generation+1} generations (no improvement for {early_stopping} generations)")
                break
            
            # Evolve population
            population = self.evolve_population(population, fitnesses)
        
        elapsed_time = time.time() - start_time
        logger.info(f"GA completed in {elapsed_time:.2f} seconds")
        
        if self.best_key is not None:
            logger.info(f"Best fitness: {self.best_fitness:.2f}")
            logger.info(f"Best key:\n{self.best_key}")
            logger.info(f"Decryption sample: {self.best_decryption[:50]}...")
            return self.best_key, self.best_decryption
        else:
            logger.warning("Failed to find a valid key")
            return None, ""

def load_language_frequencies(language: str = 'portuguese') -> Dict[str, Dict[str, float]]:
    """
    Load language n-gram frequencies from files.
    
    Args:
        language: Language to load frequencies for
        
    Returns:
        Dictionary of n-gram frequencies
    """
    frequencies = {}
    
    # Load letter frequencies
    try:
        with open(f"data/letter_frequencies.json", 'r') as f:
            frequencies['1'] = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load letter frequencies: {e}")
    
    # Load 2-gram frequencies
    try:
        with open(f"data/2gram_frequencies.json", 'r') as f:
            frequencies['2'] = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load 2-gram frequencies: {e}")
    
    # Load 3-gram frequencies
    try:
        with open(f"data/3gram_frequencies.json", 'r') as f:
            frequencies['3'] = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load 3-gram frequencies: {e}")
    
    return frequencies

def main():
    """Main function for demonstration and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hill Cipher Genetic Algorithm-based Frequency Analysis")
    parser.add_argument("--key-size", type=int, default=2, choices=[2, 3, 4, 5], help="Size of the key matrix")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext to decrypt")
    parser.add_argument("--ciphertext-file", type=str, help="File containing ciphertext")
    parser.add_argument("--generations", type=int, default=100, help="Maximum number of generations")
    parser.add_argument("--population-size", type=int, default=100, help="Population size")
    parser.add_argument("--early-stopping", type=int, default=20, help="Stop if no improvement after this many generations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get ciphertext
    ciphertext = args.ciphertext
    
    if args.ciphertext_file:
        with open(args.ciphertext_file, 'r') as f:
            ciphertext = f.read().strip()
    
    if not ciphertext:
        parser.error("Ciphertext must be provided")
    
    # Load language frequencies
    language_frequencies = load_language_frequencies()
    
    # Create GA solver
    ga = HillCipherGA(args.key_size, language_frequencies)
    ga.population_size = args.population_size
    
    # Run GA
    key, decrypted = ga.crack(ciphertext, args.generations, args.early_stopping)
    
    if key is not None:
        print(f"Recovered {args.key_size}x{args.key_size} key matrix:")
        print(key)
        print(f"Decrypted text (first 100 characters):")
        print(decrypted[:100])
        
        # Save results
        os.makedirs("results", exist_ok=True)
        with open(f"results/ga_key_{args.key_size}x{args.key_size}.txt", 'w') as f:
            f.write(str(key))
        with open(f"results/ga_decrypted_{args.key_size}x{args.key_size}.txt", 'w') as f:
            f.write(decrypted)
        
        print(f"Results saved to results/ga_key_{args.key_size}x{args.key_size}.txt and results/ga_decrypted_{args.key_size}x{args.key_size}.txt")
    else:
        print("Failed to recover key")

if __name__ == "__main__":
    main()
