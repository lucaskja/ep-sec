#!/usr/bin/env python3
"""
Unit tests for Hill Cipher Genetic Algorithm-based Frequency Analysis.
"""

import os
import sys
import unittest
import numpy as np

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakers.genetic import HillCipherGA

class TestHillCipherGA(unittest.TestCase):
    """Test cases for Hill Cipher Genetic Algorithm-based Frequency Analysis."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a simple language frequency model for testing
        self.language_frequencies = {
            '1': {'A': 0.15, 'E': 0.13, 'O': 0.10, 'S': 0.07, 'R': 0.06},
            '2': {'DE': 0.035, 'OS': 0.030, 'ES': 0.028, 'RA': 0.026, 'EN': 0.023},
            '3': {'QUE': 0.018, 'ENT': 0.014, 'COM': 0.013, 'ROS': 0.011, 'IST': 0.010}
        }
        
        # Create GA solvers for different key sizes
        self.ga_2x2 = HillCipherGA(2, self.language_frequencies)
        
        # Define test keys
        self.key_2x2 = np.array([[3, 2], [5, 3]])  # det = 9 - 10 = -1 = 25 (mod 26), which is coprime to 26
    
    def test_extract_ngrams(self):
        """Test n-gram extraction."""
        text = "HELLO"
        
        # 1-grams
        expected_1grams = {'H': 0.2, 'E': 0.2, 'L': 0.4, 'O': 0.2}
        result_1grams = self.ga_2x2.extract_ngrams(text, 1)
        self.assertEqual(result_1grams, expected_1grams)
        
        # 2-grams
        expected_2grams = {'HE': 0.25, 'EL': 0.25, 'LL': 0.25, 'LO': 0.25}
        result_2grams = self.ga_2x2.extract_ngrams(text, 2)
        self.assertEqual(result_2grams, expected_2grams)
    
    def test_decrypt(self):
        """Test decryption."""
        # Create plaintext and encrypt it
        plaintext = "HELLOWORLD"
        
        # Use a key that is definitely invertible mod 26
        key = np.array([[3, 2], [5, 3]])  # det = 9 - 10 = -1 = 25 (mod 26), which is coprime to 26
        
        # Encrypt manually
        P = self.ga_2x2.text_to_matrix(plaintext)
        C = (P @ key) % 26
        ciphertext = self.ga_2x2.matrix_to_text(C)
        
        # Decrypt using the key
        decrypted = self.ga_2x2.decrypt(ciphertext, key)
        
        # Check if decryption is correct (note: the padding may be removed in the process)
        self.assertTrue(decrypted.startswith(plaintext))
    
    def test_generate_random_key(self):
        """Test random key generation."""
        key = self.ga_2x2.generate_random_key()
        
        # Check dimensions
        self.assertEqual(key.shape, (2, 2))
        
        # Check invertibility
        self.assertTrue(self.ga_2x2.is_invertible(key))
    
    def test_crossover(self):
        """Test crossover operation."""
        parent1 = np.array([[3, 2], [5, 3]])
        parent2 = np.array([[7, 8], [11, 3]])
        
        # Force crossover
        self.ga_2x2.crossover_rate = 1.0
        
        # Perform crossover
        child = self.ga_2x2.crossover(parent1, parent2)
        
        # Check dimensions
        self.assertEqual(child.shape, (2, 2))
        
        # Check that child is invertible
        self.assertTrue(self.ga_2x2.is_invertible(child))
    
    def test_mutate(self):
        """Test mutation operation."""
        # Use a matrix that is invertible mod 26
        matrix = np.array([[3, 2], [5, 3]])
        
        # Force mutation to always happen
        self.ga_2x2.mutation_rate = 1.0
        
        # Try multiple times to account for randomness
        success = False
        for _ in range(10):
            mutated = self.ga_2x2.mutate(matrix)
            if not np.array_equal(mutated, matrix):
                success = True
                break
        
        self.assertTrue(success, "Mutation should change the matrix at least once in 10 attempts")
        
        # Check that mutated matrix is invertible
        self.assertTrue(self.ga_2x2.is_invertible(mutated))
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        population = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]])
        ]
        fitnesses = [10, 20, 30]
        
        # Set tournament size to population size for deterministic testing
        self.ga_2x2.tournament_size = len(population)
        
        # Perform tournament selection
        selected = self.ga_2x2.tournament_selection(population, fitnesses)
        
        # Check that the best individual was selected
        np.testing.assert_array_equal(selected, population[2])
    
    def test_evolve_population(self):
        """Test population evolution."""
        population = [
            np.array([[3, 2], [5, 3]]),
            np.array([[7, 8], [11, 3]]),
            np.array([[1, 0], [0, 1]])
        ]
        fitnesses = [10, 20, 30]
        
        # Set parameters for deterministic testing
        self.ga_2x2.population_size = len(population)
        self.ga_2x2.elite_size = 1
        self.ga_2x2.mutation_rate = 0
        self.ga_2x2.crossover_rate = 0
        
        # Evolve population
        new_population = self.ga_2x2.evolve_population(population, fitnesses)
        
        # Check that population size is maintained
        self.assertEqual(len(new_population), len(population))
        
        # Check that the best individual is preserved (elitism)
        np.testing.assert_array_equal(new_population[0], population[2])

if __name__ == "__main__":
    unittest.main()
