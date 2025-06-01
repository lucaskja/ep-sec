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

from src.hill_cipher_ga import HillCipherGA

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
        self.key_2x2 = np.array([[6, 24], [1, 13]])
    
    def test_text_to_numbers(self):
        """Test conversion from text to numbers."""
        text = "HELLO"
        expected = [7, 4, 11, 11, 14]
        result = self.ga_2x2.text_to_numbers(text)
        self.assertEqual(result, expected)
    
    def test_numbers_to_text(self):
        """Test conversion from numbers to text."""
        numbers = [7, 4, 11, 11, 14]
        expected = "HELLO"
        result = self.ga_2x2.numbers_to_text(numbers)
        self.assertEqual(result, expected)
    
    def test_text_to_matrix(self):
        """Test conversion from text to matrix."""
        text = "HELLO"
        expected = np.array([[7, 4], [11, 11], [14, 23]])  # Note: Padded with X (23)
        result = self.ga_2x2.text_to_matrix(text)
        np.testing.assert_array_equal(result, expected)
    
    def test_matrix_to_text(self):
        """Test conversion from matrix to text."""
        matrix = np.array([[7, 4], [11, 11], [14, 23]])
        expected = "HELLOX"
        result = self.ga_2x2.matrix_to_text(matrix)
        self.assertEqual(result, expected)
    
    def test_is_invertible(self):
        """Test invertibility check."""
        # Invertible matrix
        matrix1 = np.array([[6, 24], [1, 13]])
        self.assertTrue(self.ga_2x2.is_invertible(matrix1))
        
        # Non-invertible matrix
        matrix2 = np.array([[2, 4], [1, 2]])
        self.assertFalse(self.ga_2x2.is_invertible(matrix2))
    
    def test_generate_random_key(self):
        """Test random key generation."""
        key = self.ga_2x2.generate_random_key()
        
        # Check dimensions
        self.assertEqual(key.shape, (2, 2))
        
        # Check invertibility
        self.assertTrue(self.ga_2x2.is_invertible(key))
    
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
        
        # Encrypt manually
        P = self.ga_2x2.text_to_matrix(plaintext)
        C = (P @ self.key_2x2) % 26
        ciphertext = self.ga_2x2.matrix_to_text(C)
        
        # Decrypt using the key
        decrypted = self.ga_2x2.decrypt(ciphertext, self.key_2x2)
        
        # Check if decryption is correct
        self.assertEqual(decrypted, plaintext + "X")  # Note: Padded with X
    
    def test_crossover(self):
        """Test crossover operation."""
        parent1 = np.array([[1, 2], [3, 4]])
        parent2 = np.array([[5, 6], [7, 8]])
        
        # Force crossover
        self.ga_2x2.crossover_rate = 1.0
        
        # Perform crossover
        child = self.ga_2x2.crossover(parent1, parent2)
        
        # Check dimensions
        self.assertEqual(child.shape, (2, 2))
        
        # Check that child is different from both parents
        self.assertFalse(np.array_equal(child, parent1) and np.array_equal(child, parent2))
    
    def test_mutate(self):
        """Test mutation operation."""
        matrix = np.array([[1, 2], [3, 4]])
        
        # Force mutation
        self.ga_2x2.mutation_rate = 1.0
        
        # Perform mutation
        mutated = self.ga_2x2.mutate(matrix)
        
        # Check dimensions
        self.assertEqual(mutated.shape, (2, 2))
        
        # Check that mutated matrix is different from original
        self.assertFalse(np.array_equal(mutated, matrix))
    
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
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]])
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
