#!/usr/bin/env python3
"""
Unit tests for Hill Cipher Core Implementation.
"""

import os
import sys
import unittest
import numpy as np

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hill_cipher import HillCipher

class TestHillCipher(unittest.TestCase):
    """Test cases for Hill Cipher Core Implementation."""
    
    def setUp(self):
        """Set up test cases."""
        # Create Hill cipher instances for different key sizes
        self.hill_2x2 = HillCipher(2)
        self.hill_3x3 = HillCipher(3)
        
        # Define test keys
        self.key_2x2 = np.array([[3, 2], [5, 3]])  # det = 9 - 10 = -1 = 25 (mod 26), which is coprime to 26
        self.key_3x3 = np.array([[6, 24, 1], [13, 16, 10], [20, 17, 15]])
    
    def test_text_to_numbers(self):
        """Test conversion from text to numbers."""
        text = "HELLO"
        expected = [7, 4, 11, 11, 14]
        result = self.hill_2x2.text_to_numbers(text)
        self.assertEqual(result, expected)
    
    def test_numbers_to_text(self):
        """Test conversion from numbers to text."""
        numbers = [7, 4, 11, 11, 14]
        expected = "HELLO"
        result = self.hill_2x2.numbers_to_text(numbers)
        self.assertEqual(result, expected)
    
    def test_text_to_matrix(self):
        """Test conversion from text to matrix."""
        text = "HELLO"
        expected = np.array([[7, 4], [11, 11], [14, 23]])  # Note: Padded with X (23)
        result = self.hill_2x2.text_to_matrix(text)
        np.testing.assert_array_equal(result, expected)
    
    def test_matrix_to_text(self):
        """Test conversion from matrix to text."""
        matrix = np.array([[7, 4], [11, 11], [14, 23]])
        expected = "HELLOX"
        result = self.hill_2x2.matrix_to_text(matrix)
        self.assertEqual(result, expected)
    
    def test_mod_inverse(self):
        """Test modular multiplicative inverse."""
        # 7 * 15 = 105 = 1 (mod 26)
        self.assertEqual(self.hill_2x2.mod_inverse(7), 15)
        
        # 3 has no inverse mod 26
        with self.assertRaises(ValueError):
            self.hill_2x2.mod_inverse(13)
    
    def test_matrix_mod_inverse_2x2(self):
        """Test matrix modular inverse for 2x2 matrix."""
        # Use a matrix with determinant coprime to 26
        matrix = np.array([[3, 2], [5, 3]])  # det = 9 - 10 = -1 = 25 (mod 26), which is coprime to 26
        
        # Calculate expected inverse
        # For a 2x2 matrix [[a, b], [c, d]], the inverse mod 26 is:
        # det = (a*d - b*c) mod 26
        # det_inv = modular inverse of det mod 26
        # inverse = det_inv * [[d, -b], [-c, a]] mod 26
        det = (3*3 - 2*5) % 26  # = 9 - 10 = -1 = 25 (mod 26)
        det_inv = self.hill_2x2.mod_inverse(det)  # = 25 (since 25*25 = 625 = 1 (mod 26))
        expected_inverse = (det_inv * np.array([[3, -2], [-5, 3]])) % 26
        expected_inverse = np.array([[3*25, -2*25], [-5*25, 3*25]]) % 26
        expected_inverse = np.array([[75, -50], [-125, 75]]) % 26
        expected_inverse = np.array([[23, 2], [5, 23]]) % 26
        
        result = self.hill_2x2.matrix_mod_inverse(matrix)
        np.testing.assert_array_equal(result, expected_inverse)
    
    def test_is_invertible(self):
        """Test invertibility check."""
        # Invertible matrix
        matrix1 = np.array([[3, 2], [5, 3]])  # det = 9 - 10 = -1 = 25 (mod 26), which is coprime to 26
        self.assertTrue(self.hill_2x2.is_invertible(matrix1))
        
        # Non-invertible matrix
        matrix2 = np.array([[2, 4], [1, 2]])  # det = 4 - 4 = 0, which is not coprime to 26
        self.assertFalse(self.hill_2x2.is_invertible(matrix2))
    
    def test_encrypt_decrypt_2x2(self):
        """Test encryption and decryption with 2x2 matrix."""
        plaintext = "HELLOWORLD"
        
        # Encrypt
        ciphertext = self.hill_2x2.encrypt(plaintext, self.key_2x2)
        
        # Decrypt
        decrypted = self.hill_2x2.decrypt(ciphertext, self.key_2x2)
        
        # Check if decryption matches original plaintext
        self.assertEqual(decrypted, plaintext)
    
    def test_encrypt_decrypt_3x3(self):
        """Test encryption and decryption with 3x3 matrix."""
        plaintext = "HELLOWORLDHOWAREYOU"
        
        # Encrypt
        ciphertext = self.hill_3x3.encrypt(plaintext, self.key_3x3)
        
        # Decrypt
        decrypted = self.hill_3x3.decrypt(ciphertext, self.key_3x3)
        
        # Check if decryption matches original plaintext
        self.assertEqual(decrypted, plaintext)
    
    def test_non_invertible_key(self):
        """Test error handling for non-invertible key."""
        plaintext = "HELLO"
        non_invertible_key = np.array([[2, 4], [1, 2]])  # det = 0
        
        with self.assertRaises(ValueError):
            self.hill_2x2.encrypt(plaintext, non_invertible_key)

if __name__ == "__main__":
    unittest.main()
