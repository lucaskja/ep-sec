#!/usr/bin/env python3
"""
Unit tests for Hill Cipher Known-Plaintext Attack.
"""

import os
import sys
import unittest
import numpy as np

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakers.kpa import HillCipherKPA

class TestHillCipherKPA(unittest.TestCase):
    """Test cases for Hill Cipher Known-Plaintext Attack."""
    
    def setUp(self):
        """Set up test cases."""
        # Create KPA solvers for different key sizes
        self.kpa_2x2 = HillCipherKPA(2)
        self.kpa_3x3 = HillCipherKPA(3)
        
        # Define test keys
        self.key_2x2 = np.array([[3, 2], [5, 3]])  # det = 9 - 10 = -1 = 25 (mod 26), which is coprime to 26
        self.key_3x3 = np.array([[6, 24, 1], [13, 16, 10], [20, 17, 15]])
    
    def test_recover_key_2x2(self):
        """Test key recovery for 2x2 matrix."""
        # Create plaintext and corresponding ciphertext
        plaintext = "HELLOWORLD"
        
        # Encrypt manually
        P = self.kpa_2x2.text_to_matrix(plaintext)
        C = (P @ self.key_2x2) % 26
        ciphertext = self.kpa_2x2.matrix_to_text(C)
        
        # Recover key
        recovered_key = self.kpa_2x2.recover_key(plaintext, ciphertext)
        
        # Check if recovered key is correct
        np.testing.assert_array_equal(recovered_key, self.key_2x2)
    
    def test_recover_key_3x3(self):
        """Test key recovery for 3x3 matrix."""
        # Create plaintext and corresponding ciphertext
        plaintext = "HELLOWORLDHOWAREYOU"
        
        # Encrypt manually
        P = self.kpa_3x3.text_to_matrix(plaintext)
        C = (P @ self.key_3x3) % 26
        ciphertext = self.kpa_3x3.matrix_to_text(C)
        
        # Recover key
        recovered_key = self.kpa_3x3.recover_key(plaintext, ciphertext)
        
        # Check if recovered key is correct
        np.testing.assert_array_equal(recovered_key, self.key_3x3)
    
    def test_insufficient_data(self):
        """Test error handling for insufficient data."""
        # Not enough plaintext-ciphertext pairs for 2x2 key
        plaintext = "HEL"
        ciphertext = "XYZ"
        
        with self.assertRaises(ValueError):
            self.kpa_2x2.recover_key(plaintext, ciphertext)
    
    def test_verify_key(self):
        """Test key verification."""
        # Create plaintext and corresponding ciphertext
        plaintext = "HELLOWORLD"
        
        # Encrypt manually
        P = self.kpa_2x2.text_to_matrix(plaintext)
        C = (P @ self.key_2x2) % 26
        ciphertext = self.kpa_2x2.matrix_to_text(C)
        
        # Verify correct key
        self.assertTrue(self.kpa_2x2.verify_key(self.key_2x2, plaintext, ciphertext))
        
        # Verify incorrect key
        incorrect_key = np.array([[1, 2], [3, 4]])
        self.assertFalse(self.kpa_2x2.verify_key(incorrect_key, plaintext, ciphertext))

if __name__ == "__main__":
    unittest.main()
