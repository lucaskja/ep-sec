#!/usr/bin/env python3
"""
Hill Cipher Core Implementation

This module provides the core functionality for the Hill cipher,
including encryption and decryption operations.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Union
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hill_cipher')

class HillCipher:
    """
    Hill Cipher implementation.
    
    This class provides methods for encrypting and decrypting text
    using the Hill cipher algorithm.
    """
    
    def __init__(self, key_size: int):
        """
        Initialize the Hill Cipher.
        
        Args:
            key_size: Size of the Hill cipher key matrix (NxN)
        """
        self.key_size = key_size
        self.modulus = 26  # For English/Portuguese alphabet
        logger.info(f"Initialized Hill Cipher with key size {key_size}x{key_size}")
    
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
        det = int(round(np.linalg.det(matrix))) % mod
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
    
    def encrypt(self, plaintext: str, key: np.ndarray) -> str:
        """
        Encrypt plaintext using the Hill cipher.
        
        Args:
            plaintext: Text to encrypt
            key: Key matrix
            
        Returns:
            Encrypted text
        """
        # Check if key is valid
        if not self.is_invertible(key):
            raise ValueError("Key matrix is not invertible mod 26")
        
        # Convert plaintext to matrix
        P = self.text_to_matrix(plaintext)
        
        # Encrypt: C = P * K
        C = (P @ key) % self.modulus
        
        # Convert back to text
        return self.matrix_to_text(C)
    
    def decrypt(self, ciphertext: str, key: np.ndarray) -> str:
        """
        Decrypt ciphertext using the Hill cipher.
        
        Args:
            ciphertext: Text to decrypt
            key: Key matrix
            
        Returns:
            Decrypted text
        """
        # Convert ciphertext to matrix
        C = self.text_to_matrix(ciphertext)
        
        # Calculate inverse key
        K_inv = self.matrix_mod_inverse(key)
        
        # Decrypt: P = C * K^(-1)
        P = (C @ K_inv) % self.modulus
        
        # Convert back to text
        decrypted = self.matrix_to_text(P)
        
        # Remove padding if it exists
        # This is a heuristic - we assume padding is at the end and consists of 'X' characters
        # In a real application, you might want to use a more robust padding scheme
        if decrypted.endswith('X'):
            # Check if the last character is padding
            # For simplicity, we'll just remove trailing X's
            decrypted = decrypted.rstrip('X')
        
        return decrypted

def main():
    """Main function for demonstration and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hill Cipher Implementation")
    parser.add_argument("--key-size", type=int, default=2, choices=[2, 3, 4, 5], help="Size of the key matrix")
    parser.add_argument("--encrypt", action="store_true", help="Encrypt mode")
    parser.add_argument("--decrypt", action="store_true", help="Decrypt mode")
    parser.add_argument("--text", type=str, help="Text to encrypt/decrypt")
    parser.add_argument("--text-file", type=str, help="File containing text to encrypt/decrypt")
    parser.add_argument("--key", type=str, help="Key matrix as comma-separated values")
    parser.add_argument("--key-file", type=str, help="File containing key matrix")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check mode
    if not args.encrypt and not args.decrypt:
        parser.error("Either --encrypt or --decrypt must be specified")
    
    # Get text
    text = args.text
    if args.text_file:
        with open(args.text_file, 'r') as f:
            text = f.read().strip()
    
    if not text:
        parser.error("Text must be provided")
    
    # Get key
    key_matrix = None
    if args.key:
        # Parse key from command line
        key_values = [int(x) for x in args.key.split(',')]
        if len(key_values) != args.key_size * args.key_size:
            parser.error(f"Key must have {args.key_size * args.key_size} values")
        key_matrix = np.array(key_values).reshape(args.key_size, args.key_size)
    elif args.key_file:
        # Parse key from file
        with open(args.key_file, 'r') as f:
            key_data = f.read().strip()
            # Try to parse as comma-separated values
            try:
                key_values = [int(x) for x in key_data.split(',')]
                if len(key_values) != args.key_size * args.key_size:
                    parser.error(f"Key must have {args.key_size * args.key_size} values")
                key_matrix = np.array(key_values).reshape(args.key_size, args.key_size)
            except:
                # Try to parse as numpy array
                try:
                    key_matrix = np.loadtxt(args.key_file)
                    if key_matrix.shape != (args.key_size, args.key_size):
                        parser.error(f"Key matrix must be {args.key_size}x{args.key_size}")
                except:
                    parser.error("Failed to parse key matrix")
    
    if key_matrix is None:
        parser.error("Key matrix must be provided")
    
    # Create Hill cipher
    hill = HillCipher(args.key_size)
    
    # Encrypt or decrypt
    result = None
    if args.encrypt:
        result = hill.encrypt(text, key_matrix)
    else:
        result = hill.decrypt(text, key_matrix)
    
    # Output result
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
    else:
        print(result)

if __name__ == "__main__":
    main()
