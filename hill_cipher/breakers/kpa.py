#!/usr/bin/env python3
"""
Hill Cipher Known-Plaintext Attack Implementation

This module implements a known-plaintext attack on the Hill cipher.
It uses linear algebra to recover the encryption key when given
matching plaintext-ciphertext pairs.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Union
import math

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core Hill cipher functionality
from core.hill_cipher import HillCipher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hill_cipher_kpa')

class HillCipherKPA(HillCipher):
    """
    Hill Cipher Known-Plaintext Attack implementation.
    
    This class provides methods to recover the encryption key of a Hill cipher
    when given matching plaintext-ciphertext pairs.
    """
    
    def __init__(self, key_size: int):
        """
        Initialize the Hill Cipher KPA solver.
        
        Args:
            key_size: Size of the Hill cipher key matrix (NxN)
        """
        super().__init__(key_size)
        logger.info(f"Initialized Hill Cipher KPA solver with key size {key_size}x{key_size}")
    
    def recover_key(self, plaintext: str, ciphertext: str) -> Optional[np.ndarray]:
        """
        Recover the Hill cipher key matrix using known plaintext-ciphertext pairs.
        
        Args:
            plaintext: Known plaintext
            ciphertext: Corresponding ciphertext
            
        Returns:
            Recovered key matrix if successful, None otherwise
            
        Raises:
            ValueError: If there's insufficient data or matrices are not invertible
        """
        logger.info("Attempting to recover key from plaintext-ciphertext pair")
        
        # Convert to matrices
        P = self.text_to_matrix(plaintext)
        C = self.text_to_matrix(ciphertext)
        
        # Ensure we have enough text for key recovery
        if len(plaintext) < self.key_size * self.key_size:
            raise ValueError(f"Need at least {self.key_size * self.key_size} characters of plaintext-ciphertext pairs")
        
        # We need at least key_size blocks to solve the system
        required_blocks = self.key_size
        if P.shape[0] < required_blocks or C.shape[0] < required_blocks:
            raise ValueError(f"Need at least {required_blocks} blocks of plaintext-ciphertext pairs")
        
        try:
            # Solve the system of linear equations: P * K = C
            # We need to find K, so K = P^(-1) * C
            P_blocks = P[:required_blocks]
            C_blocks = C[:required_blocks]
            
            P_inv = self.matrix_mod_inverse(P_blocks)
            K = (P_inv @ C_blocks) % self.modulus
            
            logger.info(f"Successfully recovered {self.key_size}x{self.key_size} key matrix")
            return K
        except ValueError as e:
            logger.error(f"Failed to recover key: {e}")
            return None
    
    def verify_key(self, key: np.ndarray, plaintext: str, ciphertext: str) -> bool:
        """
        Verify if a recovered key correctly encrypts plaintext to ciphertext.
        
        Args:
            key: Key matrix to verify
            plaintext: Known plaintext
            ciphertext: Corresponding ciphertext
            
        Returns:
            True if the key is correct, False otherwise
        """
        logger.info("Verifying recovered key")
        
        # Convert to matrices
        P = self.text_to_matrix(plaintext)
        C = self.text_to_matrix(ciphertext)
        
        # Encrypt plaintext with the recovered key
        C_test = (P @ key) % self.modulus
        
        # Compare with the actual ciphertext
        if np.array_equal(C_test, C):
            logger.info("Key verification successful")
            return True
        else:
            logger.warning("Key verification failed")
            return False

def main():
    """Main function for demonstration and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hill Cipher Known-Plaintext Attack")
    parser.add_argument("--key-size", type=int, default=2, choices=[2, 3, 4, 5], help="Size of the key matrix")
    parser.add_argument("--plaintext", type=str, help="Known plaintext")
    parser.add_argument("--ciphertext", type=str, help="Corresponding ciphertext")
    parser.add_argument("--plaintext-file", type=str, help="File containing known plaintext")
    parser.add_argument("--ciphertext-file", type=str, help="File containing corresponding ciphertext")
    parser.add_argument("--output", type=str, help="Output file for recovered key")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get plaintext and ciphertext
    plaintext = args.plaintext
    ciphertext = args.ciphertext
    
    if args.plaintext_file:
        with open(args.plaintext_file, 'r') as f:
            plaintext = f.read().strip()
    
    if args.ciphertext_file:
        with open(args.ciphertext_file, 'r') as f:
            ciphertext = f.read().strip()
    
    if not plaintext or not ciphertext:
        parser.error("Plaintext and ciphertext must be provided")
    
    # Create KPA solver
    kpa = HillCipherKPA(args.key_size)
    
    try:
        # Recover key
        key = kpa.recover_key(plaintext, ciphertext)
        
        if key is not None:
            print(f"Recovered {args.key_size}x{args.key_size} key matrix:")
            print(key)
            
            # Verify key
            if kpa.verify_key(key, plaintext, ciphertext):
                print("Key verification successful")
            else:
                print("Key verification failed")
            
            # Decrypt the ciphertext with the recovered key
            decrypted = kpa.decrypt(ciphertext, key)
            print(f"Decrypted text (first 100 characters):")
            print(decrypted[:100])
            
            # Save key if requested
            if args.output:
                np.savetxt(args.output, key, fmt='%d')
                print(f"Key saved to {args.output}")
        else:
            print("Failed to recover key")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
