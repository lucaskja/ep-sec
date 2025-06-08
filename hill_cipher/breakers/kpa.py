#!/usr/bin/env python3
"""
Known Plaintext Attack (KPA) for Hill Cipher

This module implements a known plaintext attack against Hill cipher.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import numpy as np
import logging
from typing import Optional, List, Tuple
import math

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hill_cipher import HillCipher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('kpa')

class HillCipherKPA:
    """
    Known Plaintext Attack implementation for Hill cipher.
    """
    
    def __init__(self, key_size: int):
        """
        Initialize the KPA attack.
        
        Args:
            key_size: Size of the Hill cipher key matrix
        """
        self.key_size = key_size
        self.hill_cipher = HillCipher(key_size)
        logger.info(f"Initialized KPA for {key_size}x{key_size} Hill cipher")
    
    def text_to_matrix(self, text: str) -> np.ndarray:
        """
        Convert text to numerical matrix.
        
        Args:
            text: Input text
            
        Returns:
            Numerical matrix
        """
        return self.hill_cipher.text_to_matrix(text)
    
    def attack(self, ciphertext: str, plaintext: str) -> Optional[np.ndarray]:
        """
        Perform known plaintext attack.
        
        Args:
            ciphertext: Encrypted text
            plaintext: Known plaintext
            
        Returns:
            Key matrix if found, None otherwise
        """
        logger.info("Starting known plaintext attack...")
        logger.info(f"Ciphertext length: {len(ciphertext)}")
        logger.info(f"Plaintext length: {len(plaintext)}")
        
        # Convert texts to matrices
        try:
            C = self.text_to_matrix(ciphertext)
            P = self.text_to_matrix(plaintext)
            
            logger.info(f"Cipher matrix shape: {C.shape}")
            logger.info(f"Plain matrix shape: {P.shape}")
            
            # We need at least key_size blocks to solve for the key
            min_blocks = self.key_size
            if C.shape[0] < min_blocks or P.shape[0] < min_blocks:
                logger.error(f"Need at least {min_blocks} blocks, got C:{C.shape[0]}, P:{P.shape[0]}")
                return None
            
            # Try different starting positions to find a solvable system
            for start_pos in range(min(C.shape[0] - min_blocks + 1, 10)):  # Try up to 10 positions
                try:
                    # Extract blocks for the equation system
                    C_blocks = C[start_pos:start_pos + min_blocks]
                    P_blocks = P[start_pos:start_pos + min_blocks]
                    
                    logger.info(f"Trying position {start_pos}")
                    logger.info(f"C_blocks:\n{C_blocks}")
                    logger.info(f"P_blocks:\n{P_blocks}")
                    
                    # Check if P_blocks is invertible
                    if not self.hill_cipher.is_invertible(P_blocks):
                        logger.debug(f"P_blocks at position {start_pos} not invertible, trying next...")
                        continue
                    
                    # Solve: C = P * K, so K = P^(-1) * C
                    P_inv = self.hill_cipher.matrix_mod_inverse(P_blocks)
                    K = (P_inv @ C_blocks) % 26
                    
                    logger.info(f"Candidate key at position {start_pos}:\n{K}")
                    
                    # Verify the key works for the entire text
                    if self._verify_key(ciphertext, plaintext, K):
                        logger.info("✓ Key verification successful!")
                        return K
                    else:
                        logger.debug(f"Key at position {start_pos} failed verification")
                
                except Exception as e:
                    logger.debug(f"Error at position {start_pos}: {e}")
                    continue
            
            logger.error("No valid key found")
            return None
            
        except Exception as e:
            logger.error(f"Error in KPA attack: {e}")
            return None
    
    def _verify_key(self, ciphertext: str, plaintext: str, key: np.ndarray) -> bool:
        """
        Verify that a key correctly decrypts the ciphertext to plaintext.
        
        Args:
            ciphertext: Original ciphertext
            plaintext: Expected plaintext
            key: Key to verify
            
        Returns:
            True if key is correct
        """
        try:
            # Decrypt with the key
            decrypted = self.hill_cipher.decrypt(ciphertext, key)
            
            # Clean both texts for comparison
            plaintext_clean = plaintext.upper().replace(' ', '').replace('\n', '')
            decrypted_clean = decrypted.upper().replace(' ', '').replace('\n', '').rstrip('X')
            
            logger.info(f"Expected: {plaintext_clean[:50]}...")
            logger.info(f"Got:      {decrypted_clean[:50]}...")
            
            # Check for exact match or close match
            if plaintext_clean == decrypted_clean:
                return True
            elif len(plaintext_clean) <= len(decrypted_clean):
                # Check if plaintext is a prefix of decrypted (accounting for padding)
                return decrypted_clean.startswith(plaintext_clean)
            else:
                # Check if decrypted is a prefix of plaintext (truncated)
                return plaintext_clean.startswith(decrypted_clean)
        
        except Exception as e:
            logger.debug(f"Verification error: {e}")
            return False

def main():
    """Main function for testing KPA."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hill Cipher Known Plaintext Attack")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext")
    parser.add_argument("--plaintext", type=str, help="Known plaintext")
    parser.add_argument("--ciphertext-file", type=str, help="File containing ciphertext")
    parser.add_argument("--plaintext-file", type=str, help="File containing plaintext")
    parser.add_argument("--key-size", type=int, default=2, choices=[2, 3, 4, 5], 
                       help="Size of the key matrix")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get ciphertext
    ciphertext = args.ciphertext
    if args.ciphertext_file:
        with open(args.ciphertext_file, 'r') as f:
            ciphertext = f.read().strip()
    
    # Get plaintext
    plaintext = args.plaintext
    if args.plaintext_file:
        with open(args.plaintext_file, 'r') as f:
            plaintext = f.read().strip()
    
    if not ciphertext or not plaintext:
        parser.error("Both ciphertext and plaintext must be provided")
    
    # Create KPA attack
    kpa = HillCipherKPA(args.key_size)
    
    # Perform attack
    key = kpa.attack(ciphertext, plaintext)
    
    if key is not None:
        print(f"Key found: {key.flatten()}")
        print(f"Key matrix:\n{key}")
        
        # Verify by decrypting
        hill = HillCipher(args.key_size)
        decrypted = hill.decrypt(ciphertext, key)
        print(f"Decrypted text: {decrypted}")
    else:
        print("No key found")

if __name__ == "__main__":
    main()
