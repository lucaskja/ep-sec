#!/usr/bin/env python3
"""
Hill Cipher implementation.

This module contains the core functionality for encrypting and decrypting
using the Hill Cipher algorithm.
"""

import numpy as np
from typing import List
from src.utils import text_to_numbers, numbers_to_text, matrix_mod_inverse

def encrypt_hill(plaintext: str, key_matrix: np.ndarray) -> str:
    """
    Encrypts text using the Hill Cipher.
    
    Args:
        plaintext: Text to encrypt
        key_matrix: Key matrix
        
    Returns:
        Encrypted text
    """
    # Convert text to numbers
    numbers = text_to_numbers(plaintext)
    
    # Get matrix size
    n = key_matrix.shape[0]
    
    # Add padding if necessary
    if len(numbers) % n != 0:
        padding = n - (len(numbers) % n)
        numbers.extend([0] * padding)  # Add 'A's as padding
    
    # Split into blocks of size n
    blocks = [numbers[i:i+n] for i in range(0, len(numbers), n)]
    
    # Encrypt each block
    encrypted_blocks = []
    for block in blocks:
        block_vector = np.array(block)
        encrypted_block = np.dot(key_matrix, block_vector) % 26
        encrypted_blocks.extend(encrypted_block.tolist())
    
    # Convert numbers to text
    return numbers_to_text(encrypted_blocks)

def decrypt_hill(ciphertext: str, key_matrix: np.ndarray) -> str:
    """
    Decrypts text using the Hill Cipher.
    
    Args:
        ciphertext: Text to decrypt
        key_matrix: Key matrix
        
    Returns:
        Decrypted text
    """
    # Calculate inverse key matrix
    inverse_key = matrix_mod_inverse(key_matrix)
    
    # Use the inverse key to encrypt (which is equivalent to decrypting)
    return encrypt_hill(ciphertext, inverse_key)

def known_plaintext_attack(plaintext: str, ciphertext: str, matrix_size: int) -> np.ndarray:
    """
    Implements the known plaintext attack.
    
    Args:
        plaintext: Known plaintext
        ciphertext: Corresponding ciphertext
        matrix_size: Size of the matrix (2, 3, 4 or 5)
        
    Returns:
        Candidate key matrix
        
    Raises:
        ValueError: If the texts are not long enough or the matrix is not invertible
    """
    # Convert texts to numbers
    p_nums = text_to_numbers(plaintext)
    c_nums = text_to_numbers(ciphertext)
    
    # Check if the texts are long enough
    if len(p_nums) < matrix_size * matrix_size or len(c_nums) < matrix_size * matrix_size:
        raise ValueError(f"The texts must have at least {matrix_size * matrix_size} characters")
    
    # Create plaintext and ciphertext matrices
    p_blocks = []
    c_blocks = []
    
    for i in range(0, matrix_size * matrix_size, matrix_size):
        p_block = p_nums[i:i+matrix_size]
        c_block = c_nums[i:i+matrix_size]
        p_blocks.append(p_block)
        c_blocks.append(c_block)
    
    P = np.array(p_blocks).T
    C = np.array(c_blocks).T
    
    # Calculate inverse of plaintext matrix
    try:
        P_inv = matrix_mod_inverse(P)
    except ValueError:
        raise ValueError("The plaintext matrix is not invertible mod 26")
    
    # Calculate key matrix
    K = (C @ P_inv) % 26
    
    return K
