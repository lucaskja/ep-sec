#!/usr/bin/env python3
"""
Utility functions for Hill Cipher Breaker.

This module contains common utility functions used across different versions
of the Hill Cipher Breaker.
"""

import numpy as np
import re
from typing import List, Dict, Tuple
from math import gcd

# Constants
ALPHABET_SIZE = 26
LETTER_TO_NUM = {chr(65 + i): i for i in range(ALPHABET_SIZE)}  # A-Z -> 0-25
NUM_TO_LETTER = {i: chr(65 + i) for i in range(ALPHABET_SIZE)}  # 0-25 -> A-Z

def text_to_numbers(text: str) -> List[int]:
    """
    Converts text to a list of numbers (0-25).
    
    Args:
        text: Text to convert
        
    Returns:
        List of numbers
    """
    # Convert to uppercase and remove non-alphabetic characters
    text = re.sub(r'[^A-Za-z]', '', text.upper())
    return [LETTER_TO_NUM[char] for char in text]

def numbers_to_text(numbers: List[int]) -> str:
    """
    Converts numbers to text.
    
    Args:
        numbers: List of numbers to convert
        
    Returns:
        Text corresponding to the numbers
    """
    return ''.join(NUM_TO_LETTER[n % ALPHABET_SIZE] for n in numbers)

def is_invertible_matrix(matrix: np.ndarray, mod: int = ALPHABET_SIZE) -> bool:
    """
    Checks if a matrix is invertible in the given modulus.
    
    Args:
        matrix: Matrix to check
        mod: Modulus (default: 26)
        
    Returns:
        True if the matrix is invertible, False otherwise
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Calculate determinant
    det = int(round(np.linalg.det(matrix))) % mod
    
    # Check if determinant is coprime with mod
    return gcd(det, mod) == 1

def mod_inverse(a: int, m: int) -> int:
    """
    Calculates the modular multiplicative inverse of a number.
    
    Args:
        a: Number to find the inverse of
        m: Modulus
        
    Returns:
        Modular multiplicative inverse of a
        
    Raises:
        ValueError: If the inverse does not exist
    """
    if gcd(a, m) != 1:
        raise ValueError(f"Modular inverse does not exist for {a} mod {m}")
    
    # Extended Euclidean Algorithm
    def egcd(a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, x, y = egcd(b % a, a)
            return (g, y - (b // a) * x, x)
    
    _, x, _ = egcd(a % m, m)
    return (x % m + m) % m

def matrix_mod_inverse(matrix: np.ndarray, mod: int = ALPHABET_SIZE) -> np.ndarray:
    """
    Calculates the modular multiplicative inverse of a matrix.
    
    Args:
        matrix: Matrix to find the inverse of
        mod: Modulus (default: 26)
        
    Returns:
        Modular multiplicative inverse of the matrix
        
    Raises:
        ValueError: If the matrix is not invertible
    """
    # Check if the matrix is invertible
    if not is_invertible_matrix(matrix, mod):
        raise ValueError("Matrix is not invertible mod 26")
    
    # Calculate determinant
    det = int(round(np.linalg.det(matrix))) % mod
    
    # Calculate modular multiplicative inverse of determinant
    det_inv = mod_inverse(det, mod)
    
    # Calculate adjugate matrix
    if matrix.shape == (2, 2):
        # For 2x2 matrix
        adjugate = np.array([
            [matrix[1, 1], -matrix[0, 1]],
            [-matrix[1, 0], matrix[0, 0]]
        ]) % mod
    else:
        # For larger matrices
        cofactor = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
                cofactor[i, j] = ((-1) ** (i + j)) * int(round(np.linalg.det(minor)))
        adjugate = cofactor.T % mod
    
    # Calculate inverse
    inverse = (det_inv * adjugate) % mod
    
    return inverse
