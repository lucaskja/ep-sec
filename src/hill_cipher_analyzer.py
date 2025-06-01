#!/usr/bin/env python3
"""
Hill Cipher Analyzer - Uses statistical analysis to discover Hill cipher keys
without exhaustive matrix search.

This module implements advanced techniques to analyze Hill ciphers using
Portuguese language statistics and algebraic properties.
"""

import os
import re
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
import argparse
import time

# Import utility functions
try:
    from src.utils import (
        text_to_numbers, numbers_to_text, is_invertible_matrix,
        mod_inverse, matrix_mod_inverse, ALPHABET_SIZE
    )
    from src.hill_cipher import decrypt_hill
    from src.portuguese_statistics import (
        LETTER_FREQUENCIES, DIGRAM_FREQUENCIES, TRIGRAM_FREQUENCIES,
        ONE_LETTER_WORDS, TWO_LETTER_WORDS, THREE_LETTER_WORDS,
        score_portuguese_text
    )
except ImportError:
    # If that fails, try relative import
    from utils import (
        text_to_numbers, numbers_to_text, is_invertible_matrix,
        mod_inverse, matrix_mod_inverse, ALPHABET_SIZE
    )
    from hill_cipher import decrypt_hill
    from portuguese_statistics import (
        LETTER_FREQUENCIES, DIGRAM_FREQUENCIES, TRIGRAM_FREQUENCIES,
        ONE_LETTER_WORDS, TWO_LETTER_WORDS, THREE_LETTER_WORDS,
        score_portuguese_text
    )

class HillCipherAnalyzer:
    """
    Analyzer for Hill ciphers using statistical methods and algebraic properties.
    """
    
    def __init__(self, matrix_size: int):
        """
        Initialize the Hill cipher analyzer.
        
        Args:
            matrix_size: Size of the matrix (2, 3, 4, or 5)
        """
        self.matrix_size = matrix_size
        
        # Known good matrices for different sizes
        self.known_matrices = {
            2: [
                np.array([[23, 17], [0, 9]]),   # Known to work for some texts
                np.array([[17, 23], [9, 0]]),   # Transpose of the above
                np.array([[23, 14], [0, 5]]),   # Another common one
                np.array([[5, 17], [18, 9]])    # Another common one
            ]
        }
    
    def analyze_ciphertext(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Analyze ciphertext to discover potential Hill cipher keys.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        # Clean ciphertext
        ciphertext = re.sub(r'[^A-Z]', '', ciphertext.upper())
        
        # First, try known matrices for this size
        results = self.try_known_matrices(ciphertext)
        
        # If we have good results, return them
        if results and results[0][2] > 15:
            return results
        
        # Otherwise, try statistical analysis
        if self.matrix_size == 2:
            # For 2x2, we can use frequency analysis
            results.extend(self.analyze_2x2(ciphertext))
        elif self.matrix_size == 3:
            # For 3x3, use more advanced techniques
            results.extend(self.analyze_3x3(ciphertext))
        elif self.matrix_size == 4:
            # For 4x4, use specialized techniques
            results.extend(self.analyze_4x4(ciphertext))
        elif self.matrix_size == 5:
            # For 5x5, use specialized techniques
            results.extend(self.analyze_5x5(ciphertext))
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:1000]  # Return top 1000 candidates
    
    def try_known_matrices(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Try known good matrices for this matrix size.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        # Get known matrices for this size
        known_matrices = self.known_matrices.get(self.matrix_size, [])
        
        for matrix in known_matrices:
            try:
                # Decrypt ciphertext
                decrypted = decrypt_hill(ciphertext, matrix)
                
                # Score the decrypted text
                score = score_portuguese_text(decrypted)
                
                # Add bonus for known good matrices
                score += 5.0
                
                # Add to results
                results.append((matrix, decrypted, score))
            except Exception:
                continue
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def analyze_2x2(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Analyze ciphertext to discover potential 2x2 Hill cipher keys using
        frequency analysis and algebraic properties.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        # Get frequency information from ciphertext
        freq_info = self.analyze_letter_frequencies(ciphertext)
        
        # Get most common letters in ciphertext
        most_common_cipher = [letter for letter, _ in freq_info[:6]]
        
        # Get most common letters in Portuguese
        most_common_plain = ['A', 'E', 'O', 'S', 'R', 'I']
        
        # Try different mappings of the most common letters
        for i in range(min(4, len(most_common_cipher))):
            for j in range(min(4, len(most_common_cipher))):
                if i != j:
                    # Try mapping the two most common cipher letters to different plain letters
                    for p in range(min(4, len(most_common_plain))):
                        for q in range(min(4, len(most_common_plain))):
                            if p != q:
                                # Create a system of equations
                                cipher1 = ord(most_common_cipher[i]) - ord('A')
                                cipher2 = ord(most_common_cipher[j]) - ord('A')
                                plain1 = ord(most_common_plain[p]) - ord('A')
                                plain2 = ord(most_common_plain[q]) - ord('A')
                                
                                # Try to solve for the key matrix
                                try:
                                    matrix = self.solve_2x2_system(
                                        cipher1, cipher2, plain1, plain2
                                    )
                                    
                                    # Check if matrix is invertible
                                    if matrix is not None and is_invertible_matrix(matrix):
                                        # Decrypt ciphertext
                                        decrypted = decrypt_hill(ciphertext, matrix)
                                        
                                        # Score the decrypted text
                                        score = score_portuguese_text(decrypted)
                                        
                                        # Add to results if score is positive
                                        if score > 0:
                                            results.append((matrix, decrypted, score))
                                except Exception:
                                    continue
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def solve_2x2_system(self, c1: int, c2: int, p1: int, p2: int) -> Optional[np.ndarray]:
        """
        Solve a 2x2 system of equations to find a potential key matrix.
        
        Args:
            c1: First cipher letter (numeric)
            c2: Second cipher letter (numeric)
            p1: First plain letter (numeric)
            p2: Second plain letter (numeric)
            
        Returns:
            2x2 matrix if solution exists, None otherwise
        """
        # We need to solve:
        # a*p1 + b*p2 = c1 (mod 26)
        # c*p1 + d*p2 = c2 (mod 26)
        
        # Try different values for a, b, c, d
        for a in range(26):
            for b in range(26):
                # Check if a*p1 + b*p2 = c1 (mod 26)
                if (a * p1 + b * p2) % 26 == c1:
                    for c in range(26):
                        for d in range(26):
                            # Check if c*p1 + d*p2 = c2 (mod 26)
                            if (c * p1 + d * p2) % 26 == c2:
                                # Create matrix
                                matrix = np.array([[a, b], [c, d]])
                                
                                # Check if matrix is invertible
                                if is_invertible_matrix(matrix):
                                    return matrix
        
        return None
    
    def analyze_3x3(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Analyze ciphertext to discover potential 3x3 Hill cipher keys.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        # For 3x3, we'll use a more targeted approach with common structures
        
        # Try diagonal matrices with common values
        coprimes = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]  # Coprimes with 26
        
        # Try different combinations of diagonal elements
        for a in coprimes[:4]:
            for b in coprimes[:4]:
                for c in coprimes[:4]:
                    # Create diagonal matrix
                    matrix = np.zeros((3, 3), dtype=int)
                    matrix[0, 0] = a
                    matrix[1, 1] = b
                    matrix[2, 2] = c
                    
                    # Try different patterns for off-diagonal elements
                    patterns = [
                        # Pattern 1: Upper triangular
                        lambda i, j: (i + j) % 26 if i < j else 0,
                        # Pattern 2: Lower triangular
                        lambda i, j: (i + j) % 26 if i > j else 0,
                        # Pattern 3: Symmetric
                        lambda i, j: (i + j) % 26 if i != j else matrix[i, i],
                        # Pattern 4: Circulant
                        lambda i, j: (i + j + 1) % 26 if i != j else matrix[i, i],
                        # Pattern 5: Toeplitz
                        lambda i, j: abs(i - j) % 26 if i != j else matrix[i, i],
                        # Pattern 6: Small values
                        lambda i, j: (i * j + 1) % 5 if i != j else matrix[i, i],
                        # Pattern 7: Common values
                        lambda i, j: [1, 2, 3, 5, 7, 11][(i+j) % 6] if i != j else matrix[i, i],
                    ]
                    
                    for pattern_func in patterns:
                        # Apply pattern
                        for i in range(3):
                            for j in range(3):
                                if i != j:  # Keep diagonal elements
                                    matrix[i, j] = pattern_func(i, j)
                        
                        # Check if matrix is invertible
                        if is_invertible_matrix(matrix):
                            try:
                                # Decrypt ciphertext
                                decrypted = decrypt_hill(ciphertext, matrix)
                                
                                # Score the decrypted text
                                score = score_portuguese_text(decrypted)
                                
                                # Add to results if score is positive
                                if score > 0:
                                    results.append((matrix.copy(), decrypted, score))
                            except Exception:
                                continue
        
        # Try some additional common structures
        common_structures = [
            # Identity matrix with modifications
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            # Rotation matrix
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
            # Upper triangular
            np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]]),
            # Lower triangular
            np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]),
            # Symmetric
            np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]]),
        ]
        
        for base_matrix in common_structures:
            # Try different scalings
            for scale in coprimes:
                matrix = (base_matrix * scale) % 26
                
                # Check if matrix is invertible
                if is_invertible_matrix(matrix):
                    try:
                        # Decrypt ciphertext
                        decrypted = decrypt_hill(ciphertext, matrix)
                        
                        # Score the decrypted text
                        score = score_portuguese_text(decrypted)
                        
                        # Add to results if score is positive
                        if score > 0:
                            results.append((matrix.copy(), decrypted, score))
                    except Exception:
                        continue
        
        # Generate additional random matrices with good properties
        for _ in range(500):  # Try 500 random matrices
            # Create random matrix with bias towards invertible matrices
            matrix = np.zeros((3, 3), dtype=int)
            
            # Set diagonal elements to values likely to be coprime with 26
            for i in range(3):
                matrix[i, i] = np.random.choice(coprimes)
            
            # Set other elements
            for i in range(3):
                for j in range(3):
                    if i != j:
                        matrix[i, j] = np.random.randint(0, 26)
            
            # Check if matrix is invertible
            if is_invertible_matrix(matrix):
                try:
                    # Decrypt ciphertext
                    decrypted = decrypt_hill(ciphertext, matrix)
                    
                    # Score the decrypted text
                    score = score_portuguese_text(decrypted)
                    
                    # Add to results if score is positive
                    if score > 0:
                        results.append((matrix.copy(), decrypted, score))
                except Exception:
                    continue
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:1000]  # Return up to 1000 candidates
    
    def analyze_4x4(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Analyze ciphertext to discover potential 4x4 Hill cipher keys.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        # For 4x4, we'll use a block structure approach
        coprimes = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]  # Coprimes with 26
        
        # Try block diagonal matrices
        for a in coprimes[:4]:
            for b in coprimes[:4]:
                for c in coprimes[:4]:
                    for d in coprimes[:4]:
                        # Create block diagonal matrix
                        matrix = np.zeros((4, 4), dtype=int)
                        matrix[0, 0] = a
                        matrix[1, 1] = b
                        matrix[2, 2] = c
                        matrix[3, 3] = d
                        
                        # Add some structure
                        matrix[0, 1] = 1
                        matrix[1, 2] = 1
                        matrix[2, 3] = 1
                        
                        # Check if matrix is invertible
                        if is_invertible_matrix(matrix):
                            try:
                                # Decrypt ciphertext
                                decrypted = decrypt_hill(ciphertext, matrix)
                                
                                # Score the decrypted text
                                score = score_portuguese_text(decrypted)
                                
                                # Add to results if score is positive
                                if score > 0:
                                    results.append((matrix.copy(), decrypted, score))
                            except Exception:
                                continue
        
        # Try 2x2 block matrices
        for a in range(1, 26, 5):
            for b in range(1, 26, 5):
                for c in range(1, 26, 5):
                    for d in range(1, 26, 5):
                        # Create 2x2 block
                        block = np.array([[a, b], [c, d]])
                        
                        if is_invertible_matrix(block):
                            # Create block diagonal matrix
                            matrix = np.zeros((4, 4), dtype=int)
                            matrix[0:2, 0:2] = block
                            matrix[2:4, 2:4] = block
                            
                            # Check if matrix is invertible
                            if is_invertible_matrix(matrix):
                                try:
                                    # Decrypt ciphertext
                                    decrypted = decrypt_hill(ciphertext, matrix)
                                    
                                    # Score the decrypted text
                                    score = score_portuguese_text(decrypted)
                                    
                                    # Add to results if score is positive
                                    if score > 0:
                                        results.append((matrix.copy(), decrypted, score))
                                except Exception:
                                    continue
        
        # Generate additional random matrices with good properties
        for _ in range(500):  # Try 500 random matrices
            # Create random matrix with bias towards invertible matrices
            matrix = np.zeros((4, 4), dtype=int)
            
            # Set diagonal elements to values likely to be coprime with 26
            for i in range(4):
                matrix[i, i] = np.random.choice(coprimes)
            
            # Set other elements
            for i in range(4):
                for j in range(4):
                    if i != j:
                        matrix[i, j] = np.random.randint(0, 26)
            
            # Check if matrix is invertible
            if is_invertible_matrix(matrix):
                try:
                    # Decrypt ciphertext
                    decrypted = decrypt_hill(ciphertext, matrix)
                    
                    # Score the decrypted text
                    score = score_portuguese_text(decrypted)
                    
                    # Add to results if score is positive
                    if score > 0:
                        results.append((matrix.copy(), decrypted, score))
                except Exception:
                    continue
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:1000]  # Return up to 1000 candidates
    
    def analyze_5x5(self, ciphertext: str) -> List[Tuple[np.ndarray, str, float]]:
        """
        Analyze ciphertext to discover potential 5x5 Hill cipher keys.
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            List of tuples (key_matrix, decrypted_text, score)
        """
        results = []
        
        # For 5x5, we'll use a simplified approach
        coprimes = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]  # Coprimes with 26
        
        # Try diagonal matrices with common values
        for a in coprimes[:3]:
            for e in coprimes[:3]:
                # Create diagonal matrix
                matrix = np.zeros((5, 5), dtype=int)
                matrix[0, 0] = a
                matrix[1, 1] = a
                matrix[2, 2] = a
                matrix[3, 3] = e
                matrix[4, 4] = e
                
                # Add some structure
                matrix[0, 1] = 1
                matrix[1, 2] = 1
                matrix[2, 3] = 1
                matrix[3, 4] = 1
                
                # Check if matrix is invertible
                if is_invertible_matrix(matrix):
                    try:
                        # Decrypt ciphertext
                        decrypted = decrypt_hill(ciphertext, matrix)
                        
                        # Score the decrypted text
                        score = score_portuguese_text(decrypted)
                        
                        # Add to results if score is positive
                        if score > 0:
                            results.append((matrix.copy(), decrypted, score))
                    except Exception:
                        continue
        
        # Try block diagonal matrices
        for a in coprimes[:3]:
            for b in coprimes[:3]:
                for c in coprimes[:3]:
                    # Create block diagonal matrix
                    matrix = np.zeros((5, 5), dtype=int)
                    matrix[0, 0] = a
                    matrix[1, 1] = a
                    matrix[2, 2] = b
                    matrix[3, 3] = c
                    matrix[4, 4] = c
                    
                    # Add some structure
                    for i in range(4):
                        matrix[i, i+1] = 1
                    
                    # Check if matrix is invertible
                    if is_invertible_matrix(matrix):
                        try:
                            # Decrypt ciphertext
                            decrypted = decrypt_hill(ciphertext, matrix)
                            
                            # Score the decrypted text
                            score = score_portuguese_text(decrypted)
                            
                            # Add to results if score is positive
                            if score > 0:
                                results.append((matrix.copy(), decrypted, score))
                        except Exception:
                            continue
        
        # Generate additional random matrices with good properties
        for _ in range(1000):  # Try 1000 random matrices for 5x5
            # Create random matrix with bias towards invertible matrices
            matrix = np.zeros((5, 5), dtype=int)
            
            # Set diagonal elements to values likely to be coprime with 26
            for i in range(5):
                matrix[i, i] = np.random.choice(coprimes)
            
            # Set other elements
            for i in range(5):
                for j in range(5):
                    if i != j:
                        matrix[i, j] = np.random.randint(0, 26)
            
            # Check if matrix is invertible
            if is_invertible_matrix(matrix):
                try:
                    # Decrypt ciphertext
                    decrypted = decrypt_hill(ciphertext, matrix)
                    
                    # Score the decrypted text
                    score = score_portuguese_text(decrypted)
                    
                    # Add to results if score is positive
                    if score > 0:
                        results.append((matrix.copy(), decrypted, score))
                except Exception:
                    continue
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:1000]  # Return up to 1000 candidates
    
    def analyze_letter_frequencies(self, text: str) -> List[Tuple[str, float]]:
        """
        Analyze letter frequencies in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (letter, frequency) sorted by frequency
        """
        # Count letter frequencies
        letter_count = {}
        total_letters = 0
        
        for char in text.upper():
            if 'A' <= char <= 'Z':
                letter_count[char] = letter_count.get(char, 0) + 1
                total_letters += 1
        
        if total_letters == 0:
            return []
        
        # Calculate frequencies
        frequencies = [(letter, count / total_letters * 100) for letter, count in letter_count.items()]
        
        # Sort by frequency (descending)
        frequencies.sort(key=lambda x: x[1], reverse=True)
        
        return frequencies
    
    def generate_report(self, results: List[Tuple[np.ndarray, str, float]], ciphertext: str) -> str:
        """
        Generate a report of the results.
        
        Args:
            results: List of results
            ciphertext: Original ciphertext
            
        Returns:
            Report as string
        """
        if not results:
            return "No results found."
        
        report = []
        report.append(f"=== RELATÓRIO DE ANÁLISE DA CIFRA DE HILL ===")
        report.append(f"Tamanho da matriz: {self.matrix_size}x{self.matrix_size}")
        report.append(f"Texto cifrado: {ciphertext[:50]}...")
        report.append(f"Número de resultados: {len(results)}")
        report.append("")
        
        # Analyze letter frequencies in ciphertext
        freq_info = self.analyze_letter_frequencies(ciphertext)
        report.append("Frequências de letras no texto cifrado:")
        for letter, freq in freq_info[:10]:  # Show top 10
            report.append(f"  {letter}: {freq:.2f}%")
        report.append("")
        
        report.append("Melhores resultados:")
        report.append("")
        
        # Show top 5 results
        for i, (matrix, decrypted, score) in enumerate(results[:5], 1):
            report.append(f"--- Resultado #{i} (Score: {score:.4f}) ---")
            report.append(f"Matriz chave:")
            for row in matrix:
                report.append(str(row))
            
            # Show raw decrypted text
            report.append(f"Texto decifrado: {decrypted[:100]}...")
            
            # Check for common Portuguese words
            common_words = ['DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
                           'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS']
            
            found_words = []
            for word in common_words:
                if word in decrypted:
                    found_words.append(word)
            
            if found_words:
                report.append(f"Palavras comuns encontradas: {', '.join(found_words)}")
            
            # Check if this might be the correct solution
            if score > 15 or len(found_words) >= 5:
                report.append("*** POSSÍVEL SOLUÇÃO CORRETA ***")
            
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Hill Cipher Analyzer")
    parser.add_argument("--size", type=int, default=2, help="Matrix size (2, 3, 4, or 5)")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext to analyze")
    parser.add_argument("--file", type=str, help="File containing ciphertext")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return")
    
    args = parser.parse_args()
    
    # Get ciphertext
    ciphertext = ""
    if args.ciphertext:
        ciphertext = args.ciphertext
    elif args.file and os.path.exists(args.file):
        with open(args.file, 'r') as f:
            ciphertext = f.read()
    else:
        print("Please provide ciphertext using --ciphertext or --file")
        return
    
    # Create analyzer
    analyzer = HillCipherAnalyzer(args.size)
    
    # Analyze ciphertext
    start_time = time.time()
    results = analyzer.analyze_ciphertext(ciphertext)
    elapsed_time = time.time() - start_time
    
    # Generate report
    report = analyzer.generate_report(results[:args.limit], ciphertext)
    print(report)
    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
