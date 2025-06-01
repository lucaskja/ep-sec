#!/usr/bin/env python3
"""
Test script for the Hill Cipher Analyzer on 3x3 matrices.
"""

import os
import sys
import numpy as np

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.hill_cipher_analyzer import HillCipherAnalyzer
from src.hill_cipher import decrypt_hill

def test_3x3_cipher():
    """Test the analyzer on the 3x3 cipher."""
    print("=== Testing 3x3 Cipher ===")
    
    # Read ciphertext
    with open("textos_conhecidos/Cifrado/Hill/Grupo02_3_texto_cifrado.txt", 'r') as f:
        ciphertext = f.read().strip()
    
    # Create analyzer
    analyzer = HillCipherAnalyzer(3)
    
    # Analyze ciphertext
    results = analyzer.analyze_ciphertext(ciphertext)
    
    # Print results
    print(f"Found {len(results)} potential matrices")
    
    for i, (matrix, decrypted, score) in enumerate(results[:5], 1):
        print(f"\nResult #{i} (Score: {score:.4f}):")
        print(f"Matrix:")
        print(matrix)
        print(f"Decrypted: {decrypted[:50]}...")
        
        # Check for common Portuguese words
        common_words = ['DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
                       'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS']
        
        found_words = []
        for word in common_words:
            if word in decrypted:
                found_words.append(word)
        
        if found_words:
            print(f"Common words found: {', '.join(found_words)}")

def test_4x4_cipher():
    """Test the analyzer on the 4x4 cipher."""
    print("\n=== Testing 4x4 Cipher ===")
    
    # Read ciphertext
    with open("textos_conhecidos/Cifrado/Hill/Grupo02_4_texto_cifrado.txt", 'r') as f:
        ciphertext = f.read().strip()
    
    # Create analyzer
    analyzer = HillCipherAnalyzer(4)
    
    # Analyze ciphertext
    results = analyzer.analyze_ciphertext(ciphertext)
    
    # Print results
    print(f"Found {len(results)} potential matrices")
    
    for i, (matrix, decrypted, score) in enumerate(results[:5], 1):
        print(f"\nResult #{i} (Score: {score:.4f}):")
        print(f"Matrix:")
        print(matrix)
        print(f"Decrypted: {decrypted[:50]}...")
        
        # Check for common Portuguese words
        common_words = ['DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
                       'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS']
        
        found_words = []
        for word in common_words:
            if word in decrypted:
                found_words.append(word)
        
        if found_words:
            print(f"Common words found: {', '.join(found_words)}")

def test_5x5_cipher():
    """Test the analyzer on the 5x5 cipher."""
    print("\n=== Testing 5x5 Cipher ===")
    
    # Read ciphertext
    with open("textos_conhecidos/Cifrado/Hill/Grupo02_5_texto_cifrado.txt", 'r') as f:
        ciphertext = f.read().strip()
    
    # Create analyzer
    analyzer = HillCipherAnalyzer(5)
    
    # Analyze ciphertext
    results = analyzer.analyze_ciphertext(ciphertext)
    
    # Print results
    print(f"Found {len(results)} potential matrices")
    
    for i, (matrix, decrypted, score) in enumerate(results[:5], 1):
        print(f"\nResult #{i} (Score: {score:.4f}):")
        print(f"Matrix:")
        print(matrix)
        print(f"Decrypted: {decrypted[:50]}...")
        
        # Check for common Portuguese words
        common_words = ['DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
                       'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS']
        
        found_words = []
        for word in common_words:
            if word in decrypted:
                found_words.append(word)
        
        if found_words:
            print(f"Common words found: {', '.join(found_words)}")

if __name__ == "__main__":
    test_3x3_cipher()
    test_4x4_cipher()
    test_5x5_cipher()
