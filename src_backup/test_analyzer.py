#!/usr/bin/env python3
"""
Test script for the Hill Cipher Analyzer.
"""

import os
import sys
import numpy as np

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.hill_cipher_analyzer import HillCipherAnalyzer
from src.hill_cipher import decrypt_hill

def test_known_text():
    """Test the analyzer on the known text 2x2 cipher."""
    print("=== Testing Known Text 2x2 Cipher ===")
    
    # Known key for the known text
    known_key = np.array([[23, 17], [0, 9]])
    
    # Read ciphertext
    with open("textos_conhecidos/Cifrado/Hill/Grupo02_2_texto_cifrado.txt", 'r') as f:
        ciphertext = f.read().strip()
    
    # Create analyzer
    analyzer = HillCipherAnalyzer(2)
    
    # Analyze ciphertext
    results = analyzer.analyze_ciphertext(ciphertext)
    
    # Check if the known key is in the results
    found = False
    for i, (matrix, decrypted, score) in enumerate(results):
        print(f"Result #{i+1} (Score: {score:.4f}):")
        print(f"Matrix: {matrix}")
        print(f"Decrypted: {decrypted[:50]}...")
        
        # Check if this matrix is equivalent to the known key
        if np.array_equal(matrix, known_key):
            print("*** MATCH: This is the known key! ***")
            found = True
        else:
            # Check if the decryption is the same
            known_decrypted = decrypt_hill(ciphertext, known_key)
            if decrypted == known_decrypted:
                print("*** MATCH: This matrix produces the same decryption as the known key! ***")
                found = True
        
        print()
    
    if not found:
        print("The known key was not found in the results.")
        print("Known key:")
        print(known_key)
        print("Decryption with known key:")
        print(decrypt_hill(ciphertext, known_key)[:50] + "...")

def test_unknown_text():
    """Test the analyzer on the unknown text 2x2 cipher."""
    print("\n=== Testing Unknown Text 2x2 Cipher ===")
    
    # Known key for the unknown text
    known_key = np.array([[23, 14], [0, 5]])
    
    # Read ciphertext
    with open("textos_desconhecidos/Cifrado/Hill/Grupo02_2_texto_cifrado.txt", 'r') as f:
        ciphertext = f.read().strip()
    
    # Create analyzer
    analyzer = HillCipherAnalyzer(2)
    
    # Analyze ciphertext
    results = analyzer.analyze_ciphertext(ciphertext)
    
    # Check if the known key is in the results
    found = False
    for i, (matrix, decrypted, score) in enumerate(results):
        print(f"Result #{i+1} (Score: {score:.4f}):")
        print(f"Matrix: {matrix}")
        print(f"Decrypted: {decrypted[:50]}...")
        
        # Check if this matrix is equivalent to the known key
        if np.array_equal(matrix, known_key):
            print("*** MATCH: This is the known key! ***")
            found = True
        else:
            # Check if the decryption is the same
            known_decrypted = decrypt_hill(ciphertext, known_key)
            if decrypted == known_decrypted:
                print("*** MATCH: This matrix produces the same decryption as the known key! ***")
                found = True
        
        print()
    
    if not found:
        print("The known key was not found in the results.")
        print("Known key:")
        print(known_key)
        print("Decryption with known key:")
        print(decrypt_hill(ciphertext, known_key)[:50] + "...")

if __name__ == "__main__":
    test_known_text()
    test_unknown_text()
