#!/usr/bin/env python3
"""
Quick validation script for 2x2 Hill cipher with known plaintext.
"""

import sys
import os
sys.path.append('hill_cipher')

from hill_cipher.breakers.statistical_analyzer import StatisticalAnalyzer

def main():
    # Known data for 2x2 cipher
    ciphertext = "ypewhabanavprxgyekypbaonoefvdpisnxlwbabsgewuclweqktwkklkfkgyigzpbavsdxrwxacluufwjfugcwsarcoelklfowlhpnvwokmglxnpegoapjlp"
    known_plaintext = "NTAOPARANAOTERQUEENTRARNUMALUTACORPORALCOMMINHAMAEVOCETEVEQUESETRANCARNOBANHEIROEPASSOUALGUMTEMPOOUV"
    
    print("Hill Cipher 2x2 Statistical Analysis Validation")
    print("=" * 50)
    print(f"Ciphertext: {ciphertext}")
    print(f"Known plaintext: {known_plaintext}")
    print(f"Ciphertext length: {len(ciphertext)}")
    print(f"Plaintext length: {len(known_plaintext)}")
    
    # Create analyzer
    analyzer = StatisticalAnalyzer(key_size=2)
    
    # Run validation
    print("\nStarting statistical analysis...")
    success = analyzer.validate_with_known_plaintext(ciphertext, known_plaintext)
    
    if success:
        print("\n✓ Validation successful! Statistical analysis works correctly.")
    else:
        print("\n✗ Validation failed. Need to adjust the approach.")

if __name__ == "__main__":
    main()
