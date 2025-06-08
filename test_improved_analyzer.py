#!/usr/bin/env python3
"""
Test the improved statistical analyzer.
"""

import sys
import os
sys.path.append('hill_cipher')

from hill_cipher.breakers.improved_statistical_analyzer import ImprovedStatisticalAnalyzer

def main():
    # Known data for 2x2 cipher
    ciphertext = "ypewhabanavprxgyekypbaonoefvdpisnxlwbabsgewuclweqktwkklkfkgyigzpbavsdxrwxacluufwjfugcwsarcoelklfowlhpnvwokmglxnpegoapjlp"
    known_plaintext = "NTAOPARANAOTERQUEENTRARNUMALUTACORPORALCOMMINHAMAEVOCETEVEQUESETRANCARNOBANHEIROEPASSOUALGUMTEMPOOUV"
    
    print("Testing Improved Statistical Analyzer")
    print("=" * 40)
    print(f"Ciphertext: {ciphertext}")
    print(f"Known plaintext: {known_plaintext}")
    print(f"Ciphertext length: {len(ciphertext)}")
    print(f"Plaintext length: {len(known_plaintext)}")
    
    # Create improved analyzer
    analyzer = ImprovedStatisticalAnalyzer(key_size=2)
    
    # Run validation
    print("\nStarting improved statistical analysis...")
    success = analyzer.validate_with_known_plaintext(ciphertext, known_plaintext)
    
    if success:
        print("\n✓ Validation successful! Improved statistical analysis works correctly.")
    else:
        print("\n✗ Validation failed.")

if __name__ == "__main__":
    main()
