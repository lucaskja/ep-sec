#!/usr/bin/env python3
"""
Test cracking 3x3 Hill cipher only.
"""

import sys
import os
import time

sys.path.append('hill_cipher')

from hill_cipher.breakers.improved_statistical_analyzer import ImprovedStatisticalAnalyzer

def main():
    print("Testing 3x3 Hill Cipher Cracking")
    print("=" * 35)
    
    # Load 3x3 cipher texts
    base_dir = "/Users/lucaskle/Documents/USP/Seg"
    
    # Known 3x3
    known_path = os.path.join(base_dir, 'textos_conhecidos', 'Cifrado', 'Hill', '3x3_texto_cifrado.txt')
    with open(known_path, 'r') as f:
        known_3x3 = f.read().strip()
    
    # Unknown 3x3
    unknown_path = os.path.join(base_dir, 'textos_desconhecidos', 'Cifrado', 'Hill', '3x3_texto_cifrado.txt')
    with open(unknown_path, 'r') as f:
        unknown_3x3 = f.read().strip()
    
    print(f"Known 3x3: {known_3x3}")
    print(f"Unknown 3x3: {unknown_3x3}")
    print(f"Lengths: {len(known_3x3)}, {len(unknown_3x3)}")
    
    # Test with smaller number of candidates first
    analyzer = ImprovedStatisticalAnalyzer(3)
    
    print("\nTesting known 3x3 with 5000 candidates...")
    start_time = time.time()
    
    key, decrypted, score = analyzer.break_cipher_improved(
        known_3x3, 
        max_candidates=5000,
        timeout=600  # 10 minutes
    )
    
    elapsed = time.time() - start_time
    
    if key is not None:
        print(f"✓ SUCCESS in {elapsed:.1f}s")
        print(f"Key: {key.flatten()}")
        print(f"Score: {score:.2f}")
        print(f"Decrypted: {decrypted}")
    else:
        print(f"✗ FAILED after {elapsed:.1f}s")
    
    print("\nTesting unknown 3x3 with 5000 candidates...")
    start_time = time.time()
    
    key, decrypted, score = analyzer.break_cipher_improved(
        unknown_3x3, 
        max_candidates=5000,
        timeout=600  # 10 minutes
    )
    
    elapsed = time.time() - start_time
    
    if key is not None:
        print(f"✓ SUCCESS in {elapsed:.1f}s")
        print(f"Key: {key.flatten()}")
        print(f"Score: {score:.2f}")
        print(f"Decrypted: {decrypted}")
    else:
        print(f"✗ FAILED after {elapsed:.1f}s")

if __name__ == "__main__":
    main()
