#!/usr/bin/env python3
"""
Find the correct key using KPA and validate statistical approach.
"""

import sys
import os
sys.path.append('hill_cipher')

from hill_cipher.breakers.kpa import HillCipherKPA
from hill_cipher.breakers.statistical_analyzer import StatisticalAnalyzer
from hill_cipher.core.hill_cipher import HillCipher

def main():
    # Known data for 2x2 cipher
    ciphertext = "ypewhabanavprxgyekypbaonoefvdpisnxlwbabsgewuclweqktwkklkfkgyigzpbavsdxrwxacluufwjfugcwsarcoelklfowlhpnvwokmglxnpegoapjlp"
    known_plaintext = "NTAOPARANAOTERQUEENTRARNUMALUTACORPORALCOMMINHAMAEVOCETEVEQUESETRANCARNOBANHEIROEPASSOUALGUMTEMPOOUV"
    
    print("Finding Correct Key using KPA")
    print("=" * 40)
    
    # Use KPA to find the correct key
    kpa = HillCipherKPA(key_size=2)
    correct_key = kpa.attack(ciphertext, known_plaintext)
    
    if correct_key is not None:
        print(f"✓ Correct key found: {correct_key.flatten()}")
        print(f"Key matrix:\n{correct_key}")
        
        # Verify the key
        hill = HillCipher(2)
        decrypted = hill.decrypt(ciphertext, correct_key)
        print(f"Decrypted: {decrypted}")
        
        # Now test if our statistical analyzer can find this key
        print("\nTesting Statistical Analyzer")
        print("=" * 30)
        
        analyzer = StatisticalAnalyzer(key_size=2)
        
        # Test the correct key's score
        score = analyzer.score_text(decrypted)
        print(f"Score for correct decryption: {score:.2f}")
        
        # Now run statistical analysis with more candidates
        print("\nRunning statistical analysis with more candidates...")
        key, decrypted_stat, best_score = analyzer.break_cipher_statistical(
            ciphertext, 
            max_candidates=10000,  # More candidates
            use_smart_generation=True,
            early_stopping_threshold=-30.0  # More lenient threshold
        )
        
        if key is not None:
            print(f"Statistical analysis found key: {key.flatten()}")
            print(f"Matches correct key: {(key == correct_key).all()}")
        else:
            print("Statistical analysis failed to find key")
    
    else:
        print("✗ KPA failed to find correct key")

if __name__ == "__main__":
    main()
