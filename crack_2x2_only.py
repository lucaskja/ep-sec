#!/usr/bin/env python3
"""
Crack only the 2x2 Hill cipher texts first.
"""

import sys
import os
import time
import json

sys.path.append('hill_cipher')

from hill_cipher.breakers.improved_statistical_analyzer import ImprovedStatisticalAnalyzer
from hill_cipher.breakers.kpa import HillCipherKPA

def main():
    print("Hill Cipher 2x2 Cracking")
    print("=" * 30)
    
    # Known 2x2 texts
    texts = {
        'known_2x2': {
            'ciphertext': "ypewhabanavprxgyekypbaonoefvdpisnxlwbabsgewuclweqktwkklkfkgyigzpbavsdxrwxacluufwjfugcwsarcoelklfowlhpnvwokmglxnpegoapjlp",
            'known_plaintext': "NTAOPARANAOTERQUEENTRARNUMALUTACORPORALCOMMINHAMAEVOCETEVEQUESETRANCARNOBANHEIROEPASSOUALGUMTEMPOOUV"
        },
        'unknown_2x2': {
            'ciphertext': "ojabcmosjohiapckspqpepoicnaosmluqmkaqzekmiaiuayacsmsskkneranzuraojabckjouwqauuhszoehmdklanzucszocxshpuirmxehydshqcwuxubo",
            'known_plaintext': "CHUVOSAORITMODACIDADEDIMINUIASPESSOASFICAMEMCASAEOSOMCONSTANTEDACHUVACRIAUMAMELODIARELAXANTEEODIAPER"
        }
    }
    
    results = {}
    
    for name, data in texts.items():
        print(f"\nCracking {name}...")
        print(f"Ciphertext: {data['ciphertext']}")
        print(f"Length: {len(data['ciphertext'])}")
        
        start_time = time.time()
        
        # Try KPA first
        print("Trying KPA...")
        kpa = HillCipherKPA(2)
        key = kpa.attack(data['ciphertext'], data['known_plaintext'])
        
        if key is not None:
            from hill_cipher.core.hill_cipher import HillCipher
            hill = HillCipher(2)
            decrypted = hill.decrypt(data['ciphertext'], key)
            
            analyzer = ImprovedStatisticalAnalyzer(2)
            score = analyzer.score_text(decrypted)
            
            elapsed = time.time() - start_time
            
            results[name] = {
                'success': True,
                'method': 'KPA',
                'key': key.tolist(),
                'decrypted': decrypted,
                'score': score,
                'time': elapsed
            }
            
            print(f"✓ SUCCESS with KPA in {elapsed:.2f}s")
            print(f"Key: {key.flatten()}")
            print(f"Score: {score:.2f}")
            print(f"Decrypted: {decrypted}")
            
            # Validate against known plaintext
            known_clean = data['known_plaintext'].upper().replace(' ', '')
            decrypted_clean = decrypted.upper().rstrip('X')
            
            if known_clean == decrypted_clean or known_clean in decrypted_clean:
                print("✓ Validation: Matches known plaintext!")
            else:
                print("⚠ Validation: Doesn't match known plaintext")
                print(f"Expected: {known_clean[:50]}...")
                print(f"Got:      {decrypted_clean[:50]}...")
        
        else:
            print("KPA failed, trying statistical analysis...")
            
            analyzer = ImprovedStatisticalAnalyzer(2)
            key, decrypted, score = analyzer.break_cipher_improved(data['ciphertext'])
            
            elapsed = time.time() - start_time
            
            if key is not None:
                results[name] = {
                    'success': True,
                    'method': 'Statistical',
                    'key': key.tolist(),
                    'decrypted': decrypted,
                    'score': score,
                    'time': elapsed
                }
                
                print(f"✓ SUCCESS with Statistical in {elapsed:.2f}s")
                print(f"Key: {key.flatten()}")
                print(f"Score: {score:.2f}")
                print(f"Decrypted: {decrypted}")
            else:
                results[name] = {
                    'success': False,
                    'method': 'Statistical',
                    'time': elapsed
                }
                print(f"✗ FAILED after {elapsed:.2f}s")
    
    # Summary
    print("\n" + "=" * 30)
    print("SUMMARY")
    print("=" * 30)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    for name, result in results.items():
        if result['success']:
            print(f"✓ {name}: {result['method']} in {result['time']:.1f}s")
        else:
            print(f"✗ {name}: Failed")
    
    # Save results
    with open('hill_2x2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to hill_2x2_results.json")

if __name__ == "__main__":
    main()
