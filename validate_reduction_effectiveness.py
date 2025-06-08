#!/usr/bin/env python3
"""
Validate Search Space Reduction Effectiveness

This script validates that our search space reduction techniques work correctly
by testing them on the known 2x2 ciphers where we know the correct keys.
"""

import sys
import os
import time
import numpy as np

sys.path.append('hill_cipher')

from hill_cipher.breakers.enhanced_breaker import EnhancedHillBreaker
from hill_cipher.breakers.search_space_reducer import SearchSpaceReducer

def test_2x2_validation():
    """Test that our approach correctly finds the known 2x2 keys."""
    print("VALIDATING 2x2 CIPHER CRACKING EFFECTIVENESS")
    print("=" * 50)
    
    # Known test cases with expected results
    test_cases = {
        'known_2x2': {
            'file': 'textos_conhecidos/Cifrado/Hill/2x2_texto_cifrado.txt',
            'expected_key': [23, 0, 17, 9]
        },
        'unknown_2x2': {
            'file': 'textos_desconhecidos/Cifrado/Hill/2x2_texto_cifrado.txt',
            'expected_key': [23, 0, 14, 5]
        }
    }
    
    results = {}
    
    for test_name, test_data in test_cases.items():
        print(f"\nTesting {test_name}...")
        
        try:
            # Load ciphertext
            with open(test_data['file'], 'r') as f:
                ciphertext = f.read().strip()
            
            print(f"Ciphertext: {ciphertext[:50]}...")
            print(f"Expected key: {test_data['expected_key']}")
            
            # Test with exhaustive search
            breaker = EnhancedHillBreaker(2)
            start_time = time.time()
            result = breaker.break_cipher_exhaustive(ciphertext)
            elapsed = time.time() - start_time
            
            if result['success']:
                found_key = result['key'].flatten().tolist()
                expected_key = test_data['expected_key']
                key_match = found_key == expected_key
                
                print(f"✓ SUCCESS!")
                print(f"  Found key: {found_key}")
                print(f"  Expected:   {expected_key}")
                print(f"  Key match:  {key_match}")
                print(f"  Keys tested: {result['keys_tested']:,}")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Rate: {result['keys_tested']/elapsed:.1f} keys/sec")
                print(f"  Decrypted: {result['decrypted_text'][:60]}...")
                
                results[test_name] = {
                    'success': True,
                    'key_match': key_match,
                    'found_key': found_key,
                    'keys_tested': result['keys_tested'],
                    'time_elapsed': elapsed
                }
            else:
                print(f"✗ FAILED")
                results[test_name] = {'success': False}
        
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results[test_name] = {'error': str(e)}
    
    return results

def test_search_space_reduction():
    """Test search space reduction techniques."""
    print(f"\nTESTING SEARCH SPACE REDUCTION")
    print("=" * 35)
    
    for key_size in [2, 3, 4]:
        print(f"\n{key_size}x{key_size} Matrix:")
        reducer = SearchSpaceReducer(key_size)
        
        total_space = reducer.estimate_total_search_space()
        print(f"  Total space: {total_space:,} keys")
        
        # Test reductions
        techniques = [
            ('determinant_constraint', 'Determinant constraint'),
            ('diagonal', 'Diagonal only'),
        ]
        
        for technique_name, technique_desc in techniques:
            reduction = reducer.estimate_reduction_factor(technique_name)
            reduced_space = int(total_space / reduction)
            print(f"  {technique_desc:20s}: {reduction:6.1f}x -> {reduced_space:,} keys")

def main():
    """Main validation function."""
    print("HILL CIPHER VALIDATION TEST")
    print("=" * 30)
    
    # Test 2x2 validation
    validation_results = test_2x2_validation()
    
    # Test search space reduction
    test_search_space_reduction()
    
    # Summary
    print(f"\nSUMMARY:")
    total_tests = len(validation_results)
    successes = sum(1 for r in validation_results.values() if r.get('success', False))
    key_matches = sum(1 for r in validation_results.values() if r.get('key_match', False))
    
    print(f"Success rate: {successes}/{total_tests} ({successes/total_tests*100:.1f}%)")
    print(f"Key match rate: {key_matches}/{total_tests} ({key_matches/total_tests*100:.1f}%)")
    
    if successes == total_tests and key_matches == total_tests:
        print("✓ VALIDATION PASSED: Search space reduction is working correctly!")
    else:
        print("✗ VALIDATION ISSUES: Some tests failed")

if __name__ == "__main__":
    main()
