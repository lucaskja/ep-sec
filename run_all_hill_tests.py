#!/usr/bin/env python3
"""
Run all Hill cipher tests and validate search space reduction effectiveness.
"""

import sys
import os
import time
import json
import numpy as np

sys.path.append('hill_cipher')

from hill_cipher.breakers.enhanced_breaker import EnhancedHillBreaker
from hill_cipher.breakers.optimized_breaker import OptimizedHillBreaker

def load_normalized_text():
    """Load normalized text for validation."""
    try:
        with open('hill_cipher/data/normalized_text.txt', 'r', encoding='utf-8') as f:
            return f.read().upper().replace(' ', '').replace('\n', '')
    except Exception as e:
        print(f"Warning: Could not load normalized text: {e}")
        return ""

def validate_result(decrypted_text, normalized_text, expected_plaintext=None):
    """Validate decryption result."""
    if not decrypted_text:
        return False, "No decrypted text"
    
    clean_decrypted = decrypted_text.upper().replace(' ', '').replace('\n', '').rstrip('X')
    
    # Check against expected plaintext if provided
    if expected_plaintext:
        clean_expected = expected_plaintext.upper().replace(' ', '').replace('\n', '')
        if clean_decrypted == clean_expected:
            return True, "Exact match with expected plaintext"
        elif clean_decrypted in clean_expected:
            return True, "Partial match with expected plaintext"
    
    # Check against normalized text
    if normalized_text and clean_decrypted in normalized_text:
        pos = normalized_text.find(clean_decrypted)
        return True, f"Found in normalized text at position {pos}"
    
    return False, "No valid match found"

def test_2x2_validation():
    """Test 2x2 ciphers to validate our approach works."""
    print("\n" + "="*50)
    print("VALIDATING 2x2 CIPHER CRACKING")
    print("="*50)
    
    # Known results from previous successful cracks
    test_cases = {
        'known_2x2': {
            'file': 'textos_conhecidos/Cifrado/Hill/2x2_texto_cifrado.txt',
            'expected_key': [23, 0, 17, 9],
            'expected_start': 'PARAJOAOMEUFILHOQUEMESTAAIBERNARDOHAMLET'
        },
        'unknown_2x2': {
            'file': 'textos_desconhecidos/Cifrado/Hill/2x2_texto_cifrado.txt',
            'expected_key': [23, 0, 14, 5],
            'expected_start': None  # Will be determined
        }
    }
    
    normalized_text = load_normalized_text()
    results = {}
    
    for test_name, test_data in test_cases.items():
        print(f"\nTesting {test_name}...")
        
        try:
            # Load ciphertext
            with open(test_data['file'], 'r') as f:
                ciphertext = f.read().strip()
            
            print(f"Ciphertext: {ciphertext[:50]}...")
            print(f"Length: {len(ciphertext)}")
            print(f"Expected key: {test_data['expected_key']}")
            
            # Test with enhanced breaker (exhaustive search)
            breaker = EnhancedHillBreaker(2)
            start_time = time.time()
            result = breaker.break_cipher_exhaustive(ciphertext)
            elapsed = time.time() - start_time
            
            if result['success']:
                found_key = result['key'].flatten().tolist()
                expected_key = test_data['expected_key']
                
                print(f"✓ SUCCESS!")
                print(f"  Found key: {found_key}")
                print(f"  Expected:   {expected_key}")
                print(f"  Key match:  {found_key == expected_key}")
                print(f"  Keys tested: {result['keys_tested']:,}")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Score: {result['score']:.2f}")
                print(f"  Decrypted: {result['decrypted_text'][:60]}...")
                
                # Validate result
                is_valid, validation_msg = validate_result(
                    result['decrypted_text'], 
                    normalized_text, 
                    test_data.get('expected_start')
                )
                print(f"  Validation: {validation_msg}")
                
                results[test_name] = {
                    'success': True,
                    'key_match': found_key == expected_key,
                    'found_key': found_key,
                    'keys_tested': result['keys_tested'],
                    'time_elapsed': elapsed,
                    'validation': is_valid,
                    'decrypted_text': result['decrypted_text']
                }
            else:
                print(f"✗ FAILED to crack {test_name}")
                results[test_name] = {'success': False}
        
        except Exception as e:
            print(f"✗ ERROR testing {test_name}: {e}")
            results[test_name] = {'success': False, 'error': str(e)}
    
    return results

def test_3x3_optimized():
    """Test 3x3 ciphers with optimized techniques."""
    print("\n" + "="*50)
    print("TESTING 3x3 CIPHERS WITH OPTIMIZED TECHNIQUES")
    print("="*50)
    
    test_cases = {
        'known_3x3': 'textos_conhecidos/Cifrado/Hill/3x3_texto_cifrado.txt',
        'unknown_3x3': 'textos_desconhecidos/Cifrado/Hill/3x3_texto_cifrado.txt'
    }
    
    normalized_text = load_normalized_text()
    results = {}
    
    for test_name, file_path in test_cases.items():
        print(f"\nTesting {test_name}...")
        
        try:
            # Load ciphertext
            with open(file_path, 'r') as f:
                ciphertext = f.read().strip()
            
            print(f"Ciphertext: {ciphertext[:50]}...")
            print(f"Length: {len(ciphertext)}")
            
            # Test with optimized breaker
            breaker = OptimizedHillBreaker(3)
            start_time = time.time()
            
            result = breaker.break_cipher_optimized(
                ciphertext,
                max_time=900,  # 15 minutes
                max_keys_per_technique=20000,
                early_stopping_score=-50,
                use_parallel=True,
                num_processes=4
            )
            
            elapsed = time.time() - start_time
            
            if result['success'] or result['score'] > -1000:
                print(f"✓ RESULT FOUND!")
                print(f"  Success: {result['success']}")
                print(f"  Best key: {result['key']}")
                print(f"  Score: {result['score']:.2f}")
                print(f"  Technique: {result['technique_used']}")
                print(f"  Keys tested: {result['keys_tested']:,}")
                print(f"  Time: {elapsed:.2f}s")
                
                if result['decrypted_text']:
                    print(f"  Decrypted: {result['decrypted_text'][:60]}...")
                    
                    # Validate result
                    is_valid, validation_msg = validate_result(
                        result['decrypted_text'], 
                        normalized_text
                    )
                    print(f"  Validation: {validation_msg}")
                    
                    results[test_name] = {
                        'success': result['success'],
                        'found_key': result['key'],
                        'score': result['score'],
                        'technique_used': result['technique_used'],
                        'keys_tested': result['keys_tested'],
                        'time_elapsed': elapsed,
                        'validation': is_valid,
                        'decrypted_text': result['decrypted_text']
                    }
                else:
                    results[test_name] = {
                        'success': False,
                        'score': result['score'],
                        'keys_tested': result['keys_tested'],
                        'time_elapsed': elapsed
                    }
            else:
                print(f"✗ No good result found for {test_name}")
                results[test_name] = {
                    'success': False,
                    'best_score': result['score'],
                    'keys_tested': result['keys_tested'],
                    'time_elapsed': elapsed
                }
        
        except Exception as e:
            print(f"✗ ERROR testing {test_name}: {e}")
            results[test_name] = {'success': False, 'error': str(e)}
    
    return results

def test_search_space_reduction():
    """Test search space reduction effectiveness."""
    print("\n" + "="*50)
    print("TESTING SEARCH SPACE REDUCTION EFFECTIVENESS")
    print("="*50)
    
    from hill_cipher.breakers.search_space_reducer import SearchSpaceReducer
    
    for key_size in [2, 3, 4]:
        print(f"\n{key_size}x{key_size} Matrix Search Space:")
        print("-" * 30)
        
        reducer = SearchSpaceReducer(key_size)
        total_space = reducer.estimate_total_search_space()
        
        print(f"Total estimated space: {total_space:,}")
        
        # Test different reduction techniques
        techniques = [
            ('determinant_constraint', 'Determinant constraint'),
            ('diagonal', 'Diagonal matrices only'),
            ('upper_triangular', 'Upper triangular matrices')
        ]
        
        for technique_name, technique_desc in techniques:
            reduction_factor = reducer.estimate_reduction_factor(technique_name)
            reduced_space = int(total_space / reduction_factor)
            
            print(f"{technique_desc:25s}: {reduction_factor:8.1f}x -> {reduced_space:,} keys")
        
        # Combined reduction
        det_reduction = reducer.estimate_reduction_factor('determinant_constraint')
        diag_reduction = reducer.estimate_reduction_factor('diagonal')
        combined_reduction = det_reduction * diag_reduction
        combined_space = int(total_space / combined_reduction)
        
        print(f"{'Combined (det + diag)':25s}: {combined_reduction:8.1f}x -> {combined_space:,} keys")
        
        # Estimate time savings
        if key_size == 2:
            # We know 2x2 takes about 10-15 minutes for full search
            base_time_minutes = 12
            reduced_time = base_time_minutes / combined_reduction * 60  # in seconds
            print(f"Estimated time with reduction: {reduced_time:.1f} seconds (vs {base_time_minutes} minutes)")

def generate_summary_report(validation_results, optimization_results):
    """Generate a summary report."""
    print("\n" + "="*60)
    print("COMPREHENSIVE HILL CIPHER TEST SUMMARY")
    print("="*60)
    
    # 2x2 Validation Summary
    print("\n2x2 VALIDATION RESULTS:")
    print("-" * 25)
    for test_name, result in validation_results.items():
        if result.get('success'):
            key_match = result.get('key_match', False)
            validation = result.get('validation', False)
            print(f"{test_name:12s}: ✓ SUCCESS (Key match: {key_match}, Valid: {validation})")
        else:
            print(f"{test_name:12s}: ✗ FAILED")
    
    # 3x3 Optimization Summary
    print("\n3x3 OPTIMIZATION RESULTS:")
    print("-" * 27)
    for test_name, result in optimization_results.items():
        if result.get('success'):
            technique = result.get('technique_used', 'unknown')
            validation = result.get('validation', False)
            print(f"{test_name:12s}: ✓ SUCCESS ({technique}, Valid: {validation})")
        else:
            score = result.get('score', result.get('best_score', 0))
            print(f"{test_name:12s}: ✗ FAILED (Best score: {score:.2f})")
    
    # Overall Statistics
    total_2x2 = len(validation_results)
    success_2x2 = sum(1 for r in validation_results.values() if r.get('success'))
    
    total_3x3 = len(optimization_results)
    success_3x3 = sum(1 for r in optimization_results.values() if r.get('success'))
    
    print(f"\nOVERALL STATISTICS:")
    print(f"2x2 Success Rate: {success_2x2}/{total_2x2} ({success_2x2/total_2x2*100:.1f}%)")
    print(f"3x3 Success Rate: {success_3x3}/{total_3x3} ({success_3x3/total_3x3*100:.1f}%)")
    print(f"Total Success Rate: {success_2x2+success_3x3}/{total_2x2+total_3x3} "
          f"({(success_2x2+success_3x3)/(total_2x2+total_3x3)*100:.1f}%)")

def main():
    """Main function to run all tests."""
    print("COMPREHENSIVE HILL CIPHER TESTING SUITE")
    print("=" * 50)
    print("This script will test all Hill cipher cracking techniques")
    print("and validate the effectiveness of search space reduction.")
    
    start_time = time.time()
    
    # Test 1: Validate 2x2 approach works
    validation_results = test_2x2_validation()
    
    # Test 2: Test 3x3 with optimized techniques
    optimization_results = test_3x3_optimized()
    
    # Test 3: Demonstrate search space reduction
    test_search_space_reduction()
    
    # Generate summary
    generate_summary_report(validation_results, optimization_results)
    
    total_time = time.time() - start_time
    print(f"\nTotal testing time: {total_time:.2f} seconds")
    
    # Save results
    all_results = {
        'validation_results': validation_results,
        'optimization_results': optimization_results,
        'total_time': total_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    all_results = convert_numpy(all_results)
    
    with open('hill_cipher_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to hill_cipher_test_results.json")

if __name__ == "__main__":
    main()
