#!/usr/bin/env python3
"""
Test the search space reduction techniques.
"""

import sys
import time
import numpy as np

sys.path.append('hill_cipher')

from hill_cipher.breakers.search_space_reducer import SearchSpaceReducer

def test_reduction_techniques():
    """Test different reduction techniques and measure their effectiveness."""
    print("Hill Cipher Search Space Reduction Test")
    print("=" * 50)
    
    # Test for 3x3 keys
    reducer = SearchSpaceReducer(key_size=3)
    
    # Show recommendations
    print("\nReduction Recommendations:")
    recommendations = reducer.get_reduction_recommendations()
    for technique, description, reduction in recommendations:
        print(f"{technique:20s}: {description}")
        print(f"{'':20s}  Reduction factor: {reduction:.1f}x")
        print()
    
    # Test key generation speed for different techniques
    techniques = [
        ('diagonal', 'Diagonal matrices'),
        ('upper_triangular', 'Upper triangular matrices'),
        ('determinant_constraint', 'Determinant-constrained matrices'),
        ('smart_sampling', 'Smart sampling')
    ]
    
    print("Key Generation Speed Test (3x3):")
    print("-" * 40)
    
    for technique_name, technique_desc in techniques:
        print(f"\nTesting {technique_desc}...")
        
        start_time = time.time()
        count = 0
        max_keys = 1000
        
        try:
            if technique_name == 'diagonal':
                generator = reducer._generate_diagonal()
            elif technique_name == 'upper_triangular':
                generator = reducer._generate_upper_triangular()
            elif technique_name == 'determinant_constraint':
                generator = reducer.generate_keys_by_determinant_constraint([1, 3, 5])  # Test with a few determinants
            elif technique_name == 'smart_sampling':
                generator = reducer.generate_keys_smart_sampling(max_keys)
            
            for key in generator:
                count += 1
                if count >= max_keys:
                    break
                
                # Show first few keys
                if count <= 3:
                    print(f"  Key {count}: {key.flatten()}")
            
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            
            print(f"  Generated {count} keys in {elapsed:.2f}s ({rate:.1f} keys/sec)")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test the effectiveness on a sample ciphertext
    print(f"\nTesting with sample 3x3 ciphertext:")
    sample_cipher = "ysigztwrqxoegwfwveyjlcjlkpqbcggpqkdymglsavyacolzewfoxglvalewktqczasmtihavacolzewfstaocaxqvopiwkaxiwyawcjljaalrgpgqvgezmn"
    print(f"Ciphertext: {sample_cipher[:50]}...")
    
    # Test frequency-based generation
    print(f"\nTesting frequency-based generation...")
    start_time = time.time()
    count = 0
    max_keys = 1000
    
    for key in reducer.generate_keys_frequency_based(sample_cipher, max_keys):
        count += 1
        if count <= 3:
            print(f"  Key {count}: {key.flatten()}")
        if count >= max_keys:
            break
    
    elapsed = time.time() - start_time
    rate = count / elapsed if elapsed > 0 else 0
    print(f"  Generated {count} frequency-based keys in {elapsed:.2f}s ({rate:.1f} keys/sec)")

def demonstrate_reduction_effectiveness():
    """Demonstrate the effectiveness of different reduction techniques."""
    print(f"\n{'='*60}")
    print("REDUCTION EFFECTIVENESS DEMONSTRATION")
    print(f"{'='*60}")
    
    for key_size in [2, 3, 4]:
        print(f"\nKey Size: {key_size}x{key_size}")
        print("-" * 30)
        
        reducer = SearchSpaceReducer(key_size)
        total_space = reducer.estimate_total_search_space()
        
        print(f"Estimated total search space: {total_space:,}")
        
        # Calculate reductions
        techniques = [
            'determinant_constraint',
            'upper_triangular', 
            'diagonal'
        ]
        
        for technique in techniques:
            reduction = reducer.estimate_reduction_factor(technique)
            reduced_space = int(total_space / reduction)
            print(f"{technique:20s}: {reduction:8.1f}x reduction -> {reduced_space:,} keys")
        
        # Combined reduction (determinant + diagonal)
        combined_reduction = (reducer.estimate_reduction_factor('determinant_constraint') * 
                            reducer.estimate_reduction_factor('diagonal'))
        combined_space = int(total_space / combined_reduction)
        print(f"{'Combined (det+diag)':20s}: {combined_reduction:8.1f}x reduction -> {combined_space:,} keys")

if __name__ == "__main__":
    test_reduction_techniques()
    demonstrate_reduction_effectiveness()
