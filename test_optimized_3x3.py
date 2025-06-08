#!/usr/bin/env python3
"""
Test the optimized Hill cipher breaker on 3x3 ciphers.
"""

import sys
import os
import time

sys.path.append('hill_cipher')

from hill_cipher.breakers.optimized_breaker import OptimizedHillBreaker

def load_normalized_text():
    """Load the normalized text for validation."""
    normalized_path = os.path.join('hill_cipher', 'data', 'normalized_text.txt')
    try:
        with open(normalized_path, 'r', encoding='utf-8') as f:
            return f.read().upper().replace(' ', '').replace('\n', '')
    except Exception as e:
        print(f"Error loading normalized text: {e}")
        return ""

def test_cipher_optimized(ciphertext, cipher_name, normalized_text):
    """Test a cipher using the optimized breaker."""
    print(f"\n{'='*60}")
    print(f"TESTING: {cipher_name}")
    print(f"{'='*60}")
    print(f"Ciphertext: {ciphertext}")
    print(f"Length: {len(ciphertext)}")
    
    # Create optimized breaker
    breaker = OptimizedHillBreaker(key_size=3)
    
    # Run optimized breaking
    results = breaker.break_cipher_optimized(
        ciphertext,
        max_time=1800,  # 30 minutes
        max_keys_per_technique=100000,  # More keys per technique
        early_stopping_score=-100,  # More lenient early stopping
        use_parallel=True,
        num_processes=4
    )
    
    print(f"\nResults for {cipher_name}:")
    print(f"Success: {results['success']}")
    print(f"Time elapsed: {results['time_elapsed']:.1f} seconds")
    print(f"Keys tested: {results['keys_tested']:,}")
    print(f"Best score: {results['score']:.2f}")
    
    if results['key'] is not None:
        print(f"Best key: {results['key'].flatten()}")
        print(f"Technique used: {results['technique_used']}")
        print(f"Decrypted text: {results['decrypted_text']}")
        
        # Validate against normalized text
        is_valid, validation_info = breaker.validate_with_normalized_text(
            results['decrypted_text'], normalized_text
        )
        
        if is_valid:
            print(f"\n✓ VALIDATION SUCCESS!")
            print(f"  Match type: {validation_info['match_type']}")
            print(f"  Position: {validation_info['position']}")
            print(f"  Length: {validation_info['length']}")
            print(f"  Percentage: {validation_info['percentage']:.1f}%")
            
            # Show context
            if validation_info['position'] >= 0:
                start = max(0, validation_info['position'] - 20)
                end = min(len(normalized_text), validation_info['position'] + validation_info['length'] + 20)
                context = normalized_text[start:end]
                print(f"  Context: ...{context}...")
        else:
            print(f"\n✗ Validation failed")
    
    # Show technique breakdown
    print(f"\nTechnique breakdown:")
    for i, technique in enumerate(results['techniques_tried']):
        print(f"  {i+1}. {technique.get('technique_name', 'unknown')}: "
              f"{technique['keys_tested']} keys in {technique.get('technique_time', 0):.1f}s, "
              f"score {technique['score']:.2f}")
    
    return results

def main():
    print("Optimized 3x3 Hill Cipher Breaking Test")
    print("=" * 50)
    
    # Load normalized text
    normalized_text = load_normalized_text()
    if not normalized_text:
        print("Failed to load normalized text")
        return
    
    print(f"Normalized text loaded: {len(normalized_text)} characters")
    
    # Test ciphers
    ciphers = {
        'known_3x3': "ysigztwrqxoegwfwveyjlcjlkpqbcggpqkdymglsavyacolzewfoxglvalewktqczasmtihavacolzewfstaocaxqvopiwkaxiwyawcjljaalrgpgqvgezmn",
        'unknown_3x3': "aoaldaebgaoilwiuhmrhtwoagignwihpnfoommsmwmsllgwatayqcamooarehvtgjgucsmqqqntypvyzzgmelzzjjzavalkazbmnammxxlzdypazttxooshn"
    }
    
    results = {}
    
    for cipher_name, ciphertext in ciphers.items():
        result = test_cipher_optimized(ciphertext, cipher_name, normalized_text)
        results[cipher_name] = result
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    total_time = 0
    total_keys = 0
    successes = 0
    
    for name, result in results.items():
        total_time += result['time_elapsed']
        total_keys += result['keys_tested']
        if result['success']:
            successes += 1
        
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"{name:15s}: {status:7s} | Score: {result['score']:8.2f} | "
              f"Keys: {result['keys_tested']:6,} | Time: {result['time_elapsed']:6.1f}s")
    
    print(f"\nOverall Statistics:")
    print(f"Success rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Total keys tested: {total_keys:,}")
    print(f"Average rate: {total_keys/total_time:.1f} keys/second")
    
    # Save results
    import json
    output_data = {}
    for name, result in results.items():
        # Convert numpy arrays to lists for JSON serialization
        json_result = result.copy()
        if json_result['key'] is not None:
            json_result['key'] = json_result['key'].tolist()
        
        for technique in json_result['techniques_tried']:
            if technique['key'] is not None:
                technique['key'] = technique['key'].tolist()
        
        output_data[name] = json_result
    
    with open('optimized_3x3_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to optimized_3x3_results.json")

if __name__ == "__main__":
    main()
