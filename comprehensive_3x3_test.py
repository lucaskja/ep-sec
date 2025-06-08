#!/usr/bin/env python3
"""
Comprehensive 3x3 Hill cipher test with multiple approaches.
"""

import sys
import os
import time
import numpy as np
from itertools import product

sys.path.append('hill_cipher')

from hill_cipher.breakers.improved_statistical_analyzer import ImprovedStatisticalAnalyzer
from hill_cipher.core.hill_cipher import HillCipher

def load_normalized_text():
    """Load the normalized text for validation."""
    normalized_path = os.path.join('hill_cipher', 'data', 'normalized_text.txt')
    try:
        with open(normalized_path, 'r', encoding='utf-8') as f:
            return f.read().upper().replace(' ', '').replace('\n', '')
    except Exception as e:
        print(f"Error loading normalized text: {e}")
        return ""

def validate_against_normalized(decrypted_text, normalized_text, min_match_length=50):
    """Check if decrypted text matches normalized text."""
    clean_decrypted = decrypted_text.upper().replace(' ', '').replace('\n', '').rstrip('X')
    
    # Try exact match
    if clean_decrypted in normalized_text:
        pos = normalized_text.find(clean_decrypted)
        return True, pos, len(clean_decrypted), 100.0
    
    # Try partial matches
    best_match_length = 0
    best_position = -1
    
    # Check substrings of decreasing length
    for length in range(len(clean_decrypted), min_match_length - 1, -5):
        for start in range(len(clean_decrypted) - length + 1):
            substring = clean_decrypted[start:start + length]
            pos = normalized_text.find(substring)
            if pos != -1 and length > best_match_length:
                best_match_length = length
                best_position = pos
                break
        if best_match_length > 0:
            break
    
    if best_match_length >= min_match_length:
        match_percentage = (best_match_length / len(clean_decrypted)) * 100
        return True, best_position, best_match_length, match_percentage
    
    return False, -1, 0, 0.0

def targeted_search_3x3(ciphertext, normalized_text, max_time=1800):
    """
    Targeted search for 3x3 keys using multiple strategies.
    """
    print("Starting targeted search for 3x3 Hill cipher...")
    
    hill = HillCipher(3)
    analyzer = ImprovedStatisticalAnalyzer(3)
    
    start_time = time.time()
    best_key = None
    best_decrypted = None
    best_score = float('-inf')
    best_validation = (False, -1, 0, 0.0)
    
    strategies = [
        ("High candidates", 50000),
        ("Very high candidates", 100000),
        ("Exhaustive sampling", 200000)
    ]
    
    for strategy_name, max_candidates in strategies:
        if time.time() - start_time > max_time:
            break
            
        print(f"\nTrying {strategy_name} with {max_candidates} candidates...")
        strategy_start = time.time()
        
        try:
            key, decrypted, score = analyzer.break_cipher_improved(
                ciphertext,
                max_candidates=max_candidates,
                timeout=max_time - (time.time() - start_time)
            )
            
            strategy_time = time.time() - strategy_start
            
            if key is not None:
                print(f"Found key in {strategy_time:.1f}s: {key.flatten()}")
                print(f"Score: {score:.2f}")
                print(f"Decrypted: {decrypted[:80]}...")
                
                # Validate
                is_valid, pos, match_len, match_pct = validate_against_normalized(
                    decrypted, normalized_text
                )
                
                if is_valid:
                    print(f"✓ VALIDATION SUCCESS!")
                    print(f"  Position: {pos}, Length: {match_len}, Match: {match_pct:.1f}%")
                    return key, decrypted, score, (is_valid, pos, match_len, match_pct)
                else:
                    print(f"✗ Validation failed")
                    if score > best_score:
                        best_key = key
                        best_decrypted = decrypted
                        best_score = score
                        best_validation = (is_valid, pos, match_len, match_pct)
            else:
                print(f"No key found in {strategy_time:.1f}s")
        
        except Exception as e:
            print(f"Error in {strategy_name}: {e}")
    
    return best_key, best_decrypted, best_score, best_validation

def smart_brute_force_3x3(ciphertext, normalized_text, max_keys=1000000):
    """
    Smart brute force approach for 3x3 keys.
    Focus on keys that are more likely to be valid.
    """
    print(f"Starting smart brute force for 3x3 (max {max_keys} keys)...")
    
    hill = HillCipher(3)
    analyzer = ImprovedStatisticalAnalyzer(3)
    
    tested = 0
    best_key = None
    best_decrypted = None
    best_score = float('-inf')
    
    # Focus on smaller values first (more likely to be used)
    value_ranges = [
        range(0, 10),    # 0-9 first
        range(10, 20),   # 10-19 second
        range(20, 26)    # 20-25 last
    ]
    
    start_time = time.time()
    
    for value_range in value_ranges:
        print(f"Testing values in range {value_range.start}-{value_range.stop-1}...")
        
        for key_values in product(value_range, repeat=9):
            if tested >= max_keys:
                break
                
            key = np.array(key_values).reshape(3, 3)
            
            if not hill.is_invertible(key):
                continue
            
            tested += 1
            
            try:
                decrypted = hill.decrypt(ciphertext, key)
                score = analyzer.score_text(decrypted)
                
                if tested % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = tested / elapsed
                    print(f"  Tested {tested} keys ({rate:.1f} keys/sec, best score: {best_score:.2f})")
                
                if score > best_score:
                    best_score = score
                    best_key = key.copy()
                    best_decrypted = decrypted
                    
                    # Quick validation check
                    is_valid, pos, match_len, match_pct = validate_against_normalized(
                        decrypted, normalized_text
                    )
                    
                    if is_valid:
                        print(f"✓ FOUND VALID KEY: {key.flatten()}")
                        print(f"  Score: {score:.2f}")
                        print(f"  Position: {pos}, Length: {match_len}, Match: {match_pct:.1f}%")
                        print(f"  Decrypted: {decrypted}")
                        return key, decrypted, score, (is_valid, pos, match_len, match_pct)
                    
                    if score > -1000000:  # Very good score threshold
                        print(f"New best score: {score:.2f} with key {key.flatten()}")
                        print(f"  Decrypted: {decrypted[:60]}...")
            
            except Exception as e:
                continue
        
        if tested >= max_keys:
            break
    
    elapsed = time.time() - start_time
    print(f"Smart brute force completed: {tested} keys in {elapsed:.1f}s")
    
    if best_key is not None:
        is_valid, pos, match_len, match_pct = validate_against_normalized(
            best_decrypted, normalized_text
        )
        return best_key, best_decrypted, best_score, (is_valid, pos, match_len, match_pct)
    
    return None, None, float('-inf'), (False, -1, 0, 0.0)

def test_cipher_comprehensive(ciphertext, cipher_name, normalized_text):
    """Comprehensive test of a cipher using multiple approaches."""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST: {cipher_name}")
    print(f"{'='*60}")
    print(f"Ciphertext: {ciphertext}")
    print(f"Length: {len(ciphertext)}")
    
    # Method 1: Targeted statistical search
    print(f"\n--- Method 1: Targeted Statistical Search ---")
    key1, decrypted1, score1, validation1 = targeted_search_3x3(
        ciphertext, normalized_text, max_time=900  # 15 minutes
    )
    
    if validation1[0]:  # If validation successful
        return key1, decrypted1, score1, validation1
    
    # Method 2: Smart brute force
    print(f"\n--- Method 2: Smart Brute Force ---")
    key2, decrypted2, score2, validation2 = smart_brute_force_3x3(
        ciphertext, normalized_text, max_keys=500000
    )
    
    if validation2[0]:  # If validation successful
        return key2, decrypted2, score2, validation2
    
    # Return the best result
    if score1 > score2:
        return key1, decrypted1, score1, validation1
    else:
        return key2, decrypted2, score2, validation2

def main():
    print("Comprehensive 3x3 Hill Cipher Analysis")
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
        key, decrypted, score, validation = test_cipher_comprehensive(
            ciphertext, cipher_name, normalized_text
        )
        
        results[cipher_name] = {
            'key': key.tolist() if key is not None else None,
            'decrypted': decrypted,
            'score': score,
            'validation_success': validation[0],
            'validation_position': validation[1],
            'validation_length': validation[2],
            'validation_percentage': validation[3]
        }
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    for name, result in results.items():
        if result['validation_success']:
            print(f"✓ {name}: SUCCESS")
            print(f"  Key: {result['key']}")
            print(f"  Score: {result['score']:.2f}")
            print(f"  Match: {result['validation_percentage']:.1f}% at position {result['validation_position']}")
        else:
            print(f"✗ {name}: FAILED")
            print(f"  Best score: {result['score']:.2f}")
    
    # Save results
    import json
    with open('comprehensive_3x3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to comprehensive_3x3_results.json")

if __name__ == "__main__":
    main()
