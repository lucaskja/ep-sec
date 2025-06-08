#!/usr/bin/env python3
"""
Test 3x3 Hill cipher cracking with validation against normalized text.
"""

import sys
import os
import time

sys.path.append('hill_cipher')

from hill_cipher.breakers.improved_statistical_analyzer import ImprovedStatisticalAnalyzer

def load_normalized_text():
    """Load the normalized text for validation."""
    normalized_path = os.path.join('hill_cipher', 'data', 'normalized_text.txt')
    try:
        with open(normalized_path, 'r', encoding='utf-8') as f:
            return f.read().upper().replace(' ', '').replace('\n', '')
    except Exception as e:
        print(f"Error loading normalized text: {e}")
        return ""

def find_substring_in_normalized(decrypted_text, normalized_text, min_length=50):
    """
    Check if the decrypted text is a substring of the normalized text.
    
    Args:
        decrypted_text: The decrypted text to check
        normalized_text: The full normalized text
        min_length: Minimum length to consider a valid match
        
    Returns:
        Tuple of (is_match, match_position, match_length)
    """
    if len(decrypted_text) < min_length:
        return False, -1, 0
    
    # Clean the decrypted text
    clean_decrypted = decrypted_text.upper().replace(' ', '').replace('\n', '').rstrip('X')
    
    # Try to find the decrypted text in the normalized text
    position = normalized_text.find(clean_decrypted)
    if position != -1:
        return True, position, len(clean_decrypted)
    
    # Try to find partial matches (at least 80% of the text)
    min_match_length = max(min_length, int(len(clean_decrypted) * 0.8))
    
    for i in range(len(clean_decrypted) - min_match_length + 1):
        for j in range(min_match_length, len(clean_decrypted) - i + 1):
            substring = clean_decrypted[i:i+j]
            position = normalized_text.find(substring)
            if position != -1:
                return True, position, len(substring)
    
    return False, -1, 0

def test_3x3_cipher(ciphertext, cipher_name, normalized_text):
    """Test a 3x3 Hill cipher."""
    print(f"\nTesting {cipher_name}...")
    print(f"Ciphertext: {ciphertext}")
    print(f"Length: {len(ciphertext)}")
    
    # Create analyzer
    analyzer = ImprovedStatisticalAnalyzer(key_size=3)
    
    # Test with different numbers of candidates
    candidate_counts = [5000, 10000, 15000]
    
    for max_candidates in candidate_counts:
        print(f"\nTrying with {max_candidates} candidates...")
        
        start_time = time.time()
        
        try:
            key, decrypted, score = analyzer.break_cipher_improved(
                ciphertext, 
                max_candidates=max_candidates,
                timeout=600  # 10 minutes
            )
            
            elapsed = time.time() - start_time
            
            if key is not None:
                print(f"✓ Found key in {elapsed:.1f}s")
                print(f"Key: {key.flatten()}")
                print(f"Score: {score:.2f}")
                print(f"Decrypted: {decrypted[:80]}...")
                
                # Validate against normalized text
                is_match, position, match_length = find_substring_in_normalized(
                    decrypted, normalized_text
                )
                
                if is_match:
                    print(f"✓ VALIDATION SUCCESS!")
                    print(f"  Match found at position {position}")
                    print(f"  Match length: {match_length} characters")
                    print(f"  Match percentage: {match_length/len(decrypted.rstrip('X'))*100:.1f}%")
                    
                    # Show the context in normalized text
                    context_start = max(0, position - 20)
                    context_end = min(len(normalized_text), position + match_length + 20)
                    context = normalized_text[context_start:context_end]
                    print(f"  Context: ...{context}...")
                    
                    return True, key, decrypted, score
                else:
                    print(f"✗ Validation failed - not found in normalized text")
                    print(f"  Decrypted text: {decrypted}")
            else:
                print(f"✗ No key found after {elapsed:.1f}s")
        
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Error after {elapsed:.1f}s: {e}")
    
    return False, None, None, None

def main():
    print("Testing 3x3 Hill Cipher with Normalized Text Validation")
    print("=" * 60)
    
    # Load normalized text
    print("Loading normalized text...")
    normalized_text = load_normalized_text()
    
    if not normalized_text:
        print("Failed to load normalized text. Exiting.")
        return
    
    print(f"Normalized text loaded: {len(normalized_text)} characters")
    print(f"First 100 chars: {normalized_text[:100]}...")
    
    # Test ciphers
    ciphers = {
        'known_3x3': "ysigztwrqxoegwfwveyjlcjlkpqbcggpqkdymglsavyacolzewfoxglvalewktqczasmtihavacolzewfstaocaxqvopiwkaxiwyawcjljaalrgpgqvgezmn",
        'unknown_3x3': "aoaldaebgaoilwiuhmrhtwoagignwihpnfoommsmwmsllgwatayqcamooarehvtgjgucsmqqqntypvyzzgmelzzjjzavalkazbmnammxxlzdypazttxooshn"
    }
    
    results = {}
    
    for cipher_name, ciphertext in ciphers.items():
        success, key, decrypted, score = test_3x3_cipher(ciphertext, cipher_name, normalized_text)
        results[cipher_name] = {
            'success': success,
            'key': key.tolist() if key is not None else None,
            'decrypted': decrypted,
            'score': score
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    for name, result in results.items():
        if result['success']:
            print(f"✓ {name}: Successfully cracked and validated")
        else:
            print(f"✗ {name}: Failed to crack or validate")
    
    # Save results
    import json
    with open('hill_3x3_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to hill_3x3_validation_results.json")

if __name__ == "__main__":
    main()
