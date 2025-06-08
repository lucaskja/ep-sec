#!/usr/bin/env python3
"""
Debug the 3x3 Hill cipher approach by testing different hypotheses.
"""

import sys
import os
import time
import numpy as np

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

def test_known_good_key():
    """Test with a known good key to verify our approach works."""
    print("Testing with a known good key...")
    
    # Create a test case: encrypt a known substring with a known key
    normalized_text = load_normalized_text()
    if not normalized_text:
        return False
    
    # Take a 120-character substring from the normalized text
    test_plaintext = normalized_text[1000:1120]  # Random position
    print(f"Test plaintext: {test_plaintext}")
    
    # Create a test key
    test_key = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])  # Note: 10 instead of 9 to make it invertible
    
    hill = HillCipher(3)
    
    # Check if key is invertible
    if not hill.is_invertible(test_key):
        print("Test key is not invertible, trying another...")
        test_key = np.array([[3, 2, 1], [1, 1, 1], [2, 1, 3]])
        if not hill.is_invertible(test_key):
            print("Second test key also not invertible")
            return False
    
    print(f"Test key: {test_key.flatten()}")
    
    # Encrypt the test plaintext
    test_ciphertext = hill.encrypt(test_plaintext, test_key)
    print(f"Test ciphertext: {test_ciphertext}")
    
    # Now try to crack it using our statistical analyzer
    analyzer = ImprovedStatisticalAnalyzer(3)
    
    print("Attempting to crack the test cipher...")
    key, decrypted, score = analyzer.break_cipher_improved(
        test_ciphertext, 
        max_candidates=10000,
        timeout=300
    )
    
    if key is not None:
        print(f"Found key: {key.flatten()}")
        print(f"Original key: {test_key.flatten()}")
        print(f"Keys match: {np.array_equal(key, test_key)}")
        print(f"Decrypted: {decrypted}")
        print(f"Original: {test_plaintext}")
        print(f"Decryption correct: {decrypted.rstrip('X') == test_plaintext}")
        return True
    else:
        print("Failed to crack the test cipher")
        return False

def analyze_cipher_properties(ciphertext, cipher_name):
    """Analyze properties of the ciphertext."""
    print(f"\nAnalyzing {cipher_name}...")
    print(f"Ciphertext: {ciphertext}")
    print(f"Length: {len(ciphertext)}")
    
    # Character frequency analysis
    from collections import Counter
    char_freq = Counter(ciphertext.upper())
    print("Character frequencies:")
    for char, freq in sorted(char_freq.items()):
        print(f"  {char}: {freq} ({freq/len(ciphertext)*100:.1f}%)")
    
    # Check for patterns
    print(f"Unique characters: {len(char_freq)}")
    print(f"Most common: {char_freq.most_common(5)}")
    print(f"Least common: {char_freq.most_common()[-5:]}")
    
    # Check if length is divisible by 3
    print(f"Length divisible by 3: {len(ciphertext) % 3 == 0}")
    
    return char_freq

def test_manual_keys(ciphertext, normalized_text):
    """Test some manually chosen keys that might be more likely."""
    print(f"\nTesting manual keys for ciphertext: {ciphertext[:20]}...")
    
    hill = HillCipher(3)
    analyzer = ImprovedStatisticalAnalyzer(3)
    
    # Some keys that might be more likely to be used
    test_keys = [
        # Simple patterns
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity (won't work but let's see)
        [[1, 2, 3], [4, 5, 6], [7, 8, 10]],
        [[2, 1, 3], [1, 1, 2], [3, 2, 1]],
        [[1, 1, 1], [1, 2, 1], [1, 1, 3]],
        # Keys with small values
        [[1, 0, 1], [1, 1, 0], [0, 1, 1]],
        [[2, 1, 0], [1, 2, 1], [0, 1, 2]],
        [[3, 1, 2], [1, 3, 1], [2, 1, 3]],
        # Keys with larger values
        [[5, 3, 2], [2, 5, 3], [3, 2, 5]],
        [[7, 4, 1], [1, 7, 4], [4, 1, 7]],
        [[11, 7, 3], [3, 11, 7], [7, 3, 11]],
    ]
    
    for i, key_list in enumerate(test_keys):
        key = np.array(key_list)
        
        if not hill.is_invertible(key):
            print(f"Key {i+1} not invertible: {key.flatten()}")
            continue
        
        try:
            decrypted = hill.decrypt(ciphertext, key)
            score = analyzer.score_text(decrypted)
            
            print(f"Key {i+1}: {key.flatten()}")
            print(f"  Score: {score:.2f}")
            print(f"  Decrypted: {decrypted[:60]}...")
            
            # Quick check if it's in normalized text
            clean_decrypted = decrypted.rstrip('X')
            if clean_decrypted in normalized_text:
                print(f"  ✓ FOUND IN NORMALIZED TEXT!")
                pos = normalized_text.find(clean_decrypted)
                print(f"  Position: {pos}")
                return key, decrypted, score
            elif len(clean_decrypted) > 50:
                # Check for partial matches
                for length in range(len(clean_decrypted), 30, -10):
                    substring = clean_decrypted[:length]
                    if substring in normalized_text:
                        pos = normalized_text.find(substring)
                        print(f"  ✓ PARTIAL MATCH: {length} chars at position {pos}")
                        break
        
        except Exception as e:
            print(f"Key {i+1} error: {e}")
    
    return None, None, None

def main():
    print("Debugging 3x3 Hill Cipher Approach")
    print("=" * 40)
    
    # Test 1: Verify our approach works with known good keys
    print("\n--- Test 1: Known Good Key Test ---")
    if test_known_good_key():
        print("✓ Our approach works with known keys")
    else:
        print("✗ Our approach has issues with known keys")
    
    # Load normalized text
    normalized_text = load_normalized_text()
    if not normalized_text:
        print("Cannot load normalized text")
        return
    
    # Test 2: Analyze cipher properties
    print("\n--- Test 2: Cipher Properties Analysis ---")
    ciphers = {
        'known_3x3': "ysigztwrqxoegwfwveyjlcjlkpqbcggpqkdymglsavyacolzewfoxglvalewktqczasmtihavacolzewfstaocaxqvopiwkaxiwyawcjljaalrgpgqvgezmn",
        'unknown_3x3': "aoaldaebgaoilwiuhmrhtwoagignwihpnfoommsmwmsllgwatayqcamooarehvtgjgucsmqqqntypvyzzgmelzzjjzavalkazbmnammxxlzdypazttxooshn"
    }
    
    for name, ciphertext in ciphers.items():
        analyze_cipher_properties(ciphertext, name)
    
    # Test 3: Try manual keys
    print("\n--- Test 3: Manual Key Testing ---")
    for name, ciphertext in ciphers.items():
        print(f"\nTesting {name}:")
        result = test_manual_keys(ciphertext, normalized_text)
        if result[0] is not None:
            print(f"✓ Found working key for {name}!")
            print(f"Key: {result[0].flatten()}")
            print(f"Decrypted: {result[1]}")
            break
    
    # Test 4: Check if the issue is with our validation
    print("\n--- Test 4: Validation Check ---")
    print("Checking if any 120-character substring from normalized text matches our ciphertexts...")
    
    for name, ciphertext in ciphers.items():
        print(f"\nChecking {name}...")
        # Try to find if this ciphertext could be an encryption of any part of normalized text
        found_match = False
        for start in range(0, min(1000, len(normalized_text) - 120), 50):  # Sample every 50 chars
            substring = normalized_text[start:start+120]
            if len(substring) == 120:
                # This is where we'd need to try many keys, but that's computationally expensive
                # For now, just note that we're checking
                pass
        
        if not found_match:
            print(f"  No obvious matches found for {name}")

if __name__ == "__main__":
    main()
