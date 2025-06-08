#!/usr/bin/env python3
"""
Test the score of the correct key and improve statistical analysis.
"""

import sys
import os
import numpy as np
sys.path.append('hill_cipher')

from hill_cipher.breakers.statistical_analyzer import StatisticalAnalyzer
from hill_cipher.core.hill_cipher import HillCipher

def test_key_scores():
    # Known data
    ciphertext = "ypewhabanavprxgyekypbaonoefvdpisnxlwbabsgewuclweqktwkklkfkgyigzpbavsdxrwxacluufwjfugcwsarcoelklfowlhpnvwokmglxnpegoapjlp"
    known_plaintext = "NTAOPARANAOTERQUEENTRARNUMALUTACORPORALCOMMINHAMAEVOCETEVEQUESETRANCARNOBANHEIROEPASSOUALGUMTEMPOOUV"
    
    # Correct key from KPA
    correct_key = np.array([[23, 0], [17, 9]])
    
    # Key found by statistical analysis
    stat_key = np.array([[23, 13], [17, 22]])
    
    # Create analyzer and cipher
    analyzer = StatisticalAnalyzer(key_size=2)
    hill = HillCipher(2)
    
    print("Testing Key Scores")
    print("=" * 30)
    
    # Test correct key
    correct_decrypted = hill.decrypt(ciphertext, correct_key)
    correct_score = analyzer.score_text(correct_decrypted)
    print(f"Correct key {correct_key.flatten()}: score = {correct_score:.2f}")
    print(f"Correct decrypted: {correct_decrypted[:50]}...")
    
    # Test statistical key
    stat_decrypted = hill.decrypt(ciphertext, stat_key)
    stat_score = analyzer.score_text(stat_decrypted)
    print(f"Statistical key {stat_key.flatten()}: score = {stat_score:.2f}")
    print(f"Statistical decrypted: {stat_decrypted[:50]}...")
    
    print(f"\nScore difference: {correct_score - stat_score:.2f}")
    
    # Test with different scoring methods
    print("\nTesting different scoring approaches:")
    
    # Test with only bigrams
    correct_bigrams = analyzer.calculate_ngram_frequencies(correct_decrypted, 2)
    stat_bigrams = analyzer.calculate_ngram_frequencies(stat_decrypted, 2)
    
    correct_bigram_score = -analyzer.chi_squared_test(correct_bigrams, analyzer.bigram_frequencies)
    stat_bigram_score = -analyzer.chi_squared_test(stat_bigrams, analyzer.bigram_frequencies)
    
    print(f"Correct key bigram score: {correct_bigram_score:.2f}")
    print(f"Statistical key bigram score: {stat_bigram_score:.2f}")
    
    # Test with only trigrams
    correct_trigrams = analyzer.calculate_ngram_frequencies(correct_decrypted, 3)
    stat_trigrams = analyzer.calculate_ngram_frequencies(stat_decrypted, 3)
    
    correct_trigram_score = -analyzer.chi_squared_test(correct_trigrams, analyzer.trigram_frequencies)
    stat_trigram_score = -analyzer.chi_squared_test(stat_trigrams, analyzer.trigram_frequencies)
    
    print(f"Correct key trigram score: {correct_trigram_score:.2f}")
    print(f"Statistical key trigram score: {stat_trigram_score:.2f}")

def exhaustive_search_2x2():
    """Try exhaustive search for 2x2 keys."""
    ciphertext = "ypewhabanavprxgyekypbaonoefvdpisnxlwbabsgewuclweqktwkklkfkgyigzpbavsdxrwxacluufwjfugcwsarcoelklfowlhpnvwokmglxnpegoapjlp"
    correct_key = np.array([[23, 0], [17, 9]])
    
    analyzer = StatisticalAnalyzer(key_size=2)
    hill = HillCipher(2)
    
    print("\nExhaustive Search for 2x2 Keys")
    print("=" * 35)
    
    best_keys = []
    tested = 0
    
    # Test all possible 2x2 keys
    for a in range(26):
        for b in range(26):
            for c in range(26):
                for d in range(26):
                    key = np.array([[a, b], [c, d]])
                    
                    if hill.is_invertible(key):
                        tested += 1
                        try:
                            decrypted = hill.decrypt(ciphertext, key)
                            score = analyzer.score_text(decrypted, use_multiple_ngrams=True)
                            
                            best_keys.append((key.copy(), score, decrypted))
                            
                            # Check if this is the correct key
                            if np.array_equal(key, correct_key):
                                print(f"Found correct key at position {tested}: score = {score:.2f}")
                        
                        except:
                            continue
    
    print(f"Tested {tested} valid keys")
    
    # Sort by score (higher is better)
    best_keys.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 keys by score:")
    for i, (key, score, decrypted) in enumerate(best_keys[:10]):
        is_correct = np.array_equal(key, correct_key)
        marker = "✓ CORRECT" if is_correct else ""
        print(f"{i+1:2d}. Key {key.flatten()}: score = {score:8.2f} {marker}")
        print(f"    Decrypted: {decrypted[:40]}...")
    
    # Find where the correct key ranks
    for i, (key, score, decrypted) in enumerate(best_keys):
        if np.array_equal(key, correct_key):
            print(f"\nCorrect key ranks #{i+1} out of {len(best_keys)} valid keys")
            break

if __name__ == "__main__":
    test_key_scores()
    exhaustive_search_2x2()
