#!/usr/bin/env python3
"""
Example usage of Hill Cipher implementation and breakers.
"""

import os
import sys
import numpy as np
import time

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.hill_cipher import HillCipher
from breakers.kpa import HillCipherKPA
from breakers.genetic import HillCipherGA, load_language_frequencies

def example_encryption_decryption():
    """Example of Hill cipher encryption and decryption."""
    print("=== Hill Cipher Encryption and Decryption ===")
    
    # Create a Hill cipher instance
    hill = HillCipher(2)
    
    # Define a key matrix
    key = np.array([[3, 2], [5, 3]])
    
    # Define plaintext
    plaintext = "HELLOWORLDTHISISANEXAMPLE"
    
    print(f"Plaintext: {plaintext}")
    print(f"Key matrix:\n{key}")
    
    # Encrypt
    ciphertext = hill.encrypt(plaintext, key)
    print(f"Ciphertext: {ciphertext}")
    
    # Decrypt
    decrypted = hill.decrypt(ciphertext, key)
    print(f"Decrypted: {decrypted}")
    
    # Verify
    print(f"Decryption successful: {decrypted == plaintext}")
    print()

def example_known_plaintext_attack():
    """Example of known-plaintext attack."""
    print("=== Known-Plaintext Attack ===")
    
    # Create a Hill cipher instance
    hill = HillCipher(2)
    
    # Define a key matrix (this would be unknown in a real attack)
    key = np.array([[3, 2], [5, 3]])
    
    # Define plaintext and encrypt it
    plaintext = "HELLOWORLDTHISISANEXAMPLE"
    ciphertext = hill.encrypt(plaintext, key)
    
    print(f"Plaintext: {plaintext}")
    print(f"Ciphertext: {ciphertext}")
    print(f"Original key matrix:\n{key}")
    
    # Create a KPA solver
    kpa = HillCipherKPA(2)
    
    # Recover the key
    start_time = time.time()
    recovered_key = kpa.recover_key(plaintext, ciphertext)
    elapsed_time = time.time() - start_time
    
    print(f"Recovered key matrix:\n{recovered_key}")
    print(f"Key recovery time: {elapsed_time:.4f} seconds")
    
    # Verify the recovered key
    is_valid = kpa.verify_key(recovered_key, plaintext, ciphertext)
    print(f"Key verification: {is_valid}")
    
    # Decrypt using the recovered key
    decrypted = kpa.decrypt(ciphertext, recovered_key)
    print(f"Decrypted: {decrypted}")
    print()

def example_genetic_algorithm():
    """Example of genetic algorithm-based frequency analysis."""
    print("=== Genetic Algorithm-based Frequency Analysis ===")
    
    # Create a Hill cipher instance
    hill = HillCipher(2)
    
    # Define a key matrix (this would be unknown in a real attack)
    key = np.array([[3, 2], [5, 3]])
    
    # Define plaintext and encrypt it
    plaintext = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
    ciphertext = hill.encrypt(plaintext, key)
    
    print(f"Plaintext: {plaintext}")
    print(f"Ciphertext: {ciphertext}")
    print(f"Original key matrix:\n{key}")
    
    # Load language frequencies
    language_frequencies = load_language_frequencies()
    
    # Create a GA solver
    ga = HillCipherGA(2, language_frequencies)
    
    # Set GA parameters for a quick example
    ga.population_size = 50
    ga.elite_size = 5
    ga.mutation_rate = 0.2
    ga.crossover_rate = 0.8
    
    # Crack the cipher
    print("Running genetic algorithm (this may take a while)...")
    start_time = time.time()
    recovered_key, decrypted = ga.crack(ciphertext, generations=20, early_stopping=5)
    elapsed_time = time.time() - start_time
    
    print(f"GA completed in {elapsed_time:.2f} seconds")
    
    if recovered_key is not None:
        print(f"Best key found:\n{recovered_key}")
        print(f"Decryption: {decrypted[:50]}...")
    else:
        print("Failed to find a valid key")
    print()

def main():
    """Main function."""
    example_encryption_decryption()
    example_known_plaintext_attack()
    example_genetic_algorithm()

if __name__ == "__main__":
    main()
