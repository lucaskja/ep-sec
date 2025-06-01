# Hill Cipher Implementation in Python

This document demonstrates two implementations of the Hill Cipher in Python:
1. Known-plaintext attack
2. N-gram frequency analysis using genetic algorithms

## Requirements
```python
import numpy as np
from typing import List, Tuple
import random
import string
```

## 1. Known-Plaintext Attack Implementation

### Key Components

```python
class HillCipher:
    def __init__(self, key_size: int):
        self.key_size = key_size
        self.modulus = 26  # For English alphabet
        
    def text_to_matrix(self, text: str) -> np.ndarray:
        """Convert text to numerical matrix"""
        text = text.upper()
        numbers = [ord(char) - ord('A') for char in text]
        return np.array(numbers).reshape(-1, self.key_size)
    
    def matrix_to_text(self, matrix: np.ndarray) -> str:
        """Convert numerical matrix back to text"""
        numbers = matrix.flatten()
        return ''.join([chr(int(n) % self.modulus + ord('A')) for n in numbers])

    def encrypt(self, plaintext: str, key_matrix: np.ndarray) -> str:
        """Encrypt plaintext using Hill Cipher"""
        P = self.text_to_matrix(plaintext)
        C = np.dot(P, key_matrix) % self.modulus
        return self.matrix_to_text(C)

    def recover_key(self, plaintext: str, ciphertext: str) -> np.ndarray:
        """Recover key matrix using known plaintext-ciphertext pair"""
        P = self.text_to_matrix(plaintext)
        C = self.text_to_matrix(ciphertext)
        
        # Ensure we have enough text for key recovery
        if len(plaintext) < self.key_size * self.key_size:
            raise ValueError("Need more plaintext-ciphertext pairs")
            
        # Solve system of linear equations
        P_inv = np.linalg.inv(P[:self.key_size])
        key = np.dot(P_inv, C[:self.key_size]) % self.modulus
        return key
```

### Usage Example

```python
# Example usage of known-plaintext attack
cipher = HillCipher(key_size=2)

# Original key matrix (usually unknown)
original_key = np.array([[6, 24], [1, 13]])

# Known plaintext-ciphertext pair
plaintext = "HELLOWORLD"
ciphertext = cipher.encrypt(plaintext, original_key)

# Recover key
recovered_key = cipher.recover_key(plaintext, ciphertext)
print(f"Recovered Key:\n{recovered_key}")
```

## 2. N-gram Frequency Analysis Using Genetic Algorithm

### Key Components

```python
class HillCipherGA:
    def __init__(self, key_size: int, language_frequencies: dict):
        self.key_size = key_size
        self.modulus = 26
        self.language_frequencies = language_frequencies
        
    def generate_random_key(self) -> np.ndarray:
        """Generate random key matrix"""
        key = np.random.randint(0, self.modulus, (self.key_size, self.key_size))
        while np.linalg.det(key) == 0:  # Ensure matrix is invertible
            key = np.random.randint(0, self.modulus, (self.key_size, self.key_size))
        return key
    
    def calculate_fitness(self, key: np.ndarray, ciphertext: str) -> float:
        """Calculate fitness based on n-gram frequency match"""
        cipher = HillCipher(self.key_size)
        try:
            # Attempt decryption
            decrypted = cipher.decrypt(ciphertext, key)
            
            # Calculate n-gram frequencies in decrypted text
            decrypted_freq = self.calculate_ngram_frequencies(decrypted)
            
            # Compare with language frequencies
            score = self.compare_frequencies(decrypted_freq, self.language_frequencies)
            return score
        except:
            return float('-inf')
    
    def evolve(self, population: List[np.ndarray], ciphertext: str) -> np.ndarray:
        """Evolve population to find better solutions"""
        # Sort population by fitness
        population.sort(key=lambda x: self.calculate_fitness(x, ciphertext), reverse=True)
        
        # Keep best solutions
        new_population = population[:len(population)//2]
        
        # Generate new solutions through crossover and mutation
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(population[:len(population)//2], 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
            
        return new_population
    
    def crack(self, ciphertext: str, population_size: int, generations: int) -> np.ndarray:
        """Attempt to crack Hill Cipher using genetic algorithm"""
        # Initialize population
        population = [self.generate_random_key() for _ in range(population_size)]
        
        # Evolution loop
        for gen in range(generations):
            population = self.evolve(population, ciphertext)
            best_key = population[0]
            best_fitness = self.calculate_fitness(best_key, ciphertext)
            
            if best_fitness > 0.9:  # Threshold for acceptable solution
                return best_key
                
        return population[0]  # Return best found solution
```

### Usage Example

```python
# Example usage of genetic algorithm approach
# Define language n-gram frequencies (example for English)
english_frequencies = {
    'TH': 0.0356, 'HE': 0.0307, 'IN': 0.0243,
    'ER': 0.0205, 'AN': 0.0199, 'RE': 0.0185,
    # ... more n-gram frequencies
}

cipher_ga = HillCipherGA(key_size=2, language_frequencies=english_frequencies)

# Encrypted text to crack
ciphertext = "KQEREJEBCPPCJCRKIEACUZBKRVPKRBCIBQCARBJCVCKRXPKRCKVZKEX"

# Attempt to crack
best_key = cipher_ga.crack(
    ciphertext=ciphertext,
    population_size=100,
    generations=1000
)

# Decrypt using found key
cipher = HillCipher(key_size=2)
decrypted = cipher.decrypt(ciphertext, best_key)
print(f"Decrypted text: {decrypted}")
```

## Notes

- The known-plaintext attack is deterministic and will always succeed if enough plaintext-ciphertext pairs are available
- The genetic algorithm approach is probabilistic and may require multiple attempts with different parameters
- Success of the genetic algorithm depends heavily on:
  - Quality of language frequency statistics
  - Population size and number of generations
  - Length of ciphertext
  - Key size (larger keys are more difficult to crack)
  
Both implementations include error handling and validation to ensure proper usage.