# Hill Cipher Breaking: Theory and Implementation Guide

## Table of Contents
1. [Theoretical Background](#theoretical-background)
2. [Implementation Methods](#implementation-methods)
3. [Code Examples](#code-examples)
4. [Performance Optimization](#performance-optimization)
5. [Success Criteria](#success-criteria)

## Theoretical Background

### Hill Cipher Basics
The Hill Cipher is a polygraphic substitution cipher developed by Lester S. Hill in 1929, using linear algebra principles. The encryption process is represented as:
```
C = PK mod 26
```
Where:
- C is the ciphertext matrix
- P is the plaintext matrix
- K is the key matrix (must be invertible)
- Operations are performed modulo 26 (English alphabet)

## Implementation Methods

### 1. Known-Plaintext Attack

#### Theory
The known-plaintext attack exploits the Hill Cipher's linear nature through a system of equations:
```
P₁K ≡ C₁ (mod 26)
P₂K ≡ C₂ (mod 26)
...
PₙK ≡ Cₙ (mod 26)
```

Key recovery formula:
```
K ≡ P⁻¹C (mod 26)
```

#### Implementation
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
        
        if len(plaintext) < self.key_size * self.key_size:
            raise ValueError("Need more plaintext-ciphertext pairs")
            
        P_inv = np.linalg.inv(P[:self.key_size])
        key = np.dot(P_inv, C[:self.key_size]) % self.modulus
        return key
```

### 2. N-gram Frequency Analysis with Genetic Algorithms

#### Theory
This method combines statistical analysis with evolutionary computation:

1. **Language Statistics**: Uses characteristic N-gram frequency distributions
2. **Genetic Components**:
   - Chromosome: Potential key matrix
   - Fitness: Comparison with expected N-gram frequencies
   - Operators: Crossover and mutation
   - Selection: Tournament and elitism

#### Implementation
```python
class HillCipherGA:
    def __init__(self, key_size: int, language_frequencies: dict):
        self.key_size = key_size
        self.modulus = 26
        self.language_frequencies = language_frequencies
        
    def generate_random_key(self) -> np.ndarray:
        """Generate random invertible key matrix"""
        key = np.random.randint(0, self.modulus, (self.key_size, self.key_size))
        while np.linalg.det(key) == 0:
            key = np.random.randint(0, self.modulus, (self.key_size, self.key_size))
        return key
    
    def calculate_fitness(self, key: np.ndarray, ciphertext: str) -> float:
        """Calculate fitness based on n-gram frequency match"""
        cipher = HillCipher(self.key_size)
        try:
            decrypted = cipher.decrypt(ciphertext, key)
            decrypted_freq = self.calculate_ngram_frequencies(decrypted)
            score = self.compare_frequencies(decrypted_freq, self.language_frequencies)
            return score
        except:
            return float('-inf')
```

## Mathematical Complexity Analysis

### Known-Plaintext Attack
- Time Complexity: O(n³)
- Space Complexity: O(n²)

### Genetic Algorithm
- Time Complexity: O(GPn³)
- Space Complexity: O(Pn²)
Where:
  - G = generations
  - P = population size
  - n = key size

## Performance Optimization

1. **Matrix Operations**
   - Use NumPy for efficient computations
   - Implement modular arithmetic carefully
   - Handle singular matrices

2. **Genetic Algorithm Tuning**
   - Population size: 4n² to 10n²
   - Mutation rate: Adaptive, starting at 1/n²
   - Crossover rate: 0.6-0.8
   - Generation limit: 100n-1000n

3. **Parallel Processing**
   ```python
   # Example of parallel fitness evaluation
   from multiprocessing import Pool
   
   with Pool() as pool:
       fitness_scores = pool.map(calculate_fitness, population)
   ```

## Usage Examples

### Known-Plaintext Attack
```python
# Example usage
cipher = HillCipher(key_size=2)
original_key = np.array([[6, 24], [1, 13]])
plaintext = "HELLOWORLD"
ciphertext = cipher.encrypt(plaintext, original_key)
recovered_key = cipher.recover_key(plaintext, ciphertext)
print(f"Recovered Key:\n{recovered_key}")
```

### Genetic Algorithm
```python
# Example usage
english_frequencies = {
    'TH': 0.0356, 'HE': 0.0307, 'IN': 0.0243,
    'ER': 0.0205, 'AN': 0.0199, 'RE': 0.0185,
}

cipher_ga = HillCipherGA(key_size=2, language_frequencies=english_frequencies)
best_key = cipher_ga.crack(
    ciphertext="KQEREJEBCPPCJCR",
    population_size=100,
    generations=1000
)
```

## Success Criteria

### Known-Plaintext Attack
- Exact key recovery
- Successful decryption verification

### Genetic Algorithm
- Fitness threshold achievement
- N-gram frequency match
- Readable plaintext recovery

## Notes
- Implementation includes error handling
- Unit tests recommended for both approaches
- Security considerations for production use
- Regular monitoring and logging suggested