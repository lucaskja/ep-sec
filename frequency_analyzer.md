# Breaking the Hill Cipher with Frequency Analysis in Portuguese

This document provides an **in-depth**, step-by-step guide to breaking Hill Ciphers of sizes **3×3, 4×4, and 5×5** using frequency analysis tailored for the Portuguese language. We will cover:

1. **Background on the Hill cipher**  
2. **Preprocessing steps** (cleaning, normalization, mapping)  
3. **Portuguese letter and n-gram frequencies**  
4. **General frequency-analysis strategy**  
5. **Detailed implementation** (code examples in Python)  
   - Counting ciphertext n-grams  
   - Selecting candidate plaintext n-grams  
   - Building and inverting matrices mod 26  
   - Recovering the key matrix  
   - Validating and iterating  
6. **Special considerations** for 3×3, 4×4, and 5×5 cases  
7. **Pitfalls, tips, and optimizations**  

> **Note:** All code examples assume usage of Python 3.8+ and the packages:
> - `numpy` (for matrix operations)  
> - `sympy` (for modular inversion and exact arithmetic)  
> - `unidecode` (for accent stripping)  
> - `collections` (for counting n-grams)  
>
> You can install them via:
>
> ```bash
> pip install numpy sympy unidecode
> ```

---

## 1. Background: How the Hill Cipher Works

The Hill cipher is a classical polygraphic substitution cipher based on linear algebra.  For a block size \(n\), the cipher uses an invertible \(n \times n\) key matrix \(K\) over the ring \(\mathbb{Z}_{26}\).  Encryption and decryption go as follows:

1. **Alphabet mapping**  
   - Map letters A → 0, B → 1, …, Z → 25.  
   - For Portuguese text, strip accents and special characters first (e.g., Á→A, Ç→C).

2. **Plaintext blocks**  
   - Split plaintext into non-overlapping blocks of length \(n\).  If the last block is shorter, pad with a filler letter (e.g., “X” → 23).  

3. **Encryption**  
   - Let \(\mathbf{p} \in \mathbb{Z}_{26}^n\) be the column vector representing an \(n\)-letter plaintext block.  
   - Compute \(\mathbf{c} = K \cdot \mathbf{p} \pmod{26}\).  
   - Map \(\mathbf{c}\) back to letters → ciphertext block.

4. **Decryption**  
   - If \(K\) is invertible mod 26, compute \(K^{-1} \bmod 26\).  
   - Given a ciphertext block \(\mathbf{c}\), plaintext \(\mathbf{p} = K^{-1} \cdot \mathbf{c} \bmod 26\).  

> **Key requirement:** \(\det(K)\) must be coprime with 26 (i.e., \(\gcd(\det(K), 26) = 1\)).  

---

## 2. Preprocessing the Text

Before any analysis, **normalize** plaintext and ciphertext:

1. **Remove non-letter characters**: punctuation, digits, whitespace.  
2. **Strip accents** (e.g., `Á → A`, `Ç → C`, `É → E`)  
3. **Convert to uppercase** (A–Z)  
4. **Map letters to numbers**:  
   \[
     \text{ORD}(\text{‘A’}) = 65 \;\longrightarrow\; 0,\quad
     \text{ORD}(\text{‘B’}) = 66 \;\longrightarrow\; 1, \;\dots,\;
     \text{ORD}(\text{‘Z’}) = 90 \;\longrightarrow\; 25.
   \]
5. **Split into blocks** of size \(n\).  If the final block is incomplete, pad with “X” (23) or another agreed-upon filler.

### 2.1. Example: Preprocessing Function

```python
import re
from unidecode import unidecode

def preprocess(text: str) -> str:
    """
    Remove non-letters, strip accents, convert to uppercase, 
    and return only A–Z characters.
    """
    # 1. Strip accents (e.g., "á" -> "a", "ç" -> "c")
    no_accents = unidecode(text)
    # 2. Keep only letters A–Z or a–z
    letters_only = re.sub(r'[^A-Za-z]', '', no_accents)
    # 3. Convert to uppercase
    normalized = letters_only.upper()
    return normalized

def letters_to_numbers(s: str) -> list[int]:
    """
    Convert string of A–Z letters to list of integers 0–25.
    """
    return [ord(ch) - ord('A') for ch in s]

def numbers_to_letters(nums: list[int]) -> str:
    """
    Convert list of integers 0–25 back to string A–Z.
    """
    return ''.join(chr((n % 26) + ord('A')) for n in nums)

def chunkify(numbers: list[int], block_size: int) -> list[list[int]]:
    """
    Split list of numbers into consecutive blocks of length block_size.
    Pads the final block with X=23 if needed.
    """
    blocks = []
    i = 0
    while i < len(numbers):
        block = numbers[i:i+block_size]
        if len(block) < block_size:
            # Pad with X (23)
            block += [23] * (block_size - len(block))
        blocks.append(block)
        i += block_size
    return blocks

# Example usage:
raw = "Olá, mundo! Este é um texto em Português."
cleaned = preprocess(raw)
nums = letters_to_numbers(cleaned)
blocks = chunkify(nums, 3)  # e.g., for 3×3 Hill cipher
print("Preprocessed:", cleaned)
print("Numeric:", nums)
print("Blocks (size 3):", blocks)
````

---

## 3. Portuguese Letter and n-Gram Frequencies

Breaking a Hill cipher by frequency analysis exploits the fact that even though blocks of $n$ letters are linearly transformed, **high-frequency $n$-grams in plaintext tend to map to high-frequency $n$-grams in ciphertext** (permuted by the unknown key). You match the most frequent ciphertext $n$-grams to likely plaintext $n$-grams to recover $K$.

### 3.1. Single-Letter Frequencies (for reference)

Below are approximate letter frequencies in modern Portuguese (percentages).  While Hill uses $n$-grams, knowing single-letter frequency helps sanity-check decrypted output.

| Letter | Frequency (%) | Letter | Frequency (%) |
| :----: | :-----------: | :----: | :-----------: |
|    E   |     14.63     |    R   |      6.53     |
|    A   |     12.57     |    I   |      6.18     |
|    O   |     10.73     |    N   |      5.05     |
|    S   |      7.81     |    D   |      4.99     |
|    U   |      4.63     |    M   |      4.74     |
|    T   |      4.34     |    C   |      3.88     |
|    L   |      2.78     |    P   |      2.52     |
|    V   |      1.67     |    G   |      1.30     |
|    H   |      1.28     |    Q   |      1.10     |
|    B   |      1.04     |    F   |      1.02     |
|    Z   |      0.47     |    J   |      0.40     |
|    X   |      0.21     |    K   |      0.02     |
|    Y   |      0.02     |    W   |      0.01     |

> **Source:** [UFRJ Criptoanálise (Portuguese)](https://www.gta.ufrj.br/grad/06_2/alexandre/criptoanalise.html)

### 3.2. Common Portuguese n-Gram Frequencies

#### 3.2.1. Bigrams (2-grams)

| Rank | Digram | Approx. % of all digrams |
| :--: | :----: | :----------------------: |
|   1  |   DE   |           3.50%          |
|   2  |   OS   |           3.00%          |
|   3  |   ES   |           2.80%          |
|   4  |   RA   |           2.60%          |
|   5  |   EN   |           2.30%          |
|   6  |   SE   |           2.00%          |
|   7  |   ER   |           1.90%          |
|   8  |   AN   |           1.80%          |
|   9  |   AS   |           1.70%          |
|  10  |   OC   |           1.50%          |

#### 3.2.2. Trigrams (3-grams)

Based on large newspaper corpora, some of the most frequent Portuguese trigrams are:

QUE (~1.8%), ENT (~1.4%), COM (~1.3%), ROS (~1.1%), IST (~1.0%), ADO (~0.9%), ELA (~0.9%), PRA (~0.8%), INH (~0.8%), EST (~0.8%), etc.

> **Tip:** For a 3×3 Hill cipher, use a corpus to compute your own trigram frequencies, or find a published list.

#### 3.2.3. 4-grams and 5-grams

For block sizes 4 or 5, the absolute frequencies drop, but common Portuguese 4-grams include:

DESE (~0.5%), OSSE (~0.4%), ROTA (~0.4%), ADOU (~0.3%), etc.

Common 5-grams:

PORTE (~0.3%), LIGAR (~0.25%), QUESE (~0.25%), etc.

> **Advice:** The larger $n$, the sparser the frequency data.  For 4×4 or 5×5 Hill, you often combine bigram/trigram knowledge with plaintext guesses (e.g., known salutations, “BOMDIA”, “SAUDA”, “FELIZ”).

---

## 4. General Frequency-Analysis Strategy

Regardless of $n$, the core approach is:

1. **Compute ciphertext n-gram frequencies**

   * Slide a window of size $n$ across the normalized ciphertext (non-overlapping, if you assume block alignment).
   * Count how many times each unique $n$-gram appears.

2. **Identify the top $k$ most frequent ciphertext blocks**

   * $k$ should be at least $n+1$.  For a 3×3 key, you need at least 3 distinct ciphertext trigrams to attempt recovery; but having 5–6 gives flexibility.

3. **Select candidate plaintext n-grams**

   * Based on Portuguese frequency tables, pick the top $k$ plaintext $n$-grams.
   * Form candidate pairings: map each top ciphertext $n$-gram to one of the top plaintext $n$-grams.

4. **Form $P_{\text{stack}}$ and $C_{\text{stack}}$** for each candidate set of $n$ pairings:

   * Convert each plaintext $n$-gram to a column vector in $\mathbb{Z}_{26}^n$.
   * Stack these $n$ column vectors to form an $n \times n$ matrix $P_{\text{stack}}$.
   * Do the same for the corresponding ciphertext $n$-grams → $C_{\text{stack}}$.

5. **Check if $P_{\text{stack}}$ is invertible mod 26**

   * Compute $\det(P_{\text{stack}}) \bmod 26$.
   * If $\gcd(\det, 26) \neq 1$, it is not invertible.  Discard this candidate set.

6. **Compute $P_{\text{stack}}^{-1} \bmod 26$**

   * Use a function to compute modular inverse of the matrix.

7. **Recover candidate key:**

   $$
     K \;\equiv\; C_{\text{stack}} \cdot P_{\text{stack}}^{-1} \pmod{26}.
   $$

8. **Validate the candidate key:**

   * Check $\det(K)\bmod 26$ is invertible.
   * Decrypt a portion of ciphertext:
     $\mathbf{p} = K^{-1} \cdot \mathbf{c} \bmod 26$.
   * Convert numeric $\mathbf{p}$ back to letters.
   * If the output “looks like” Portuguese (words, grammar), you probably found the correct key.
   * If it is gibberish, go back and try a different pairing of ciphertext→plaintext $n$-grams.

> **Note:** Because actual ciphertext may deviate from ideal frequency distributions, you must systematically iterate through several plausible pairings.

---

## 5. Detailed Implementation (Python)

Below is a comprehensive, commented Python implementation.  We will gradually build up from helper functions to complete key recovery for $n=3,4,5$.

### 5.1. Required Imports

```python
import re
from unidecode import unidecode
from collections import Counter
import numpy as np
import sympy as sp

# For typing clarity (Python 3.9+)
from typing import List, Tuple, Dict
```

### 5.2. Text Preprocessing

```python
def preprocess(text: str) -> str:
    """
    Remove non-letter chars, strip accents, uppercase.
    Returns only characters A–Z.
    """
    no_accents = unidecode(text)
    letters_only = re.sub(r'[^A-Za-z]', '', no_accents)
    return letters_only.upper()

def letters_to_numbers(s: str) -> List[int]:
    """Map A→0, B→1, …, Z→25."""
    return [ord(ch) - ord('A') for ch in s]

def numbers_to_letters(nums: List[int]) -> str:
    """Map 0→A, 1→B, …, 25→Z (wrap mod 26)."""
    return ''.join(chr((n % 26) + ord('A')) for n in nums)

def chunkify(numbers: List[int], block_size: int) -> List[List[int]]:
    """
    Split into non-overlapping blocks of size `block_size`.
    Pads the final block with X=23 (if needed).
    """
    blocks = []
    i = 0
    while i < len(numbers):
        block = numbers[i:i + block_size]
        if len(block) < block_size:
            block += [23] * (block_size - len(block))  # pad with X
        blocks.append(block)
        i += block_size
    return blocks
```

### 5.3. Counting Ciphertext n-Gram Frequencies

```python
def count_ngrams(text: str, block_size: int) -> Counter:
    """
    Count non-overlapping n-gram (size=block_size) frequencies in `text`.
    `text` must be preprocessed (uppercase A–Z).
    """
    numbers = letters_to_numbers(text)
    blocks = chunkify(numbers, block_size)
    # Convert each numeric block back to a string for counting
    str_blocks = [''.join(numbers_to_letters(block)) for block in blocks]
    return Counter(str_blocks)

# Example usage:
cipher = preprocess("UFQZRXDMUFQZRXDM…")  # truncated for illustration
bigram_counts = count_ngrams(cipher, 2)
trigram_counts = count_ngrams(cipher, 3)
# Get the top 10 most common trigrams:
top_10_trigrams = trigram_counts.most_common(10)
print("Top 10 ciphertext trigrams:", top_10_trigrams)
```

> **Important:** We assume the Hill cipher encrypts blocks *exactly* in multiples of $n$.  If the original plaintext was padded or if the adversary doesn’t know the exact block alignment, you might also look at **all overlapping** n-grams.  For pure textbook Hill, non-overlapping is correct.

### 5.4. Candidate Plaintext n-Gram List (Portuguese)

You need a curated list of the **most frequent Portuguese n-grams**.  Below are examples—feel free to expand or generate from a large corpus.

#### 5.4.1. Trigram Example List (Top \~10)

```python
# Example Portuguese trigram frequencies (descending order).
# Ideally generate this from a huge corpus. For illustration:
top_portuguese_trigrams = [
    "QUE", "ENT", "COM", "ROS", "IST", "ADO", 
    "ELA", "PRA", "INH", "EST", "NTE", "ERA", "CON", "IAL"
]
```

#### 5.4.2. 4-gram Example List

```python
top_portuguese_4grams = [
    "DESE", "OSSE", "ROTA", "ADOU", "CIAO", 
    "PORT", "PROT", "ENTE", "MENT", "PREM"
]
```

#### 5.4.3. 5-gram Example List

```python
top_portuguese_5grams = [
    "PORTE", "LIGAR", "QUESE", "ENTRE", "CONSE", 
    "SAUDA", "FELIZ", "BOMDI", "MUNDO", "TEXTO"
]
```

> **Tip:** If you have a large Portuguese text corpus, compute frequencies directly using a similar `Counter` approach (sliding window) and pick the top $n$ n-grams.

### 5.5. Converting n-grams to Numeric Matrices

To build $P_{\text{stack}}$ and $C_{\text{stack}}$, we need to map each $n$-gram (string of length $n$) to an $n \times 1$ numeric column vector.

```python
def ngram_to_vector(ngram: str) -> np.ndarray:
    """
    Convert an n-letter string to an (n x 1) column vector of ints mod 26.
    """
    nums = letters_to_numbers(ngram)
    return np.array(nums, dtype=int).reshape(-1, 1)

def build_stack_matrix(ngrams: List[str]) -> np.ndarray:
    """
    Given a list of n distinct n-grams (all length n),
    return an (n x n) matrix whose columns are the numeric vectors of these n-grams.
    """
    n = len(ngrams[0])
    if any(len(ng) != n for ng in ngrams):
        raise ValueError("All ngrams must have the same length n.")
    # Stack each column
    cols = [ngram_to_vector(ng) for ng in ngrams]
    return np.hstack(cols) % 26
```

### 5.6. Checking Matrix Invertibility mod 26 & Computing Inverse

```python
def matrix_mod_inv(M: np.ndarray, mod: int = 26) -> np.ndarray:
    """
    Compute the modular inverse of an n x n integer matrix M modulo `mod`.
    Returns M^{-1} mod `mod` as an ndarray of shape (n, n).
    Raises ValueError if M is not invertible mod `mod`.
    """
    # Convert to a sympy Matrix for exact arithmetic
    sym_M = sp.Matrix(M.tolist())
    det = int(sym_M.det() % mod)
    if np.gcd(det, mod) != 1:
        raise ValueError(f"Matrix determinant {det} is not invertible mod {mod}")
    inv_det = sp.mod_inverse(det, mod)  # modular inverse of det
    # Compute adjugate: adj(M) = det * M^{-1} (over integers)
    adj = sym_M.adjugate()
    # M^{-1} = inv_det * adj(M) mod mod
    inv_M = (inv_det * adj) % mod
    # Convert back to numpy array of ints
    inv_np = np.array(inv_M.tolist(), dtype=int) % mod
    return inv_np

# Example:
P_stack = np.array([[3, 14],
                    [4, 18]])  # from "DE" ([3,4]) and "OS" ([14,18])
try:
    P_inv = matrix_mod_inv(P_stack, 26)
    print("P_stack inverse mod 26:\n", P_inv)
except ValueError as e:
    print("Not invertible:", e)
```

> **Explanation:**
>
> * We lift `M` into a `sympy.Matrix` to compute exact determinant and adjugate.
> * Check $\gcd(\det(M), 26) = 1$.  If not, no inverse exists.
> * Compute $\det(M)^{-1} \bmod 26$ with `sp.mod_inverse`.
> * `adjugate(M)` in sympy returns the classical adjoint.  Then $M^{-1} \equiv (\det(M)^{-1} \cdot \text{adj}(M)) \bmod 26$.
> * Convert back to `numpy.ndarray` of dtype `int`.

### 5.7. Recovering the Key Matrix $K$

Once we have:

* A candidate plaintext stack $P_{\text{stack}}$ (size $n \times n$),
* The corresponding ciphertext stack $C_{\text{stack}}$ (size $n \times n$),

compute:

$$
  K \;\equiv\; C_{\text{stack}} \times P_{\text{stack}}^{-1} \pmod{26}.
$$

```python
def recover_key_matrix(
    P_stack: np.ndarray,
    C_stack: np.ndarray,
    mod: int = 26
) -> np.ndarray:
    """
    Given P_stack (n x n) and C_stack (n x n), compute key K such that
    C_stack ≡ K * P_stack (mod mod).  Returns K (n x n).
    """
    # 1. Compute inverse of P_stack mod 26
    P_inv = matrix_mod_inv(P_stack, mod)
    # 2. K = C_stack * P_inv mod 26
    K = (C_stack.dot(P_inv)) % mod
    # Validate invertibility of K
    sym_K = sp.Matrix(K.tolist())
    det_K = int(sym_K.det() % mod)
    if np.gcd(det_K, mod) != 1:
        # Technically, K must be invertible mod 26 to be valid
        raise ValueError(f"Recovered K has determinant {det_K} not invertible mod {mod}.")
    return K

# Example (2x2 toy demonstration):
P_stack_2x2 = np.array([[3, 14], [4, 18]])  # hypothetical plaintext stack
C_stack_2x2 = np.array([[16, 19], [25, 23]])  # hypothetical ciphertext stack
try:
    K_candidate = recover_key_matrix(P_stack_2x2, C_stack_2x2)
    print("Candidate key matrix K:\n", K_candidate)
except ValueError as e:
    print("Invalid key:", e)
```

---

## 6. Breaking 3×3 Hill Cipher (Detailed Steps)

For an **$n=3$** Hill cipher:

1. **Preprocess the ciphertext**

   ```python
   cipher_raw = "...your ciphertext here..."
   cipher = preprocess(cipher_raw)
   ```

2. **Count ciphertext trigrams**

   ```python
   tri_counts = count_ngrams(cipher, 3)
   top_cipher_trigrams = [t for t, _ in tri_counts.most_common(8)]
   ```

   > Pick the top 6–8 most frequent ciphertext trigrams as candidates.

3. **Select candidate plaintext trigrams**

   ```python
   top_plain_trigrams = ["QUE", "ENT", "COM", "ROS", "IST", "ADO", "ELA", "PRA"]
   ```

4. **Generate all combinations of 3 distinct ciphertext trigrams** and 3 distinct plaintext trigrams from the top lists.  E.g.:

   ```python
   import itertools

   def generate_candidate_triplets(cipher_tris, plain_tris):
       for c_triplet in itertools.permutations(cipher_tris, 3):
           for p_triplet in itertools.permutations(plain_tris, 3):
               yield list(c_triplet), list(p_triplet)
   ```

5. **For each candidate $(C^{(1)},C^{(2)},C^{(3)}) ↔ (P^{(1)},P^{(2)},P^{(3)})$:**

   ```python
   for c_trip, p_trip in generate_candidate_triplets(top_cipher_trigrams, top_plain_trigrams):
       # (a) Build P_stack, C_stack
       P_stack = build_stack_matrix(p_trip)    # shape (3,3)
       C_stack = build_stack_matrix(c_trip)    # shape (3,3)

       try:
           # (b) Attempt to recover K
           K_candidate = recover_key_matrix(P_stack, C_stack)
       except ValueError:
           # P_stack not invertible or K invalid; skip this combination
           continue

       # (c) Test K_candidate by decrypting some ciphertext
       # Convert ciphertext into numeric blocks:
       cipher_nums = letters_to_numbers(cipher)
       cipher_blocks = chunkify(cipher_nums, 3)  # list of length-3 lists

       # Build K_inverse once
       K_inv = matrix_mod_inv(K_candidate, 26)
       # Decrypt first 10 blocks as a sanity check:
       decrypted = []
       for block in cipher_blocks[:10]:
           c_vec = np.array(block, dtype=int).reshape(3, 1)
           p_vec = (K_inv.dot(c_vec)) % 26
           decrypted.append(numbers_to_letters(p_vec.flatten().tolist()))
       plain_snippet = "".join(decrypted)

       # (d) Check if plain_snippet looks like Portuguese:
       if looks_like_portuguese(plain_snippet):
           print("Likely key found!")
           print("K =\n", K_candidate)
           break
   ```

   * **`looks_like_portuguese(...)`**: Implement a quick check—e.g., check whether common Portuguese words (“QUE”, “DE”, “DO”, “DA”, “E”, “EM”, “O”, “A”) appear in the plaintext snippet.
   * If the snippet contains valid words, consider it a **candidate**.  Decrypt a larger sample or entire text to confirm.

### 6.1. Helper: Checking Readability (Heuristic)

```python
def looks_like_portuguese(plaintext: str) -> bool:
    """
    Heuristic: check if decrypted snippet contains common Portuguese bigrams
    or words. Returns True if it likely is valid Portuguese.
    """
    portuguese_common = ["DE", "QUE", "E ", " A ", " O ", " DA", " DO", "EM ", "PARA", "COM"]
    # If at least 3 of these substrings appear, we call it valid.
    count = sum(1 for w in portuguese_common if w in plaintext)
    return count >= 3

# Example usage (assuming we have plain_snippet from above):
if looks_like_portuguese(plain_snippet):
    print("Decryption seems valid Portuguese.")
```

### 6.2. Putting It All Together (3×3 Script)

```python
def break_hill_3x3(ciphertext: str) -> Tuple[np.ndarray, str]:
    """
    Attempt to break a 3x3 Hill cipher on `ciphertext`.
    Returns (K, decrypted_text) if successful, else raises RuntimeError.
    """
    # 1. Preprocess
    cipher = preprocess(ciphertext)

    # 2. Count ciphertext trigrams
    tri_counts = count_ngrams(cipher, 3)
    top_cipher_trigrams = [t for t, _ in tri_counts.most_common(8)]

    # 3. Candidate Portuguese trigrams
    top_plain_trigrams = ["QUE", "ENT", "COM", "ROS", "IST", "ADO", "ELA", "PRA"]

    # 4. Iterate over triplets
    for c_trip, p_trip in generate_candidate_triplets(top_cipher_trigrams, top_plain_trigrams):
        P_stack = build_stack_matrix(p_trip)
        C_stack = build_stack_matrix(c_trip)
        try:
            K_candidate = recover_key_matrix(P_stack, C_stack)
        except ValueError:
            continue

        # 5. Decrypt entire ciphertext using K_candidate
        cipher_nums = letters_to_numbers(cipher)
        cipher_blocks = chunkify(cipher_nums, 3)
        K_inv = matrix_mod_inv(K_candidate, 26)

        decrypted_blocks = []
        for blk in cipher_blocks:
            p_vec = (K_inv.dot(np.array(blk).reshape(3, 1))) % 26
            decrypted_blocks.append(numbers_to_letters(p_vec.flatten().tolist()))
        decrypted_text = "".join(decrypted_blocks)

        # 6. Quick Portuguese check
        if looks_like_portuguese(decrypted_text[:200]):  # check first 200 letters
            return K_candidate, decrypted_text

    raise RuntimeError("Failed to find a valid 3x3 key with given assumptions.")
```

---

## 7. Breaking 4×4 Hill Cipher

For **$n=4$**, operations are analogous, but you must work with **4-letter blocks**.  Because 4-grams are less frequent than 3-grams, you will need to:

1. Count ciphertext 4-gram frequencies.
2. Choose at least 4 candidate plaintext 4-grams (e.g., “DESE”, “OSSE”, “ROTA”, “ADOU”, “PORT”, “ENTE”, “MENT”).
3. Generate combinations of 4 ciphertext 4-grams and 4 plaintext 4-grams.
4. Build $4\times4$ matrices $P_{\text{stack}}$ and $C_{\text{stack}}$.
5. Check invertibility ($\gcd(\det(P_{\text{stack}}), 26)=1$).
6. Compute $K = C_{\text{stack}} \cdot P_{\text{stack}}^{-1} \bmod 26$.
7. Decrypt and validate.

### 7.1. Example Candidate Plaintext 4-grams

```python
top_plain_4grams = [
    "DESE", "OSSE", "ROTA", "ADOU", "PORT", "ENTE", "MENT", "COMO", "BOAS", "CASO"
]
```

### 7.2. Full Script Skeleton (4×4)

```python
def break_hill_4x4(ciphertext: str) -> Tuple[np.ndarray, str]:
    """
    Attempt to break a 4x4 Hill cipher. Returns (K, decrypted_text).
    """
    cipher = preprocess(ciphertext)
    quad_counts = count_ngrams(cipher, 4)
    top_cipher_quads = [q for q, _ in quad_counts.most_common(10)]

    top_plain_4grams = ["DESE", "OSSE", "ROTA", "ADOU", "PORT", "ENTE", "MENT", "COMO", "BOAS", "CASO"]

    for c_quad_combo in itertools.permutations(top_cipher_quads, 4):
        for p_quad_combo in itertools.permutations(top_plain_4grams, 4):
            P_stack = build_stack_matrix(p_quad_combo)  # shape (4,4)
            C_stack = build_stack_matrix(c_quad_combo)  # shape (4,4)

            try:
                K_candidate = recover_key_matrix(P_stack, C_stack)
            except ValueError:
                continue

            # Decrypt entire text
            cipher_nums = letters_to_numbers(cipher)
            cipher_blocks = chunkify(cipher_nums, 4)
            K_inv = matrix_mod_inv(K_candidate, 26)

            decrypted_blocks = []
            for blk in cipher_blocks:
                p_vec = (K_inv.dot(np.array(blk).reshape(4, 1))) % 26
                decrypted_blocks.append(numbers_to_letters(p_vec.flatten().tolist()))
            decrypted_text = "".join(decrypted_blocks)

            if looks_like_portuguese(decrypted_text[:200]):
                return K_candidate, decrypted_text

    raise RuntimeError("Failed to break 4x4 Hill cipher.")
```

> **Warning:** The number of permutations is $\binom{10}{4} \times 4! \approx 5{,}040 \times 24 = 120{,}960$.  Combined with another $\approx 120{,}960$ from ciphertext side, you get ∼$1.46\times10^{10}$ total combinations!  **That’s infeasible.**
>
> **Optimization strategies:**
>
> * Narrow both ciphertext and plaintext lists to top 5–6 candidates only.
> * Use bigram/trigram heuristics to guess likely 4-grams (e.g., “DESE” almost always appears in words like “DESEJO”, “DESEMPREGO”).
> * Instead of full permutations, try selective pairings:
>
>   * Map the single most frequent ciphertext 4-gram to “DESE.”
>   * Then map next two to “OSSE,” “ENTE,” etc.  Only if that fails, back off.
> * Stop early when a plausible key is found.

---

## 8. Breaking 5×5 Hill Cipher

For **$n=5$**, the conceptual steps are identical, but:

* You need **five** linearly independent ciphertext 5-grams and **five** plaintext 5-grams.
* 5-gram frequencies in Portuguese are much lower → top candidates may have frequency < 0.1%.
* The search space grows combinatorially, so you **must** prune aggressively.

### 8.1. Example Candidate Plaintext 5-grams

```python
top_plain_5grams = [
    "PORTE", "LIGAR", "QUESE", "ENTRE", "CONSE", 
    "SAUDA", "FELIZ", "BOMDI", "MUNDO", "TEXTO"
]
```

### 8.2. Script Skeleton (5×5)

```python
def break_hill_5x5(ciphertext: str) -> Tuple[np.ndarray, str]:
    """
    Attempt to break a 5x5 Hill cipher. Returns (K, decrypted_text).
    """
    cipher = preprocess(ciphertext)
    penta_counts = count_ngrams(cipher, 5)
    top_cipher_pentas = [p for p, _ in penta_counts.most_common(7)]

    top_plain_5grams = ["PORTE", "LIGAR", "QUESE", "ENTRE", "CONSE", "SAUDA", "FELIZ"]

    # Reduce combinations: try heuristic pairings first
    candidate_cipher_sets = list(itertools.combinations(top_cipher_pentas, 5))
    candidate_plain_sets = list(itertools.combinations(top_plain_5grams, 5))

    for c_penta_set in candidate_cipher_sets:
        for p_penta_set in candidate_plain_sets:
            # Permute only if needed; try direct alignment first
            P_stack = build_stack_matrix(list(p_penta_set))
            C_stack = build_stack_matrix(list(c_penta_set))
            try:
                K_candidate = recover_key_matrix(P_stack, C_stack)
            except ValueError:
                continue

            # Decrypt snippet
            cipher_nums = letters_to_numbers(cipher)
            cipher_blocks = chunkify(cipher_nums, 5)
            K_inv = matrix_mod_inv(K_candidate, 26)

            decrypted_blocks = []
            for blk in cipher_blocks[:20]:  # just test first 20 blocks (~100 letters)
                p_vec = (K_inv.dot(np.array(blk).reshape(5, 1))) % 26
                decrypted_blocks.append(numbers_to_letters(p_vec.flatten().tolist()))
            decrypted_snippet = "".join(decrypted_blocks)

            if looks_like_portuguese(decrypted_snippet):
                # Assume full text is correct; decrypt all
                decrypted_full = []
                for blk in cipher_blocks:
                    p_vec = (K_inv.dot(np.array(blk).reshape(5, 1))) % 26
                    decrypted_full.append(numbers_to_letters(p_vec.flatten().tolist()))
                return K_candidate, "".join(decrypted_full)

    raise RuntimeError("Failed to break 5x5 Hill cipher.")
```

> **Note:** We used `itertools.combinations` instead of `permutations` to cut down on search.  Once you have a promising set of 5 plaintext/5 ciphertext 5-grams, you may try permuting their order if direct index alignment fails.

---

## 9. Pitfalls and Tips

1. **Matrix Invertibility**

   * Always check $\gcd(\det(P_{\text{stack}}), 26) = 1$.  If not, discard.
   * After computing $K$, check $\gcd(\det(K), 26) = 1$.  Otherwise $K$ won’t decrypt properly.

2. **Permutation vs. Combination**

   * For an $n×n$ key, you need $n$ linearly independent $(C^{(i)} \leftrightarrow P^{(i)})$ pairs.
   * The order matters when building matrices—each plaintext vector must align with its ciphertext counterpart in the same column index.
   * In code, use `permutations` if you need to try different orderings, but that multiplies the search by $n!$.  Use heuristics to prune early.

3. **Choice of Plaintext n-grams**

   * Bigram/trigram lists are easy to get, but 4-grams/5-grams are sparser.
   * Mix common n-grams with dictionary words/prefixes.  For 5×5, you might guess “BOMDI” (de “BOM DIA”) or “SAUDA” (de “SAUDAÇÃO”).
   * If you have a suspected plaintext keyword (e.g., “PRAZER” appears in the message), include it.

4. **Caesar/Shift Offsets**

   * Sometimes the actual plaintext may begin or end with a known header (e.g., “CARTA”, “EXCELENTÍSSIMO”).  If you know a common prefix, use that as one column of $P_{\text{stack}}$.

5. **Spacing and Alignment**

   * Hill ciphers ignore word boundaries; you must align your non-overlapping blocks from the first letter.
   * If the adversary padded the message with a random letter or length is unknown, consider also sliding a window of size $n$ over the ciphertext (overlapping blocks).  This yields more candidate n-grams but complicates counting.

6. **Validation Beyond Heuristics**

   * Once you think you have $K$, decrypt the entire ciphertext.  Run a Portuguese spellchecker or dictionary lookup on the output to get a more robust confidence measure.

7. **Performance Considerations**

   * For 3×3, a few hundred candidate triplets is manageable.
   * For 4×4 and 5×5, the naive search space is gigantic; you must prune both ciphertext and plaintext lists aggressively (top 5–8 only) and avoid full Cartesian/permutation expansions.
   * Consider heuristics:

     * Always pair the single most frequent ciphertext n-gram with the most frequent plaintext n-gram (“DESE” or “QUE”).
     * If that pairing fails (matrix non-invertible or nonsense output), fix the most frequent mapping and vary the rest.

---

## 10. Example Walk-Through (3×3 Full Example)

Assume we have a 3×3 Hill cipher of the form:

$$
K = 
\begin{pmatrix}
k_{11} & k_{12} & k_{13} \\
k_{21} & k_{22} & k_{23} \\
k_{31} & k_{32} & k_{33}
\end{pmatrix}
$$

and an intercepted ciphertext (after preprocessing):

```
CIPHERTEXT (cleaned, uppercase, no accents):
QZTXQZTXYLKQZTXYLKRXP...
```

1. **Count trigram frequencies** → Suppose top 5 ciphertext trigrams are:

   ```
   1) QZT (freq=48)
   2) XQZ (freq=46)
   3) TXY (freq=39)
   4) LKQ (freq=27)
   5) XRL (freq=24)
   ```

2. **Select top Portuguese trigrams**:

   ```
   "QUE", "ENT", "COM", "ROS", "IST", ...
   ```

3. **Hypothesize mappings** (Trial 1):

   ```
   C^(1)=QZT ↔ P^(1)=QUE
   C^(2)=XQZ ↔ P^(2)=ENT
   C^(3)=TXY ↔ P^(3)=COM
   ```

4. **Convert to numeric**:

   ```
   P^(1) = "QUE" → [16, 20, 4]^T
   P^(2) = "ENT" → [4, 13, 19]^T
   P^(3) = "COM" → [2, 14, 12]^T

   C^(1) = "QZT" → [16, 25, 19]^T
   C^(2) = "XQZ" → [23, 16, 25]^T
   C^(3) = "TXY" → [19, 23, 24]^T
   ```

5. **Build $P_{\text{stack}}$, $C_{\text{stack}}$**:

   ```python
   P_stack = np.array([[16, 4, 2],
                       [20, 13, 14],
                       [ 4, 19, 12]])  # shape (3,3)

   C_stack = np.array([[16, 23, 19],
                       [25, 16, 23],
                       [19, 25, 24]])  # shape (3,3)
   ```

6. **Check det(P\_stack) mod 26**:

   ```python
   sym_P = sp.Matrix(P_stack.tolist())
   detP = int(sym_P.det() % 26)  # Suppose detP = 7
   gcd(detP, 26) = 1 → invertible
   ```

7. **Compute $P_{\text{stack}}^{-1} \bmod 26$**:

   ```python
   P_inv = matrix_mod_inv(P_stack, 26)
   # Suppose P_inv = some 3×3 matrix mod 26
   ```

8. **Compute candidate key $K$**:

   ```python
   K_candidate = (C_stack.dot(P_inv)) % 26
   # Suppose K_candidate =
   # [[ 5, 17, 20],
   #  [ 8, 21, 15],
   #  [12,  3, 19]]
   ```

9. **Validate $\det(K)\bmod 26$**:

   ```python
   sym_K = sp.Matrix(K_candidate.tolist())
   detK = int(sym_K.det() % 26)  # Suppose detK = 25 → gcd(25,26)=1 → invertible
   ```

10. **Decrypt first few blocks**:

    ```python
    K_inv = matrix_mod_inv(K_candidate, 26)
    cipher_blocks = chunkify(letters_to_numbers(cipher), 3)  # List of 3-vectors

    decrypted_blocks = []
    for blk in cipher_blocks[:10]:
        p_vec = (K_inv.dot(np.array(blk).reshape(3, 1))) % 26
        decrypted_blocks.append(numbers_to_letters(p_vec.flatten().tolist()))
    snippet = "".join(decrypted_blocks)
    # Suppose snippet = "QUEELOSRUUIDOQUE..."
    ```

11. **Heuristic check** → look for “QUE”, “DE”, “DO”, “EM”, etc.

    * “QUE” appears → plausible.
    * “ELOSRU” is nonsense → maybe this mapping is wrong.

12. **Iterate** → try next combination of $(C^{(1)},C^{(2)},C^{(4)}) ↔ (QUE,ENT,COM)$, etc.

    * Eventually, you find a combination that yields valid Portuguese.

---

## 11. Conclusion

By systematically matching high-frequency ciphertext blocks with known high-frequency Portuguese blocks—then solving the resulting linear system—you can recover the Hill key matrix $K$.  Key points:

* **Preprocess** text carefully (strip accents, uppercase, remove non-letters).
* **Count n-gram frequencies** in ciphertext.
* **Build candidate pairs** $(C^{(i)} \leftrightarrow P^{(i)})$.
* **Form $n\times n$ stacks** $P_{\text{stack}}, C_{\text{stack}}$.
* **Check invertibility** of $P_{\text{stack}}$ (and of $K$ once obtained).
* **Decrypt** and **validate** with a Portuguese readability heuristic.
* **Iterate** over a manageable set of top candidates—prune aggressively for $n=4,5$.

This approach generalizes to any block size $n$.  The larger $n$, the more n-grams you need for stacking, and the sparser the frequency data—so you must combine frequency information with educated plaintext guesses.

---

## 12. References

* UFRJ Criptoanálise (Portuguese):
  [https://www.gta.ufrj.br/grad/06\_2/alexandre/criptoanalise.html](https://www.gta.ufrj.br/grad/06_2/alexandre/criptoanalise.html)
* Menezes, Oorschot & Vanstone, *Handbook of Applied Cryptography*, 1996 (Chapter on Classical Ciphers)
* Stallings, *Cryptography and Network Security*, 7th Ed., 2017 (Section on Polygraphic Ciphers)

---

**Good luck** with your cryptanalysis!

```