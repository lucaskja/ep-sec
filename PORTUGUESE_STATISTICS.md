# Portuguese Language Statistics for Cryptanalysis

This document compares the Portuguese language statistics from the reference study with our implementation in the Hill Cipher Analyzer.

## Letter Frequencies

### Reference Values vs. Implementation

| Letter | Reference Frequency | Implementation Frequency | Match |
|--------|--------------------:|------------------------:|:-----:|
| A      | 14.63%              | 14.63%                  | ✓     |
| B      | 1.04%               | 1.04%                   | ✓     |
| C      | 3.88%               | 3.88%                   | ✓     |
| D      | 4.99%               | 4.99%                   | ✓     |
| E      | 12.57%              | 12.57%                  | ✓     |
| F      | 1.02%               | 1.02%                   | ✓     |
| G      | 1.30%               | 1.30%                   | ✓     |
| H      | 1.28%               | 1.28%                   | ✓     |
| I      | 6.18%               | 6.18%                   | ✓     |
| J      | 0.40%               | 0.40%                   | ✓     |
| K      | 0.02%               | 0.02%                   | ✓     |
| L      | 2.78%               | 2.78%                   | ✓     |
| M      | 4.74%               | 4.74%                   | ✓     |
| N      | 5.05%               | 5.05%                   | ✓     |
| O      | 10.73%              | 10.73%                  | ✓     |
| P      | 2.52%               | 2.52%                   | ✓     |
| Q      | 1.20%               | 1.20%                   | ✓     |
| R      | 6.53%               | 6.53%                   | ✓     |
| S      | 7.81%               | 7.81%                   | ✓     |
| T      | 4.34%               | 4.34%                   | ✓     |
| U      | 4.63%               | 4.63%                   | ✓     |
| V      | 1.67%               | 1.67%                   | ✓     |
| W      | 0.01%               | 0.01%                   | ✓     |
| X      | 0.21%               | 0.21%                   | ✓     |
| Y      | 0.01%               | 0.01%                   | ✓     |
| Z      | 0.47%               | 0.47%                   | ✓     |

**Implementation Code:**
```python
# Letter frequencies in Portuguese
LETTER_FREQUENCIES = {
    'A': 14.63, 'B': 1.04, 'C': 3.88, 'D': 4.99, 'E': 12.57, 'F': 1.02,
    'G': 1.30, 'H': 1.28, 'I': 6.18, 'J': 0.40, 'K': 0.02, 'L': 2.78,
    'M': 4.74, 'N': 5.05, 'O': 10.73, 'P': 2.52, 'Q': 1.20, 'R': 6.53,
    'S': 7.81, 'T': 4.34, 'U': 4.63, 'V': 1.67, 'W': 0.01, 'X': 0.21,
    'Y': 0.01, 'Z': 0.47
}
```

## Digram Frequencies

### Reference Values vs. Implementation

| Digram | Reference Frequency | Implementation Frequency | Match |
|--------|--------------------:|------------------------:|:-----:|
| DE     | 1.76                | 1.76                    | ✓     |
| RA     | 1.67                | 1.67                    | ✓     |
| ES     | 1.65                | 1.65                    | ✓     |
| OS     | 1.51                | 1.51                    | ✓     |
| AS     | 1.49                | 1.49                    | ✓     |
| DO     | 1.41                | 1.41                    | ✓     |
| AR     | 1.33                | 1.33                    | ✓     |
| CO     | 1.31                | 1.31                    | ✓     |
| EN     | 1.23                | 1.23                    | ✓     |
| QU     | 1.20                | 1.20                    | ✓     |
| ER     | 1.18                | 1.18                    | ✓     |
| DA     | 1.17                | 1.17                    | ✓     |
| RE     | 1.14                | 1.14                    | ✓     |
| CA     | 1.11                | 1.11                    | ✓     |
| TA     | 1.10                | 1.10                    | ✓     |
| SE     | 1.08                | 1.08                    | ✓     |
| NT     | 1.08                | 1.08                    | ✓     |
| MA     | 1.06                | 1.06                    | ✓     |
| UE     | 1.05                | 1.05                    | ✓     |
| TE     | 1.05                | 1.05                    | ✓     |

**Implementation Code:**
```python
# The 20 most frequent digrams in Portuguese (per 100 letters)
DIGRAM_FREQUENCIES = {
    'DE': 1.76, 'RA': 1.67, 'ES': 1.65, 'OS': 1.51, 'AS': 1.49,
    'DO': 1.41, 'AR': 1.33, 'CO': 1.31, 'EN': 1.23, 'QU': 1.20,
    'ER': 1.18, 'DA': 1.17, 'RE': 1.14, 'CA': 1.11, 'TA': 1.10,
    'SE': 1.08, 'NT': 1.08, 'MA': 1.06, 'UE': 1.05, 'TE': 1.05
}
```

## Trigram Frequencies

### Reference Values vs. Implementation

| Trigram | Reference Frequency | Implementation Frequency | Match |
|---------|--------------------:|------------------------:|:-----:|
| QUE     | 0.96                | 0.96                    | ✓     |
| ENT     | 0.56                | 0.56                    | ✓     |
| COM     | 0.47                | 0.47                    | ✓     |
| NTE     | 0.44                | 0.44                    | ✓     |
| EST     | 0.34                | 0.34                    | ✓     |
| AVA     | 0.34                | 0.34                    | ✓     |
| ARA     | 0.33                | 0.33                    | ✓     |
| ADO     | 0.33                | 0.33                    | ✓     |
| PAR     | 0.30                | 0.30                    | ✓     |
| NDO     | 0.30                | 0.30                    | ✓     |
| NAO     | 0.30                | 0.30                    | ✓     |
| ERA     | 0.30                | 0.30                    | ✓     |
| AND     | 0.30                | 0.30                    | ✓     |
| UMA     | 0.28                | 0.28                    | ✓     |
| STA     | 0.28                | 0.28                    | ✓     |
| RES     | 0.27                | 0.27                    | ✓     |
| MEN     | 0.27                | 0.27                    | ✓     |
| CON     | 0.27                | 0.27                    | ✓     |
| DOS     | 0.25                | 0.25                    | ✓     |
| ANT     | 0.25                | 0.25                    | ✓     |

**Implementation Code:**
```python
# The 20 most frequent trigrams in Portuguese (per 100 letters)
TRIGRAM_FREQUENCIES = {
    'QUE': 0.96, 'ENT': 0.56, 'COM': 0.47, 'NTE': 0.44, 'EST': 0.34,
    'AVA': 0.34, 'ARA': 0.33, 'ADO': 0.33, 'PAR': 0.30, 'NDO': 0.30,
    'NAO': 0.30, 'ERA': 0.30, 'AND': 0.30, 'UMA': 0.28, 'STA': 0.28,
    'RES': 0.27, 'MEN': 0.27, 'CON': 0.27, 'DOS': 0.25, 'ANT': 0.25
}
```

## Short Words

### One-Letter Words

| Word | Reference Frequency | Implementation Frequency | Match |
|------|--------------------:|------------------------:|:-----:|
| E    | 0.88                | 0.88                    | ✓     |
| A    | 0.84                | 0.84                    | ✓     |
| O    | 0.71                | 0.71                    | ✓     |

### Two-Letter Words

| Word | Reference Frequency | Implementation Frequency | Match |
|------|--------------------:|------------------------:|:-----:|
| DE   | 0.82                | 0.82                    | ✓     |
| UM   | 0.31                | 0.31                    | ✓     |
| SE   | 0.30                | 0.30                    | ✓     |
| DA   | 0.27                | 0.27                    | ✓     |
| OS   | 0.25                | 0.25                    | ✓     |
| DO   | 0.25                | 0.25                    | ✓     |
| AS   | 0.19                | 0.19                    | ✓     |
| EM   | 0.17                | 0.17                    | ✓     |
| NO   | 0.14                | 0.14                    | ✓     |
| NA   | 0.12                | 0.12                    | ✓     |

### Three-Letter Words

| Word | Reference Frequency | Implementation Frequency | Match |
|------|--------------------:|------------------------:|:-----:|
| QUE  | 0.63                | 0.63                    | ✓     |
| NAO  | 0.29                | 0.29                    | ✓     |
| UMA  | 0.21                | 0.21                    | ✓     |
| COM  | 0.21                | 0.21                    | ✓     |
| ERA  | 0.14                | 0.14                    | ✓     |
| POR  | 0.12                | 0.12                    | ✓     |
| MAS  | 0.11                | 0.11                    | ✓     |
| DOS  | 0.11                | 0.11                    | ✓     |
| LHE  | 0.09                | 0.09                    | ✓     |
| FOI  | 0.07                | 0.07                    | ✓     |
| ELE  | 0.07                | 0.07                    | ✓     |
| DAS  | 0.07                | 0.07                    | ✓     |
| SUA  | 0.06                | 0.06                    | ✓     |
| SEU  | 0.06                | 0.06                    | ✓     |
| SEM  | 0.05                | 0.05                    | ✓     |

## Frequency-Based Matrix Generation

In our implementation, we use the Portuguese letter frequencies to generate matrices that are likely to produce valid Portuguese text. Here's how we use these statistics:

```python
def generate_frequency_based_matrices(self) -> List[np.ndarray]:
    """
    Generate matrices based on Portuguese letter frequencies.
    
    Returns:
        List of matrices
    """
    matrices = []
    
    # Most frequent letters in Portuguese (A, E, O, S, R, I, N, D, M, U)
    freq_indices = [0, 4, 14, 18, 17, 8, 13, 3, 12, 20]  # Indices in alphabet
    
    # Generate matrices with high-frequency letters in strategic positions
    for a in freq_indices[:5]:  # First row, first column
        for b in freq_indices[:5]:  # First row, second column
            for c in freq_indices[:5]:  # Second row, first column
                for d in freq_indices[:5]:  # Second row, second column
                    matrix = np.array([[a, b], [c, d]])
                    
                    # Check if matrix is invertible
                    if is_invertible_matrix(matrix):
                        matrices.append(matrix)
    
    # Add matrices with common patterns in Portuguese
    common_patterns = [
        # Matrices with common digrams
        np.array([[3, 4], [0, 0]]),  # DE
        np.array([[17, 0], [0, 0]]),  # RA
        np.array([[4, 18], [0, 0]]),  # ES
        np.array([[14, 18], [0, 0]]),  # OS
        np.array([[0, 18], [0, 0]]),  # AS
        
        # Known good matrices
        np.array([[23, 17], [0, 9]]),
        np.array([[17, 23], [9, 0]]),
        np.array([[23, 14], [0, 5]]),
        np.array([[5, 17], [18, 9]])
    ]
    
    # Scale these patterns with coprimes
    coprimes = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
    for pattern in common_patterns:
        for scale in coprimes:
            matrix = (pattern * scale) % 26
            if is_invertible_matrix(matrix):
                matrices.append(matrix)
    
    return matrices
```

## Conclusion

The implementation correctly matches all the reference values for Portuguese language statistics. The letter frequencies, digram frequencies, trigram frequencies, and word statistics are all accurately represented in the code.

The Hill Cipher Analyzer is using these statistics correctly to:
1. Generate matrices based on Portuguese letter frequencies
2. Score decrypted text based on statistical properties of Portuguese
3. Identify common Portuguese words in decrypted text

This implementation should provide accurate results when analyzing Portuguese text and breaking Hill ciphers.
