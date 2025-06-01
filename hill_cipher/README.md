# Hill Cipher Implementation and Breakers

This project implements the Hill cipher encryption algorithm and various methods to break it, including known-plaintext attacks and genetic algorithm-based frequency analysis.

## Project Structure

```
hill_cipher/
├── core/                      # Core functionality
│   └── hill_cipher.py         # Hill cipher implementation
├── breakers/                  # Cipher breaking implementations
│   ├── kpa.py                 # Known-plaintext attack
│   ├── genetic.py             # Genetic algorithm-based frequency analysis
│   └── hill_breaker.py        # Unified interface for breaking Hill ciphers
├── utils/                     # Utility functions
│   └── text_processing.py     # Text processing utilities
├── scripts/                   # Scripts for running the code
│   ├── run_hill_breaker.sh    # Bash script to run Hill cipher breaker
│   └── run_hill_breaker.ps1   # PowerShell script to run Hill cipher breaker
├── data/                      # Data files
│   ├── letter_frequencies.json # Letter frequencies for Portuguese
│   ├── 2gram_frequencies.json # 2-gram frequencies for Portuguese
│   ├── 3gram_frequencies.json # 3-gram frequencies for Portuguese
│   └── normalized_text.txt    # Normalized text for validation
└── tests/                     # Unit tests
    ├── test_hill_cipher.py    # Tests for Hill cipher implementation
    ├── test_kpa.py            # Tests for known-plaintext attack
    └── test_genetic.py        # Tests for genetic algorithm-based frequency analysis
```

## Features

### Hill Cipher Implementation

- Support for key sizes 2x2, 3x3, 4x4, and 5x5
- Encryption and decryption operations
- Matrix operations in modulo 26
- Validation of key invertibility

### Known-Plaintext Attack

- Recovery of encryption key from plaintext-ciphertext pairs
- Verification of recovered keys
- Support for all key sizes

### Genetic Algorithm-based Frequency Analysis

- Breaking Hill cipher without known plaintext
- Using n-gram frequency analysis for Portuguese language
- Population evolution with crossover and mutation
- Fitness evaluation based on language statistics
- Parallel processing for performance optimization

## Usage

### Hill Cipher Encryption/Decryption

```bash
python core/hill_cipher.py --encrypt --text "HELLO" --key "3,2,5,3" --key-size 2
python core/hill_cipher.py --decrypt --text "ENCRYPTED_TEXT" --key "3,2,5,3" --key-size 2
```

### Known-Plaintext Attack

```bash
python breakers/kpa.py --plaintext "KNOWN_PLAINTEXT" --ciphertext "CORRESPONDING_CIPHERTEXT" --key-size 2
```

### Genetic Algorithm-based Frequency Analysis

```bash
python breakers/genetic.py --ciphertext "ENCRYPTED_TEXT" --key-size 2 --generations 100
```

### Unified Hill Cipher Breaker

```bash
python breakers/hill_breaker.py --ciphertext "ENCRYPTED_TEXT" --key-size 2 --method auto
```

### Running on Multiple Files

#### On Unix/Linux/macOS:

```bash
./scripts/run_hill_breaker.sh
```

#### On Windows:

```powershell
.\scripts\run_hill_breaker.ps1
```

## Requirements

- Python 3.6+
- NumPy
- Multiprocessing (for parallel processing)

## Testing

Run the unit tests to verify the implementation:

```bash
python -m unittest discover tests
```

## Author

Lucas Kledeglau Jahchan Alves

## License

This project is licensed under the MIT License - see the LICENSE file for details.
