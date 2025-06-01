# Hill Cipher Implementation and Breakers

This project has been reorganized into a cleaner structure. Please see the `hill_cipher` directory for the new implementation.

## New Project Structure

The project has been reorganized into a more modular and maintainable structure:

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

- Hill Cipher implementation with support for key sizes 2x2, 3x3, 4x4, and 5x5
- Known-plaintext attack implementation
- Genetic algorithm-based frequency analysis
- Text processing utilities
- Unit tests for all components

## Usage

See the README.md file in the `hill_cipher` directory for detailed usage instructions.

## Author

Lucas Kledeglau Jahchan Alves
