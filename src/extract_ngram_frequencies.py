#!/usr/bin/env python3
"""
Script to extract n-gram frequencies from a Portuguese text file.
This script reads a text file, normalizes it (removing accents, etc.),
and calculates frequencies for letters, digrams, trigrams, quadgrams, and pentagrams.
"""

import os
import re
import json
from collections import Counter
from typing import Dict, List, Tuple

def remove_accents(text: str) -> str:
    """
    Remove accents from text (á -> a, ç -> c, etc.)
    
    Args:
        text: Input text
        
    Returns:
        Text without accents
    """
    # Define accent mappings
    accent_map = {
        'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c', 'ñ': 'n',
        'Á': 'A', 'À': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
        'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
        'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
        'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
        'Ç': 'C', 'Ñ': 'N'
    }
    
    # Replace accented characters
    for accented, plain in accent_map.items():
        text = text.replace(accented, plain)
    
    return text

def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing accents, converting to uppercase,
    and keeping only A-Z characters.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Remove accents (á -> a, ç -> c, etc.)
    no_accents = remove_accents(text)
    
    # Convert to uppercase
    uppercase = no_accents.upper()
    
    # Keep only A-Z characters
    letters_only = re.sub(r'[^A-Z]', '', uppercase)
    
    return letters_only

def count_ngrams(text: str, n: int) -> Counter:
    """
    Count n-grams in text.
    
    Args:
        text: Input text
        n: Size of n-grams
        
    Returns:
        Counter with n-gram frequencies
    """
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    return Counter(ngrams)

def calculate_frequencies(counts: Counter, total: int) -> Dict[str, float]:
    """
    Calculate frequencies from counts.
    
    Args:
        counts: Counter with n-gram counts
        total: Total number of n-grams
        
    Returns:
        Dictionary with n-gram frequencies
    """
    return {item: count / total for item, count in counts.items()}

def get_top_ngrams(frequencies: Dict[str, float], top_n: int = 20) -> List[Tuple[str, float]]:
    """
    Get top n-grams by frequency.
    
    Args:
        frequencies: Dictionary with n-gram frequencies
        top_n: Number of top n-grams to return
        
    Returns:
        List of (n-gram, frequency) tuples
    """
    return sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:top_n]

def main():
    """Main function."""
    # File path
    file_path = "textos_conhecidos/textos/avesso_da_pele.txt"
    
    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try with Latin-1 encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            text = f.read()
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    print(f"Processed text length: {len(processed_text)} characters")
    
    # Calculate letter frequencies
    letter_counts = Counter(processed_text)
    total_letters = len(processed_text)
    letter_frequencies = calculate_frequencies(letter_counts, total_letters)
    
    # Calculate n-gram frequencies
    ngram_results = {}
    for n in range(2, 6):  # 2=digrams, 3=trigrams, 4=quadgrams, 5=pentagrams
        ngram_counts = count_ngrams(processed_text, n)
        total_ngrams = len(processed_text) - n + 1
        ngram_frequencies = calculate_frequencies(ngram_counts, total_ngrams)
        ngram_results[n] = ngram_frequencies
    
    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save letter frequencies
    with open("data/letter_frequencies.json", 'w') as f:
        json.dump(letter_frequencies, f, indent=2)
    
    # Save n-gram frequencies
    for n, frequencies in ngram_results.items():
        with open(f"data/{n}gram_frequencies.json", 'w') as f:
            json.dump(frequencies, f, indent=2)
    
    # Save top n-grams for each n
    top_ngrams = {}
    top_ngrams[1] = get_top_ngrams(letter_frequencies, 26)  # All letters
    for n in range(2, 6):
        top_ngrams[n] = get_top_ngrams(ngram_results[n], 20)
    
    with open("data/top_ngrams.json", 'w') as f:
        json.dump({str(n): [(ngram, freq) for ngram, freq in ngrams] 
                  for n, ngrams in top_ngrams.items()}, f, indent=2)
    
    # Print top n-grams
    print("\nTop n-grams:")
    for n in range(1, 6):
        print(f"\nTop {n}-grams:")
        for ngram, freq in top_ngrams[n]:
            print(f"  {ngram}: {freq:.6f}")
    
    # Generate Python code for use in frequency analyzer
    with open("src/portuguese_ngrams.py", 'w') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""\n')
        f.write("Portuguese n-gram frequencies extracted from avesso_da_pele.txt.\n")
        f.write("This file is automatically generated by extract_ngram_frequencies.py.\n")
        f.write('"""\n\n')
        
        # Write letter frequencies
        f.write("# Letter frequencies\n")
        f.write("LETTER_FREQUENCIES = {\n")
        for letter, freq in sorted(top_ngrams[1]):
            f.write(f"    '{letter}': {freq:.6f},\n")
        f.write("}\n\n")
        
        # Write n-gram lists for each n
        for n in range(2, 6):
            name = {2: "DIGRAMS", 3: "TRIGRAMS", 4: "QUADGRAMS", 5: "PENTAGRAMS"}[n]
            f.write(f"# Top {n}-grams\n")
            f.write(f"{name} = [\n")
            for ngram, _ in top_ngrams[n]:
                f.write(f"    '{ngram}',\n")
            f.write("]\n\n")
        
        # Write dictionary for use in frequency analyzer
        f.write("# Dictionary for use in frequency analyzer\n")
        f.write("COMMON_NGRAMS = {\n")
        for n in range(2, 6):
            f.write(f"    {n}: {[ngram for ngram, _ in top_ngrams[n]]},\n")
        f.write("}\n")
    
    print("\nResults saved to:")
    print("  - data/letter_frequencies.json")
    print("  - data/2gram_frequencies.json")
    print("  - data/3gram_frequencies.json")
    print("  - data/4gram_frequencies.json")
    print("  - data/5gram_frequencies.json")
    print("  - data/top_ngrams.json")
    print("  - src/portuguese_ngrams.py")

if __name__ == "__main__":
    main()
