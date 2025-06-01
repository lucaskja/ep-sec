#!/usr/bin/env python3
"""
Script to find common substrings in the known text file that could be used for breaking the Hill cipher.
This script analyzes the avesso_da_pele.txt file to find frequent substrings that might appear in the ciphertext.
"""

import os
import re
import argparse
from collections import Counter
from typing import List, Tuple, Dict

def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing non-alphabetic characters and converting to uppercase.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Remove accents (manually since we're not using unidecode)
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
    
    for accented, plain in accent_map.items():
        text = text.replace(accented, plain)
    
    # Convert to uppercase and remove non-alphabetic characters
    return re.sub(r'[^A-Za-z]', '', text).upper()

def find_common_substrings(text: str, min_length: int = 5, max_length: int = 15, top_n: int = 20) -> Dict[int, List[Tuple[str, int]]]:
    """
    Find common substrings in text of various lengths.
    
    Args:
        text: Input text
        min_length: Minimum substring length
        max_length: Maximum substring length
        top_n: Number of top substrings to return for each length
        
    Returns:
        Dictionary mapping substring length to list of (substring, count) tuples
    """
    results = {}
    
    for length in range(min_length, max_length + 1):
        # Extract all substrings of the current length
        substrings = [text[i:i+length] for i in range(len(text) - length + 1)]
        
        # Count occurrences
        counts = Counter(substrings)
        
        # Get top N most common
        top_substrings = counts.most_common(top_n)
        
        # Only include substrings that appear more than once
        top_substrings = [(s, c) for s, c in top_substrings if c > 1]
        
        if top_substrings:
            results[length] = top_substrings
    
    return results

def find_repeated_patterns(text: str, min_length: int = 10, max_length: int = 30, min_occurrences: int = 3) -> List[Tuple[str, int]]:
    """
    Find repeated patterns in text that might be useful for known plaintext attacks.
    
    Args:
        text: Input text
        min_length: Minimum pattern length
        max_length: Maximum pattern length
        min_occurrences: Minimum number of occurrences to consider
        
    Returns:
        List of (pattern, count) tuples
    """
    patterns = []
    
    for length in range(min_length, max_length + 1):
        # Extract all substrings of the current length
        substrings = [text[i:i+length] for i in range(len(text) - length + 1)]
        
        # Count occurrences
        counts = Counter(substrings)
        
        # Get patterns that occur at least min_occurrences times
        frequent_patterns = [(s, c) for s, c in counts.items() if c >= min_occurrences]
        
        patterns.extend(frequent_patterns)
    
    # Sort by length (longer patterns first) and then by frequency
    return sorted(patterns, key=lambda x: (-len(x[0]), -x[1]))

def find_common_words(text: str, min_length: int = 3, top_n: int = 50) -> List[Tuple[str, int]]:
    """
    Find common words in text.
    
    Args:
        text: Input text
        min_length: Minimum word length
        top_n: Number of top words to return
        
    Returns:
        List of (word, count) tuples
    """
    # Add spaces before preprocessing to preserve word boundaries
    words = re.findall(r'\b[A-Za-z]+\b', text.lower())
    
    # Preprocess words
    processed_words = [preprocess_text(word) for word in words]
    
    # Filter by length
    filtered_words = [word for word in processed_words if len(word) >= min_length]
    
    # Count occurrences
    counts = Counter(filtered_words)
    
    # Get top N most common
    return counts.most_common(top_n)

def find_potential_known_plaintext(text: str) -> List[str]:
    """
    Find potential known plaintext segments for Hill cipher breaking.
    
    Args:
        text: Input text
        
    Returns:
        List of potential known plaintext segments
    """
    # Process text
    processed_text = preprocess_text(text)
    
    # Find repeated patterns
    patterns = find_repeated_patterns(processed_text, min_occurrences=3)
    
    # Find common words
    words = find_common_words(text)
    
    # Combine patterns and words
    potential_segments = []
    
    # Add top patterns
    for pattern, count in patterns[:10]:
        potential_segments.append(pattern)
    
    # Add top words
    for word, count in words[:20]:
        if len(word) >= 5 and word not in potential_segments:
            potential_segments.append(word)
    
    # Add some common phrases from the text
    common_phrases = [
        "VOCE", "MINHA", "ESTAVA", "PORQUE", "QUANDO", "AQUELE", "SEMPRE",
        "MINHAMAE", "PROFESSOR", "ESCOLA", "ALUNOS", "POLICIA", "RASKOLNIKOV"
    ]
    
    for phrase in common_phrases:
        if phrase not in potential_segments and phrase in processed_text:
            potential_segments.append(phrase)
    
    return potential_segments

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Find common substrings in a text file")
    parser.add_argument("--file", default="textos_conhecidos/textos/avesso_da_pele.txt", help="Path to text file")
    parser.add_argument("--min-length", type=int, default=5, help="Minimum substring length")
    parser.add_argument("--max-length", type=int, default=15, help="Maximum substring length")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top substrings to return for each length")
    parser.add_argument("--output", default="data/known_plaintext_segments.py", help="Output file path")
    
    args = parser.parse_args()
    
    # Read file
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try with Latin-1 encoding if UTF-8 fails
        with open(args.file, 'r', encoding='latin-1') as f:
            text = f.read()
    
    # Find common words
    print("Finding common words...")
    common_words = find_common_words(text)
    
    print(f"\nTop {len(common_words)} common words:")
    for word, count in common_words[:20]:
        print(f"  {word}: {count}")
    
    # Preprocess text for substring analysis
    processed_text = preprocess_text(text)
    
    print(f"\nProcessed text length: {len(processed_text)} characters")
    
    # Find common substrings
    print(f"Finding common substrings (length {args.min_length}-{args.max_length})...")
    common_substrings = find_common_substrings(
        processed_text, 
        min_length=args.min_length, 
        max_length=args.max_length,
        top_n=args.top_n
    )
    
    # Print results
    print("\nCommon substrings by length:")
    for length in sorted(common_substrings.keys()):
        print(f"\nLength {length}:")
        for substring, count in common_substrings[length][:5]:  # Show only top 5 for each length
            print(f"  {substring}: {count}")
    
    # Find potential known plaintext segments
    print("\nFinding potential known plaintext segments...")
    potential_segments = find_potential_known_plaintext(text)
    
    print(f"\nFound {len(potential_segments)} potential known plaintext segments:")
    for i, segment in enumerate(potential_segments[:20]):  # Show only top 20
        print(f"  {i+1}. {segment}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save results to file
    with open(args.output, 'w') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""\n')
        f.write("Known plaintext segments from avesso_da_pele.txt.\n")
        f.write("This file is automatically generated by find_common_substrings.py.\n")
        f.write('"""\n\n')
        
        # Write potential known plaintext segments
        f.write("# Potential known plaintext segments for Hill cipher breaking\n")
        f.write("KNOWN_PLAINTEXT_SEGMENTS = [\n")
        for segment in potential_segments:
            f.write(f"    '{segment}',\n")
        f.write("]\n\n")
        
        # Write common words
        f.write("# Common words\n")
        f.write("COMMON_WORDS = [\n")
        for word, count in common_words[:50]:
            f.write(f"    # {count} occurrences\n")
            f.write(f"    '{word}',\n")
        f.write("]\n\n")
        
        # Write common substrings
        f.write("# Common substrings by length\n")
        f.write("COMMON_SUBSTRINGS = {\n")
        for length in sorted(common_substrings.keys()):
            f.write(f"    {length}: [\n")
            for substring, count in common_substrings[length][:10]:  # Top 10 for each length
                f.write(f"        # {count} occurrences\n")
                f.write(f"        '{substring}',\n")
            f.write("    ],\n")
        f.write("}\n")
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
