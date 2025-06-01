#!/usr/bin/env python3
"""
Text Processing Utilities for Hill Cipher

This module provides utility functions for text processing,
including normalization, n-gram extraction, and frequency analysis.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('text_processing')

def normalize_text(text: str) -> str:
    """
    Normalize text by removing non-alphabetic characters and converting to uppercase.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Remove non-alphabetic characters and convert to uppercase
    normalized = re.sub(r'[^A-Za-z]', '', text).upper()
    return normalized

def extract_ngrams(text: str, n: int) -> List[str]:
    """
    Extract n-grams from text.
    
    Args:
        text: Input text
        n: Size of n-grams
        
    Returns:
        List of n-grams
    """
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def count_ngrams(text: str, n: int) -> List[Tuple[str, int]]:
    """
    Count n-grams in text.
    
    Args:
        text: Input text
        n: Size of n-grams
        
    Returns:
        List of (n-gram, count) tuples, sorted by frequency
    """
    ngrams = extract_ngrams(text, n)
    counter = Counter(ngrams)
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)

def calculate_ngram_frequencies(text: str, n: int) -> Dict[str, float]:
    """
    Calculate n-gram frequencies in text.
    
    Args:
        text: Input text
        n: Size of n-grams
        
    Returns:
        Dictionary mapping n-grams to their frequencies
    """
    ngrams = extract_ngrams(text, n)
    counter = Counter(ngrams)
    total = len(ngrams)
    return {ngram: count / total for ngram, count in counter.items()}

def save_ngram_frequencies(frequencies: Dict[str, float], output_file: str) -> None:
    """
    Save n-gram frequencies to a JSON file.
    
    Args:
        frequencies: Dictionary mapping n-grams to their frequencies
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(frequencies, f, indent=2)

def load_ngram_frequencies(n: int, language: str = 'portuguese') -> Dict[str, float]:
    """
    Load n-gram frequencies from a JSON file.
    
    Args:
        n: Size of n-grams
        language: Language of the n-grams
        
    Returns:
        Dictionary mapping n-grams to their frequencies
    """
    # Define the base directory for data files
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # Define the file path
    if n == 1:
        file_path = os.path.join(base_dir, "letter_frequencies.json")
    else:
        file_path = os.path.join(base_dir, f"{n}gram_frequencies.json")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {n}-gram frequencies: {e}")
        return {}

def extract_ngram_frequencies(input_file: str, output_dir: str, n_values: List[int] = [1, 2, 3]) -> None:
    """
    Extract n-gram frequencies from a text file and save them to JSON files.
    
    Args:
        input_file: Input text file
        output_dir: Output directory for JSON files
        n_values: List of n values to extract
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_file, 'r') as f:
        text = f.read()
    
    # Normalize text
    normalized_text = normalize_text(text)
    
    # Extract and save n-gram frequencies
    for n in n_values:
        frequencies = calculate_ngram_frequencies(normalized_text, n)
        
        # Define output file name
        if n == 1:
            output_file = os.path.join(output_dir, "letter_frequencies.json")
        else:
            output_file = os.path.join(output_dir, f"{n}gram_frequencies.json")
        
        # Save frequencies
        save_ngram_frequencies(frequencies, output_file)
        logger.info(f"Saved {n}-gram frequencies to {output_file}")

def main():
    """Main function for demonstration and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text Processing Utilities for Hill Cipher")
    parser.add_argument("--normalize", type=str, help="Normalize text file")
    parser.add_argument("--extract-ngrams", type=str, help="Extract n-grams from text file")
    parser.add_argument("--n", type=int, default=2, help="Size of n-grams")
    parser.add_argument("--output", type=str, help="Output file or directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if args.normalize:
        # Normalize text file
        with open(args.normalize, 'r') as f:
            text = f.read()
        
        normalized_text = normalize_text(text)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(normalized_text)
            logger.info(f"Normalized text saved to {args.output}")
        else:
            print(normalized_text)
    
    elif args.extract_ngrams:
        # Extract n-grams from text file
        if not args.output:
            parser.error("Output directory must be provided for n-gram extraction")
        
        extract_ngram_frequencies(args.extract_ngrams, args.output, [args.n])

if __name__ == "__main__":
    main()
