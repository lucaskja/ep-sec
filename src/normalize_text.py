#!/usr/bin/env python3
"""
Script to normalize text by:
1. Removing spaces and newlines
2. Converting to uppercase
3. Removing punctuation
4. Normalizing accented characters (á -> a, ç -> c, etc.)
"""

import os
import re
import argparse
import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalize text by removing spaces, newlines, punctuation,
    converting to uppercase, and normalizing accented characters.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Convert to uppercase
    text = text.upper()
    
    # Remove accents (á -> a, ç -> c, etc.)
    # This uses unicodedata to decompose accented characters and then remove the accent marks
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Remove all non-alphabetic characters (spaces, newlines, punctuation, etc.)
    text = re.sub(r'[^A-Z]', '', text)
    
    return text

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Normalize text file")
    parser.add_argument("--input", default="textos_conhecidos/textos/avesso_da_pele.txt", help="Input text file")
    parser.add_argument("--output", default="data/normalized_text.txt", help="Output text file")
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try with Latin-1 encoding if UTF-8 fails
        with open(args.input, 'r', encoding='latin-1') as f:
            text = f.read()
    
    # Normalize text
    normalized_text = normalize_text(text)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Write output file
    with open(args.output, 'w') as f:
        f.write(normalized_text)
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Original length: {len(text)} characters")
    print(f"Normalized length: {len(normalized_text)} characters")
    print("Normalization complete!")

if __name__ == "__main__":
    main()
