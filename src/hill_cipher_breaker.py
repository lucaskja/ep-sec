#!/usr/bin/env python3
"""
Hill Cipher Breaker - Main Module

This module provides a unified interface for breaking Hill ciphers
using various techniques including:
1. Known-plaintext attack
2. Genetic algorithm-based frequency analysis
3. N-gram frequency analysis

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import numpy as np
import logging
import argparse
import time
from typing import List, Tuple, Dict, Optional, Union

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.hill_cipher_kpa import HillCipherKPA
from src.hill_cipher_ga import HillCipherGA, load_language_frequencies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hill_cipher_breaker')

def break_hill_cipher(ciphertext: str, key_size: int, 
                      plaintext: Optional[str] = None,
                      method: str = 'auto',
                      generations: int = 100,
                      early_stopping: int = 20,
                      verbose: bool = False) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Break Hill cipher using the specified method.
    
    Args:
        ciphertext: Encrypted text
        key_size: Size of the key matrix
        plaintext: Known plaintext (for KPA method)
        method: Attack method ('kpa', 'ga', 'auto')
        generations: Maximum number of generations for GA
        early_stopping: Stop if no improvement after this many generations
        verbose: Enable verbose output
        
    Returns:
        Tuple of (recovered key, decrypted text)
    """
    # Set logging level based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine the best method to use
    if method == 'auto':
        if plaintext and len(plaintext) >= key_size * key_size:
            method = 'kpa'
        else:
            method = 'ga'
    
    logger.info(f"Breaking Hill cipher using {method} method")
    
    # Apply the selected method
    if method == 'kpa':
        if not plaintext:
            logger.error("Plaintext is required for KPA method")
            return None, None
        
        # Use known-plaintext attack
        kpa = HillCipherKPA(key_size)
        try:
            key = kpa.recover_key(plaintext, ciphertext)
            if key is not None:
                # Decrypt the ciphertext with the recovered key
                decrypted = kpa.decrypt(ciphertext, key)
                return key, decrypted
            else:
                logger.warning("KPA failed to recover key")
                return None, None
        except ValueError as e:
            logger.error(f"KPA error: {e}")
            return None, None
    
    elif method == 'ga':
        # Use genetic algorithm
        language_frequencies = load_language_frequencies()
        ga = HillCipherGA(key_size, language_frequencies)
        key, decrypted = ga.crack(ciphertext, generations, early_stopping)
        return key, decrypted
    
    else:
        logger.error(f"Unknown method: {method}")
        return None, None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Hill Cipher Breaker")
    parser.add_argument("--key-size", type=int, default=2, choices=[2, 3, 4, 5], help="Size of the key matrix")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext to decrypt")
    parser.add_argument("--ciphertext-file", type=str, help="File containing ciphertext")
    parser.add_argument("--plaintext", type=str, help="Known plaintext (for KPA method)")
    parser.add_argument("--plaintext-file", type=str, help="File containing known plaintext")
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "kpa", "ga"], help="Attack method")
    parser.add_argument("--generations", type=int, default=100, help="Maximum number of generations for GA")
    parser.add_argument("--early-stopping", type=int, default=20, help="Stop if no improvement after this many generations")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Get ciphertext
    ciphertext = args.ciphertext
    if args.ciphertext_file:
        with open(args.ciphertext_file, 'r') as f:
            ciphertext = f.read().strip()
    
    if not ciphertext:
        parser.error("Ciphertext must be provided")
    
    # Get plaintext if provided
    plaintext = args.plaintext
    if args.plaintext_file:
        with open(args.plaintext_file, 'r') as f:
            plaintext = f.read().strip()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Break the cipher
    start_time = time.time()
    key, decrypted = break_hill_cipher(
        ciphertext=ciphertext,
        key_size=args.key_size,
        plaintext=plaintext,
        method=args.method,
        generations=args.generations,
        early_stopping=args.early_stopping,
        verbose=args.verbose
    )
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    if key is not None:
        print(f"Recovered {args.key_size}x{args.key_size} key matrix:")
        print(key)
        print(f"Decrypted text (first 100 characters):")
        print(decrypted[:100])
        
        # Save results
        key_path = os.path.join(args.output_dir, f"key_{args.key_size}x{args.key_size}.txt")
        decrypted_path = os.path.join(args.output_dir, f"decrypted_{args.key_size}x{args.key_size}.txt")
        
        with open(key_path, 'w') as f:
            f.write(str(key))
        with open(decrypted_path, 'w') as f:
            f.write(decrypted)
        
        print(f"Results saved to {key_path} and {decrypted_path}")
    else:
        print("Failed to recover key")

if __name__ == "__main__":
    main()
