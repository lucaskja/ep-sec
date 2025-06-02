#!/usr/bin/env python3
"""
Hill Cipher Breaker - Main Module

This module provides a unified interface for breaking Hill ciphers
using various techniques including:
1. Known-plaintext attack
2. Genetic algorithm-based frequency analysis

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
from breakers.kpa import HillCipherKPA
from breakers.genetic import HillCipherGA, load_language_frequencies

# Configure logging
logger = logging.getLogger('hill_cipher_breaker')

def break_hill_cipher(ciphertext: str, key_size: int, 
                      plaintext: Optional[str] = None,
                      method: str = 'auto',
                      generations: int = 100,
                      early_stopping: int = 20,
                      max_attempts: int = 10,
                      verbose: bool = False,
                      quiet: bool = False) -> Tuple[Optional[np.ndarray], Optional[str]]:
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
        quiet: Reduce output verbosity
        
    Returns:
        Tuple of (recovered key, decrypted text)
    """
    # Determine the best method to use
    if method == 'auto':
        if plaintext and len(plaintext) >= key_size * key_size:
            method = 'kpa'
        else:
            method = 'ga'
    
    logger.info(f"Breaking Hill cipher using {method} method")
    
    # Load normalized text for validation
    normalized_text = ""
    try:
        normalized_text_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                           "data", "normalized_text.txt")
        with open(normalized_text_path, 'r') as f:
            normalized_text = f.read()
        logger.info(f"Loaded normalized text ({len(normalized_text)} characters)")
    except Exception as e:
        logger.warning(f"Failed to load normalized text: {e}")
    
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
                
                # Validate against normalized text if available
                if normalized_text:
                    is_valid = False
                    for length in range(15, min(100, len(decrypted)), 5):
                        for start_pos in range(0, min(200, len(decrypted) - length), 20):
                            substring = decrypted[start_pos:start_pos+length]
                            if substring in normalized_text:
                                logger.info(f"Found matching substring in normalized text: '{substring}'")
                                is_valid = True
                                break
                        if is_valid:
                            break
                    
                    if is_valid:
                        logger.info("Decryption validated against normalized text")
                    else:
                        logger.warning("Decryption could not be validated against normalized text")
                
                return key, decrypted
            else:
                logger.warning("KPA failed to recover key, falling back to GA method")
                # Fall back to GA method
                method = 'ga'
        except ValueError as e:
            logger.error(f"KPA error: {e}, falling back to GA method")
            # Fall back to GA method
            method = 'ga'
    
    if method == 'ga':
        # Use genetic algorithm
        language_frequencies = load_language_frequencies()
        ga = HillCipherGA(key_size, language_frequencies)
        
        # Set GA to be less verbose if quiet mode is enabled
        if quiet:
            ga.verbose = False
        
        # Run GA with multiple attempts until a solution is found
        key, decrypted = ga.crack(ciphertext, generations, early_stopping, max_attempts=10)
        
        # Validate against normalized text if available
        if key is not None and normalized_text:
            is_valid = False
            for length in range(15, min(100, len(decrypted)), 5):
                for start_pos in range(0, min(200, len(decrypted) - length), 20):
                    substring = decrypted[start_pos:start_pos+length]
                    if substring in normalized_text:
                        logger.info(f"Found matching substring in normalized text: '{substring}'")
                        is_valid = True
                        break
                if is_valid:
                    break
            
            if is_valid:
                logger.info("Decryption validated against normalized text")
            else:
                logger.warning("Decryption could not be validated against normalized text")
        
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
    parser.add_argument("--population-size", type=int, default=1000, help="Population size for GA")
    parser.add_argument("--early-stopping", type=int, default=20, help="Stop if no improvement after this many generations")
    parser.add_argument("--max-attempts", type=int, default=10, help="Maximum number of GA attempts with different initial populations")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--log-file", type=str, help="File to save logs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    
    # Set up logging to file if specified
    if args.log_file:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=args.log_file,
            filemode='w'
        )
        # Add console handler with a higher log level
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING if args.quiet else logging.INFO)
        console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logging.getLogger('').addHandler(console)
    else:
        # Just log to console
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
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
        max_attempts=args.max_attempts,
        verbose=args.verbose,
        quiet=args.quiet
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
        
        np.savetxt(key_path, key, fmt='%d')
        with open(decrypted_path, 'w') as f:
            f.write(decrypted)
        
        print(f"Results saved to {key_path} and {decrypted_path}")
    else:
        print("Failed to recover key")

if __name__ == "__main__":
    main()
