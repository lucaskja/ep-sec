#!/usr/bin/env python3
"""
Enhanced Hill Cipher Breaker

This module provides an enhanced Hill cipher breaker that combines
multiple techniques including statistical analysis, genetic algorithms,
and known-plaintext attacks.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Union
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hill_cipher import HillCipher
from breakers.statistical_analyzer import StatisticalAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('enhanced_breaker')

class EnhancedHillBreaker:
    """
    Enhanced Hill cipher breaker using multiple attack methods.
    """
    
    def __init__(self, key_size: int, data_dir: str = None):
        """
        Initialize the enhanced breaker.
        
        Args:
            key_size: Size of the Hill cipher key matrix
            data_dir: Directory containing frequency data files
        """
        self.key_size = key_size
        self.hill_cipher = HillCipher(key_size)
        self.statistical_analyzer = StatisticalAnalyzer(key_size, data_dir)
        
        logger.info(f"Initialized Enhanced Hill Breaker for {key_size}x{key_size} cipher")
    
    def break_cipher(self, ciphertext: str, 
                    known_plaintext: str = None,
                    method: str = 'auto',
                    max_candidates: int = 10000,
                    timeout: int = 300) -> Dict:
        """
        Break Hill cipher using the best available method.
        
        Args:
            ciphertext: Encrypted text
            known_plaintext: Known plaintext (if available)
            method: Attack method ('statistical', 'kpa', 'auto')
            max_candidates: Maximum number of candidates for statistical attack
            timeout: Maximum time in seconds
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        results = {
            'success': False,
            'key': None,
            'decrypted_text': None,
            'score': float('-inf'),
            'method_used': method,
            'time_elapsed': 0,
            'attempts': 0
        }
        
        logger.info(f"Starting Hill cipher attack using method: {method}")
        logger.info(f"Ciphertext length: {len(ciphertext)}")
        logger.info(f"Key size: {self.key_size}x{self.key_size}")
        
        try:
            if method == 'kpa' and known_plaintext:
                # Known-plaintext attack
                results.update(self._kpa_attack(ciphertext, known_plaintext))
            
            elif method == 'statistical' or (method == 'auto' and not known_plaintext):
                # Statistical analysis attack
                results.update(self._statistical_attack(ciphertext, max_candidates, timeout))
            
            elif method == 'auto':
                # Try multiple methods
                if known_plaintext:
                    # Try KPA first
                    kpa_results = self._kpa_attack(ciphertext, known_plaintext)
                    if kpa_results['success']:
                        results.update(kpa_results)
                        results['method_used'] = 'kpa'
                    else:
                        # Fall back to statistical
                        stat_results = self._statistical_attack(ciphertext, max_candidates, timeout)
                        results.update(stat_results)
                        results['method_used'] = 'statistical'
                else:
                    # Only statistical analysis available
                    results.update(self._statistical_attack(ciphertext, max_candidates, timeout))
                    results['method_used'] = 'statistical'
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        except Exception as e:
            logger.error(f"Error during cipher breaking: {e}")
            results['error'] = str(e)
        
        results['time_elapsed'] = time.time() - start_time
        
        # Log results
        if results['success']:
            logger.info(f"✓ Cipher broken successfully!")
            logger.info(f"Method: {results['method_used']}")
            logger.info(f"Time: {results['time_elapsed']:.2f} seconds")
            logger.info(f"Score: {results['score']:.2f}")
            logger.info(f"Key: {results['key'].flatten() if results['key'] is not None else 'None'}")
            logger.info(f"Decrypted: {results['decrypted_text'][:50]}...")
        else:
            logger.warning("✗ Failed to break cipher")
        
        return results
    
    def _kpa_attack(self, ciphertext: str, known_plaintext: str) -> Dict:
        """
        Perform known-plaintext attack.
        
        Args:
            ciphertext: Encrypted text
            known_plaintext: Known plaintext
            
        Returns:
            Attack results
        """
        logger.info("Attempting known-plaintext attack...")
        
        try:
            # Import KPA module
            from breakers.kpa import HillCipherKPA
            
            kpa = HillCipherKPA(self.key_size)
            key = kpa.attack(ciphertext, known_plaintext)
            
            if key is not None:
                # Verify the key works
                decrypted = self.hill_cipher.decrypt(ciphertext, key)
                score = self.statistical_analyzer.score_text(decrypted)
                
                return {
                    'success': True,
                    'key': key,
                    'decrypted_text': decrypted,
                    'score': score,
                    'attempts': 1
                }
            else:
                return {
                    'success': False,
                    'attempts': 1
                }
        
        except ImportError:
            logger.warning("KPA module not available")
            return {'success': False, 'attempts': 0}
        except Exception as e:
            logger.error(f"KPA attack failed: {e}")
            return {'success': False, 'attempts': 1, 'error': str(e)}
    
    def _statistical_attack(self, ciphertext: str, max_candidates: int, timeout: int) -> Dict:
        """
        Perform statistical analysis attack.
        
        Args:
            ciphertext: Encrypted text
            max_candidates: Maximum number of candidates to test
            timeout: Maximum time in seconds
            
        Returns:
            Attack results
        """
        logger.info("Attempting statistical analysis attack...")
        
        # Adjust candidates based on key size and timeout
        adjusted_candidates = self._adjust_candidates_for_key_size(max_candidates)
        
        start_time = time.time()
        
        def progress_callback(current, total, best_score):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError("Statistical attack timeout")
        
        try:
            key, decrypted, score = self.statistical_analyzer.break_cipher_statistical(
                ciphertext,
                max_candidates=adjusted_candidates,
                use_smart_generation=True,
                early_stopping_threshold=self._get_early_stopping_threshold(),
                progress_callback=progress_callback
            )
            
            if key is not None:
                return {
                    'success': True,
                    'key': key,
                    'decrypted_text': decrypted,
                    'score': score,
                    'attempts': adjusted_candidates
                }
            else:
                return {
                    'success': False,
                    'attempts': adjusted_candidates
                }
        
        except TimeoutError:
            logger.warning(f"Statistical attack timed out after {timeout} seconds")
            return {'success': False, 'attempts': adjusted_candidates, 'timeout': True}
        except Exception as e:
            logger.error(f"Statistical attack failed: {e}")
            return {'success': False, 'attempts': adjusted_candidates, 'error': str(e)}
    
    def _adjust_candidates_for_key_size(self, base_candidates: int) -> int:
        """
        Adjust the number of candidates based on key size.
        
        Args:
            base_candidates: Base number of candidates
            
        Returns:
            Adjusted number of candidates
        """
        # Reduce candidates for larger key sizes due to exponentially larger search space
        multipliers = {2: 1.0, 3: 0.5, 4: 0.2, 5: 0.1}
        multiplier = multipliers.get(self.key_size, 0.1)
        
        adjusted = int(base_candidates * multiplier)
        logger.info(f"Adjusted candidates from {base_candidates} to {adjusted} for {self.key_size}x{self.key_size} key")
        
        return max(adjusted, 1000)  # Minimum 1000 candidates
    
    def _get_early_stopping_threshold(self) -> float:
        """
        Get early stopping threshold based on key size.
        
        Returns:
            Early stopping threshold
        """
        # Larger keys might have slightly lower scores due to more complex patterns
        thresholds = {2: -30.0, 3: -40.0, 4: -50.0, 5: -60.0}
        return thresholds.get(self.key_size, -50.0)
    
    def validate_solution(self, ciphertext: str, key: np.ndarray, 
                         known_plaintext: str = None) -> Dict:
        """
        Validate a potential solution.
        
        Args:
            ciphertext: Original ciphertext
            key: Potential key matrix
            known_plaintext: Known plaintext for validation
            
        Returns:
            Validation results
        """
        try:
            # Decrypt with the key
            decrypted = self.hill_cipher.decrypt(ciphertext, key)
            
            # Score the decrypted text
            score = self.statistical_analyzer.score_text(decrypted)
            
            # Check against known plaintext if available
            exact_match = False
            if known_plaintext:
                known_clean = known_plaintext.upper().replace(' ', '').replace('\n', '')
                decrypted_clean = decrypted.upper().replace(' ', '').replace('\n', '').rstrip('X')
                exact_match = (known_clean == decrypted_clean)
            
            return {
                'valid': True,
                'decrypted_text': decrypted,
                'score': score,
                'exact_match': exact_match,
                'key_matrix': key.tolist()
            }
        
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def batch_attack(self, cipher_files: List[str], 
                    known_plaintexts: Dict[str, str] = None,
                    output_dir: str = None) -> Dict:
        """
        Perform batch attack on multiple cipher files.
        
        Args:
            cipher_files: List of cipher file paths
            known_plaintexts: Dictionary mapping file names to known plaintexts
            output_dir: Directory to save results
            
        Returns:
            Batch attack results
        """
        if known_plaintexts is None:
            known_plaintexts = {}
        
        results = {}
        
        for cipher_file in cipher_files:
            logger.info(f"Processing {cipher_file}...")
            
            try:
                # Read ciphertext
                with open(cipher_file, 'r') as f:
                    ciphertext = f.read().strip()
                
                # Get known plaintext if available
                filename = os.path.basename(cipher_file)
                known_plaintext = known_plaintexts.get(filename)
                
                # Attack the cipher
                result = self.break_cipher(
                    ciphertext,
                    known_plaintext=known_plaintext,
                    method='auto'
                )
                
                result['filename'] = filename
                result['ciphertext'] = ciphertext
                results[filename] = result
                
                # Save individual result if output directory specified
                if output_dir and result['success']:
                    os.makedirs(output_dir, exist_ok=True)
                    result_file = os.path.join(output_dir, f"{filename}_result.json")
                    
                    # Convert numpy arrays to lists for JSON serialization
                    json_result = result.copy()
                    if json_result['key'] is not None:
                        json_result['key'] = json_result['key'].tolist()
                    
                    with open(result_file, 'w') as f:
                        json.dump(json_result, f, indent=2)
            
            except Exception as e:
                logger.error(f"Error processing {cipher_file}: {e}")
                results[os.path.basename(cipher_file)] = {
                    'success': False,
                    'error': str(e),
                    'filename': os.path.basename(cipher_file)
                }
        
        return results

def main():
    """Main function for testing the enhanced breaker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Hill Cipher Breaker")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext to analyze")
    parser.add_argument("--ciphertext-file", type=str, help="File containing ciphertext")
    parser.add_argument("--key-size", type=int, default=2, choices=[2, 3, 4, 5], 
                       help="Size of the key matrix")
    parser.add_argument("--known-plaintext", type=str, help="Known plaintext")
    parser.add_argument("--method", type=str, default='auto', 
                       choices=['auto', 'statistical', 'kpa'],
                       help="Attack method")
    parser.add_argument("--max-candidates", type=int, default=10000, 
                       help="Maximum number of candidates for statistical attack")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Timeout in seconds")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get ciphertext
    ciphertext = args.ciphertext
    if args.ciphertext_file:
        with open(args.ciphertext_file, 'r') as f:
            ciphertext = f.read().strip()
    
    if not ciphertext:
        parser.error("Ciphertext must be provided")
    
    # Create breaker
    breaker = EnhancedHillBreaker(args.key_size)
    
    # Attack the cipher
    results = breaker.break_cipher(
        ciphertext,
        known_plaintext=args.known_plaintext,
        method=args.method,
        max_candidates=args.max_candidates,
        timeout=args.timeout
    )
    
    # Output results
    if args.output:
        # Convert numpy arrays to lists for JSON serialization
        json_results = results.copy()
        if json_results['key'] is not None:
            json_results['key'] = json_results['key'].tolist()
        
        with open(args.output, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    # Print summary
    if results['success']:
        print(f"✓ Cipher broken successfully!")
        print(f"Method: {results['method_used']}")
        print(f"Time: {results['time_elapsed']:.2f} seconds")
        print(f"Score: {results['score']:.2f}")
        if results['key'] is not None:
            print(f"Key: {results['key'].flatten()}")
        print(f"Decrypted text: {results['decrypted_text']}")
    else:
        print("✗ Failed to break cipher")
        if 'error' in results:
            print(f"Error: {results['error']}")

if __name__ == "__main__":
    main()
