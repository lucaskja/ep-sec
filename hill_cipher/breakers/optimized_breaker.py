#!/usr/bin/env python3
"""
Optimized Hill Cipher Breaker with Search Space Reduction

This module implements an optimized Hill cipher breaker that uses
multiple search space reduction techniques and parallel processing.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import numpy as np
import logging
import time
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hill_cipher import HillCipher
from breakers.statistical_analyzer import StatisticalAnalyzer
from breakers.search_space_reducer import SearchSpaceReducer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('optimized_breaker')

class OptimizedHillBreaker:
    """
    Optimized Hill cipher breaker with advanced search space reduction.
    """
    
    def __init__(self, key_size: int, data_dir: str = None):
        """
        Initialize the optimized breaker.
        
        Args:
            key_size: Size of the Hill cipher key matrix
            data_dir: Directory containing frequency data files
        """
        self.key_size = key_size
        self.hill_cipher = HillCipher(key_size)
        self.statistical_analyzer = StatisticalAnalyzer(key_size, data_dir)
        self.search_reducer = SearchSpaceReducer(key_size)
        
        # Performance tracking
        self.keys_tested = 0
        self.start_time = None
        
        logger.info(f"Initialized OptimizedHillBreaker for {key_size}x{key_size} cipher")
    
    def break_cipher_optimized(self, ciphertext: str, 
                              max_time: int = 1800,
                              max_keys_per_technique: int = 50000,
                              early_stopping_score: float = -1000,
                              use_parallel: bool = True,
                              num_processes: int = None) -> Dict:
        """
        Break Hill cipher using optimized techniques.
        
        Args:
            ciphertext: Encrypted text
            max_time: Maximum time in seconds
            max_keys_per_technique: Maximum keys to test per technique
            early_stopping_score: Stop if score exceeds this threshold
            use_parallel: Whether to use parallel processing
            num_processes: Number of processes (None for auto)
            
        Returns:
            Dictionary with results
        """
        self.start_time = time.time()
        self.keys_tested = 0
        
        logger.info(f"Starting optimized Hill cipher breaking")
        logger.info(f"Ciphertext length: {len(ciphertext)}")
        logger.info(f"Max time: {max_time}s, Early stopping: {early_stopping_score}")
        
        results = {
            'success': False,
            'key': None,
            'decrypted_text': None,
            'score': float('-inf'),
            'technique_used': None,
            'keys_tested': 0,
            'time_elapsed': 0,
            'techniques_tried': []
        }
        
        # Define techniques in order of efficiency
        techniques = [
            ('diagonal', 'Diagonal matrices'),
            ('determinant_constraint', 'Determinant-constrained matrices'),
            ('frequency_based', 'Frequency-based generation'),
            ('upper_triangular', 'Upper triangular matrices'),
            ('lower_triangular', 'Lower triangular matrices'),
            ('sparse', 'Sparse matrices'),
            ('smart_sampling', 'Smart sampling')
        ]
        
        best_result = None
        
        for technique_name, technique_desc in techniques:
            if time.time() - self.start_time > max_time:
                logger.info("Time limit reached")
                break
            
            logger.info(f"Trying technique: {technique_desc}")
            technique_start = time.time()
            
            try:
                if use_parallel and num_processes != 1:
                    result = self._break_with_technique_parallel(
                        ciphertext, technique_name, max_keys_per_technique,
                        early_stopping_score, num_processes
                    )
                else:
                    result = self._break_with_technique_sequential(
                        ciphertext, technique_name, max_keys_per_technique,
                        early_stopping_score
                    )
                
                technique_time = time.time() - technique_start
                result['technique_time'] = technique_time
                result['technique_name'] = technique_name
                
                results['techniques_tried'].append(result)
                
                logger.info(f"Technique {technique_name} completed in {technique_time:.1f}s")
                logger.info(f"Keys tested: {result['keys_tested']}, Best score: {result['score']:.2f}")
                
                if result['success'] or (best_result is None or result['score'] > best_result['score']):
                    best_result = result
                
                # Early stopping if we found a very good solution
                if result['score'] > early_stopping_score:
                    logger.info(f"Early stopping - excellent score achieved: {result['score']:.2f}")
                    break
                
            except Exception as e:
                logger.error(f"Error in technique {technique_name}: {e}")
                continue
        
        # Update final results
        if best_result:
            results.update({
                'success': best_result['score'] > early_stopping_score,
                'key': best_result['key'],
                'decrypted_text': best_result['decrypted_text'],
                'score': best_result['score'],
                'technique_used': best_result.get('technique_name', 'unknown')
            })
        
        results['keys_tested'] = self.keys_tested
        results['time_elapsed'] = time.time() - self.start_time
        
        logger.info(f"Optimized breaking completed in {results['time_elapsed']:.1f}s")
        logger.info(f"Total keys tested: {results['keys_tested']}")
        logger.info(f"Best score: {results['score']:.2f}")
        
        return results
    
    def _break_with_technique_sequential(self, ciphertext: str, technique: str, 
                                       max_keys: int, early_stopping_score: float) -> Dict:
        """Break cipher using a single technique sequentially."""
        result = {
            'success': False,
            'key': None,
            'decrypted_text': None,
            'score': float('-inf'),
            'keys_tested': 0
        }
        
        # Get key generator for the technique
        key_generator = self._get_key_generator(technique, ciphertext, max_keys)
        
        keys_tested_this_technique = 0
        
        for key in key_generator:
            if time.time() - self.start_time > 1800:  # Global time limit
                break
            
            if keys_tested_this_technique >= max_keys:
                break
            
            try:
                decrypted = self.hill_cipher.decrypt(ciphertext, key)
                score = self.statistical_analyzer.score_text(decrypted)
                
                keys_tested_this_technique += 1
                self.keys_tested += 1
                
                if score > result['score']:
                    result.update({
                        'key': key.copy(),
                        'decrypted_text': decrypted,
                        'score': score
                    })
                    
                    if score > early_stopping_score:
                        result['success'] = True
                        break
                
                # Progress logging
                if keys_tested_this_technique % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.keys_tested / elapsed
                    logger.info(f"  {technique}: {keys_tested_this_technique} keys, "
                              f"rate: {rate:.1f} keys/sec, best: {result['score']:.2f}")
            
            except Exception as e:
                continue
        
        result['keys_tested'] = keys_tested_this_technique
        return result
    
    def _break_with_technique_parallel(self, ciphertext: str, technique: str, 
                                     max_keys: int, early_stopping_score: float,
                                     num_processes: int = None) -> Dict:
        """Break cipher using a single technique with parallel processing."""
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes
        
        logger.info(f"Using {num_processes} processes for parallel search")
        
        # Generate keys in batches
        batch_size = max(100, max_keys // (num_processes * 10))
        key_generator = self._get_key_generator(technique, ciphertext, max_keys)
        
        result = {
            'success': False,
            'key': None,
            'decrypted_text': None,
            'score': float('-inf'),
            'keys_tested': 0
        }
        
        keys_tested_this_technique = 0
        
        try:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit batches of work
                futures = []
                current_batch = []
                
                for key in key_generator:
                    if time.time() - self.start_time > 1800:  # Global time limit
                        break
                    
                    if keys_tested_this_technique >= max_keys:
                        break
                    
                    current_batch.append(key.copy())
                    keys_tested_this_technique += 1
                    
                    if len(current_batch) >= batch_size:
                        future = executor.submit(
                            self._process_key_batch, 
                            current_batch, ciphertext
                        )
                        futures.append(future)
                        current_batch = []
                
                # Submit remaining keys
                if current_batch:
                    future = executor.submit(
                        self._process_key_batch, 
                        current_batch, ciphertext
                    )
                    futures.append(future)
                
                # Process results
                for future in as_completed(futures):
                    try:
                        batch_result = future.result(timeout=60)  # 1 minute timeout per batch
                        
                        self.keys_tested += batch_result['keys_processed']
                        
                        if batch_result['best_score'] > result['score']:
                            result.update({
                                'key': batch_result['best_key'],
                                'decrypted_text': batch_result['best_decrypted'],
                                'score': batch_result['best_score']
                            })
                            
                            if batch_result['best_score'] > early_stopping_score:
                                result['success'] = True
                                # Cancel remaining futures
                                for f in futures:
                                    f.cancel()
                                break
                    
                    except Exception as e:
                        logger.warning(f"Batch processing error: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Parallel processing error: {e}")
            # Fall back to sequential processing
            return self._break_with_technique_sequential(
                ciphertext, technique, max_keys, early_stopping_score
            )
        
        result['keys_tested'] = keys_tested_this_technique
        return result
    
    def _process_key_batch(self, keys: List[np.ndarray], ciphertext: str) -> Dict:
        """Process a batch of keys (for parallel processing)."""
        # Create new instances for this process
        hill_cipher = HillCipher(self.key_size)
        statistical_analyzer = StatisticalAnalyzer(self.key_size)
        
        best_key = None
        best_decrypted = None
        best_score = float('-inf')
        keys_processed = 0
        
        for key in keys:
            try:
                decrypted = hill_cipher.decrypt(ciphertext, key)
                score = statistical_analyzer.score_text(decrypted)
                
                keys_processed += 1
                
                if score > best_score:
                    best_score = score
                    best_key = key.copy()
                    best_decrypted = decrypted
            
            except Exception as e:
                continue
        
        return {
            'keys_processed': keys_processed,
            'best_key': best_key,
            'best_decrypted': best_decrypted,
            'best_score': best_score
        }
    
    def _get_key_generator(self, technique: str, ciphertext: str, max_keys: int) -> Iterator[np.ndarray]:
        """Get key generator for the specified technique."""
        if technique == 'diagonal':
            return self.search_reducer.generate_keys_with_structure_constraints()
        elif technique == 'determinant_constraint':
            return self.search_reducer.generate_keys_by_determinant_constraint()
        elif technique == 'frequency_based':
            return self.search_reducer.generate_keys_frequency_based(ciphertext, max_keys)
        elif technique == 'upper_triangular':
            return self.search_reducer._generate_upper_triangular()
        elif technique == 'lower_triangular':
            return self.search_reducer._generate_lower_triangular()
        elif technique == 'sparse':
            return self.search_reducer._generate_sparse_matrices()
        elif technique == 'smart_sampling':
            return self.search_reducer.generate_keys_smart_sampling(max_keys)
        else:
            raise ValueError(f"Unknown technique: {technique}")
    
    def validate_with_normalized_text(self, decrypted_text: str, normalized_text: str) -> Tuple[bool, Dict]:
        """
        Validate decrypted text against normalized text.
        
        Args:
            decrypted_text: The decrypted text
            normalized_text: The normalized reference text
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        clean_decrypted = decrypted_text.upper().replace(' ', '').replace('\n', '').rstrip('X')
        
        # Exact match
        if clean_decrypted in normalized_text:
            pos = normalized_text.find(clean_decrypted)
            return True, {
                'match_type': 'exact',
                'position': pos,
                'length': len(clean_decrypted),
                'percentage': 100.0
            }
        
        # Partial match
        best_match_length = 0
        best_position = -1
        
        # Check substrings of decreasing length
        min_length = max(50, len(clean_decrypted) // 2)
        for length in range(len(clean_decrypted), min_length - 1, -5):
            for start in range(len(clean_decrypted) - length + 1):
                substring = clean_decrypted[start:start + length]
                pos = normalized_text.find(substring)
                if pos != -1 and length > best_match_length:
                    best_match_length = length
                    best_position = pos
                    break
            if best_match_length > 0:
                break
        
        if best_match_length >= min_length:
            percentage = (best_match_length / len(clean_decrypted)) * 100
            return True, {
                'match_type': 'partial',
                'position': best_position,
                'length': best_match_length,
                'percentage': percentage
            }
        
        return False, {
            'match_type': 'none',
            'position': -1,
            'length': 0,
            'percentage': 0.0
        }

def main():
    """Main function for testing the optimized breaker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Hill Cipher Breaker")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext to analyze")
    parser.add_argument("--ciphertext-file", type=str, help="File containing ciphertext")
    parser.add_argument("--key-size", type=int, default=3, choices=[2, 3, 4, 5], 
                       help="Size of the key matrix")
    parser.add_argument("--max-time", type=int, default=1800, 
                       help="Maximum time in seconds")
    parser.add_argument("--max-keys", type=int, default=50000, 
                       help="Maximum keys per technique")
    parser.add_argument("--early-stopping", type=float, default=-1000, 
                       help="Early stopping score threshold")
    parser.add_argument("--no-parallel", action="store_true", 
                       help="Disable parallel processing")
    parser.add_argument("--processes", type=int, 
                       help="Number of processes for parallel execution")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate against normalized text")
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
    breaker = OptimizedHillBreaker(args.key_size)
    
    # Break the cipher
    results = breaker.break_cipher_optimized(
        ciphertext,
        max_time=args.max_time,
        max_keys_per_technique=args.max_keys,
        early_stopping_score=args.early_stopping,
        use_parallel=not args.no_parallel,
        num_processes=args.processes
    )
    
    # Validate if requested
    if args.validate and results['success']:
        normalized_path = os.path.join('hill_cipher', 'data', 'normalized_text.txt')
        try:
            with open(normalized_path, 'r') as f:
                normalized_text = f.read().upper().replace(' ', '').replace('\n', '')
            
            is_valid, validation_info = breaker.validate_with_normalized_text(
                results['decrypted_text'], normalized_text
            )
            
            results['validation'] = {
                'is_valid': is_valid,
                'info': validation_info
            }
            
            if is_valid:
                print(f"✓ Validation successful: {validation_info['match_type']} match")
                print(f"  Position: {validation_info['position']}")
                print(f"  Length: {validation_info['length']}")
                print(f"  Percentage: {validation_info['percentage']:.1f}%")
        
        except Exception as e:
            print(f"Validation error: {e}")
    
    # Output results
    if args.output:
        # Convert numpy arrays to lists for JSON serialization
        json_results = results.copy()
        if json_results['key'] is not None:
            json_results['key'] = json_results['key'].tolist()
        
        # Convert technique results
        for technique_result in json_results['techniques_tried']:
            if technique_result['key'] is not None:
                technique_result['key'] = technique_result['key'].tolist()
        
        with open(args.output, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    # Print summary
    print(f"\nOptimized Hill Cipher Breaking Results")
    print(f"=" * 50)
    print(f"Success: {results['success']}")
    print(f"Time elapsed: {results['time_elapsed']:.1f} seconds")
    print(f"Keys tested: {results['keys_tested']:,}")
    print(f"Best score: {results['score']:.2f}")
    
    if results['key'] is not None:
        print(f"Best key: {results['key'].flatten()}")
        print(f"Technique used: {results['technique_used']}")
        print(f"Decrypted text: {results['decrypted_text'][:100]}...")
    
    print(f"\nTechniques tried: {len(results['techniques_tried'])}")
    for i, technique in enumerate(results['techniques_tried']):
        print(f"  {i+1}. {technique.get('technique_name', 'unknown')}: "
              f"{technique['keys_tested']} keys, score {technique['score']:.2f}")

if __name__ == "__main__":
    main()
