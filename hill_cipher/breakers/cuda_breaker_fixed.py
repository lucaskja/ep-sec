#!/usr/bin/env python3
"""
Fixed CUDA Hill Cipher Breaker
Focuses on correctness and reasonable GPU utilization

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hill_cipher import HillCipher
from breakers.statistical_analyzer import StatisticalAnalyzer

# CUDA imports with fallback
CUDA_AVAILABLE = False

try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("[OK] CuPy available - GPU acceleration enabled")
except ImportError:
    print("[WARNING] No CUDA library available - falling back to CPU")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cuda_breaker_fixed')

class FixedCudaHillBreaker:
    """Fixed CUDA Hill cipher breaker focusing on correctness."""
    
    def __init__(self, key_size: int):
        self.key_size = key_size
        self.hill_cipher = HillCipher(key_size)
        self.statistical_analyzer = StatisticalAnalyzer(key_size)
        self.cuda_available = CUDA_AVAILABLE
        
        if self.cuda_available:
            self._setup_gpu()
        
        logger.info(f"Initialized FixedCudaHillBreaker for {key_size}x{key_size} cipher")
        logger.info(f"GPU acceleration: {'Enabled' if self.cuda_available else 'Disabled'}")
    
    def _setup_gpu(self):
        """Setup GPU resources."""
        try:
            device = cp.cuda.Device()
            total_bytes = cp.cuda.runtime.memGetInfo()[1]
            logger.info(f"GPU Device: {device}")
            logger.info(f"GPU Memory: {total_bytes / (1024**3):.1f} GB total")
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")
            self.cuda_available = False
    
    def break_cipher(self, ciphertext: str, 
                    max_keys: int = 50000,
                    batch_size: int = 1024) -> Dict:
        """Break Hill cipher with focus on finding correct keys."""
        
        if not self.cuda_available:
            return self._break_cipher_cpu(ciphertext, max_keys)
        
        logger.info(f"Starting CUDA Hill cipher breaking")
        logger.info(f"Max keys: {max_keys}, Batch size: {batch_size}")
        
        start_time = time.time()
        best_key = None
        best_score = float('-inf')
        best_decrypted = ""
        keys_tested = 0
        
        # Generate all possible keys systematically for small matrices
        if self.key_size == 2:
            key_generator = self._generate_2x2_keys_systematic()
        else:
            key_generator = self._generate_keys_systematic(max_keys)
        
        # Process in reasonable batches
        key_batch = []
        
        for key in key_generator:
            key_batch.append(key)
            
            if len(key_batch) >= batch_size:
                # Process batch
                batch_results = self._process_gpu_batch(ciphertext, key_batch)
                keys_tested += len(key_batch)
                
                # Find best in batch
                for i, (score, decrypted) in enumerate(batch_results):
                    if score > best_score:
                        best_score = score
                        best_key = key_batch[i]
                        best_decrypted = decrypted
                
                # Log progress
                elapsed = time.time() - start_time
                rate = keys_tested / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {keys_tested} keys ({rate:.1f} keys/sec), best score: {best_score:.2f}")
                
                # Early stopping for very good scores
                if best_score > -50:
                    logger.info(f"Early stopping - excellent score: {best_score:.2f}")
                    break
                
                key_batch = []
                
                # Stop if we've tested enough keys
                if keys_tested >= max_keys:
                    break
        
        # Process remaining keys
        if key_batch:
            batch_results = self._process_gpu_batch(ciphertext, key_batch)
            keys_tested += len(key_batch)
            
            for i, (score, decrypted) in enumerate(batch_results):
                if score > best_score:
                    best_score = score
                    best_key = key_batch[i]
                    best_decrypted = decrypted
        
        elapsed_time = time.time() - start_time
        
        return {
            'success': best_key is not None,
            'key': best_key.tolist() if best_key is not None else None,
            'score': best_score,
            'decrypted_text': best_decrypted,
            'keys_tested': keys_tested,
            'time_elapsed': elapsed_time,
            'keys_per_second': keys_tested / elapsed_time if elapsed_time > 0 else 0
        }
    
    def _generate_2x2_keys_systematic(self):
        """Generate 2x2 keys systematically to ensure we find the correct one."""
        logger.info("Generating 2x2 keys systematically")
        
        # Generate keys in a systematic way
        for a in range(26):
            for b in range(26):
                for c in range(26):
                    for d in range(26):
                        key = np.array([[a, b], [c, d]], dtype=np.int32)
                        
                        # Check if key is valid (invertible)
                        det = (a * d - b * c) % 26
                        if det != 0 and self._gcd(det, 26) == 1:
                            yield key
    
    def _generate_keys_systematic(self, max_keys: int):
        """Generate keys systematically for larger matrices."""
        logger.info(f"Generating up to {max_keys} keys systematically")
        
        generated = 0
        # For larger matrices, use random generation with validity checking
        while generated < max_keys:
            if self.key_size == 3:
                key = np.random.randint(0, 26, (3, 3), dtype=np.int32)
            elif self.key_size == 4:
                key = np.random.randint(0, 26, (4, 4), dtype=np.int32)
            elif self.key_size == 5:
                key = np.random.randint(0, 26, (5, 5), dtype=np.int32)
            else:
                key = np.random.randint(0, 26, (self.key_size, self.key_size), dtype=np.int32)
            
            # Check if key is valid
            try:
                det = int(np.round(np.linalg.det(key))) % 26
                if det != 0 and self._gcd(det, 26) == 1:
                    yield key
                    generated += 1
            except:
                continue
    
    def _process_gpu_batch(self, ciphertext: str, key_batch: List[np.ndarray]) -> List[Tuple[float, str]]:
        """Process batch using GPU for matrix operations, CPU for scoring."""
        if not self.cuda_available:
            return self._process_cpu_batch(ciphertext, key_batch)
        
        results = []
        
        # Convert ciphertext to numbers once
        cipher_nums = np.array([ord(c) - ord('A') for c in ciphertext.upper()])
        cipher_gpu = cp.asarray(cipher_nums)
        
        for key in key_batch:
            try:
                # Use GPU for matrix operations
                key_gpu = cp.asarray(key)
                
                # Decrypt on GPU
                decrypted_gpu = self._decrypt_gpu(cipher_gpu, key_gpu)
                
                # Convert back to CPU for scoring
                decrypted_cpu = cp.asnumpy(decrypted_gpu)
                decrypted_text = ''.join(chr(int(c) + ord('A')) for c in decrypted_cpu)
                
                # Score on CPU for accuracy
                score = self.statistical_analyzer.score_text(decrypted_text)
                results.append((score, decrypted_text))
                
            except Exception as e:
                results.append((float('-inf'), ''))
        
        return results
    
    def _decrypt_gpu(self, cipher_gpu: cp.ndarray, key_matrix: cp.ndarray) -> cp.ndarray:
        """Decrypt text using GPU matrix operations."""
        try:
            # Calculate modular inverse of key matrix
            det = cp.linalg.det(key_matrix)
            det_int = int(cp.round(det)) % 26
            
            if det_int == 0 or self._gcd(det_int, 26) != 1:
                raise ValueError("Key not invertible")
            
            # Calculate inverse matrix
            inv_key = self._gpu_matrix_inverse_mod26(key_matrix, det_int)
            
            # Decrypt in blocks
            text_length = len(cipher_gpu)
            result = cp.zeros(text_length, dtype=cp.int32)
            
            for i in range(0, text_length, self.key_size):
                block_end = min(i + self.key_size, text_length)
                block = cipher_gpu[i:block_end]
                
                if len(block) == self.key_size:
                    decrypted_block = cp.dot(inv_key, block) % 26
                    result[i:block_end] = decrypted_block
                else:
                    # Handle partial block
                    padded_block = cp.zeros(self.key_size, dtype=cp.int32)
                    padded_block[:len(block)] = block
                    decrypted_block = cp.dot(inv_key, padded_block) % 26
                    result[i:block_end] = decrypted_block[:len(block)]
            
            return result
            
        except Exception:
            # Return original if decryption fails
            return cipher_gpu
    
    def _gpu_matrix_inverse_mod26(self, matrix: cp.ndarray, det_int: int) -> cp.ndarray:
        """GPU matrix inverse modulo 26."""
        try:
            inv_det = self._mod_inverse(det_int, 26)
            
            if self.key_size == 2:
                # Optimized 2x2 inverse
                adj_matrix = cp.array([[matrix[1,1], -matrix[0,1]], 
                                     [-matrix[1,0], matrix[0,0]]])
                inv_matrix = (adj_matrix * inv_det) % 26
            else:
                # General case - use CPU for reliability
                matrix_cpu = cp.asnumpy(matrix)
                inv_cpu = self.hill_cipher._matrix_inverse_mod26(matrix_cpu)
                inv_matrix = cp.asarray(inv_cpu)
            
            return inv_matrix
            
        except Exception:
            return cp.eye(self.key_size, dtype=cp.int32)
    
    def _process_cpu_batch(self, ciphertext: str, key_batch: List[np.ndarray]) -> List[Tuple[float, str]]:
        """CPU batch processing fallback."""
        results = []
        
        for key in key_batch:
            try:
                decrypted = self.hill_cipher.decrypt(ciphertext, key)
                score = self.statistical_analyzer.score_text(decrypted)
                results.append((score, decrypted))
            except:
                results.append((float('-inf'), ''))
        
        return results
    
    def _break_cipher_cpu(self, ciphertext: str, max_keys: int) -> Dict:
        """CPU fallback implementation."""
        logger.info("Using CPU fallback")
        
        start_time = time.time()
        best_key = None
        best_score = float('-inf')
        best_decrypted = ""
        keys_tested = 0
        
        if self.key_size == 2:
            key_generator = self._generate_2x2_keys_systematic()
        else:
            key_generator = self._generate_keys_systematic(max_keys)
        
        for key in key_generator:
            try:
                decrypted = self.hill_cipher.decrypt(ciphertext, key)
                score = self.statistical_analyzer.score_text(decrypted)
                
                if score > best_score:
                    best_score = score
                    best_key = key
                    best_decrypted = decrypted
                
                keys_tested += 1
                
                if keys_tested % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = keys_tested / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {keys_tested} keys ({rate:.1f} keys/sec), best score: {best_score:.2f}")
                
                # Early stopping
                if best_score > -50:
                    logger.info(f"Early stopping - excellent score: {best_score:.2f}")
                    break
                
                if keys_tested >= max_keys:
                    break
                
            except Exception:
                keys_tested += 1
                continue
        
        elapsed_time = time.time() - start_time
        
        return {
            'success': best_key is not None,
            'key': best_key.tolist() if best_key is not None else None,
            'score': best_score,
            'decrypted_text': best_decrypted,
            'keys_tested': keys_tested,
            'time_elapsed': elapsed_time,
            'keys_per_second': keys_tested / elapsed_time if elapsed_time > 0 else 0
        }
    
    def _gcd(self, a: int, b: int) -> int:
        """Calculate GCD."""
        while b:
            a, b = b, a % b
        return a
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Calculate modular inverse."""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a % m, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Fixed CUDA Hill Cipher Breaker')
    parser.add_argument('--ciphertext', required=True, help='Ciphertext to decrypt')
    parser.add_argument('--key-size', type=int, required=True, help='Size of the key matrix (2, 3, 4, or 5)')
    parser.add_argument('--max-keys', type=int, default=50000, help='Maximum number of keys to test')
    parser.add_argument('--batch-size', type=int, default=1024, help='GPU batch size')
    
    args = parser.parse_args()
    
    print(f"[FIXED] CUDA Hill Cipher Breaker")
    print(f"Breaking {args.key_size}x{args.key_size} Hill cipher...")
    print(f"GPU acceleration: {'Enabled' if CUDA_AVAILABLE else 'Disabled'}")
    print(f"Focus: Correctness and systematic key generation")
    
    # Initialize breaker
    breaker = FixedCudaHillBreaker(args.key_size)
    
    # Break cipher
    result = breaker.break_cipher(
        args.ciphertext,
        max_keys=args.max_keys,
        batch_size=args.batch_size
    )
    
    # Display results
    print("\nResults:")
    print(f"Success: {result['success']}")
    print(f"Keys tested: {result['keys_tested']:,}")
    print(f"Time elapsed: {result['time_elapsed']:.2f}s")
    print(f"Keys per second: {result['keys_per_second']:.1f}")
    
    if result['success']:
        print(f"Best key: {result['key']}")
        print(f"Best score: {result['score']:.2f}")
        print(f"Decrypted text: {result['decrypted_text'][:200]}...")

if __name__ == "__main__":
    main()
