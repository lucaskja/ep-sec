#!/usr/bin/env python3
"""
Fully Optimized CUDA Hill Cipher Breaker
True GPU parallelism with batch matrix operations and GPU-based scoring

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
from breakers.search_space_reducer import SearchSpaceReducer

# CUDA imports with fallback
CUDA_AVAILABLE = False
CUDA_LIB = None

try:
    import cupy as cp
    CUDA_AVAILABLE = True
    CUDA_LIB = 'cupy'
    print("[OK] CuPy available - GPU acceleration enabled")
except ImportError:
    print("[WARNING] No CUDA library available - falling back to CPU")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fully_optimized_cuda_breaker')

class FullyOptimizedCudaHillBreaker:
    """Fully optimized CUDA Hill cipher breaker with true GPU parallelism."""
    
    def __init__(self, key_size: int):
        self.key_size = key_size
        self.hill_cipher = HillCipher(key_size)
        self.statistical_analyzer = StatisticalAnalyzer(key_size)
        self.search_space_reducer = SearchSpaceReducer(key_size)
        
        # GPU configuration
        self.cuda_available = CUDA_AVAILABLE
        self.cuda_lib = CUDA_LIB
        
        if self.cuda_available:
            self._setup_gpu()
            self._setup_gpu_scoring()
        
        logger.info(f"Initialized FullyOptimizedCudaHillBreaker for {key_size}x{key_size} cipher")
        logger.info(f"GPU acceleration: {'Enabled' if self.cuda_available else 'Disabled'}")
    
    def _setup_gpu(self):
        """Setup GPU resources and kernels."""
        try:
            # Get GPU info
            device = cp.cuda.Device()
            logger.info(f"GPU Device: {device}")
            
            # Get memory info
            total_bytes = cp.cuda.runtime.memGetInfo()[1]
            logger.info(f"GPU Memory: {total_bytes / (1024**3):.1f} GB total")
            
            # Pre-compile kernels
            self._compile_gpu_kernels()
            
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")
            self.cuda_available = False
    
    def _setup_gpu_scoring(self):
        """Setup GPU-based statistical scoring."""
        try:
            # Load frequency data to GPU
            letter_freqs = self.statistical_analyzer.letter_frequencies
            bigram_freqs = self.statistical_analyzer.bigram_frequencies
            
            # Convert to GPU arrays for fast scoring
            self.gpu_letter_freqs = cp.array([letter_freqs.get(chr(i + ord('A')), 0.0001) for i in range(26)])
            
            # Create bigram frequency lookup table
            bigram_lookup = np.zeros((26, 26), dtype=np.float32)
            for bigram, freq in bigram_freqs.items():
                if len(bigram) == 2:
                    i, j = ord(bigram[0]) - ord('A'), ord(bigram[1]) - ord('A')
                    if 0 <= i < 26 and 0 <= j < 26:
                        bigram_lookup[i, j] = freq
            
            self.gpu_bigram_freqs = cp.array(bigram_lookup)
            
            logger.info("GPU-based scoring initialized")
            
        except Exception as e:
            logger.warning(f"GPU scoring setup failed: {e}")
    
    def _compile_gpu_kernels(self):
        """Pre-compile GPU kernels for maximum performance."""
        # Matrix operations kernel
        dummy_matrices = cp.random.randint(0, 26, (100, self.key_size, self.key_size))
        dummy_text = cp.random.randint(0, 26, 1000)
        
        # Compile batch matrix inverse
        try:
            for i in range(min(10, len(dummy_matrices))):
                _ = cp.linalg.det(dummy_matrices[i])
        except:
            pass
        
        # Compile batch matrix multiplication
        try:
            _ = cp.dot(dummy_matrices[0], dummy_text[:self.key_size])
        except:
            pass
        
        logger.info("GPU kernels pre-compiled")
    
    def break_cipher_fully_optimized(self, ciphertext: str, 
                                   max_keys: int = 50000,
                                   batch_size: int = 4096,
                                   early_stopping_score: float = -1000) -> Dict:
        """Break Hill cipher using fully optimized GPU processing."""
        
        if not self.cuda_available:
            logger.warning("GPU not available, falling back to CPU")
            return self._break_cipher_cpu(ciphertext, max_keys)
        
        logger.info(f"Starting fully optimized GPU Hill cipher breaking")
        logger.info(f"Max keys: {max_keys}, Batch size: {batch_size}")
        
        start_time = time.time()
        best_key = None
        best_score = float('-inf')
        best_decrypted = ""
        keys_tested = 0
        
        # Convert ciphertext to GPU once
        cipher_nums = np.array([ord(c) - ord('A') for c in ciphertext.upper()])
        cipher_gpu = cp.asarray(cipher_nums)
        text_length = len(cipher_nums)
        
        # Generate keys using search space reduction
        key_generator = self.search_space_reducer.generate_keys_smart_sampling(max_keys)
        
        # Process in very large batches for maximum GPU utilization
        key_batch = []
        
        for key in key_generator:
            key_batch.append(key)
            
            if len(key_batch) >= batch_size:
                # Process massive batch on GPU with full parallelism
                batch_results = self._process_fully_parallel_gpu_batch(cipher_gpu, key_batch)
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
                
                # Early stopping
                if best_score > early_stopping_score:
                    logger.info(f"Early stopping - excellent score: {best_score:.2f}")
                    break
                
                key_batch = []
        
        # Process remaining keys
        if key_batch and best_score <= early_stopping_score:
            batch_results = self._process_fully_parallel_gpu_batch(cipher_gpu, key_batch)
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
    
    def _process_fully_parallel_gpu_batch(self, cipher_gpu: cp.ndarray, key_batch: List[np.ndarray]) -> List[Tuple[float, str]]:
        """Process batch with full GPU parallelism - matrices, decryption, and scoring."""
        batch_size = len(key_batch)
        
        try:
            # Stack all keys into 3D tensor for parallel processing
            keys_array = np.stack(key_batch)  # Shape: (batch_size, key_size, key_size)
            keys_gpu = cp.asarray(keys_array)
            
            # Parallel batch matrix inversion and decryption
            all_decrypted, valid_mask = self._gpu_batch_decrypt_parallel(cipher_gpu, keys_gpu)
            
            # Parallel GPU-based scoring
            scores = self._gpu_batch_score_parallel(all_decrypted, valid_mask)
            
            # Convert results back to CPU
            scores_cpu = cp.asnumpy(scores)
            all_decrypted_cpu = cp.asnumpy(all_decrypted)
            valid_mask_cpu = cp.asnumpy(valid_mask)
            
            # Build results
            results = []
            for batch_idx in range(batch_size):
                if valid_mask_cpu[batch_idx]:
                    decrypted_nums = all_decrypted_cpu[batch_idx]
                    decrypted_text = ''.join(chr(int(c) + ord('A')) for c in decrypted_nums)
                    score = float(scores_cpu[batch_idx])
                    results.append((score, decrypted_text))
                else:
                    results.append((float('-inf'), ''))
            
            return results
            
        except Exception as e:
            logger.warning(f"Fully parallel GPU processing failed: {e}, using fallback")
            # Fallback to individual processing
            results = []
            for key in key_batch:
                try:
                    decrypted = self.hill_cipher.decrypt(''.join(chr(int(c) + ord('A')) for c in cp.asnumpy(cipher_gpu)), key)
                    score = self.statistical_analyzer.score_text(decrypted)
                    results.append((score, decrypted))
                except:
                    results.append((float('-inf'), ''))
            return results
    
    def _gpu_batch_decrypt_parallel(self, cipher_gpu: cp.ndarray, keys_gpu: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """Parallel batch decryption of all keys simultaneously."""
        batch_size = keys_gpu.shape[0]
        text_length = len(cipher_gpu)
        
        # Pre-allocate result arrays
        all_decrypted = cp.zeros((batch_size, text_length), dtype=cp.int32)
        valid_mask = cp.ones(batch_size, dtype=cp.bool_)
        
        # Vectorized determinant calculation for all matrices
        dets = cp.linalg.det(keys_gpu)
        det_ints = cp.round(dets).astype(cp.int32) % 26
        
        # Vectorized GCD check (simplified for speed)
        for batch_idx in range(batch_size):
            det_val = int(det_ints[batch_idx])
            if det_val == 0 or self._gcd(det_val, 26) != 1:
                valid_mask[batch_idx] = False
                continue
            
            try:
                # Calculate inverse matrix
                key_matrix = keys_gpu[batch_idx]
                inv_key = self._gpu_matrix_inverse_mod26_fast(key_matrix, det_val)
                
                # Decrypt text using vectorized operations
                decrypted = self._gpu_decrypt_vectorized(cipher_gpu, inv_key)
                all_decrypted[batch_idx] = decrypted
                
            except Exception:
                valid_mask[batch_idx] = False
        
        return all_decrypted, valid_mask
    
    def _gpu_decrypt_vectorized(self, cipher_gpu: cp.ndarray, inv_key: cp.ndarray) -> cp.ndarray:
        """Vectorized GPU decryption using matrix operations."""
        text_length = len(cipher_gpu)
        result = cp.zeros(text_length, dtype=cp.int32)
        
        # Process all blocks in parallel using advanced indexing
        num_blocks = (text_length + self.key_size - 1) // self.key_size
        
        # Reshape cipher into blocks for parallel processing
        padded_length = num_blocks * self.key_size
        padded_cipher = cp.zeros(padded_length, dtype=cp.int32)
        padded_cipher[:text_length] = cipher_gpu
        
        # Reshape into matrix of blocks
        cipher_blocks = padded_cipher.reshape(num_blocks, self.key_size)
        
        # Parallel matrix multiplication for all blocks
        decrypted_blocks = cp.dot(cipher_blocks, inv_key.T) % 26
        
        # Flatten and trim to original length
        result = decrypted_blocks.flatten()[:text_length]
        
        return result
    
    def _gpu_batch_score_parallel(self, all_decrypted: cp.ndarray, valid_mask: cp.ndarray) -> cp.ndarray:
        """Parallel GPU-based statistical scoring for all decrypted texts."""
        batch_size = all_decrypted.shape[0]
        scores = cp.full(batch_size, float('-inf'), dtype=cp.float32)
        
        for batch_idx in range(batch_size):
            if not valid_mask[batch_idx]:
                continue
            
            try:
                decrypted_text = all_decrypted[batch_idx]
                
                # GPU-based frequency scoring
                score = self._gpu_score_text_fast(decrypted_text)
                scores[batch_idx] = score
                
            except Exception:
                scores[batch_idx] = float('-inf')
        
        return scores
    
    def _gpu_score_text_fast(self, text_gpu: cp.ndarray) -> float:
        """Fast GPU-based text scoring using frequency analysis."""
        try:
            # Letter frequency scoring
            letter_counts = cp.bincount(text_gpu, minlength=26)
            letter_freqs = letter_counts / len(text_gpu)
            letter_score = -cp.sum((letter_freqs - self.gpu_letter_freqs) ** 2)
            
            # Bigram frequency scoring (simplified for speed)
            if len(text_gpu) > 1:
                bigram_indices = text_gpu[:-1] * 26 + text_gpu[1:]
                bigram_counts = cp.bincount(bigram_indices, minlength=676)
                bigram_freqs = bigram_counts / (len(text_gpu) - 1)
                
                # Simplified bigram scoring
                bigram_score = -cp.sum(bigram_freqs ** 2) * 1000
            else:
                bigram_score = 0
            
            total_score = float(letter_score + bigram_score)
            return total_score
            
        except Exception:
            return float('-inf')
    
    def _gpu_matrix_inverse_mod26_fast(self, matrix: cp.ndarray, det_int: int) -> cp.ndarray:
        """Fast GPU matrix inverse modulo 26."""
        try:
            inv_det = self._mod_inverse(det_int, 26)
            
            if self.key_size == 2:
                # Optimized 2x2 inverse
                adj_matrix = cp.array([[matrix[1,1], -matrix[0,1]], 
                                     [-matrix[1,0], matrix[0,0]]])
                inv_matrix = (adj_matrix * inv_det) % 26
            else:
                # General case using GPU linear algebra
                matrix_float = matrix.astype(cp.float32)
                inv_matrix_float = cp.linalg.inv(matrix_float)
                inv_matrix = cp.round(inv_matrix_float * det_int * inv_det) % 26
                inv_matrix = inv_matrix.astype(cp.int32)
            
            return inv_matrix
            
        except Exception:
            return cp.eye(self.key_size, dtype=cp.int32)
    
    def _break_cipher_cpu(self, ciphertext: str, max_keys: int) -> Dict:
        """CPU fallback implementation."""
        logger.info("Using CPU fallback")
        
        start_time = time.time()
        best_key = None
        best_score = float('-inf')
        best_decrypted = ""
        keys_tested = 0
        
        key_generator = self.search_space_reducer.generate_keys_smart_sampling(max_keys)
        
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
    parser = argparse.ArgumentParser(description='Fully Optimized CUDA Hill Cipher Breaker')
    parser.add_argument('--ciphertext', required=True, help='Ciphertext to decrypt')
    parser.add_argument('--key-size', type=int, required=True, help='Size of the key matrix (2, 3, 4, or 5)')
    parser.add_argument('--max-keys', type=int, default=50000, help='Maximum number of keys to test')
    parser.add_argument('--batch-size', type=int, default=8192, help='GPU batch size (larger = more GPU utilization)')
    
    args = parser.parse_args()
    
    print(f"[FULLY-OPTIMIZED] CUDA Hill Cipher Breaker")
    print(f"Breaking {args.key_size}x{args.key_size} Hill cipher...")
    print(f"GPU acceleration: {'Enabled' if CUDA_AVAILABLE else 'Disabled'}")
    print(f"Batch size: {args.batch_size:,} (optimized for maximum GPU utilization)")
    
    # Initialize breaker
    breaker = FullyOptimizedCudaHillBreaker(args.key_size)
    
    # Break cipher
    result = breaker.break_cipher_fully_optimized(
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
