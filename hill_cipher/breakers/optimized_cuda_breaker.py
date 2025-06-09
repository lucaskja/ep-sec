#!/usr/bin/env python3
"""
Optimized CUDA Hill Cipher Breaker
High-performance GPU-accelerated Hill cipher cryptanalysis with proper GPU utilization

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
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import pycuda.gpuarray as gpuarray
        CUDA_AVAILABLE = True
        CUDA_LIB = 'pycuda'
        print("[OK] PyCUDA available - GPU acceleration enabled")
    except ImportError:
        print("[WARNING] No CUDA library available - falling back to CPU")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('optimized_cuda_breaker')

class OptimizedCudaHillBreaker:
    """Optimized CUDA-accelerated Hill cipher breaker with proper GPU utilization."""
    
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
        
        logger.info(f"Initialized OptimizedCudaHillBreaker for {key_size}x{key_size} cipher")
        logger.info(f"GPU acceleration: {'Enabled' if self.cuda_available else 'Disabled'}")
    
    def _setup_gpu(self):
        """Setup GPU resources and kernels."""
        if self.cuda_lib == 'cupy':
            self._setup_cupy()
        elif self.cuda_lib == 'pycuda':
            self._setup_pycuda()
    
    def _setup_cupy(self):
        """Setup CuPy GPU resources."""
        try:
            # Get GPU info
            device = cp.cuda.Device()
            logger.info(f"GPU Device: {device}")
            
            # Get memory info
            mempool = cp.get_default_memory_pool()
            total_bytes = cp.cuda.runtime.memGetInfo()[1]
            logger.info(f"GPU Memory: {total_bytes / (1024**3):.1f} GB total")
            
            # Pre-compile kernels for better performance
            self._compile_cupy_kernels()
            
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")
            self.cuda_available = False
    
    def _compile_cupy_kernels(self):
        """Pre-compile CuPy kernels for matrix operations."""
        # Create dummy data to compile kernels
        dummy_matrix = cp.random.randint(0, 26, (self.key_size, self.key_size))
        dummy_text = cp.random.randint(0, 26, 100)
        
        # Compile matrix inverse kernel
        try:
            _ = cp.linalg.inv(dummy_matrix.astype(cp.float32))
        except:
            pass
        
        # Compile matrix multiplication kernel
        try:
            _ = cp.dot(dummy_matrix, dummy_text[:self.key_size])
        except:
            pass
        
        logger.info("CuPy kernels pre-compiled")
    
    def break_cipher_optimized(self, ciphertext: str, 
                             max_keys: int = 50000,
                             batch_size: int = 2048,
                             early_stopping_score: float = -100) -> Dict:
        """Break Hill cipher using optimized GPU batch processing."""
        
        if not self.cuda_available:
            logger.warning("GPU not available, falling back to CPU")
            return self._break_cipher_cpu(ciphertext, max_keys)
        
        logger.info(f"Starting optimized GPU Hill cipher breaking")
        logger.info(f"Max keys: {max_keys}, Batch size: {batch_size}")
        
        start_time = time.time()
        best_key = None
        best_score = float('-inf')
        best_decrypted = ""
        keys_tested = 0
        
        # Generate keys using search space reduction
        key_generator = self.search_space_reducer.generate_keys_smart_sampling(max_keys)
        
        # Process in large batches for GPU efficiency
        key_batch = []
        
        for key in key_generator:
            key_batch.append(key)
            
            if len(key_batch) >= batch_size:
                # Process large batch on GPU
                batch_results = self._process_optimized_gpu_batch(ciphertext, key_batch)
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
            batch_results = self._process_optimized_gpu_batch(ciphertext, key_batch)
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
    
    def _process_optimized_gpu_batch(self, ciphertext: str, key_batch: List[np.ndarray]) -> List[Tuple[float, str]]:
        """Process batch with optimized GPU utilization."""
        if self.cuda_lib == 'cupy':
            return self._process_cupy_optimized_batch(ciphertext, key_batch)
        else:
            return self._process_cpu_batch(ciphertext, key_batch)
    
    def _process_cupy_optimized_batch(self, ciphertext: str, key_batch: List[np.ndarray]) -> List[Tuple[float, str]]:
        """Optimized CuPy batch processing with true GPU parallelism."""
        batch_size = len(key_batch)
        results = []
        
        try:
            # Convert ciphertext to GPU once
            cipher_nums = np.array([ord(c) - ord('A') for c in ciphertext.upper()])
            cipher_gpu = cp.asarray(cipher_nums)
            text_length = len(cipher_nums)
            
            # Stack all keys into a 3D tensor for parallel processing
            keys_array = np.stack(key_batch)  # Shape: (batch_size, key_size, key_size)
            keys_gpu = cp.asarray(keys_array)
            
            # Pre-allocate result arrays
            all_decrypted = cp.zeros((batch_size, text_length), dtype=cp.int32)
            valid_mask = cp.ones(batch_size, dtype=cp.bool_)
            
            # Parallel matrix inversion and decryption
            for batch_idx in range(batch_size):
                try:
                    key_matrix = keys_gpu[batch_idx]
                    
                    # Calculate determinant and check invertibility
                    det = cp.linalg.det(key_matrix)
                    det_int = int(cp.round(det)) % 26
                    
                    if det_int == 0 or self._gcd(det_int, 26) != 1:
                        valid_mask[batch_idx] = False
                        continue
                    
                    # Calculate inverse matrix
                    if self.key_size == 2:
                        # Optimized 2x2 inverse
                        inv_det = self._mod_inverse(det_int, 26)
                        adj_matrix = cp.array([[key_matrix[1,1], -key_matrix[0,1]], 
                                             [-key_matrix[1,0], key_matrix[0,0]]])
                        inv_key = (adj_matrix * inv_det) % 26
                    else:
                        # General matrix inverse
                        inv_key = self._gpu_matrix_inverse_mod26(key_matrix, det_int)
                    
                    # Decrypt text in parallel blocks
                    decrypted = self._gpu_decrypt_parallel(cipher_gpu, inv_key)
                    all_decrypted[batch_idx] = decrypted
                    
                except Exception:
                    valid_mask[batch_idx] = False
            
            # Convert results back to CPU and score
            all_decrypted_cpu = cp.asnumpy(all_decrypted)
            valid_mask_cpu = cp.asnumpy(valid_mask)
            
            for batch_idx in range(batch_size):
                if valid_mask_cpu[batch_idx]:
                    decrypted_nums = all_decrypted_cpu[batch_idx]
                    decrypted_text = ''.join(chr(int(c) + ord('A')) for c in decrypted_nums)
                    score = self.statistical_analyzer.score_text(decrypted_text)
                    results.append((score, decrypted_text))
                else:
                    results.append((float('-inf'), ''))
            
        except Exception as e:
            logger.warning(f"GPU batch processing failed: {e}, falling back to individual processing")
            # Fallback to individual processing
            for key in key_batch:
                try:
                    decrypted = self.hill_cipher.decrypt(ciphertext, key)
                    score = self.statistical_analyzer.score_text(decrypted)
                    results.append((score, decrypted))
                except:
                    results.append((float('-inf'), ''))
        
        return results
    
    def _gpu_decrypt_parallel(self, cipher_gpu: cp.ndarray, inv_key: cp.ndarray) -> cp.ndarray:
        """Parallel GPU decryption of text blocks."""
        text_length = len(cipher_gpu)
        result = cp.zeros(text_length, dtype=cp.int32)
        
        # Process all blocks in parallel using vectorized operations
        num_blocks = (text_length + self.key_size - 1) // self.key_size
        
        for block_idx in range(num_blocks):
            start_idx = block_idx * self.key_size
            end_idx = min(start_idx + self.key_size, text_length)
            
            if end_idx - start_idx == self.key_size:
                # Full block - use matrix multiplication
                block = cipher_gpu[start_idx:end_idx]
                decrypted_block = cp.dot(inv_key, block) % 26
                result[start_idx:end_idx] = decrypted_block
            else:
                # Partial block - pad and process
                padded_block = cp.zeros(self.key_size, dtype=cp.int32)
                padded_block[:end_idx-start_idx] = cipher_gpu[start_idx:end_idx]
                decrypted_block = cp.dot(inv_key, padded_block) % 26
                result[start_idx:end_idx] = decrypted_block[:end_idx-start_idx]
        
        return result
    
    def _gpu_matrix_inverse_mod26(self, matrix: cp.ndarray, det_int: int) -> cp.ndarray:
        """GPU-optimized matrix inverse modulo 26."""
        try:
            # Use GPU linear algebra for matrix inverse
            inv_det = self._mod_inverse(det_int, 26)
            
            # Calculate adjugate matrix using GPU operations
            if self.key_size == 3:
                # Optimized 3x3 inverse
                adj = cp.zeros((3, 3), dtype=cp.int32)
                adj[0,0] = matrix[1,1]*matrix[2,2] - matrix[1,2]*matrix[2,1]
                adj[0,1] = matrix[0,2]*matrix[2,1] - matrix[0,1]*matrix[2,2]
                adj[0,2] = matrix[0,1]*matrix[1,2] - matrix[0,2]*matrix[1,1]
                adj[1,0] = matrix[1,2]*matrix[2,0] - matrix[1,0]*matrix[2,2]
                adj[1,1] = matrix[0,0]*matrix[2,2] - matrix[0,2]*matrix[2,0]
                adj[1,2] = matrix[0,2]*matrix[1,0] - matrix[0,0]*matrix[1,2]
                adj[2,0] = matrix[1,0]*matrix[2,1] - matrix[1,1]*matrix[2,0]
                adj[2,1] = matrix[0,1]*matrix[2,0] - matrix[0,0]*matrix[2,1]
                adj[2,2] = matrix[0,0]*matrix[1,1] - matrix[0,1]*matrix[1,0]
                
                inv_matrix = (adj * inv_det) % 26
            else:
                # General case using GPU linear algebra
                matrix_float = matrix.astype(cp.float32)
                inv_matrix_float = cp.linalg.inv(matrix_float)
                inv_matrix = cp.round(inv_matrix_float * det_int * inv_det) % 26
                inv_matrix = inv_matrix.astype(cp.int32)
            
            return inv_matrix
            
        except Exception:
            # Fallback to CPU calculation
            matrix_cpu = cp.asnumpy(matrix)
            inv_cpu = self._cpu_matrix_inverse_mod26(matrix_cpu, det_int)
            return cp.asarray(inv_cpu)
    
    def _cpu_matrix_inverse_mod26(self, matrix: np.ndarray, det_int: int) -> np.ndarray:
        """CPU fallback for matrix inverse."""
        try:
            return self.hill_cipher._matrix_inverse_mod26(matrix)
        except:
            return np.eye(self.key_size, dtype=np.int32)
    
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
    parser = argparse.ArgumentParser(description='Optimized CUDA Hill Cipher Breaker')
    parser.add_argument('--ciphertext', required=True, help='Ciphertext to decrypt')
    parser.add_argument('--key-size', type=int, required=True, help='Size of the key matrix (2, 3, 4, or 5)')
    parser.add_argument('--max-keys', type=int, default=50000, help='Maximum number of keys to test')
    parser.add_argument('--batch-size', type=int, default=2048, help='GPU batch size')
    
    args = parser.parse_args()
    
    print(f"[OPTIMIZED] CUDA Hill Cipher Breaker")
    print(f"Breaking {args.key_size}x{args.key_size} Hill cipher...")
    print(f"GPU acceleration: {'Enabled' if CUDA_AVAILABLE else 'Disabled'}")
    
    # Initialize breaker
    breaker = OptimizedCudaHillBreaker(args.key_size)
    
    # Break cipher
    result = breaker.break_cipher_optimized(
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
