#!/usr/bin/env python3
"""
CUDA-Accelerated Hill Cipher Breaker

This module implements GPU-accelerated Hill cipher breaking using CUDA.
Provides significant speedup for large key spaces (3x3, 4x4, 5x5).

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hill_cipher import HillCipher
from breakers.statistical_analyzer import StatisticalAnalyzer
from breakers.search_space_reducer import SearchSpaceReducer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('cuda_breaker')

# Try to import CUDA libraries
CUDA_AVAILABLE = False
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    CUDA_LIB = 'cupy'
    logger.info("CUDA support available with CuPy")
except ImportError:
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import pycuda.gpuarray as gpuarray
        CUDA_AVAILABLE = True
        CUDA_LIB = 'pycuda'
        logger.info("CUDA support available with PyCUDA")
    except ImportError:
        logger.warning("CUDA libraries not available. Install cupy-cuda11x or pycuda for GPU acceleration.")

class CudaHillBreaker:
    """CUDA-accelerated Hill cipher breaker."""
    
    def __init__(self, key_size: int, data_dir: str = None, use_gpu: bool = True):
        """Initialize the CUDA Hill breaker."""
        self.key_size = key_size
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        
        # Initialize CPU components
        self.hill_cipher = HillCipher(key_size)
        self.statistical_analyzer = StatisticalAnalyzer(key_size, data_dir)
        self.search_reducer = SearchSpaceReducer(key_size)
        
        if self.use_gpu:
            self._initialize_gpu()
        
        logger.info(f"Initialized CudaHillBreaker for {key_size}x{key_size} cipher")
        logger.info(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
    
    def _initialize_gpu(self):
        """Initialize GPU components."""
        try:
            if CUDA_LIB == 'cupy':
                device = cp.cuda.Device()
                logger.info(f"GPU Device: {device}")
                logger.info(f"GPU Memory: {device.mem_info[1] / 1024**3:.1f} GB total")
            else:
                device = cuda.Device(0)
                logger.info(f"GPU Device: {device.name()}")
                logger.info(f"GPU Memory: {device.total_memory() / 1024**3:.1f} GB")
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            self.use_gpu = False
    
    def break_cipher_gpu_batch(self, ciphertext: str, 
                              max_keys: int = 50000,
                              batch_size: int = 512,
                              early_stopping_score: float = -100) -> Dict:
        """Break Hill cipher using GPU batch processing."""
        if not self.use_gpu:
            logger.warning("GPU not available, falling back to CPU")
            return self._break_cipher_cpu_fallback(ciphertext, max_keys, early_stopping_score)
        
        logger.info(f"Starting GPU-accelerated Hill cipher breaking")
        logger.info(f"Max keys: {max_keys}, Batch size: {batch_size}")
        
        start_time = time.time()
        
        # Generate keys using search space reduction
        key_generator = self.search_reducer.generate_keys_smart_sampling(max_keys)
        
        best_key = None
        best_decrypted = None
        best_score = float('-inf')
        keys_tested = 0
        
        # Process keys in batches
        key_batch = []
        
        for key in key_generator:
            key_batch.append(key)
            
            if len(key_batch) >= batch_size:
                # Process batch on GPU
                batch_results = self._process_gpu_batch(ciphertext, key_batch)
                keys_tested += len(key_batch)
                
                # Find best result in batch
                for i, (score, decrypted) in enumerate(batch_results):
                    if score > best_score:
                        best_score = score
                        best_key = key_batch[i].copy()
                        best_decrypted = decrypted
                        
                        if score > early_stopping_score:
                            logger.info(f"Early stopping - excellent score: {score:.2f}")
                            break
                
                # Progress update
                elapsed = time.time() - start_time
                rate = keys_tested / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {keys_tested} keys ({rate:.1f} keys/sec), best score: {best_score:.2f}")
                
                key_batch = []
                
                if best_score > early_stopping_score:
                    break
        
        # Process remaining keys
        if key_batch and best_score <= early_stopping_score:
            batch_results = self._process_gpu_batch(ciphertext, key_batch)
            keys_tested += len(key_batch)
            
            for i, (score, decrypted) in enumerate(batch_results):
                if score > best_score:
                    best_score = score
                    best_key = key_batch[i].copy()
                    best_decrypted = decrypted
        
        elapsed = time.time() - start_time
        
        return {
            'success': best_score > early_stopping_score,
            'key': best_key,
            'decrypted_text': best_decrypted,
            'score': best_score,
            'keys_tested': keys_tested,
            'time_elapsed': elapsed,
            'keys_per_second': keys_tested / elapsed if elapsed > 0 else 0,
            'gpu_acceleration': True
        }
    
    def _process_gpu_batch(self, ciphertext: str, key_batch: List[np.ndarray]) -> List[Tuple[float, str]]:
        """Process a batch of keys on GPU."""
        if CUDA_LIB == 'cupy':
            return self._process_cupy_batch(ciphertext, key_batch)
        else:
            return self._process_cpu_batch(ciphertext, key_batch)  # Fallback for now
    
    def _process_cupy_batch(self, ciphertext: str, key_batch: List[np.ndarray]) -> List[Tuple[float, str]]:
        """Process batch using CuPy."""
        results = []
        
        # Convert ciphertext to numbers
        cipher_nums = np.array([ord(c) - ord('A') for c in ciphertext.upper()])
        cipher_gpu = cp.asarray(cipher_nums)
        
        for key in key_batch:
            try:
                # Convert key to GPU
                key_gpu = cp.asarray(key)
                
                # Decrypt on GPU
                decrypted_gpu = self._decrypt_cupy(cipher_gpu, key_gpu)
                
                # Convert back to CPU and string
                decrypted_cpu = cp.asnumpy(decrypted_gpu)
                decrypted_text = ''.join(chr(int(c) + ord('A')) for c in decrypted_cpu)
                
                # Score on CPU
                score = self.statistical_analyzer.score_text(decrypted_text)
                results.append((score, decrypted_text))
                
            except Exception:
                results.append((float('-inf'), ''))
        
        return results
    
    def _decrypt_cupy(self, cipher_gpu: cp.ndarray, key_matrix: cp.ndarray) -> cp.ndarray:
        """Decrypt text using CuPy matrix operations."""
        try:
            # Calculate modular inverse of key matrix
            det = cp.linalg.det(key_matrix)
            det_int = int(cp.round(det)) % 26
            
            if det_int == 0 or self._gcd(det_int, 26) != 1:
                raise ValueError("Key not invertible")
            
            # Simple inverse calculation for small matrices
            if self.key_size == 2:
                inv_det = self._mod_inverse(det_int, 26)
                adj_matrix = cp.array([[key_matrix[1,1], -key_matrix[0,1]], 
                                     [-key_matrix[1,0], key_matrix[0,0]]])
                inv_key = (adj_matrix * inv_det) % 26
            else:
                # For larger matrices, use approximation
                inv_key = cp.linalg.inv(key_matrix.astype(cp.float32))
                inv_key = cp.round(inv_key * det_int * self._mod_inverse(det_int, 26)) % 26
                inv_key = inv_key.astype(cp.int32)
            
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
    
    def _process_cpu_batch(self, ciphertext: str, key_batch: List[np.ndarray]) -> List[Tuple[float, str]]:
        """Fallback CPU batch processing."""
        results = []
        
        for key in key_batch:
            try:
                decrypted = self.hill_cipher.decrypt(ciphertext, key)
                score = self.statistical_analyzer.score_text(decrypted)
                results.append((score, decrypted))
            except Exception:
                results.append((float('-inf'), ''))
        
        return results
    
    def _break_cipher_cpu_fallback(self, ciphertext: str, max_keys: int, early_stopping_score: float) -> Dict:
        """Fallback to CPU-only processing."""
        from breakers.optimized_breaker import OptimizedHillBreaker
        
        cpu_breaker = OptimizedHillBreaker(self.key_size)
        return cpu_breaker.break_cipher_optimized(
            ciphertext,
            max_keys_per_technique=max_keys,
            early_stopping_score=early_stopping_score,
            use_parallel=True
        )
    
    def _gcd(self, a: int, b: int) -> int:
        """Calculate greatest common divisor."""
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
        
        _, x, _ = extended_gcd(a, m)
        return (x % m + m) % m
    
    def get_gpu_info(self) -> Dict:
        """Get GPU information."""
        if not CUDA_AVAILABLE:
            return {'available': False, 'reason': 'CUDA libraries not installed'}
        
        try:
            if CUDA_LIB == 'cupy':
                device = cp.cuda.Device()
                return {
                    'available': True,
                    'library': 'CuPy',
                    'device_name': str(device),
                    'memory_total': device.mem_info[1],
                    'memory_free': device.mem_info[0]
                }
            else:
                device = cuda.Device(0)
                return {
                    'available': True,
                    'library': 'PyCUDA',
                    'device_name': device.name(),
                    'memory_total': device.total_memory()
                }
        except Exception as e:
            return {'available': False, 'reason': str(e)}

def main():
    """Main function for testing CUDA breaker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CUDA-Accelerated Hill Cipher Breaker")
    parser.add_argument("--ciphertext", type=str, help="Ciphertext to analyze")
    parser.add_argument("--key-size", type=int, default=3, choices=[2, 3, 4, 5])
    parser.add_argument("--max-keys", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--gpu-info", action="store_true")
    
    args = parser.parse_args()
    
    breaker = CudaHillBreaker(args.key_size, use_gpu=not args.no_gpu)
    
    if args.gpu_info:
        import json
        gpu_info = breaker.get_gpu_info()
        print("GPU Information:")
        print(json.dumps(gpu_info, indent=2))
        return
    
    if not args.ciphertext:
        print("Please provide --ciphertext or use --gpu-info")
        return
    
    print(f"Breaking {args.key_size}x{args.key_size} Hill cipher...")
    print(f"GPU acceleration: {'Enabled' if breaker.use_gpu else 'Disabled'}")
    
    result = breaker.break_cipher_gpu_batch(
        args.ciphertext,
        max_keys=args.max_keys,
        batch_size=args.batch_size
    )
    
    print(f"\nResults:")
    print(f"Success: {result['success']}")
    print(f"Keys tested: {result['keys_tested']:,}")
    print(f"Time elapsed: {result['time_elapsed']:.2f}s")
    print(f"Keys per second: {result['keys_per_second']:.1f}")
    
    if result['key'] is not None:
        print(f"Best key: {result['key'].flatten()}")
        print(f"Best score: {result['score']:.2f}")
        print(f"Decrypted text: {result['decrypted_text'][:100]}...")

if __name__ == "__main__":
    main()
