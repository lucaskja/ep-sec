#!/usr/bin/env python3
"""
GPU Utilization Test Script
Quick test to verify the optimized CUDA breaker is using GPU properly

Author: Lucas Kledeglau Jahchan Alves
"""

import sys
import time
import subprocess
from pathlib import Path

def test_gpu_utilization():
    """Test GPU utilization with both versions"""
    
    print("[TEST] GPU Utilization Test")
    print("=" * 30)
    
    # Test ciphertext (same as your successful 2x2 test)
    test_ciphertext = "ypewhabanavprxgyekypbaonoefvdpisnxlwbabsgewuclweqktwkklkfkgyigzpbavsdxrwxacluufwjfugcwsarcoelklfowlhpnvwokmglxnpegoapjlp"
    
    print("Testing with 2x2 cipher for quick comparison...")
    print(f"Ciphertext length: {len(test_ciphertext)} characters")
    print()
    
    # Test original CUDA breaker
    print("[1] Testing Original CUDA Breaker:")
    print("Expected: Low GPU utilization (1-5%)")
    
    original_cmd = [
        sys.executable,
        "hill_cipher/breakers/cuda_breaker.py",
        "--ciphertext", test_ciphertext,
        "--key-size", "2",
        "--max-keys", "5000",
        "--batch-size", "512"
    ]
    
    print("Command:", " ".join(original_cmd))
    print("[RUNNING] Running original version...")
    
    start_time = time.time()
    try:
        result = subprocess.run(original_cmd, capture_output=True, text=True, timeout=300)
        original_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[OK] Original completed in {original_time:.1f}s")
            # Extract performance info
            lines = result.stdout.split('\n')
            for line in lines:
                if "keys/sec" in line.lower():
                    print(f"   Performance: {line.strip()}")
        else:
            print(f"[ERROR] Original failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Original version timed out")
        original_time = 300
    
    print()
    
    # Test optimized CUDA breaker
    print("[2] Testing Optimized CUDA Breaker:")
    print("Expected: High GPU utilization (80-95%)")
    
    optimized_cmd = [
        sys.executable,
        "hill_cipher/breakers/optimized_cuda_breaker.py",
        "--ciphertext", test_ciphertext,
        "--key-size", "2",
        "--max-keys", "5000",
        "--batch-size", "2048"
    ]
    
    print("Command:", " ".join(optimized_cmd))
    print("[RUNNING] Running optimized version...")
    
    start_time = time.time()
    try:
        result = subprocess.run(optimized_cmd, capture_output=True, text=True, timeout=300)
        optimized_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[OK] Optimized completed in {optimized_time:.1f}s")
            # Extract performance info
            lines = result.stdout.split('\n')
            for line in lines:
                if "keys/sec" in line.lower() or "keys per second" in line.lower():
                    print(f"   Performance: {line.strip()}")
        else:
            print(f"[ERROR] Optimized failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Optimized version timed out")
        optimized_time = 300
    
    print()
    print("[COMPARISON] Results:")
    if 'original_time' in locals() and 'optimized_time' in locals():
        if optimized_time > 0:
            speedup = original_time / optimized_time
            print(f"   Original time: {original_time:.1f}s")
            print(f"   Optimized time: {optimized_time:.1f}s")
            print(f"   Speedup: {speedup:.1f}x")
            
            if speedup > 2:
                print("   [EXCELLENT] Significant improvement achieved!")
            elif speedup > 1.2:
                print("   [GOOD] Good improvement")
            else:
                print("   [WARNING] Limited improvement - check GPU utilization")
    
    print()
    print("[TIPS] Monitoring:")
    print("   - Monitor GPU usage with: nvidia-smi -l 1")
    print("   - Optimized version should show 80-95% GPU utilization")
    print("   - Original version typically shows 1-5% GPU utilization")
    print("   - If both show low utilization, check CUDA installation")

if __name__ == "__main__":
    try:
        test_gpu_utilization()
    except KeyboardInterrupt:
        print("\n[CANCELLED] Test cancelled by user")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
