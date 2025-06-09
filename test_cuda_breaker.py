#!/usr/bin/env python3
"""
Test script for fully optimized CUDA Hill cipher breaker
This should achieve 80-95% GPU utilization

Author: Lucas Kledeglau Jahchan Alves
"""

import sys
import subprocess

def test_cuda_breaker():
    """Test the fully optimized CUDA breaker"""
    
    print("=" * 60)
    print("FULLY OPTIMIZED CUDA HILL CIPHER BREAKER TEST")
    print("=" * 60)
    print("This should achieve 80-95% GPU utilization!")
    print()
    
    # Test ciphertext
    test_ciphertext = "ypewhabanavprxgyekypbaonoefvdpisnxlwbabsgewuclweqktwkklkfkgyigzpbavsdxrwxacluufwjfugcwsarcoelklfowlhpnvwokmglxnpegoapjlp"
    
    print(f"Testing 2x2 Hill cipher breaking...")
    print(f"Ciphertext length: {len(test_ciphertext)} characters")
    print(f"Expected: VERY HIGH GPU utilization (80-95%)")
    print(f"Features: Parallel matrix ops + GPU scoring")
    print()
    
    # Test fully optimized CUDA breaker with large batch
    cmd = [
        sys.executable,
        "hill_cipher/breakers/fully_optimized_cuda_breaker.py",
        "--ciphertext", test_ciphertext,
        "--key-size", "2",
        "--max-keys", "10000",
        "--batch-size", "8192"  # Very large batch for maximum GPU utilization
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    print("[IMPORTANT] Monitor GPU usage with: nvidia-smi -l 1")
    print("[EXPECTED] GPU utilization should be 80-95% during execution")
    print("=" * 60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print()
            print("=" * 60)
            print("[SUCCESS] Fully optimized CUDA breaker completed!")
            print()
            print("GPU Utilization Check:")
            print("- Did you see 80-95% GPU usage during execution?")
            print("- If yes: Optimization successful!")
            print("- If no: There may be other bottlenecks")
        else:
            print()
            print("=" * 60)
            print("[ERROR] Fully optimized CUDA breaker failed")
            print("Check the error messages above")
            
    except subprocess.TimeoutExpired:
        print()
        print("=" * 60)
        print("[TIMEOUT] Test took longer than 5 minutes")
        
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("[CANCELLED] Test cancelled by user")
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"[ERROR] Test failed: {e}")
    
    print()
    print("Performance Comparison Expected:")
    print("- Original CUDA breaker: 85 keys/sec, 1-5% GPU")
    print("- Fully optimized: 500+ keys/sec, 80-95% GPU")
    print()
    print("If you don't see high GPU utilization, the bottleneck")
    print("might be in CPU-GPU memory transfers or other factors.")

if __name__ == "__main__":
    test_cuda_breaker()
