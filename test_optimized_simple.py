#!/usr/bin/env python3
"""
Simple Windows-compatible test for optimized CUDA breaker
No Unicode characters - Windows Command Prompt safe

Author: Lucas Kledeglau Jahchan Alves
"""

import sys
import subprocess

def test_optimized_breaker():
    """Simple test of the optimized CUDA breaker"""
    
    print("=" * 50)
    print("OPTIMIZED CUDA HILL CIPHER BREAKER TEST")
    print("=" * 50)
    
    # Test ciphertext (same as successful 2x2 test)
    test_ciphertext = "ypewhabanavprxgyekypbaonoefvdpisnxlwbabsgewuclweqktwkklkfkgyigzpbavsdxrwxacluufwjfugcwsarcoelklfowlhpnvwokmglxnpegoapjlp"
    
    print(f"Testing 2x2 Hill cipher breaking...")
    print(f"Ciphertext length: {len(test_ciphertext)} characters")
    print(f"Expected: High GPU utilization (80-95%)")
    print()
    
    # Test optimized CUDA breaker
    cmd = [
        sys.executable,
        "hill_cipher/breakers/optimized_cuda_breaker.py",
        "--ciphertext", test_ciphertext,
        "--key-size", "2",
        "--max-keys", "5000",
        "--batch-size", "2048"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    print("Running optimized CUDA breaker...")
    print("Monitor GPU usage with: nvidia-smi -l 1")
    print("=" * 50)
    
    try:
        # Run the command
        result = subprocess.run(cmd, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print()
            print("=" * 50)
            print("[SUCCESS] Optimized CUDA breaker completed successfully!")
            print("Check GPU utilization - should be 80-95% during execution")
        else:
            print()
            print("=" * 50)
            print("[ERROR] Optimized CUDA breaker failed")
            print("Check the error messages above")
            
    except subprocess.TimeoutExpired:
        print()
        print("=" * 50)
        print("[TIMEOUT] Test took longer than 5 minutes")
        print("This might indicate GPU utilization issues")
        
    except KeyboardInterrupt:
        print()
        print("=" * 50)
        print("[CANCELLED] Test cancelled by user")
        
    except Exception as e:
        print()
        print("=" * 50)
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_optimized_breaker()
