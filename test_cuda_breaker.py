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
    print("FIXED CUDA HILL CIPHER BREAKER TEST")
    print("=" * 60)
    print("Focus: Correctness first, then reasonable GPU utilization")
    print()
    
    # Test ciphertext
    test_ciphertext = "ypewhabanavprxgyekypbaonoefvdpisnxlwbabsgewuclweqktwkklkfkgyigzpbavsdxrwxacluufwjfugcwsarcoelklfowlhpnvwokmglxnpegoapjlp"
    
    print(f"Testing 2x2 Hill cipher breaking...")
    print(f"Ciphertext length: {len(test_ciphertext)} characters")
    print(f"Expected: Correct key [23, 0, 17, 9] and readable Portuguese text")
    print(f"Features: Systematic key generation + accurate scoring")
    print()
    
    # Test fixed CUDA breaker
    cmd = [
        sys.executable,
        "hill_cipher/breakers/cuda_breaker_fixed.py",
        "--ciphertext", test_ciphertext,
        "--key-size", "2",
        "--max-keys", "10000",
        "--batch-size", "1024"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    print("[IMPORTANT] Monitor GPU usage with: nvidia-smi -l 1")
    print("[EXPECTED] Should find correct key and readable Portuguese text")
    print("=" * 60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print()
            print("=" * 60)
            print("[SUCCESS] Fixed CUDA breaker completed!")
            print()
            print("Correctness Check:")
            print("- Did it find key [23, 0, 17, 9] or similar?")
            print("- Is the decrypted text readable Portuguese?")
            print("- If yes: Fix successful!")
        else:
            print()
            print("=" * 60)
            print("[ERROR] Fixed CUDA breaker failed")
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
    print("Expected Results:")
    print("- Correct key: [23, 0, 17, 9] (as 2x2 matrix)")
    print("- Readable Portuguese text starting with known phrases")
    print("- Reasonable performance: 200-500 keys/sec")
    print()
    print("This version prioritizes finding the CORRECT answer")
    print("over maximum GPU utilization.")

if __name__ == "__main__":
    test_cuda_breaker()
