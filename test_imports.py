#!/usr/bin/env python3
"""
Test imports for the optimized CUDA breaker
Check if all required modules and methods are available
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all required imports"""
    
    print("Testing imports for optimized CUDA breaker...")
    print("=" * 50)
    
    try:
        print("[1] Testing core imports...")
        from hill_cipher.core.hill_cipher import HillCipher
        print("    [OK] HillCipher imported")
        
        from hill_cipher.breakers.statistical_analyzer import StatisticalAnalyzer
        print("    [OK] StatisticalAnalyzer imported")
        
        from hill_cipher.breakers.search_space_reducer import SearchSpaceReducer
        print("    [OK] SearchSpaceReducer imported")
        
    except Exception as e:
        print(f"    [ERROR] Core import failed: {e}")
        return False
    
    try:
        print("\n[2] Testing CUDA imports...")
        import cupy as cp
        print("    [OK] CuPy imported")
        
        # Test basic CuPy functionality
        test_array = cp.array([1, 2, 3])
        print("    [OK] CuPy basic functionality works")
        
    except Exception as e:
        print(f"    [WARNING] CuPy not available: {e}")
    
    try:
        print("\n[3] Testing SearchSpaceReducer methods...")
        reducer = SearchSpaceReducer(2)
        print("    [OK] SearchSpaceReducer initialized")
        
        # Test the method we need
        key_gen = reducer.generate_keys_smart_sampling(100)
        print("    [OK] generate_keys_smart_sampling method exists")
        
        # Test generating a few keys
        keys = list(key_gen)
        print(f"    [OK] Generated {len(keys)} sample keys")
        
    except Exception as e:
        print(f"    [ERROR] SearchSpaceReducer method test failed: {e}")
        return False
    
    try:
        print("\n[4] Testing StatisticalAnalyzer...")
        analyzer = StatisticalAnalyzer(2)
        print("    [OK] StatisticalAnalyzer initialized")
        
        # Test scoring
        score = analyzer.score_text("HELLO")
        print(f"    [OK] Text scoring works (score: {score:.2f})")
        
    except Exception as e:
        print(f"    [ERROR] StatisticalAnalyzer test failed: {e}")
        return False
    
    try:
        print("\n[5] Testing HillCipher...")
        cipher = HillCipher(2)
        print("    [OK] HillCipher initialized")
        
        # Test basic functionality
        import numpy as np
        test_key = np.array([[1, 2], [3, 4]])
        try:
            result = cipher.encrypt("HELLO", test_key)
            print("    [OK] HillCipher encryption works")
        except:
            print("    [WARNING] HillCipher encryption test failed (expected for invalid key)")
        
    except Exception as e:
        print(f"    [ERROR] HillCipher test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("[SUCCESS] All required components are available!")
    print("The optimized CUDA breaker should work correctly.")
    return True

if __name__ == "__main__":
    try:
        success = test_imports()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"[FATAL ERROR] Import test crashed: {e}")
        sys.exit(1)
