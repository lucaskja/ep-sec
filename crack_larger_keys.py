#!/usr/bin/env python3
"""
Crack the larger Hill cipher texts (3x3, 4x4, 5x5) using statistical analysis.
"""

import sys
import os
import time
import json

sys.path.append('hill_cipher')

from hill_cipher.breakers.improved_statistical_analyzer import ImprovedStatisticalAnalyzer

def load_cipher_text(filename):
    """Load cipher text from file."""
    base_dir = "/Users/lucaskle/Documents/USP/Seg"
    
    # Try known texts first
    known_path = os.path.join(base_dir, 'textos_conhecidos', 'Cifrado', 'Hill', filename)
    if os.path.exists(known_path):
        with open(known_path, 'r') as f:
            return f.read().strip(), 'known'
    
    # Try unknown texts
    unknown_path = os.path.join(base_dir, 'textos_desconhecidos', 'Cifrado', 'Hill', filename)
    if os.path.exists(unknown_path):
        with open(unknown_path, 'r') as f:
            return f.read().strip(), 'unknown'
    
    return None, None

def crack_cipher(ciphertext, key_size, max_candidates=None, timeout=None):
    """Crack a cipher using statistical analysis."""
    print(f"  Ciphertext: {ciphertext}")
    print(f"  Length: {len(ciphertext)}")
    
    # Set default parameters based on key size
    if max_candidates is None:
        max_candidates = {3: 20000, 4: 10000, 5: 5000}.get(key_size, 5000)
    
    if timeout is None:
        timeout = {3: 1800, 4: 3600, 5: 5400}.get(key_size, 1800)  # 30min, 1hr, 1.5hr
    
    print(f"  Using {max_candidates} candidates with {timeout}s timeout")
    
    start_time = time.time()
    
    try:
        analyzer = ImprovedStatisticalAnalyzer(key_size)
        key, decrypted, score = analyzer.break_cipher_improved(
            ciphertext, 
            max_candidates=max_candidates,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        if key is not None:
            return {
                'success': True,
                'key': key.tolist(),
                'decrypted': decrypted,
                'score': score,
                'time': elapsed,
                'candidates_tested': max_candidates
            }
        else:
            return {
                'success': False,
                'time': elapsed,
                'candidates_tested': max_candidates,
                'error': 'No suitable key found'
            }
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'time': elapsed,
            'error': str(e)
        }

def main():
    print("Hill Cipher Larger Keys Cracking (3x3, 4x4, 5x5)")
    print("=" * 55)
    
    # Define the files to crack
    files_to_crack = [
        ('3x3_texto_cifrado.txt', 3),
        ('4x4_texto_cifrado.txt', 4),
        ('5x5_texto_cifrado.txt', 5)
    ]
    
    results = {}
    
    for filename, key_size in files_to_crack:
        print(f"\nProcessing {filename} ({key_size}x{key_size})...")
        
        # Load from both known and unknown directories
        for directory in ['known', 'unknown']:
            full_filename = filename
            if directory == 'unknown':
                # Check if file exists in unknown directory
                ciphertext, source = load_cipher_text(filename)
                if source != 'unknown':
                    continue
            else:
                ciphertext, source = load_cipher_text(filename)
                if source != 'known':
                    continue
            
            result_key = f"{directory}_{filename}"
            print(f"\n  Cracking {result_key}...")
            
            if ciphertext:
                result = crack_cipher(ciphertext, key_size)
                results[result_key] = result
                
                if result['success']:
                    print(f"  ✓ SUCCESS in {result['time']:.1f}s")
                    print(f"    Key: {result['key']}")
                    print(f"    Score: {result['score']:.2f}")
                    print(f"    Decrypted: {result['decrypted'][:60]}...")
                else:
                    print(f"  ✗ FAILED after {result['time']:.1f}s")
                    if 'error' in result:
                        print(f"    Error: {result['error']}")
            else:
                print(f"  ⚠ File not found: {filename}")
                results[result_key] = {
                    'success': False,
                    'error': 'File not found'
                }
    
    # Summary
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    
    successful = sum(1 for r in results.values() if r.get('success', False))
    total = len([r for r in results.values() if 'error' not in r or r['error'] != 'File not found'])
    
    print(f"Files processed: {len(results)}")
    print(f"Valid attempts: {total}")
    print(f"Successful cracks: {successful}")
    if total > 0:
        print(f"Success rate: {successful/total*100:.1f}%")
    
    print("\nDetailed results:")
    for name, result in results.items():
        if result.get('success'):
            print(f"✓ {name}: {result['time']:.1f}s, score: {result['score']:.2f}")
        elif 'error' in result and result['error'] == 'File not found':
            print(f"⚠ {name}: File not found")
        else:
            print(f"✗ {name}: Failed ({result.get('time', 0):.1f}s)")
    
    # Save results
    output_file = 'hill_larger_keys_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    # Show successful decryptions
    print("\nSuccessful Decryptions:")
    print("-" * 30)
    for name, result in results.items():
        if result.get('success'):
            print(f"\n{name}:")
            print(f"Key: {result['key']}")
            print(f"Decrypted text: {result['decrypted']}")

if __name__ == "__main__":
    main()
