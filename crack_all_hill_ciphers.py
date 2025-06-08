#!/usr/bin/env python3
"""
Comprehensive Hill Cipher Cracking Script

This script attempts to crack all Hill cipher texts using the improved statistical analysis.
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple

sys.path.append('hill_cipher')

from hill_cipher.breakers.improved_statistical_analyzer import ImprovedStatisticalAnalyzer
from hill_cipher.breakers.kpa import HillCipherKPA

def load_cipher_files() -> Dict[str, Dict]:
    """Load all cipher files and organize them."""
    base_dir = "/Users/lucaskle/Documents/USP/Seg"
    
    files = {
        'known': {},
        'unknown': {}
    }
    
    # Known texts directory
    known_dir = os.path.join(base_dir, 'textos_conhecidos', 'Cifrado', 'Hill')
    for filename in os.listdir(known_dir):
        if filename.endswith('.txt') and not filename.startswith('.'):
            filepath = os.path.join(known_dir, filename)
            with open(filepath, 'r') as f:
                content = f.read().strip()
            
            key_size = extract_key_size(filename)
            files['known'][filename] = {
                'content': content,
                'key_size': key_size,
                'filepath': filepath
            }
    
    # Unknown texts directory
    unknown_dir = os.path.join(base_dir, 'textos_desconhecidos', 'Cifrado', 'Hill')
    for filename in os.listdir(unknown_dir):
        if filename.endswith('.txt') and not filename.startswith('.'):
            filepath = os.path.join(unknown_dir, filename)
            with open(filepath, 'r') as f:
                content = f.read().strip()
            
            key_size = extract_key_size(filename)
            files['unknown'][filename] = {
                'content': content,
                'key_size': key_size,
                'filepath': filepath
            }
    
    return files

def extract_key_size(filename: str) -> int:
    """Extract key size from filename."""
    if '2x2' in filename:
        return 2
    elif '3x3' in filename:
        return 3
    elif '4x4' in filename:
        return 4
    elif '5x5' in filename:
        return 5
    else:
        return 2

def get_known_plaintexts() -> Dict[str, str]:
    """Get known plaintexts for validation."""
    return {
        '2x2_texto_cifrado.txt': "NTAOPARANAOTERQUEENTRARNUMALUTACORPORALCOMMINHAMAEVOCETEVEQUESETRANCARNOBANHEIROEPASSOUALGUMTEMPOOUV",
        '2x2_texto_cifrado.txt_unknown': "CHUVOSAORITMODACIDADEDIMINUIASPESSOASFICAMEMCASAEOSOMCONSTANTEDACHUVACRIAUMAMELODIARELAXANTEEODIAPER"
    }

def crack_cipher(ciphertext: str, key_size: int, known_plaintext: str = None) -> Dict:
    """Crack a single cipher."""
    result = {
        'success': False,
        'key': None,
        'decrypted_text': None,
        'score': float('-inf'),
        'method': None,
        'time_elapsed': 0,
        'error': None
    }
    
    start_time = time.time()
    
    try:
        # Try KPA first if we have known plaintext
        if known_plaintext:
            print(f"  Trying KPA attack...")
            kpa = HillCipherKPA(key_size)
            key = kpa.attack(ciphertext, known_plaintext)
            
            if key is not None:
                from hill_cipher.core.hill_cipher import HillCipher
                hill = HillCipher(key_size)
                decrypted = hill.decrypt(ciphertext, key)
                
                analyzer = ImprovedStatisticalAnalyzer(key_size)
                score = analyzer.score_text(decrypted)
                
                result.update({
                    'success': True,
                    'key': key.tolist(),
                    'decrypted_text': decrypted,
                    'score': score,
                    'method': 'KPA',
                    'time_elapsed': time.time() - start_time
                })
                return result
        
        # Use statistical analysis
        print(f"  Trying statistical analysis...")
        analyzer = ImprovedStatisticalAnalyzer(key_size)
        
        # Set parameters based on key size
        if key_size == 2:
            # Use exhaustive search for 2x2
            key, decrypted, score = analyzer.break_cipher_improved(ciphertext)
        else:
            # Use optimized search for larger keys
            max_candidates = {3: 30000, 4: 15000, 5: 10000}.get(key_size, 10000)
            key, decrypted, score = analyzer.break_cipher_improved(
                ciphertext, max_candidates=max_candidates, timeout=1800  # 30 minutes
            )
        
        if key is not None:
            result.update({
                'success': True,
                'key': key.tolist(),
                'decrypted_text': decrypted,
                'score': score,
                'method': 'Statistical',
                'time_elapsed': time.time() - start_time
            })
        else:
            result.update({
                'success': False,
                'method': 'Statistical',
                'time_elapsed': time.time() - start_time,
                'error': 'No suitable key found'
            })
    
    except Exception as e:
        result.update({
            'success': False,
            'time_elapsed': time.time() - start_time,
            'error': str(e)
        })
    
    return result

def main():
    print("Hill Cipher Comprehensive Cracking")
    print("=" * 50)
    
    # Load all cipher files
    print("Loading cipher files...")
    files = load_cipher_files()
    known_plaintexts = get_known_plaintexts()
    
    print(f"Found {len(files['known'])} known texts and {len(files['unknown'])} unknown texts")
    
    results = {
        'known': {},
        'unknown': {},
        'summary': {}
    }
    
    # Process known texts
    print("\nProcessing known texts:")
    print("-" * 30)
    
    for filename, file_info in files['known'].items():
        print(f"\nCracking {filename} ({file_info['key_size']}x{file_info['key_size']})...")
        
        known_plaintext = known_plaintexts.get(filename)
        if known_plaintext:
            print(f"  Using known plaintext for validation")
        
        result = crack_cipher(
            file_info['content'], 
            file_info['key_size'], 
            known_plaintext
        )
        
        results['known'][filename] = result
        
        if result['success']:
            print(f"  ✓ SUCCESS: {result['method']} in {result['time_elapsed']:.1f}s")
            print(f"    Key: {result['key']}")
            print(f"    Score: {result['score']:.2f}")
            print(f"    Decrypted: {result['decrypted_text'][:50]}...")
        else:
            print(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")
    
    # Process unknown texts
    print("\nProcessing unknown texts:")
    print("-" * 30)
    
    for filename, file_info in files['unknown'].items():
        print(f"\nCracking {filename} ({file_info['key_size']}x{file_info['key_size']})...")
        
        # Check if we have known plaintext for this unknown text
        known_plaintext = known_plaintexts.get(f"{filename}_unknown")
        if known_plaintext:
            print(f"  Using known plaintext for validation")
        
        result = crack_cipher(
            file_info['content'], 
            file_info['key_size'], 
            known_plaintext
        )
        
        results['unknown'][filename] = result
        
        if result['success']:
            print(f"  ✓ SUCCESS: {result['method']} in {result['time_elapsed']:.1f}s")
            print(f"    Key: {result['key']}")
            print(f"    Score: {result['score']:.2f}")
            print(f"    Decrypted: {result['decrypted_text'][:50]}...")
        else:
            print(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")
    
    # Generate summary
    all_results = {**results['known'], **results['unknown']}
    successful = [r for r in all_results.values() if r['success']]
    
    results['summary'] = {
        'total_files': len(all_results),
        'successful': len(successful),
        'success_rate': len(successful) / len(all_results) * 100 if all_results else 0,
        'by_key_size': {},
        'by_method': {},
        'total_time': sum(r['time_elapsed'] for r in all_results.values())
    }
    
    # Success by key size
    for result in all_results.values():
        key_size = extract_key_size(list(all_results.keys())[list(all_results.values()).index(result)])
        if key_size not in results['summary']['by_key_size']:
            results['summary']['by_key_size'][key_size] = {'total': 0, 'successful': 0}
        results['summary']['by_key_size'][key_size]['total'] += 1
        if result['success']:
            results['summary']['by_key_size'][key_size]['successful'] += 1
    
    # Success by method
    for result in successful:
        method = result['method']
        if method not in results['summary']['by_method']:
            results['summary']['by_method'][method] = 0
        results['summary']['by_method'][method] += 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total files processed: {results['summary']['total_files']}")
    print(f"Successful cracks: {results['summary']['successful']}")
    print(f"Success rate: {results['summary']['success_rate']:.1f}%")
    print(f"Total time: {results['summary']['total_time']:.1f} seconds")
    
    print("\nSuccess by key size:")
    for key_size, stats in results['summary']['by_key_size'].items():
        rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {key_size}x{key_size}: {stats['successful']}/{stats['total']} ({rate:.1f}%)")
    
    print("\nSuccess by method:")
    for method, count in results['summary']['by_method'].items():
        print(f"  {method}: {count}")
    
    # Save results
    output_file = "hill_cipher_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
