#!/usr/bin/env python3
"""
Comprehensive Hill Cipher Testing Script

This script tests all Hill cipher cracking techniques on all available ciphertexts
and validates the search space reduction effectiveness.

Author: Lucas Kledeglau Jahchan Alves
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add hill_cipher to path
sys.path.append('hill_cipher')

from hill_cipher.breakers.enhanced_breaker import EnhancedHillBreaker
from hill_cipher.breakers.optimized_breaker import OptimizedHillBreaker
from hill_cipher.breakers.search_space_reducer import SearchSpaceReducer
from hill_cipher.core.hill_cipher import HillCipher

class ComprehensiveHillTester:
    """Comprehensive tester for all Hill cipher techniques."""
    
    def __init__(self):
        """Initialize the tester."""
        self.results = {}
        self.normalized_text = self.load_normalized_text()
        
        # Load all test ciphers
        self.test_ciphers = self.load_all_ciphers()
        
        print("Comprehensive Hill Cipher Tester Initialized")
        print(f"Loaded {len(self.test_ciphers)} test ciphers")
        print(f"Normalized text: {len(self.normalized_text)} characters")
    
    def load_normalized_text(self) -> str:
        """Load the normalized text for validation."""
        try:
            with open('hill_cipher/data/normalized_text.txt', 'r', encoding='utf-8') as f:
                return f.read().upper().replace(' ', '').replace('\n', '')
        except Exception as e:
            print(f"Warning: Could not load normalized text: {e}")
            return ""
    
    def load_all_ciphers(self) -> Dict[str, Dict]:
        """Load all available cipher texts."""
        ciphers = {}
        
        # 2x2 ciphers
        try:
            with open('textos_conhecidos/Cifrado/Hill/2x2_texto_cifrado.txt', 'r') as f:
                ciphers['known_2x2'] = {
                    'ciphertext': f.read().strip(),
                    'key_size': 2,
                    'type': 'known',
                    'expected_key': [23, 0, 17, 9],  # From previous successful crack
                    'expected_plaintext': 'PARAJOAOMEUFILHOQUEMESTAAIBERNARDOHAMLETAPELEASVEZESVOCEFAZIAUMPENSAMENTOEMORAVANELEAFASTAVASECONSTRUIAUMACASAASSIMLONGINQUADENTRODESIERAESSEOSEUMODODELIDARCOMASCOISASHOJEPREFIROPENSARQUEVOCEPARTIUPARAREGRESSARAMIMEUNAOQUERIAAPENASASUAAUSENCIACOMOLEGADOEUQUERIAUMTIPODEPRESENCAAINDAQUEDOLORIDAETRISTEEAPESARDETUDONESTACASANESTEAPARTAMENTOVOCESERASEMPREUMCORPOQUENAOVAIPARARDEMORRERSERASEMPREOPAIQUESERECUSAAPARTIRNAVERDADEVOCENUNCASOUBEIREMBORAATEOFIMVOCEACREDITOUQUEOSLIVROSPODERIAMFAZERALGOPELASPESSOASNOENTANTOVOCEENTROUESAIUDAVIDAEELACONTINUOUASPERAHANOSOBJETOSMEMORIASDEVOCEMASP'
                }
        except Exception as e:
            print(f"Warning: Could not load known 2x2: {e}")
        
        try:
            with open('textos_desconhecidos/Cifrado/Hill/2x2_texto_cifrado.txt', 'r') as f:
                ciphers['unknown_2x2'] = {
                    'ciphertext': f.read().strip(),
                    'key_size': 2,
                    'type': 'unknown',
                    'expected_key': [23, 0, 14, 5],  # From previous successful crack
                    'expected_plaintext': None  # Will be determined
                }
        except Exception as e:
            print(f"Warning: Could not load unknown 2x2: {e}")
        
        # 3x3 ciphers
        try:
            with open('textos_conhecidos/Cifrado/Hill/3x3_texto_cifrado.txt', 'r') as f:
                ciphers['known_3x3'] = {
                    'ciphertext': f.read().strip(),
                    'key_size': 3,
                    'type': 'known',
                    'expected_key': None,  # Unknown
                    'expected_plaintext': None
                }
        except Exception as e:
            print(f"Warning: Could not load known 3x3: {e}")
        
        try:
            with open('textos_desconhecidos/Cifrado/Hill/3x3_texto_cifrado.txt', 'r') as f:
                ciphers['unknown_3x3'] = {
                    'ciphertext': f.read().strip(),
                    'key_size': 3,
                    'type': 'unknown',
                    'expected_key': None,  # Unknown
                    'expected_plaintext': None
                }
        except Exception as e:
            print(f"Warning: Could not load unknown 3x3: {e}")
        
        # 4x4 and 5x5 ciphers (if they exist)
        for size in [4, 5]:
            for cipher_type in ['conhecidos', 'desconhecidos']:
                try:
                    path = f'textos_{cipher_type}/Cifrado/Hill/{size}x{size}_texto_cifrado.txt'
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            key = f'{"known" if cipher_type == "conhecidos" else "unknown"}_{size}x{size}'
                            ciphers[key] = {
                                'ciphertext': f.read().strip(),
                                'key_size': size,
                                'type': cipher_type.replace('conhecidos', 'known').replace('desconhecidos', 'unknown'),
                                'expected_key': None,
                                'expected_plaintext': None
                            }
                except Exception as e:
                    continue
        
        return ciphers
    
    def validate_result(self, decrypted_text: str, expected_plaintext: str = None) -> Tuple[bool, Dict]:
        """Validate a decryption result."""
        if not decrypted_text:
            return False, {'error': 'No decrypted text'}
        
        clean_decrypted = decrypted_text.upper().replace(' ', '').replace('\n', '').rstrip('X')
        
        # If we have expected plaintext, check exact match
        if expected_plaintext:
            clean_expected = expected_plaintext.upper().replace(' ', '').replace('\n', '')
            if clean_decrypted == clean_expected:
                return True, {'match_type': 'exact_expected', 'length': len(clean_decrypted)}
            elif clean_decrypted in clean_expected or clean_expected in clean_decrypted:
                return True, {'match_type': 'partial_expected', 'length': len(clean_decrypted)}
        
        # Check against normalized text
        if self.normalized_text:
            if clean_decrypted in self.normalized_text:
                pos = self.normalized_text.find(clean_decrypted)
                return True, {
                    'match_type': 'exact_normalized',
                    'position': pos,
                    'length': len(clean_decrypted)
                }
            
            # Check for partial matches
            best_match_length = 0
            best_position = -1
            min_length = max(30, len(clean_decrypted) // 2)
            
            for length in range(len(clean_decrypted), min_length - 1, -5):
                for start in range(len(clean_decrypted) - length + 1):
                    substring = clean_decrypted[start:start + length]
                    pos = self.normalized_text.find(substring)
                    if pos != -1 and length > best_match_length:
                        best_match_length = length
                        best_position = pos
                        break
                if best_match_length > 0:
                    break
            
            if best_match_length >= min_length:
                percentage = (best_match_length / len(clean_decrypted)) * 100
                return True, {
                    'match_type': 'partial_normalized',
                    'position': best_position,
                    'length': best_match_length,
                    'percentage': percentage
                }
        
        return False, {'match_type': 'none'}
    
    def test_search_space_reduction(self, cipher_name: str, cipher_data: Dict) -> Dict:
        """Test search space reduction effectiveness."""
        print(f"\nTesting search space reduction for {cipher_name}...")
        
        key_size = cipher_data['key_size']
        ciphertext = cipher_data['ciphertext']
        
        reducer = SearchSpaceReducer(key_size)
        
        # Get reduction recommendations
        recommendations = reducer.get_reduction_recommendations()
        
        # Test different techniques
        techniques_to_test = ['diagonal', 'determinant_constraint', 'smart_sampling']
        if key_size == 2:
            techniques_to_test.append('exhaustive')
        
        results = {
            'cipher_name': cipher_name,
            'key_size': key_size,
            'recommendations': recommendations,
            'technique_results': {}
        }
        
        for technique in techniques_to_test:
            print(f"  Testing {technique}...")
            start_time = time.time()
            
            try:
                if technique == 'exhaustive' and key_size == 2:
                    # Use enhanced breaker for exhaustive search
                    breaker = EnhancedHillBreaker(key_size)
                    result = breaker.break_cipher_exhaustive(ciphertext)
                    
                    technique_result = {
                        'success': result['success'],
                        'key': result['key'].tolist() if result['key'] is not None else None,
                        'decrypted_text': result['decrypted_text'],
                        'score': result['score'],
                        'keys_tested': result['keys_tested'],
                        'time_elapsed': result['time_elapsed']
                    }
                else:
                    # Use optimized breaker with specific technique
                    breaker = OptimizedHillBreaker(key_size)
                    
                    # Limit keys for testing
                    max_keys = 1000 if technique == 'smart_sampling' else 500
                    
                    # Create a simple test by generating keys and testing a few
                    hill_cipher = HillCipher(key_size)
                    keys_tested = 0
                    best_score = float('-inf')
                    best_key = None
                    best_decrypted = None
                    
                    if technique == 'diagonal':
                        key_generator = reducer._generate_diagonal()
                    elif technique == 'determinant_constraint':
                        key_generator = reducer.generate_keys_by_determinant_constraint([1, 3, 5])
                    elif technique == 'smart_sampling':
                        key_generator = reducer.generate_keys_smart_sampling(max_keys)
                    
                    for key in key_generator:
                        if keys_tested >= max_keys:
                            break
                        
                        try:
                            decrypted = hill_cipher.decrypt(ciphertext, key)
                            # Simple scoring based on letter frequency
                            score = -sum((decrypted.count(c) - len(decrypted) * 0.1) ** 2 for c in 'AEIOU')
                            
                            keys_tested += 1
                            
                            if score > best_score:
                                best_score = score
                                best_key = key.copy()
                                best_decrypted = decrypted
                        except:
                            continue
                    
                    technique_result = {
                        'success': best_key is not None,
                        'key': best_key.tolist() if best_key is not None else None,
                        'decrypted_text': best_decrypted,
                        'score': best_score,
                        'keys_tested': keys_tested,
                        'time_elapsed': time.time() - start_time
                    }
                
                # Validate result
                if technique_result['decrypted_text']:
                    is_valid, validation_info = self.validate_result(
                        technique_result['decrypted_text'],
                        cipher_data.get('expected_plaintext')
                    )
                    technique_result['validation'] = {
                        'is_valid': is_valid,
                        'info': validation_info
                    }
                
                results['technique_results'][technique] = technique_result
                
                print(f"    {technique}: {keys_tested} keys in {technique_result['time_elapsed']:.2f}s, "
                      f"score: {best_score:.2f}")
                
            except Exception as e:
                print(f"    Error in {technique}: {e}")
                results['technique_results'][technique] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def test_all_ciphers(self) -> Dict:
        """Test all available ciphers with all techniques."""
        print("\n" + "="*60)
        print("COMPREHENSIVE HILL CIPHER TESTING")
        print("="*60)
        
        all_results = {}
        
        for cipher_name, cipher_data in self.test_ciphers.items():
            print(f"\n{'='*50}")
            print(f"TESTING: {cipher_name}")
            print(f"{'='*50}")
            print(f"Key size: {cipher_data['key_size']}x{cipher_data['key_size']}")
            print(f"Ciphertext: {cipher_data['ciphertext'][:50]}...")
            print(f"Length: {len(cipher_data['ciphertext'])}")
            
            # Test search space reduction
            reduction_results = self.test_search_space_reduction(cipher_name, cipher_data)
            
            all_results[cipher_name] = {
                'cipher_data': cipher_data,
                'reduction_results': reduction_results
            }
        
        return all_results
    
    def validate_2x2_effectiveness(self) -> Dict:
        """Specifically validate that our techniques work on known 2x2 ciphers."""
        print("\n" + "="*60)
        print("VALIDATING 2x2 SEARCH SPACE REDUCTION EFFECTIVENESS")
        print("="*60)
        
        validation_results = {}
        
        for cipher_name in ['known_2x2', 'unknown_2x2']:
            if cipher_name not in self.test_ciphers:
                continue
            
            cipher_data = self.test_ciphers[cipher_name]
            print(f"\nValidating {cipher_name}...")
            print(f"Expected key: {cipher_data['expected_key']}")
            
            # Test with enhanced breaker (exhaustive search)
            breaker = EnhancedHillBreaker(2)
            result = breaker.break_cipher_exhaustive(cipher_data['ciphertext'])
            
            validation_results[cipher_name] = {
                'expected_key': cipher_data['expected_key'],
                'found_key': result['key'].tolist() if result['key'] is not None else None,
                'success': result['success'],
                'keys_tested': result['keys_tested'],
                'time_elapsed': result['time_elapsed'],
                'decrypted_text': result['decrypted_text']
            }
            
            # Check if we found the expected key
            if result['key'] is not None and cipher_data['expected_key']:
                key_match = np.array_equal(result['key'], np.array(cipher_data['expected_key']).reshape(2, 2))
                validation_results[cipher_name]['key_match'] = key_match
                
                print(f"  Expected: {cipher_data['expected_key']}")
                print(f"  Found:    {result['key'].flatten()}")
                print(f"  Match:    {key_match}")
                print(f"  Success:  {result['success']}")
                print(f"  Keys tested: {result['keys_tested']}")
                print(f"  Time: {result['time_elapsed']:.2f}s")
                
                if result['decrypted_text']:
                    print(f"  Decrypted: {result['decrypted_text'][:100]}...")
                    
                    # Validate against expected plaintext
                    if cipher_data.get('expected_plaintext'):
                        is_valid, validation_info = self.validate_result(
                            result['decrypted_text'],
                            cipher_data['expected_plaintext']
                        )
                        validation_results[cipher_name]['plaintext_validation'] = {
                            'is_valid': is_valid,
                            'info': validation_info
                        }
                        print(f"  Plaintext validation: {is_valid}")
            else:
                print(f"  Failed to crack cipher")
        
        return validation_results
    
    def generate_summary_report(self, all_results: Dict, validation_results: Dict) -> str:
        """Generate a comprehensive summary report."""
        report = []
        report.append("COMPREHENSIVE HILL CIPHER TESTING REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        total_ciphers = len(all_results)
        successful_cracks = 0
        
        report.append("SUMMARY STATISTICS")
        report.append("-" * 20)
        report.append(f"Total ciphers tested: {total_ciphers}")
        
        # Analyze results by key size
        by_key_size = {}
        for cipher_name, results in all_results.items():
            key_size = results['cipher_data']['key_size']
            if key_size not in by_key_size:
                by_key_size[key_size] = {'total': 0, 'successful': 0}
            
            by_key_size[key_size]['total'] += 1
            
            # Check if any technique was successful
            for technique, tech_result in results['reduction_results']['technique_results'].items():
                if tech_result.get('validation', {}).get('is_valid', False):
                    by_key_size[key_size]['successful'] += 1
                    successful_cracks += 1
                    break
        
        for key_size, stats in sorted(by_key_size.items()):
            success_rate = (stats['successful'] / stats['total']) * 100 if stats['total'] > 0 else 0
            report.append(f"{key_size}x{key_size} ciphers: {stats['successful']}/{stats['total']} "
                         f"({success_rate:.1f}% success rate)")
        
        report.append(f"Overall success rate: {successful_cracks}/{total_ciphers} "
                     f"({successful_cracks/total_ciphers*100:.1f}%)")
        report.append("")
        
        # 2x2 Validation Results
        report.append("2x2 VALIDATION RESULTS")
        report.append("-" * 25)
        for cipher_name, val_result in validation_results.items():
            report.append(f"{cipher_name}:")
            report.append(f"  Key match: {val_result.get('key_match', False)}")
            report.append(f"  Success: {val_result.get('success', False)}")
            report.append(f"  Keys tested: {val_result.get('keys_tested', 0)}")
            report.append(f"  Time: {val_result.get('time_elapsed', 0):.2f}s")
            if 'plaintext_validation' in val_result:
                pv = val_result['plaintext_validation']
                report.append(f"  Plaintext valid: {pv['is_valid']}")
            report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS BY CIPHER")
        report.append("-" * 30)
        
        for cipher_name, results in all_results.items():
            report.append(f"\n{cipher_name.upper()}:")
            cipher_data = results['cipher_data']
            report.append(f"  Key size: {cipher_data['key_size']}x{cipher_data['key_size']}")
            report.append(f"  Ciphertext length: {len(cipher_data['ciphertext'])}")
            
            # Show technique results
            for technique, tech_result in results['reduction_results']['technique_results'].items():
                if 'error' in tech_result:
                    report.append(f"  {technique}: ERROR - {tech_result['error']}")
                else:
                    success = tech_result.get('success', False)
                    keys_tested = tech_result.get('keys_tested', 0)
                    time_elapsed = tech_result.get('time_elapsed', 0)
                    score = tech_result.get('score', 0)
                    
                    status = "SUCCESS" if success else "FAILED"
                    report.append(f"  {technique}: {status} - {keys_tested} keys, "
                                 f"{time_elapsed:.2f}s, score: {score:.2f}")
                    
                    # Show validation if available
                    if 'validation' in tech_result:
                        val = tech_result['validation']
                        if val['is_valid']:
                            info = val['info']
                            report.append(f"    ✓ Validation: {info.get('match_type', 'unknown')}")
                        else:
                            report.append(f"    ✗ Validation failed")
        
        return "\n".join(report)
    
    def run_comprehensive_test(self) -> None:
        """Run the complete comprehensive test."""
        start_time = time.time()
        
        # Test all ciphers
        all_results = self.test_all_ciphers()
        
        # Validate 2x2 effectiveness
        validation_results = self.validate_2x2_effectiveness()
        
        # Generate report
        report = self.generate_summary_report(all_results, validation_results)
        
        total_time = time.time() - start_time
        
        # Print report
        print("\n" + "="*60)
        print(report)
        print("="*60)
        print(f"Total testing time: {total_time:.2f} seconds")
        
        # Save results
        output_data = {
            'all_results': all_results,
            'validation_results': validation_results,
            'summary_report': report,
            'total_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        output_data = convert_numpy(output_data)
        
        with open('comprehensive_hill_test_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nDetailed results saved to comprehensive_hill_test_results.json")

def main():
    """Main function."""
    print("Starting Comprehensive Hill Cipher Testing...")
    
    tester = ComprehensiveHillTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
