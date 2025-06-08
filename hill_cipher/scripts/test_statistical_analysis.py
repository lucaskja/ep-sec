#!/usr/bin/env python3
"""
Test Script for Statistical Analysis of Hill Cipher

This script tests the statistical analysis approach on known Hill cipher texts
and then applies it to unknown texts.

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import logging
import time
import json
from typing import Dict, List

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakers.enhanced_breaker import EnhancedHillBreaker
from breakers.statistical_analyzer import StatisticalAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_statistical')

class HillCipherTester:
    """
    Comprehensive tester for Hill cipher statistical analysis.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the tester.
        
        Args:
            base_dir: Base directory containing the project
        """
        if base_dir is None:
            # Assume we're in hill_cipher/scripts/
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_dir = base_dir
        
        self.known_texts_dir = os.path.join(self.base_dir, 'textos_conhecidos', 'Cifrado', 'Hill')
        self.unknown_texts_dir = os.path.join(self.base_dir, 'textos_desconhecidos', 'Cifrado', 'Hill')
        self.results_dir = os.path.join(self.base_dir, 'hill_cipher', 'results')
        
        # Known plaintexts for validation
        self.known_plaintexts = {
            '2x2_texto_cifrado.txt': "NTAOPARANAOTERQUEENTRARNUMALUTACORPORALCOMMINHAMAEVOCETEVEQUESETRANCARNOBANHEIROEPASSOUALGUMTEMPOOUV",
            # For unknown texts directory
            '2x2_texto_cifrado.txt_unknown': "CHUVOSAORITMODACIDADEDIMINUIASPESSOASFICAMEMCASAEOSOMCONSTANTEDACHUVACRIAUMAMELODIARELAXANTEEODIAPER"
        }
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Initialized tester with base directory: {self.base_dir}")
    
    def load_ciphertext(self, filename: str, directory: str) -> str:
        """
        Load ciphertext from file.
        
        Args:
            filename: Name of the cipher file
            directory: Directory containing the file
            
        Returns:
            Ciphertext content
        """
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return ""
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return ""
    
    def extract_key_size(self, filename: str) -> int:
        """
        Extract key size from filename.
        
        Args:
            filename: Cipher filename
            
        Returns:
            Key size
        """
        if '2x2' in filename:
            return 2
        elif '3x3' in filename:
            return 3
        elif '4x4' in filename:
            return 4
        elif '5x5' in filename:
            return 5
        else:
            logger.warning(f"Could not determine key size from {filename}, defaulting to 2")
            return 2
    
    def test_known_texts(self) -> Dict:
        """
        Test the statistical analysis on texts with known plaintexts.
        
        Returns:
            Test results
        """
        logger.info("=" * 60)
        logger.info("TESTING ON KNOWN TEXTS")
        logger.info("=" * 60)
        
        results = {}
        
        # Test files in known texts directory
        for filename in os.listdir(self.known_texts_dir):
            if filename.endswith('.txt') and not filename.startswith('.'):
                logger.info(f"\nTesting {filename}...")
                
                # Load ciphertext
                ciphertext = self.load_ciphertext(filename, self.known_texts_dir)
                if not ciphertext:
                    continue
                
                # Get key size
                key_size = self.extract_key_size(filename)
                
                # Get known plaintext if available
                known_plaintext = self.known_plaintexts.get(filename)
                
                # Create breaker
                breaker = EnhancedHillBreaker(key_size)
                
                # Test the cipher
                start_time = time.time()
                result = breaker.break_cipher(
                    ciphertext,
                    known_plaintext=known_plaintext,
                    method='statistical',  # Force statistical method for testing
                    max_candidates=5000 if key_size == 2 else 2000,
                    timeout=300
                )
                
                # Validate if we have known plaintext
                if known_plaintext and result['success']:
                    validation = breaker.validate_solution(ciphertext, result['key'], known_plaintext)
                    result['validation'] = validation
                
                result['filename'] = filename
                result['key_size'] = key_size
                result['ciphertext_length'] = len(ciphertext)
                results[filename] = result
                
                # Log results
                if result['success']:
                    logger.info(f"✓ SUCCESS: {filename}")
                    logger.info(f"  Key: {result['key'].flatten()}")
                    logger.info(f"  Score: {result['score']:.2f}")
                    logger.info(f"  Time: {result['time_elapsed']:.2f}s")
                    if 'validation' in result and result['validation']['exact_match']:
                        logger.info(f"  ✓ VALIDATION: Exact match with known plaintext!")
                    elif 'validation' in result:
                        logger.info(f"  ⚠ VALIDATION: No exact match")
                else:
                    logger.info(f"✗ FAILED: {filename}")
                    if 'error' in result:
                        logger.info(f"  Error: {result['error']}")
        
        return results
    
    def test_unknown_texts(self) -> Dict:
        """
        Test the statistical analysis on unknown texts.
        
        Returns:
            Test results
        """
        logger.info("=" * 60)
        logger.info("TESTING ON UNKNOWN TEXTS")
        logger.info("=" * 60)
        
        results = {}
        
        # Test files in unknown texts directory
        for filename in os.listdir(self.unknown_texts_dir):
            if filename.endswith('.txt') and not filename.startswith('.'):
                logger.info(f"\nTesting {filename}...")
                
                # Load ciphertext
                ciphertext = self.load_ciphertext(filename, self.unknown_texts_dir)
                if not ciphertext:
                    continue
                
                # Get key size
                key_size = self.extract_key_size(filename)
                
                # Check if we have known plaintext for this unknown text
                known_plaintext = self.known_plaintexts.get(f"{filename}_unknown")
                
                # Create breaker
                breaker = EnhancedHillBreaker(key_size)
                
                # Adjust parameters based on key size
                max_candidates = {2: 5000, 3: 3000, 4: 1500, 5: 1000}.get(key_size, 1000)
                timeout = {2: 300, 3: 600, 4: 900, 5: 1200}.get(key_size, 300)
                
                # Test the cipher
                start_time = time.time()
                result = breaker.break_cipher(
                    ciphertext,
                    known_plaintext=known_plaintext,
                    method='statistical',
                    max_candidates=max_candidates,
                    timeout=timeout
                )
                
                # Validate if we have known plaintext
                if known_plaintext and result['success']:
                    validation = breaker.validate_solution(ciphertext, result['key'], known_plaintext)
                    result['validation'] = validation
                
                result['filename'] = filename
                result['key_size'] = key_size
                result['ciphertext_length'] = len(ciphertext)
                results[filename] = result
                
                # Log results
                if result['success']:
                    logger.info(f"✓ SUCCESS: {filename}")
                    logger.info(f"  Key: {result['key'].flatten()}")
                    logger.info(f"  Score: {result['score']:.2f}")
                    logger.info(f"  Time: {result['time_elapsed']:.2f}s")
                    logger.info(f"  Decrypted: {result['decrypted_text'][:50]}...")
                    if 'validation' in result and result['validation']['exact_match']:
                        logger.info(f"  ✓ VALIDATION: Exact match with known plaintext!")
                    elif 'validation' in result:
                        logger.info(f"  ⚠ VALIDATION: No exact match")
                else:
                    logger.info(f"✗ FAILED: {filename}")
                    if 'error' in result:
                        logger.info(f"  Error: {result['error']}")
        
        return results
    
    def save_results(self, results: Dict, filename: str):
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            filename: Output filename
        """
        output_path = os.path.join(self.results_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, result in results.items():
            json_result = result.copy()
            if 'key' in json_result and json_result['key'] is not None:
                json_result['key'] = json_result['key'].tolist()
            json_results[key] = json_result
        
        try:
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def generate_summary_report(self, known_results: Dict, unknown_results: Dict):
        """
        Generate a summary report of all tests.
        
        Args:
            known_results: Results from known texts
            unknown_results: Results from unknown texts
        """
        logger.info("=" * 60)
        logger.info("SUMMARY REPORT")
        logger.info("=" * 60)
        
        # Known texts summary
        known_success = sum(1 for r in known_results.values() if r['success'])
        known_total = len(known_results)
        logger.info(f"Known texts: {known_success}/{known_total} successful")
        
        # Unknown texts summary
        unknown_success = sum(1 for r in unknown_results.values() if r['success'])
        unknown_total = len(unknown_results)
        logger.info(f"Unknown texts: {unknown_success}/{unknown_total} successful")
        
        # Success by key size
        all_results = {**known_results, **unknown_results}
        by_key_size = {}
        for result in all_results.values():
            key_size = result['key_size']
            if key_size not in by_key_size:
                by_key_size[key_size] = {'success': 0, 'total': 0}
            by_key_size[key_size]['total'] += 1
            if result['success']:
                by_key_size[key_size]['success'] += 1
        
        logger.info("\nSuccess rate by key size:")
        for key_size in sorted(by_key_size.keys()):
            stats = by_key_size[key_size]
            rate = stats['success'] / stats['total'] * 100
            logger.info(f"  {key_size}x{key_size}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
        
        # Average times
        successful_results = [r for r in all_results.values() if r['success']]
        if successful_results:
            avg_time = sum(r['time_elapsed'] for r in successful_results) / len(successful_results)
            logger.info(f"\nAverage time for successful attacks: {avg_time:.2f} seconds")
        
        # Best scores
        if successful_results:
            best_score = max(r['score'] for r in successful_results)
            worst_score = min(r['score'] for r in successful_results)
            logger.info(f"Score range: {worst_score:.2f} to {best_score:.2f}")
    
    def run_comprehensive_test(self):
        """
        Run comprehensive test on all available texts.
        """
        logger.info("Starting comprehensive Hill cipher statistical analysis test")
        
        # Test known texts first
        known_results = self.test_known_texts()
        self.save_results(known_results, 'known_texts_results.json')
        
        # Test unknown texts
        unknown_results = self.test_unknown_texts()
        self.save_results(unknown_results, 'unknown_texts_results.json')
        
        # Generate summary
        self.generate_summary_report(known_results, unknown_results)
        
        # Save combined results
        combined_results = {
            'known_texts': known_results,
            'unknown_texts': unknown_results,
            'summary': {
                'known_success_rate': sum(1 for r in known_results.values() if r['success']) / len(known_results) if known_results else 0,
                'unknown_success_rate': sum(1 for r in unknown_results.values() if r['success']) / len(unknown_results) if unknown_results else 0,
                'total_tests': len(known_results) + len(unknown_results),
                'total_successes': sum(1 for r in known_results.values() if r['success']) + sum(1 for r in unknown_results.values() if r['success'])
            }
        }
        
        self.save_results(combined_results, 'comprehensive_test_results.json')
        
        return combined_results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Hill Cipher Statistical Analysis")
    parser.add_argument("--base-dir", type=str, help="Base directory containing the project")
    parser.add_argument("--known-only", action="store_true", help="Test only known texts")
    parser.add_argument("--unknown-only", action="store_true", help="Test only unknown texts")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create tester
    tester = HillCipherTester(args.base_dir)
    
    if args.known_only:
        results = tester.test_known_texts()
        tester.save_results(results, 'known_texts_only_results.json')
    elif args.unknown_only:
        results = tester.test_unknown_texts()
        tester.save_results(results, 'unknown_texts_only_results.json')
    else:
        tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
