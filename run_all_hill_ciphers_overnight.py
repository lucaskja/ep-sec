#!/usr/bin/env python3
"""
Overnight Hill Cipher Breaking Script
Runs all Hill cipher tests in sequence using CUDA acceleration
Saves detailed results and logs for analysis

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import sys
import json
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path

class HillCipherBatchRunner:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.results_dir = self.base_dir / "overnight_results"
        self.logs_dir = self.results_dir / "logs"
        self.setup_directories()
        self.setup_logging()
        
        # Test cases with fully optimized GPU settings
        self.test_cases = [
            {
                "name": "3x3_known",
                "file": "textos_conhecidos/Cifrado/Hill/3x3_texto_cifrado.txt",
                "key_size": 3,
                "max_keys": 20000,
                "batch_size": 8192,
                "expected_time": "5-15 minutes"
            },
            {
                "name": "3x3_unknown", 
                "file": "textos_desconhecidos/Cifrado/Hill/3x3_texto_cifrado.txt",
                "key_size": 3,
                "max_keys": 20000,
                "batch_size": 8192,
                "expected_time": "5-15 minutes"
            },
            {
                "name": "4x4_known",
                "file": "textos_conhecidos/Cifrado/Hill/4x4_texto_cifrado.txt", 
                "key_size": 4,
                "max_keys": 200000,
                "batch_size": 4096,
                "expected_time": "15-60 minutes"
            },
            {
                "name": "4x4_unknown",
                "file": "textos_desconhecidos/Cifrado/Hill/4x4_texto_cifrado.txt",
                "key_size": 4, 
                "max_keys": 200000,
                "batch_size": 4096,
                "expected_time": "15-60 minutes"
            },
            {
                "name": "5x5_known",
                "file": "textos_conhecidos/Cifrado/Hill/5x5_texto_cifrado.txt",
                "key_size": 5,
                "max_keys": 2000000,
                "batch_size": 2048,
                "expected_time": "1-4 hours"
            },
            {
                "name": "5x5_unknown",
                "file": "textos_desconhecidos/Cifrado/Hill/5x5_texto_cifrado.txt",
                "key_size": 5,
                "max_keys": 2000000,
                "batch_size": 2048,
                "expected_time": "1-4 hours"
            }
        ]
        
        self.results = {}
        self.start_time = None
        self.total_time = None
        
    def setup_directories(self):
        """Create necessary directories for results and logs"""
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"hill_cipher_overnight_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*80)
        self.logger.info("FULLY OPTIMIZED HILL CIPHER OVERNIGHT BREAKING SESSION STARTED")
        self.logger.info("="*80)
        
    def read_ciphertext(self, file_path):
        """Read ciphertext from file"""
        full_path = self.base_dir / file_path
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Clean the text - remove spaces, newlines, keep only letters
                cleaned = ''.join(c.upper() for c in content if c.isalpha())
                self.logger.info(f"Read {len(cleaned)} characters from {file_path}")
                return cleaned
        except FileNotFoundError:
            self.logger.error(f"File not found: {full_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None
            
    def run_cuda_breaker(self, ciphertext, key_size, max_keys, batch_size, test_name):
        """Run the fully optimized CUDA breaker for a specific test case"""
        self.logger.info(f"Starting {test_name} - {key_size}x{key_size} matrix")
        self.logger.info(f"Ciphertext length: {len(ciphertext)} characters")
        self.logger.info(f"Max keys to test: {max_keys:,}")
        self.logger.info(f"GPU batch size: {batch_size:,} (fully optimized)")
        
        # Prepare command for fully optimized CUDA breaker
        cuda_breaker_path = self.base_dir / "hill_cipher" / "breakers" / "fully_optimized_cuda_breaker.py"
        cmd = [
            sys.executable,
            str(cuda_breaker_path),
            "--ciphertext", ciphertext,
            "--key-size", str(key_size),
            "--max-keys", str(max_keys),
            "--batch-size", str(batch_size)
        ]
        
        start_time = time.time()
        
        try:
            # Run the command and capture output
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=86400  # 24 hour timeout
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Parse results
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
            # Extract key information from output
            key_found = None
            score = None
            keys_tested = None
            decrypted_text = None
            
            if success and stdout:
                lines = stdout.split('\n')
                for line in lines:
                    if "Best key:" in line:
                        try:
                            key_part = line.split("Best key:")[1].strip()
                            key_found = key_part
                        except:
                            pass
                    elif "Best score:" in line:
                        try:
                            score = float(line.split("Best score:")[1].strip())
                        except:
                            pass
                    elif "Keys tested:" in line:
                        try:
                            keys_tested = int(line.split("Keys tested:")[1].strip().replace(',', ''))
                        except:
                            pass
                    elif "Decrypted text:" in line:
                        try:
                            decrypted_text = line.split("Decrypted text:")[1].strip()
                        except:
                            pass
            
            # Log results
            if success:
                self.logger.info(f"SUCCESS - {test_name} completed in {elapsed_time:.1f}s")
                if key_found:
                    self.logger.info(f"   Key found: {key_found}")
                if score is not None:
                    self.logger.info(f"   Score: {score:.2f}")
                if keys_tested:
                    self.logger.info(f"   Keys tested: {keys_tested:,}")
                if decrypted_text:
                    preview = decrypted_text[:100] + "..." if len(decrypted_text) > 100 else decrypted_text
                    self.logger.info(f"   Decrypted preview: {preview}")
            else:
                self.logger.error(f"FAILED - {test_name} failed after {elapsed_time:.1f}s")
                if stderr:
                    self.logger.error(f"   Error: {stderr}")
            
            return {
                "success": success,
                "elapsed_time": elapsed_time,
                "key_found": key_found,
                "score": score,
                "keys_tested": keys_tested,
                "decrypted_text": decrypted_text,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"TIMEOUT - {test_name} exceeded 24 hour limit")
            return {
                "success": False,
                "elapsed_time": 86400,
                "error": "Timeout after 24 hours",
                "timeout": True
            }
        except Exception as e:
            self.logger.error(f"ERROR - {test_name} crashed: {e}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "error": str(e),
                "exception": True
            }
    
    def save_results(self):
        """Save comprehensive results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"hill_cipher_results_{timestamp}.json"
        
        summary = {
            "session_info": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now().isoformat(),
                "total_time_seconds": self.total_time,
                "total_time_formatted": self.format_time(self.total_time) if self.total_time else None
            },
            "test_cases": self.test_cases,
            "results": self.results,
            "summary": self.generate_summary()
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Results saved to: {results_file}")
        return results_file
    
    def generate_summary(self):
        """Generate summary statistics"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r.get('success', False))
        failed_tests = total_tests - successful_tests
        
        total_keys_tested = sum(r.get('keys_tested', 0) for r in self.results.values() if r.get('keys_tested'))
        total_time = sum(r.get('elapsed_time', 0) for r in self.results.values())
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": f"{(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%",
            "total_keys_tested": total_keys_tested,
            "total_processing_time": total_time,
            "total_processing_time_formatted": self.format_time(total_time)
        }
    
    def format_time(self, seconds):
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def run_all_tests(self):
        """Run all test cases in sequence"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting overnight Hill cipher breaking session")
        self.logger.info(f"Total test cases: {len(self.test_cases)}")
        
        for i, test_case in enumerate(self.test_cases, 1):
            self.logger.info("="*60)
            self.logger.info(f"TEST {i}/{len(self.test_cases)}: {test_case['name']}")
            self.logger.info(f"Expected time: {test_case['expected_time']}")
            self.logger.info("="*60)
            
            # Read ciphertext
            ciphertext = self.read_ciphertext(test_case['file'])
            if not ciphertext:
                self.logger.error(f"Skipping {test_case['name']} - could not read ciphertext")
                self.results[test_case['name']] = {
                    "success": False,
                    "error": "Could not read ciphertext file",
                    "elapsed_time": 0
                }
                continue
            
            # Run the test
            result = self.run_cuda_breaker(
                ciphertext=ciphertext,
                key_size=test_case['key_size'],
                max_keys=test_case['max_keys'],
                batch_size=test_case['batch_size'],
                test_name=test_case['name']
            )
            
            # Store result with test case info
            result['test_case'] = test_case
            result['ciphertext_length'] = len(ciphertext)
            self.results[test_case['name']] = result
            
            # Save intermediate results
            self.save_results()
            
            self.logger.info(f"Completed {i}/{len(self.test_cases)} tests")
            
            # Brief pause between tests
            if i < len(self.test_cases):
                self.logger.info("Pausing 10 seconds before next test...")
                time.sleep(10)
        
        # Final summary
        end_time = datetime.now()
        self.total_time = (end_time - self.start_time).total_seconds()
        
        self.logger.info("="*80)
        self.logger.info("FULLY OPTIMIZED OVERNIGHT SESSION COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Total time: {self.format_time(self.total_time)}")
        
        summary = self.generate_summary()
        self.logger.info(f"Tests completed: {summary['successful_tests']}/{summary['total_tests']}")
        self.logger.info(f"Success rate: {summary['success_rate']}")
        self.logger.info(f"Total keys tested: {summary['total_keys_tested']:,}")
        
        # Save final results
        results_file = self.save_results()
        self.logger.info(f"Final results saved to: {results_file}")
        
        return self.results

def main():
    """Main function"""
    print("Fully Optimized Hill Cipher Overnight Breaking Session")
    print("=" * 60)
    print("Using fully optimized CUDA breaker with maximum GPU utilization")
    
    # Initialize runner
    runner = HillCipherBatchRunner()
    
    try:
        # Run all tests
        results = runner.run_all_tests()
        
        print("\nFully optimized overnight session completed successfully!")
        print(f"Check results in: {runner.results_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        runner.logger.info("Session interrupted by user")
        runner.save_results()
        return 1
    except Exception as e:
        runner.logger.error(f"Session failed with error: {e}")
        runner.save_results()
        return 1

if __name__ == "__main__":
    sys.exit(main())
