#!/usr/bin/env python3
"""
Hill Cipher Progress Checker
Monitor the overnight breaking session progress

Author: Lucas Kledeglau Jahchan Alves
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

def check_progress():
    """Check the current progress of the overnight session"""
    base_dir = Path.cwd()
    results_dir = base_dir / "overnight_results"
    logs_dir = results_dir / "logs"
    
    print("🔍 Hill Cipher Progress Check")
    print("=" * 40)
    
    if not results_dir.exists():
        print("❌ No overnight session found")
        print("Run run_overnight.bat or run_overnight.ps1 first")
        return
    
    # Find latest results file
    result_files = list(results_dir.glob("hill_cipher_results_*.json"))
    if not result_files:
        print("⏳ Session started but no results yet")
        print("Check logs for current status...")
    else:
        latest_results = max(result_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_results, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Latest Results: {latest_results.name}")
        print(f"⏰ Last Updated: {datetime.fromtimestamp(latest_results.stat().st_mtime)}")
        print()
        
        # Show summary
        if 'summary' in data:
            summary = data['summary']
            print(f"✅ Completed: {summary['successful_tests']}/{summary['total_tests']} tests")
            print(f"📈 Success Rate: {summary['success_rate']}")
            print(f"🔑 Keys Tested: {summary['total_keys_tested']:,}")
            print(f"⏱️  Processing Time: {summary.get('total_processing_time_formatted', 'N/A')}")
            print()
        
        # Show individual results
        if 'results' in data:
            print("📋 Test Status:")
            for test_name, result in data['results'].items():
                status = "✅" if result.get('success') else "❌"
                time_str = f"{result.get('elapsed_time', 0):.1f}s"
                if result.get('success') and result.get('key_found'):
                    print(f"  {status} {test_name}: {time_str} - Key: {result['key_found']}")
                else:
                    print(f"  {status} {test_name}: {time_str}")
    
    # Check latest log
    log_files = list(logs_dir.glob("hill_cipher_overnight_*.log")) if logs_dir.exists() else []
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        print()
        print(f"📝 Latest Log: {latest_log.name}")
        
        # Show last few lines
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                print("📄 Recent Activity:")
                for line in lines[-5:]:
                    print(f"  {line.strip()}")
        except:
            print("  Could not read log file")
    
    print()
    print("💡 Tip: Run this script periodically to monitor progress")

if __name__ == "__main__":
    try:
        check_progress()
    except KeyboardInterrupt:
        print("\n👋 Progress check cancelled")
    except Exception as e:
        print(f"❌ Error checking progress: {e}")
