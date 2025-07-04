#!/usr/bin/env python3
"""Test FinBERT scorer functionality"""

import subprocess
import sys

try:
    print("Testing FinBERT scorer...")
    result = subprocess.run([
        sys.executable, 'score_tweets_finbert.py', '--test-mode'
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode == 0:
        print("✅ FinBERT scorer test successful")
        print(result.stdout)
    else:
        print(f"❌ FinBERT scorer test failed: {result.stderr}")
        
except Exception as e:
    print(f"❌ FinBERT scorer test error: {e}")
