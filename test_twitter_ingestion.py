#!/usr/bin/env python3
"""Test Twitter ingestion functionality"""

import subprocess
import sys

try:
    print("Testing Twitter ingestion (dry-run)...")
    result = subprocess.run([
        sys.executable, 'scripts/twitter_stream_ingest.py', 
        '--dry-run', '--duration', '10'
    ], capture_output=True, text=True, timeout=30)
    
    if result.returncode == 0:
        print("✅ Twitter ingestion test successful")
        print(result.stdout)
    else:
        print(f"❌ Twitter ingestion test failed: {result.stderr}")
        
except Exception as e:
    print(f"❌ Twitter ingestion test error: {e}")
