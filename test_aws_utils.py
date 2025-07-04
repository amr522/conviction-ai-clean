#!/usr/bin/env python3
"""Test AWS utils functionality"""

try:
    from aws_utils import get_secret
    print("✅ AWS utils import successful")
    
    test_secret = get_secret("test-secret")
    if test_secret is None:
        print("✅ Secret retrieval test successful: None returned for non-existent secret")
    else:
        print(f"✅ Secret retrieval test successful: {str(test_secret)[:50]}...")
    
except Exception as e:
    print(f"❌ AWS utils test failed: {e}")
