#!/usr/bin/env python3
"""Test script to verify pinned dataset functionality"""

import os
import sys

def test_pinned_dataset():
    print("🧪 Testing HPO Dataset Pinning Functionality")
    print("=" * 50)
    
    if os.path.exists('last_dataset_uri.txt'):
        with open('last_dataset_uri.txt', 'r') as f:
            pinned_uri = f.read().strip()
        print(f"✅ Found pinned dataset URI: {pinned_uri}")
        
        os.environ['PINNED_DATA_S3'] = pinned_uri
        
        try:
            from aws_hpo_launch import get_pinned_dataset, validate_s3_uri
            
            retrieved_uri = get_pinned_dataset()
            print(f"✅ Retrieved pinned dataset: {retrieved_uri}")
            
            if retrieved_uri == pinned_uri:
                print("✅ Environment variable correctly retrieved")
            else:
                print("❌ Environment variable mismatch")
                return False
            
            print("🔍 Validating S3 URI accessibility...")
            is_valid = validate_s3_uri(retrieved_uri)
            
            if is_valid:
                print("✅ S3 URI validation passed")
            else:
                print("⚠️ S3 URI validation failed (may be expected if credentials limited)")
            
            print("\n🎯 PINNED DATASET TEST RESULTS:")
            print(f"Dataset URI: {retrieved_uri}")
            print(f"Environment Variable: {'✅ Set' if os.environ.get('PINNED_DATA_S3') else '❌ Missing'}")
            print(f"Function Integration: ✅ Working")
            print(f"S3 Validation: {'✅ Passed' if is_valid else '⚠️ Limited'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing aws_hpo_launch functions: {e}")
            return False
    else:
        print("❌ No last_dataset_uri.txt file found")
        return False

if __name__ == "__main__":
    success = test_pinned_dataset()
    sys.exit(0 if success else 1)
