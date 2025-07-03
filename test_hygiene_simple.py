#!/usr/bin/env python3
"""
Simple test script to verify HPO training script hygiene without SageMaker dependencies
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

def test_cli_precedence():
    """Test CLI argument precedence without importing SageMaker"""
    print("🧪 Testing CLI Argument Precedence")
    print("-" * 40)
    
    test_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    
    try:
        os.chdir(test_dir)
        
        with patch.dict('sys.modules', {
            'sagemaker': MagicMock(),
            'sagemaker.tuner': MagicMock(),
            'sagemaker.parameter': MagicMock(),
            'sagemaker.estimator': MagicMock()
        }):
            sys.path.insert(0, original_cwd)
            from aws_hpo_launch import get_input_data_s3
            
            cli_arg = "s3://cli-bucket/cli-data.csv"
            env_var = "s3://env-bucket/env-data.csv"
            file_data = "s3://file-bucket/file-data.csv"
            
            with open('last_dataset_uri.txt', 'w') as f:
                f.write(file_data)
            
            with patch.dict(os.environ, {'PINNED_DATA_S3': env_var}):
                result = get_input_data_s3(cli_arg)
                assert result == cli_arg, f"Expected {cli_arg}, got {result}"
                print("✅ CLI argument precedence test passed")
            
            with patch.dict(os.environ, {'PINNED_DATA_S3': env_var}):
                result = get_input_data_s3(None)
                assert result == env_var, f"Expected {env_var}, got {result}"
                print("✅ Environment variable precedence test passed")
            
            with patch.dict(os.environ, {}, clear=True):
                result = get_input_data_s3(None)
                assert result == file_data, f"Expected {file_data}, got {result}"
                print("✅ File fallback precedence test passed")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(test_dir)

def test_s3_uri_validation():
    """Test S3 URI validation regex"""
    print("\n🧪 Testing S3 URI Validation")
    print("-" * 40)
    
    try:
        with patch.dict('sys.modules', {
            'sagemaker': MagicMock(),
            'sagemaker.tuner': MagicMock(),
            'sagemaker.parameter': MagicMock(),
            'sagemaker.estimator': MagicMock()
        }):
            sys.path.insert(0, os.getcwd())
            from aws_hpo_launch import validate_s3_uri
            
            with patch('aws_hpo_launch.boto3.client') as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.return_value = mock_s3
                mock_s3.head_object.return_value = {}
                
                valid_uris = [
                    "s3://bucket/path/file.csv",
                    "s3://my-bucket/deep/nested/path/data.parquet",
                    "s3://bucket123/folder_name/file-name.csv"
                ]
                
                for uri in valid_uris:
                    try:
                        result = validate_s3_uri(uri)
                        assert result == True, f"Valid URI should pass: {uri}"
                        print(f"✅ Valid URI passed: {uri}")
                    except SystemExit:
                        print(f"❌ Valid URI caused SystemExit: {uri}")
                        return False
                
                invalid_uris = [
                    "http://bucket/path/file.csv",
                    "s3://",
                    "s3://bucket",
                    "bucket/path/file.csv",
                    "",
                    None
                ]
                
                for uri in invalid_uris:
                    try:
                        validate_s3_uri(uri)
                        print(f"❌ Invalid URI should have caused SystemExit: {uri}")
                        return False
                    except SystemExit:
                        print(f"✅ Invalid URI correctly caused SystemExit: {uri}")
                    except Exception as e:
                        if uri is None or uri == "":
                            print(f"✅ Empty URI correctly handled: {uri}")
                        else:
                            print(f"❌ Unexpected error for {uri}: {e}")
                            return False
                
                return True
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_dry_run_functionality():
    """Test dry-run mode functionality"""
    print("\n🧪 Testing Dry-Run Functionality")
    print("-" * 40)
    
    try:
        with patch.dict('sys.modules', {
            'sagemaker': MagicMock(),
            'sagemaker.tuner': MagicMock(),
            'sagemaker.parameter': MagicMock(),
            'sagemaker.estimator': MagicMock()
        }):
            sys.path.insert(0, os.getcwd())
            from aws_hpo_launch import launch_aapl_hpo, launch_full_universe_hpo
            
            with patch('aws_hpo_launch.boto3.client') as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.return_value = mock_s3
                mock_s3.head_object.return_value = {}
                
                valid_uri = "s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv"
                
                result = launch_aapl_hpo(valid_uri, dry_run=True)
                assert result is not None, "Dry run should return job name"
                assert "dry-run" in result, "Dry run job name should contain 'dry-run'"
                print("✅ AAPL dry-run test passed")
                
                result = launch_full_universe_hpo(valid_uri, dry_run=True)
                assert result is not None, "Dry run should return job name"
                assert "dry-run" in result, "Dry run job name should contain 'dry-run'"
                print("✅ Full Universe dry-run test passed")
                
                return True
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run all hygiene tests"""
    print("🚀 HPO Training Script Hygiene Tests")
    print("=" * 50)
    
    tests = [
        test_cli_precedence,
        test_s3_uri_validation,
        test_dry_run_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            break
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All hygiene tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
