#!/usr/bin/env python3
"""
Test enhanced HPO hygiene functionality without SageMaker dependencies
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

def test_enhanced_hygiene():
    """Test enhanced hygiene functionality with mocked SageMaker"""
    print("🧪 Testing Enhanced HPO Hygiene")
    print("=" * 50)
    
    test_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    
    try:
        os.chdir(test_dir)
        
        with patch.dict('sys.modules', {
            'sagemaker': MagicMock(),
            'sagemaker.tuner': MagicMock(),
            'sagemaker.parameter': MagicMock(),
            'sagemaker.estimator': MagicMock(),
            'boto3': MagicMock()
        }):
            sys.path.insert(0, original_cwd)
            
            mock_boto3 = MagicMock()
            mock_s3_client = MagicMock()
            mock_boto3.client.return_value = mock_s3_client
            mock_s3_client.head_object.return_value = {}
            mock_s3_client.list_objects_v2.return_value = {'Contents': [{'Key': 'test.csv'}]}
            
            with patch('boto3.client', return_value=mock_s3_client):
                from aws_hpo_launch import get_input_data_s3, validate_s3_uri, launch_aapl_hpo, launch_full_universe_hpo
                
                print("✅ Step 2.1: CLI Argument Precedence")
                
                cli_arg = "s3://cli-bucket/cli-data.csv"
                env_var = "s3://env-bucket/env-data.csv"
                file_data = "s3://file-bucket/file-data.csv"
                
                with open('last_dataset_uri.txt', 'w') as f:
                    f.write(file_data)
                
                with patch.dict(os.environ, {'PINNED_DATA_S3': env_var, 'LAST_DATA_S3': env_var}):
                    result = get_input_data_s3(cli_arg)
                    assert result == cli_arg, f"Expected {cli_arg}, got {result}"
                    print(f"  ✅ CLI argument precedence: {result}")
                
                with patch.dict(os.environ, {'PINNED_DATA_S3': env_var}, clear=True):
                    result = get_input_data_s3(None)
                    assert result == env_var, f"Expected {env_var}, got {result}"
                    print(f"  ✅ PINNED_DATA_S3 precedence: {result}")
                
                with patch.dict(os.environ, {'LAST_DATA_S3': env_var}, clear=True):
                    result = get_input_data_s3(None)
                    assert result == env_var, f"Expected {env_var}, got {result}"
                    print(f"  ✅ LAST_DATA_S3 precedence: {result}")
                
                with patch.dict(os.environ, {}, clear=True):
                    result = get_input_data_s3(None)
                    assert result == file_data, f"Expected {file_data}, got {result}"
                    print(f"  ✅ File fallback precedence: {result}")
                
                print("\n✅ Step 2.2: S3 URI Validation")
                
                valid_uris = [
                    "s3://bucket/path/file.csv",
                    "s3://my-bucket/deep/nested/path/data.parquet",
                    "s3://bucket123/folder_name/file-name.csv"
                ]
                
                for uri in valid_uris:
                    try:
                        result = validate_s3_uri(uri)
                        assert result == True, f"Valid URI should pass: {uri}"
                        print(f"  ✅ Valid URI passed: {uri}")
                    except SystemExit:
                        print(f"  ❌ Valid URI caused SystemExit: {uri}")
                        return False
                
                invalid_uris = [
                    "http://bucket/path/file.csv",
                    "s3://",
                    "s3://bucket",
                    "bucket/path/file.csv"
                ]
                
                for uri in invalid_uris:
                    try:
                        validate_s3_uri(uri)
                        print(f"  ❌ Invalid URI should have caused SystemExit: {uri}")
                        return False
                    except SystemExit:
                        print(f"  ✅ Invalid URI correctly caused SystemExit: {uri}")
                    except Exception as e:
                        print(f"  ❌ Unexpected error for {uri}: {e}")
                        return False
                
                print("\n✅ Step 2.3: Dry-Run Functionality")
                
                valid_uri = "s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv"
                
                result = launch_aapl_hpo(valid_uri, dry_run=True)
                assert result is not None, "Dry run should return job name"
                assert "dry-run" in result, "Dry run job name should contain 'dry-run'"
                print(f"  ✅ AAPL dry-run test passed: {result}")
                
                result = launch_full_universe_hpo(valid_uri, dry_run=True)
                assert result is not None, "Dry run should return job name"
                assert "dry-run" in result, "Dry run job name should contain 'dry-run'"
                print(f"  ✅ Full Universe dry-run test passed: {result}")
                
                print("\n✅ All enhanced hygiene tests passed!")
                return True
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    success = test_enhanced_hygiene()
    sys.exit(0 if success else 1)
