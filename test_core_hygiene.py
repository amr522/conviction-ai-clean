#!/usr/bin/env python3
"""
Test core HPO hygiene functionality without SageMaker dependencies
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

def test_core_hygiene():
    """Test core hygiene functionality with mocked SageMaker"""
    print("🧪 Testing core HPO hygiene without SageMaker dependencies...")
    
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
            
            print('✅ CLI precedence test:')
            result = get_input_data_s3('s3://cli-test/data.csv')
            print(f'  CLI arg result: {result}')
            assert result == 's3://cli-test/data.csv', "CLI argument should have highest precedence"
            
            os.environ['PINNED_DATA_S3'] = 's3://env-test/data.csv'
            result = get_input_data_s3(None)
            print(f'  Env var result: {result}')
            assert result == 's3://env-test/data.csv', "Environment variable should have second precedence"
            
            with open('last_dataset_uri.txt', 'w') as f:
                f.write('s3://file-test/data.csv')
            del os.environ['PINNED_DATA_S3']
            result = get_input_data_s3(None)
            print(f'  File fallback result: {result}')
            assert result == 's3://file-test/data.csv', "File should be used as fallback"
            
            print('✅ All hygiene tests passed without SageMaker dependencies')
            return True
            
    except Exception as e:
        print(f'❌ Test failed: {e}')
        return False
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    success = test_core_hygiene()
    sys.exit(0 if success else 1)
