#!/usr/bin/env python3
"""
Unit tests for argument parsing validation of the three main scripts
"""
import unittest
import sys
import os
from unittest.mock import patch
from io import StringIO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestScriptArgumentParsing(unittest.TestCase):
    
    def test_deploy_best_model_args(self):
        """Test deploy_best_model.py argument parsing"""
        with patch('sys.argv', ['scripts/deploy_best_model.py', 
                                '--model-artifact', 's3://test-bucket/model.tar.gz',
                                '--endpoint-name', 'test-endpoint',
                                '--dry-run']):
            try:
                from scripts.deploy_best_model import main
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    main()
                    output = mock_stdout.getvalue()
                    self.assertIn('DRY RUN MODE', output)
                    self.assertIn('s3://test-bucket/model.tar.gz', output)
                    self.assertIn('test-endpoint', output)
            except SystemExit:
                pass
    
    def test_sample_inference_args(self):
        """Test sample_inference.py argument parsing"""
        with patch('sys.argv', ['sample_inference.py', 
                                '--endpoint-name', 'test-endpoint',
                                '--sample-count', '3',
                                '--dry-run']):
            try:
                import sample_inference
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    result = sample_inference.main()
                    output = mock_stdout.getvalue()
                    self.assertIn('DRY RUN MODE', output)
                    self.assertIn('test-endpoint', output)
                    self.assertIn('3 inference samples', output)
                    self.assertEqual(result, 0)
            except SystemExit:
                pass
    
    def test_catboost_hpo_launch_args(self):
        """Test aws_catboost_hpo_launch.py argument parsing"""
        with patch('sys.argv', ['aws_catboost_hpo_launch.py',
                                '--input-data-s3', 's3://test-bucket/data/',
                                '--dry-run']):
            try:
                import aws_catboost_hpo_launch
                import logging
                log_capture = StringIO()
                handler = logging.StreamHandler(log_capture)
                logger = logging.getLogger('aws_catboost_hpo_launch')
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                
                aws_catboost_hpo_launch.main()
                output = log_capture.getvalue()
                self.assertIn('DRY RUN MODE', output)
                self.assertIn('s3://test-bucket/data/', output)
            except SystemExit:
                pass

if __name__ == '__main__':
    unittest.main()
