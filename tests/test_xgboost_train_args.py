import unittest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from xgboost_train import parse_args


class TestXGBoostTrainArgs(unittest.TestCase):
    """Unit tests for xgboost_train.py argument parsing"""
    
    def test_parse_args_train_required(self):
        """Test that --train argument is required when SM_CHANNEL_TRAINING not set"""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, 'argv', ['xgboost_train.py', '--model-dir', '/tmp/model']):
                with self.assertRaises(SystemExit):
                    parse_args()
    
    def test_parse_args_validation_defaults_to_none(self):
        """Test that --validation defaults to None"""
        with patch.object(sys, 'argv', ['xgboost_train.py', '--train', '/tmp/train.csv', '--model-dir', '/tmp/model']):
            args = parse_args()
            self.assertIsNone(args.validation)
    
    def test_parse_args_model_dir_default(self):
        """Test that --model-dir uses environment variable default"""
        with patch.dict(os.environ, {'SM_MODEL_DIR': '/opt/ml/model'}, clear=True):
            with patch.object(sys, 'argv', ['xgboost_train.py', '--train', '/tmp/train.csv']):
                args = parse_args()
                self.assertEqual(args.model_dir, '/opt/ml/model')
    
    def test_parse_args_all_arguments(self):
        """Test parsing with all arguments provided"""
        with patch.object(sys, 'argv', [
            'xgboost_train.py',
            '--train', '/tmp/train.csv',
            '--validation', '/tmp/val.csv',
            '--model-dir', '/custom/model'
        ]):
            args = parse_args()
            self.assertEqual(args.train, '/tmp/train.csv')
            self.assertEqual(args.validation, '/tmp/val.csv')
            self.assertEqual(args.model_dir, '/custom/model')
    
    def test_parse_args_hyperparameters(self):
        """Test that hyperparameters are parsed correctly"""
        with patch.object(sys, 'argv', [
            'xgboost_train.py',
            '--train', '/tmp/train.csv',
            '--model-dir', '/tmp/model',
            '--max_depth', '6',
            '--eta', '0.1',
            '--min_child_weight', '3'
        ]):
            args = parse_args()
            self.assertEqual(args.max_depth, 6)
            self.assertEqual(args.eta, 0.1)
            self.assertEqual(args.min_child_weight, 3)


if __name__ == '__main__':
    unittest.main()
