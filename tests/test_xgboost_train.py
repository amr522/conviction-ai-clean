#!/usr/bin/env python3
"""
Test XGBoost training with synthetic CSV data
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestXGBoostTrain(unittest.TestCase):
    """Test XGBoost training functionality"""
    
    def test_train_model_with_tiny_csv(self):
        """Test training with tiny synthetic CSV and assert AUC >= 0.5"""
        from xgboost_train import train_model, parse_args
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = os.path.join(temp_dir, 'model')
            train_dir = os.path.join(temp_dir, 'train')
            os.makedirs(model_dir)
            os.makedirs(train_dir)
            
            tiny_csv_path = os.path.join(train_dir, 'train.csv')
            
            xgb_content = '1,0.5,0.3,0.8\n0,0.2,0.7,0.1'
            
            with open(tiny_csv_path, 'w') as dst:
                dst.write(xgb_content)
            
            mock_args = MagicMock()
            mock_args.model_dir = model_dir
            mock_args.train = train_dir
            mock_args.validation = None
            mock_args.max_depth = 3
            mock_args.eta = 0.1
            mock_args.subsample = 1.0
            mock_args.colsample_bytree = 1.0
            mock_args.min_child_weight = 1
            mock_args.gamma = 0.0
            mock_args.alpha = 0.0
            mock_args.reg_lambda = 1.0
            mock_args.num_round = 10
            
            auc = train_model(mock_args)
            
            self.assertGreaterEqual(auc, 0.5, f"AUC {auc} should be >= 0.5")
            
            model_path = os.path.join(model_dir, 'xgboost-model')
            self.assertTrue(os.path.exists(model_path), "Model file should be created")

if __name__ == '__main__':
    unittest.main()
