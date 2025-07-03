#!/usr/bin/env python3
"""
Test hyperparameter validation script
"""
import sys
sys.path.append('.')
from aws_hpo_launch import get_hyperparameter_ranges

ranges = get_hyperparameter_ranges()
print('XGBoost parameters found:')
xgb_params = ['max_depth', 'eta', 'subsample', 'colsample_bytree', 'gamma', 'alpha', 'lambda', 'min_child_weight']
for param in xgb_params:
    if param in ranges:
        print(f'  ✅ {param}: {ranges[param]}')
    else:
        print(f'  ❌ {param}: MISSING')

print('\nLightGBM parameters check:')
lgb_params = ['num_leaves', 'feature_fraction', 'bagging_fraction', 'min_child_samples']
for param in lgb_params:
    if param in ranges:
        print(f'  ❌ {param}: STILL PRESENT - {ranges[param]}')
    else:
        print(f'  ✅ {param}: REMOVED')
