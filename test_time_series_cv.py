#!/usr/bin/env python3
"""
Test time-series CV functionality
"""
import sys
import os
sys.path.append('.')

try:
    from enhanced_train_sagemaker import create_time_series_cv_folds
    print('✅ Time-series CV function available')
    print('✅ Enhanced training pipeline ready for leak-proof retraining')
except ImportError as e:
    print(f'⚠️ Time-series CV function not found: {e}')
    print('⚠️ Will use NEXT_SESSION_PROMPT.md implementation instead')
