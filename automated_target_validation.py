#!/usr/bin/env python3
"""
Automated target validation check for ML pipeline
Ensures direction column exists and has proper distribution
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TargetValidationError(Exception):
    """Custom exception for target validation failures"""
    pass

class AutomatedTargetValidator:
    """Automated validation of target columns in processed data files"""
    
    def __init__(self, data_dir: str, min_positive_ratio: float = 0.3, max_positive_ratio: float = 0.7):
        self.data_dir = Path(data_dir)
        self.min_positive_ratio = min_positive_ratio
        self.max_positive_ratio = max_positive_ratio
        self.validation_results = {}
        
    def validate_single_file(self, file_path: Path) -> Dict:
        """Validate a single CSV file for proper target schema (XGBoost format)"""
        try:
            logger.info(f"Validating {file_path.name}")
            
            df = pd.read_csv(file_path, nrows=1000)  # Sample for efficiency
            
            target_col = df.iloc[:, 0]
            first_col_name = df.columns[0]
            
            logger.info(f"First column name: '{first_col_name}', values: {target_col.unique()[:10]}")
            
            unique_values = set(target_col.dropna().unique())
            
            if not unique_values.issubset({0, 1, 0.0, 1.0}):
                raise TargetValidationError(
                    f"Invalid target values in first column of {file_path.name}: {unique_values}. "
                    f"Expected only 0 and 1 for XGBoost format."
                )
            
            positive_count = (target_col == 1).sum()
            total_count = len(target_col.dropna())
            positive_ratio = positive_count / total_count if total_count > 0 else 0
            
            if positive_ratio < self.min_positive_ratio or positive_ratio > self.max_positive_ratio:
                logger.warning(
                    f"Suspicious target distribution in {file_path.name}: "
                    f"{positive_ratio:.3f} positive ratio. Expected between "
                    f"{self.min_positive_ratio} and {self.max_positive_ratio}"
                )
            
            has_direction_column = 'direction' in df.columns
            direction_correlation = None
            
            if has_direction_column:
                direction_col = df['direction']
                direction_correlation = np.corrcoef(target_col, direction_col)[0, 1]
                logger.info(f"Found 'direction' column with correlation to target: {direction_correlation:.3f}")
            else:
                logger.info("No 'direction' column found - using first column as target (XGBoost format)")
            
            return {
                'file': file_path.name,
                'status': 'PASS',
                'total_samples': len(df),
                'target_samples': total_count,
                'positive_ratio': positive_ratio,
                'has_direction_column': has_direction_column,
                'direction_correlation': direction_correlation,
                'first_column_name': first_col_name,
                'unique_target_values': sorted(list(unique_values)),
                'format': 'XGBoost' if not has_direction_column else 'Named'
            }
            
        except Exception as e:
            logger.error(f"Validation failed for {file_path.name}: {e}")
            return {
                'file': file_path.name,
                'status': 'FAIL',
                'error': str(e),
                'has_direction_column': 'direction' in df.columns if 'df' in locals() else False
            }
    
    def validate_all_files(self) -> Dict:
        """Validate all CSV files in the data directory"""
        logger.info(f"Starting automated target validation in {self.data_dir}")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            raise TargetValidationError(f"No CSV files found in {self.data_dir}")
        
        logger.info(f"Found {len(csv_files)} CSV files to validate")
        
        results = []
        failed_files = []
        
        for csv_file in csv_files:
            result = self.validate_single_file(csv_file)
            results.append(result)
            
            if result['status'] == 'FAIL':
                failed_files.append(csv_file.name)
        
        passed_files = [r for r in results if r['status'] == 'PASS']
        
        summary = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'total_files': len(csv_files),
            'passed_files': len(passed_files),
            'failed_files': len(failed_files),
            'pass_rate': len(passed_files) / len(csv_files) if csv_files else 0,
            'failed_file_list': failed_files,
            'detailed_results': results
        }
        
        if passed_files:
            positive_ratios = [r['positive_ratio'] for r in passed_files if r['positive_ratio'] is not None]
            correlations = [r['direction_correlation'] for r in passed_files if r['direction_correlation'] is not None]
            
            summary.update({
                'avg_positive_ratio': np.mean(positive_ratios) if positive_ratios else None,
                'std_positive_ratio': np.std(positive_ratios) if positive_ratios else None,
                'avg_correlation': np.mean(correlations) if correlations else None,
                'min_correlation': np.min(correlations) if correlations else None
            })
        
        self.validation_results = summary
        return summary
    
    def save_results(self, output_file: str):
        """Save validation results to JSON file"""
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy_types(self.validation_results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Validation results saved to {output_file}")
    
    def assert_pipeline_ready(self):
        """Assert that the pipeline is ready for training (fail fast)"""
        if not self.validation_results:
            raise TargetValidationError("No validation results available. Run validate_all_files() first.")
        
        if self.validation_results['pass_rate'] < 0.95:
            raise TargetValidationError(
                f"Pipeline validation failed: Only {self.validation_results['pass_rate']:.1%} of files passed validation. "
                f"Failed files: {self.validation_results['failed_file_list']}"
            )
        
        avg_correlation = self.validation_results.get('avg_correlation')
        if avg_correlation is not None and avg_correlation < 0.7:
            raise TargetValidationError(
                f"Pipeline validation failed: Average correlation between direction and target_1d is too low: "
                f"{avg_correlation:.3f}. Expected > 0.7"
            )
        
        logger.info("✅ Pipeline validation passed - ready for training")

def validate_intraday_data():
    """Validate intraday data structure in S3"""
    import boto3
    
    s3 = boto3.client('s3')
    bucket = "hpo-bucket-773934887314"
    
    logger.info("🔍 Validating intraday data structure in S3...")
    
    required_timeframes = ['5min', '10min', '60min']
    symbols_found = set()
    validation_results = {
        'symbols_with_complete_data': [],
        'symbols_with_missing_data': [],
        'total_symbols_found': 0,
        'validation_passed': False
    }
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix='intraday/')
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.csv'):
                        parts = key.split('/')
                        if len(parts) >= 4:
                            symbol = parts[1]
                            timeframe = parts[2]
                            symbols_found.add(symbol)
        
        logger.info(f"📊 Found {len(symbols_found)} symbols with intraday data")
        
        for symbol in symbols_found:
            complete_data = True
            for timeframe in required_timeframes:
                prefix = f"intraday/{symbol}/{timeframe}/"
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                if 'Contents' not in response or len(response['Contents']) == 0:
                    complete_data = False
                    break
            
            if complete_data:
                validation_results['symbols_with_complete_data'].append(symbol)
            else:
                validation_results['symbols_with_missing_data'].append(symbol)
        
        validation_results['total_symbols_found'] = len(symbols_found)
        validation_results['validation_passed'] = len(validation_results['symbols_with_missing_data']) == 0
        
        if validation_results['validation_passed']:
            logger.info("✅ Intraday data validation passed")
        else:
            logger.warning(f"⚠️ Intraday data validation failed. Missing data for: {validation_results['symbols_with_missing_data']}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ Intraday data validation failed: {e}")
        return {'validation_passed': False, 'error': str(e)}

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated target validation for ML pipeline")
    parser.add_argument("--input-dir", default="data/sagemaker_input/46_models/2025-07-02-03-05-02", help="Directory containing CSV files to validate")
    parser.add_argument("--output-file", default="target_validation_results.json", help="Output JSON file for results")
    parser.add_argument("--min-positive-ratio", type=float, default=0.3, help="Minimum acceptable positive ratio")
    parser.add_argument("--max-positive-ratio", type=float, default=0.7, help="Maximum acceptable positive ratio")
    parser.add_argument("--fail-fast", action="store_true", help="Exit with error code if validation fails")
    parser.add_argument("--intraday", action="store_true", help="Validate intraday data structure in S3")
    
    args = parser.parse_args()
    
    try:
        if args.intraday:
            results = validate_intraday_data()
            
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n{'='*60}")
            print("INTRADAY DATA VALIDATION SUMMARY")
            print(f"{'='*60}")
            print(f"Total symbols found: {results.get('total_symbols_found', 0)}")
            print(f"Symbols with complete data: {len(results.get('symbols_with_complete_data', []))}")
            print(f"Symbols with missing data: {len(results.get('symbols_with_missing_data', []))}")
            
            if results.get('symbols_with_missing_data'):
                print(f"Missing data for: {', '.join(results['symbols_with_missing_data'])}")
            
            if results.get('validation_passed'):
                print("✅ Intraday validation completed successfully")
            else:
                print("❌ Intraday validation failed")
                if args.fail_fast:
                    sys.exit(1)
        else:
            validator = AutomatedTargetValidator(
                data_dir=args.input_dir,
                min_positive_ratio=args.min_positive_ratio,
                max_positive_ratio=args.max_positive_ratio
            )
            
            results = validator.validate_all_files()
            
            validator.save_results(args.output_file)
            
            print(f"\n{'='*60}")
            print("AUTOMATED TARGET VALIDATION SUMMARY")
            print(f"{'='*60}")
            print(f"Total files: {results['total_files']}")
            print(f"Passed: {results['passed_files']}")
            print(f"Failed: {results['failed_files']}")
            print(f"Pass rate: {results['pass_rate']:.1%}")
            
            if results['failed_files'] > 0:
                print(f"Failed files: {', '.join(results['failed_file_list'])}")
            
            if results.get('avg_positive_ratio'):
                print(f"Average positive ratio: {results['avg_positive_ratio']:.3f}")
            
            if results.get('avg_correlation'):
                print(f"Average correlation: {results['avg_correlation']:.3f}")
            
            if args.fail_fast:
                validator.assert_pipeline_ready()
            
            print("✅ Validation completed successfully")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if args.fail_fast:
            sys.exit(1)
        else:
            print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    main()
