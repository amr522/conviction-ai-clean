#!/usr/bin/env python3
"""
Data drift monitoring using Evidently
"""
import os
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

def setup_logging():
    """Setup logging configuration"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/evidently_log.txt'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_reference_data(reference_path="datasets/processed/reference.parquet", use_delta=False, timestamp_as_of=None):
    """Load reference dataset for drift comparison with Delta Lake support"""
    if use_delta:
        # Try Delta Lake format first
        delta_path = reference_path.replace('.parquet', '.delta')
        try:
            from utils.delta_writer import read_delta_table
            df = read_delta_table(delta_path, timestamp_as_of=timestamp_as_of)
            if df is not None:
                return df
        except Exception as e:
            logger.warning(f"Failed to read Delta reference data: {e}")
    
    # Fallback to Parquet
    if os.path.exists(reference_path):
        return pd.read_parquet(reference_path)
    
    # Fallback: use data from 30 days ago as reference
    ref_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    fallback_path = f"datasets/processed/{ref_date}.parquet"
    
    if os.path.exists(fallback_path):
        return pd.read_parquet(fallback_path)
    
    raise FileNotFoundError(f"No reference data found at {reference_path} or {fallback_path}")

def generate_drift_report(current_data, reference_data, start_date, output_dir="metrics"):
    """Generate Evidently data drift report"""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Evidently report
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric()
    ])
    
    # Run drift analysis
    logger.info(f"Running drift analysis for {start_date}")
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save HTML report
    report_path = f"{output_dir}/data_drift_report_{start_date}.html"
    report.save_html(report_path)
    logger.info(f"Drift report saved to {report_path}")
    
    # Extract drift results
    report_dict = report.as_dict()
    dataset_drift = report_dict['metrics'][1]['result']['dataset_drift']
    
    # Log drift status
    if dataset_drift:
        logger.warning(f"Drift detected: True for {start_date}")
    else:
        logger.info(f"Drift detected: False for {start_date}")
    
    return report_path, dataset_drift

def monitor_data_drift(start_date, current_data_path=None):
    """Main function to monitor data drift"""
    logger = setup_logging()
    
    try:
        # Load current data
        if current_data_path is None:
            current_data_path = f"datasets/processed/{start_date}.parquet"
        
        if not os.path.exists(current_data_path):
            logger.error(f"Current data not found: {current_data_path}")
            return False, None
        
        current_data = pd.read_parquet(current_data_path)
        logger.info(f"Loaded current data: {current_data.shape}")
        
        # Load reference data
        reference_data = load_reference_data()
        logger.info(f"Loaded reference data: {reference_data.shape}")
        
        # Ensure same columns
        common_cols = list(set(current_data.columns) & set(reference_data.columns))
        current_data = current_data[common_cols]
        reference_data = reference_data[common_cols]
        
        # Generate drift report
        report_path, drift_detected = generate_drift_report(
            current_data, reference_data, start_date
        )
        
        return drift_detected, report_path
        
    except Exception as e:
        logger.error(f"Error in drift monitoring: {str(e)}")
        return False, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor data drift using Evidently")
    parser.add_argument("--start-date", required=True, help="Date to analyze (YYYY-MM-DD)")
    parser.add_argument("--current-data", help="Path to current data file")
    
    args = parser.parse_args()
    
    drift_detected, report_path = monitor_data_drift(args.start_date, args.current_data)
    
    if drift_detected:
        print(f"⚠️ Data drift detected for {args.start_date}")
        exit(1)
    else:
        print(f"✅ No data drift detected for {args.start_date}")
        exit(0)