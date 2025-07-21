#!/usr/bin/env python3
"""
mock_glue_etl_script.py - Simulates running the Glue ETL job locally for smoke testing

This script simulates a successful ETL job run and generates mock outputs to test
the end-to-end pipeline.
"""

import argparse
import logging
import time
import os
import json
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mock Glue ETL job for Conviction-AI')
    
    parser.add_argument('--JOB_NAME', type=str, default="dry_run",
                        help='Name of the Glue job')
    
    parser.add_argument('--raw-prefix', type=str, default="s3://convictionai-data/conviction-ai/raw/",
                        help='S3 prefix for raw data')
    
    parser.add_argument('--output-path', type=str, default="s3://convictionai-data/conviction-ai/clean/train_dataset_dry/",
                        help='S3 output path for processed data')
    
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Number of sample rows to process')
    
    parser.add_argument('--null-threshold', type=float, default=0.2,
                        help='Maximum allowed null percentage')
    
    parser.add_argument('--validate-schema', action='store_true',
                        help='Validate the schema of the data')
    
    return parser.parse_args()

def simulate_etl_job(args):
    """
    Simulate a Glue ETL job run.
    
    This function:
    1. Simulates reading data from S3
    2. Simulates processing the data
    3. Simulates writing the results back to S3
    """
    try:
        logger.info(f"🚀 Starting ETL job: {args.JOB_NAME}")
        logger.info(f"Reading data from: {args.raw_prefix}")
        logger.info(f"Sample size: {args.sample_size}")
        logger.info(f"Null threshold: {args.null_threshold}")
        
        # Simulate reading data
        logger.info("Reading data sources...")
        time.sleep(1)
        
        # List of mock data sources
        data_sources = [
            "stocks_daily", 
            "options_daily", 
            "forex_data", 
            "vix_data", 
            "news_sentiment"
        ]
        
        # Simulate processing each data source
        for source in data_sources:
            logger.info(f"Processing {source} data...")
            # Simulate data processing time
            time.sleep(0.5)
            
            # Generate random statistics
            rows_processed = random.randint(100, 500)
            null_percentage = random.uniform(0.0, args.null_threshold * 0.9)  # Keep below threshold
            
            logger.info(f"  - Processed {rows_processed} rows")
            logger.info(f"  - Null percentage: {null_percentage:.4f}")
            
            if args.validate_schema:
                logger.info(f"  - Schema validation passed for {source}")
        
        # Simulate data merging
        logger.info("Merging all data sources...")
        time.sleep(1)
        
        # Simulate writing output
        logger.info(f"Writing processed data to: {args.output_path}")
        time.sleep(1)
        
        # Simulate data quality checks
        logger.info("Running data quality checks...")
        time.sleep(1)
        
        logger.info("✅ Data quality checks passed")
        
        # Create a mock stats file to track what was done
        stats = {
            "job_name": args.JOB_NAME,
            "raw_prefix": args.raw_prefix,
            "output_path": args.output_path,
            "sample_size": args.sample_size,
            "null_threshold": args.null_threshold,
            "validate_schema": args.validate_schema,
            "data_sources_processed": data_sources,
            "total_rows_processed": random.randint(500, 2000),
            "execution_time_seconds": random.randint(10, 30),
            "success": True
        }
        
        # Save to project root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stats_path = os.path.join(root_dir, "etl_stats.json")
        
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"ETL statistics saved to {stats_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in mock ETL job: {str(e)}")
        return False

if __name__ == "__main__":
    args = parse_arguments()
    
    success = simulate_etl_job(args)
    if success:
        logger.info("Mock ETL job completed successfully")
        exit(0)  # Successful exit
    else:
        logger.error("Mock ETL job failed")
        exit(1)  # Error exit
