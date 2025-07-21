#!/usr/bin/env python3
"""
mock_run_sagemaker_autopilot.py - Simulates running SageMaker Autopilot for smoke testing

This script simulates a successful SageMaker Autopilot job run and generates a valid endpoint_info.json
file that can be used for the subsequent steps in the smoke test pipeline.
"""

import argparse
import logging
import json
from datetime import datetime
import os
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mock SageMaker Autopilot for Conviction-AI')
    
    parser.add_argument('--target-column', type=str, default="return",
                        help='Target column to predict (default: return)')
    
    parser.add_argument('--problem-type', type=str, default="Regression",
                        choices=['Regression', 'BinaryClassification', 'MulticlassClassification'],
                        help='Problem type (default: Regression)')
    
    parser.add_argument('--max-candidates', type=int, default=3,
                        help='Maximum number of model candidates (default: 3)')
    
    return parser.parse_args()

def simulate_sagemaker_autopilot(args):
    """
    Simulate a SageMaker Autopilot job run.
    
    This function:
    1. Simulates launching a SageMaker Autopilot job
    2. Simulates waiting for the job to complete
    3. Simulates deploying the best model to an endpoint
    4. Creates a valid endpoint_info.json file
    """
    try:
        # Generate a unique job name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        autopilot_job_name = f"conviction-automl-{timestamp}"
        endpoint_name = f"conviction-ai-endpoint-{timestamp}"
        
        logger.info(f"🚀 Simulating AutoML job: {autopilot_job_name}")
        
        # Simulate job creation
        logger.info(f"AutoML job created: arn:aws:sagemaker:us-east-1:123456789012:automl-job/{autopilot_job_name}")
        
        # Simulate waiting for the job to complete
        logger.info("Waiting for AutoML job to complete...")
        
        # Simulate different status updates
        status_updates = ["Starting", "InProgress", "InProgress", "InProgress", "Completed"]
        for status in status_updates:
            logger.info(f"Status: {status}")
            time.sleep(2)  # Brief pause to simulate time passing
        
        # Simulate getting the best candidate
        logger.info("AutoML job completed. Finding best candidate...")
        
        # Generate random metric value based on problem type
        if args.problem_type == "Regression":
            metric_name = "RMSE"
            metric_value = round(random.uniform(0.01, 0.5), 4)
        else:
            metric_name = "F1"
            metric_value = round(random.uniform(0.7, 0.95), 4)
            
        candidate_name = f"candidate-{random.randint(0, args.max_candidates-1)}"
        
        logger.info(f"🎯 Best candidate: {candidate_name} with {metric_name}={metric_value:.4f}")
        
        # Simulate model deployment
        logger.info(f"Deploying model to endpoint: {endpoint_name}")
        
        # Simulate endpoint creation and deployment
        logger.info("Waiting for endpoint deployment to complete...")
        
        # Simulate endpoint status updates
        endpoint_status_updates = ["Creating", "Creating", "InService"]
        for status in endpoint_status_updates:
            logger.info(f"Endpoint status: {status}")
            time.sleep(2)  # Brief pause to simulate time passing
        
        logger.info(f"✅ Deployed endpoint: {endpoint_name}")
        
        # Save endpoint information for future reference
        endpoint_info = {
            "job_name": autopilot_job_name,
            "endpoint_name": endpoint_name,
            "best_candidate": candidate_name,
            "metric_value": float(metric_value),
            "problem_type": args.problem_type,
            "target_column": args.target_column,
            "created_at": datetime.now().isoformat()
        }
        
        # Save to project root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        endpoint_info_path = os.path.join(root_dir, "endpoint_info.json")
        
        with open(endpoint_info_path, "w") as f:
            json.dump(endpoint_info, f, indent=2)
        
        logger.info(f"Endpoint information saved to {endpoint_info_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in mock SageMaker AutoML process: {str(e)}")
        return False

if __name__ == "__main__":
    args = parse_arguments()
    
    logger.info(f"Starting mock SageMaker Autopilot run with target column: {args.target_column}")
    logger.info(f"Problem type: {args.problem_type}")
    logger.info(f"Max candidates: {args.max_candidates}")
    
    success = simulate_sagemaker_autopilot(args)
    if success:
        logger.info("Mock SageMaker AutoML process completed successfully")
        exit(0)  # Successful exit
    else:
        logger.error("Mock SageMaker AutoML process failed")
        exit(1)  # Error exit
