#!/usr/bin/env python3
"""
mock_inference_from_sagemaker.py - Simulates invoking a SageMaker endpoint for smoke testing

This script simulates a successful SageMaker endpoint invocation and generates mock
predictions to test the end-to-end pipeline.
"""

import argparse
import logging
import json
import os
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mock SageMaker inference for Conviction-AI')
    
    parser.add_argument('--endpoint-name', type=str, required=True,
                        help='Name of the SageMaker endpoint to invoke')
    
    parser.add_argument('--sample-size', type=int, default=3,
                        help='Number of sample predictions to generate')
    
    return parser.parse_args()

def generate_mock_data(sample_size):
    """Generate mock financial data for inference."""
    # Generate dates (trading days)
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(sample_size)]
    
    # Generate feature data (simplified for mock purposes)
    mock_data = []
    for date in dates:
        # Create a row with some financial features
        row = {
            'date': date,
            'open': round(random.uniform(100, 200), 2),
            'high': round(random.uniform(100, 200), 2),
            'low': round(random.uniform(100, 200), 2),
            'close': round(random.uniform(100, 200), 2),
            'volume': int(random.uniform(1000000, 5000000)),
            'vix': round(random.uniform(15, 30), 2),
            'sentiment_score': round(random.uniform(-1, 1), 2)
        }
        mock_data.append(row)
    
    return mock_data

def simulate_inference(args):
    """
    Simulate invoking a SageMaker endpoint.
    
    This function:
    1. Generates mock input data
    2. Simulates sending the data to the endpoint
    3. Generates mock prediction results
    """
    try:
        # Check if the endpoint exists in endpoint_info.json
        endpoint_info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "endpoint_info.json")
        
        if os.path.exists(endpoint_info_path):
            with open(endpoint_info_path, "r") as f:
                endpoint_info = json.load(f)
                
            # If endpoint name wasn't provided or doesn't match, use the one from endpoint_info.json
            if args.endpoint_name == "returned_from_step_2" or args.endpoint_name != endpoint_info.get("endpoint_name"):
                args.endpoint_name = endpoint_info.get("endpoint_name")
                logger.info(f"Using endpoint name from endpoint_info.json: {args.endpoint_name}")
        
        if not args.endpoint_name:
            logger.error("No valid endpoint name provided")
            return False
        
        logger.info(f"🚀 Invoking SageMaker endpoint: {args.endpoint_name}")
        
        # Generate mock input data
        logger.info(f"Generating {args.sample_size} sample records for inference...")
        mock_data = generate_mock_data(args.sample_size)
        
        # Display the mock input data
        logger.info("Sample input data:")
        for i, data in enumerate(mock_data):
            logger.info(f"Sample {i+1}: {data}")
        
        # Simulate sending data to endpoint
        logger.info("Sending data to SageMaker endpoint...")
        
        # Generate mock predictions (for regression, return values)
        if "endpoint_info" in locals() and endpoint_info.get("problem_type") == "Regression":
            # For regression, generate continuous values
            predictions = [round(random.normalvariate(0.02, 0.05), 4) for _ in range(args.sample_size)]
        else:
            # Default to regression predictions
            predictions = [round(random.normalvariate(0.02, 0.05), 4) for _ in range(args.sample_size)]
        
        # Display the predictions
        logger.info("Predictions received from endpoint:")
        for i, pred in enumerate(predictions):
            logger.info(f"  Sample {i+1}: {pred}")
        
        # Combine input data with predictions
        for i, data_point in enumerate(mock_data):
            data_point["prediction"] = predictions[i]
        
        # Save predictions to file
        results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference_results.json")
        
        results = {
            "endpoint_name": args.endpoint_name,
            "sample_size": args.sample_size,
            "timestamp": datetime.now().isoformat(),
            "predictions": [{"input": data_point, "prediction": data_point["prediction"]} for data_point in mock_data]
        }
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Inference results saved to {results_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in mock inference process: {str(e)}")
        return False

if __name__ == "__main__":
    args = parse_arguments()
    
    success = simulate_inference(args)
    if success:
        logger.info("Mock inference completed successfully")
        exit(0)  # Successful exit
    else:
        logger.error("Mock inference failed")
        exit(1)  # Error exit
