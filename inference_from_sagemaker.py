#!/usr/bin/env python3
"""
inference_from_sagemaker.py - Script to perform inference using a deployed SageMaker endpoint

This script sends data to a SageMaker endpoint for inference and processes the results.
"""

import argparse
import boto3
import json
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Perform inference using a SageMaker endpoint')
    
    parser.add_argument('--endpoint-name', type=str, required=True,
                        help='Name of the SageMaker endpoint to use for inference')
    
    parser.add_argument('--input-file', type=str, default=None,
                        help='Path to input CSV/JSON file with data for inference')
    
    parser.add_argument('--output-file', type=str, default='predictions.csv',
                        help='Path to save inference results (default: predictions.csv)')
    
    parser.add_argument('--content-type', type=str, default='application/json',
                        choices=['application/json', 'text/csv'],
                        help='Content type for the request (default: application/json)')
    
    parser.add_argument('--accept-type', type=str, default='application/json',
                        choices=['application/json', 'text/csv'],
                        help='Accept type for the response (default: application/json)')
    
    parser.add_argument('--sample-size', type=int, default=10,
                        help='Number of sample records to generate if no input file (default: 10)')
    
    parser.add_argument('--region', type=str, default=os.environ.get('AWS_REGION', 'us-east-1'),
                        help='AWS region for SageMaker endpoint (default: from env or us-east-1)')
    
    return parser.parse_args()

def load_input_data(input_file, sample_size):
    """
    Load data from input file or generate sample data if no file provided.
    
    Args:
        input_file: Path to input file (CSV or JSON)
        sample_size: Number of sample records to generate if no input file
        
    Returns:
        List of records for inference
    """
    if input_file:
        logger.info(f"Loading data from {input_file}")
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
            return df.head(sample_size).to_dict('records')
        elif input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                data = json.load(f)
            return data[:sample_size] if isinstance(data, list) else [data]
        else:
            raise ValueError(f"Unsupported file format: {input_file}")
    else:
        # Generate sample data
        logger.info(f"Generating {sample_size} sample records for inference")
        records = []
        for i in range(sample_size):
            # Create a sample record with random features
            record = {
                f'feature_{j}': np.random.normal() 
                for j in range(5)
            }
            records.append(record)
        return records

def invoke_endpoint(endpoint_name, payload, region, content_type='application/json', accept_type='application/json'):
    """
    Invoke a SageMaker endpoint for inference.
    
    Args:
        endpoint_name: Name of the SageMaker endpoint
        payload: Data payload for inference
        region: AWS region
        content_type: Content type for the request
        accept_type: Accept type for the response
        
    Returns:
        Inference results from the endpoint
    """
    try:
        # Create boto3 session with credentials from environment
        session = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=region
        )
        
        # Create SageMaker runtime client
        runtime = session.client('sagemaker-runtime')
        
        # Prepare payload based on content type
        if content_type == 'text/csv':
            if isinstance(payload, list) and isinstance(payload[0], dict):
                # Convert list of dictionaries to CSV string
                df = pd.DataFrame(payload)
                payload_data = df.to_csv(index=False)
            else:
                # Already a CSV string
                payload_data = payload
        else:
            # JSON payload
            payload_data = json.dumps(payload)
        
        # Invoke endpoint
        logger.info(f"Invoking endpoint: {endpoint_name}")
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Accept=accept_type,
            Body=payload_data
        )
        
        # Parse response
        response_body = response['Body'].read().decode('utf-8')
        
        # For CSV responses, try to parse as JSON first
        if accept_type == 'text/csv':
            try:
                result = json.loads(response_body)
            except json.JSONDecodeError:
                # If not valid JSON, keep as CSV string
                result = response_body
        else:
            # JSON response
            try:
                result = json.loads(response_body)
            except json.JSONDecodeError:
                # If not valid JSON, return as is
                result = response_body
        
        return result
        
    except Exception as e:
        logger.error(f"Error invoking endpoint: {str(e)}")
        raise

def main():
    """Main function to run inference."""
    args = parse_arguments()
    
    try:
        # Load or generate input data
        records = load_input_data(args.input_file, args.sample_size)
        
        # Invoke endpoint
        predictions = invoke_endpoint(
            args.endpoint_name, 
            records, 
            args.region,
            content_type=args.content_type,
            accept_type=args.accept_type
        )
        
        # Process and save results
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = args.output_file or f"predictions_{timestamp}.csv"
        
        # Process predictions to extract values
        if isinstance(predictions, str):
            # Try to parse as JSON if it's a string
            try:
                predictions = json.loads(predictions)
            except:
                # If not JSON, split by newlines and try to convert to float
                try:
                    predictions = [float(p.strip()) for p in predictions.split("\n") if p.strip()]
                except:
                    # Keep as is if can't convert
                    predictions = [predictions]
        
        # Handle different prediction formats
        if isinstance(predictions, list):
            if len(predictions) > 0 and isinstance(predictions[0], dict):
                # Extract prediction values from list of dicts
                values = []
                for p in predictions:
                    if isinstance(p, dict):
                        if 'predicted_label' in p:
                            values.append(p['predicted_label'])
                        elif 'prediction' in p:
                            values.append(p['prediction'])
                        else:
                            values.append(list(p.values())[0])
                    else:
                        # Not a dict, use the value directly
                        values.append(p)
            else:
                # Already a list of values
                values = predictions
        else:
            # Single value or dict
            if isinstance(predictions, dict):
                if 'predicted_label' in predictions:
                    values = [predictions['predicted_label']]
                elif 'prediction' in predictions:
                    values = [predictions['prediction']]
                else:
                    values = [list(predictions.values())[0]]
            else:
                values = [predictions]
        
        # Create input dataframe
        df_input = pd.DataFrame(records)
        
        # Make sure values list is same length as input data
        if len(values) == 1 and len(df_input) > 1:
            values = values * len(df_input)  # Replicate the single value
        elif len(values) < len(df_input):
            # Pad with None if we have fewer predictions than inputs
            values = values + [None] * (len(df_input) - len(values))
        elif len(values) > len(df_input):
            # Truncate if we have more predictions than inputs
            values = values[:len(df_input)]
        
        # Add prediction column to input dataframe
        df_input['prediction'] = values
        
        # Save to file
        df_input.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        
        # Print sample predictions
        logger.info(f"Sample predictions (first 5):")
        for i, pred in enumerate(values[:5]):
            logger.info(f"  Record {i}: {pred}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error performing inference: {str(e)}")
        return False

if __name__ == "__main__":
    main()