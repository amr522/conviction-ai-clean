#!/usr/bin/env python3
"""
Prediction script for the stacked model endpoint.

This script takes an input CSV file and makes predictions using
the stacked model endpoint deployed by stacking_pipeline.sh.
"""

import os
import sys
import json
import argparse
import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with stacked model endpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with features')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file for predictions')
    parser.add_argument('--endpoint', type=str,
                        help='SageMaker endpoint name (if not specified, reads from stacked_endpoint_info.json)')
    parser.add_argument('--region', type=str,
                        help='AWS region (defaults to AWS_REGION env var)')
    
    return parser.parse_args()

def get_endpoint_name(args):
    """Get the endpoint name from args or from the stacked_endpoint_info.json file."""
    if args.endpoint:
        return args.endpoint
    
    # Try to read from endpoint info file
    try:
        with open('stacked_endpoint_info.json', 'r') as f:
            endpoint_info = json.load(f)
            return endpoint_info['endpoint_name']
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading endpoint info: {e}")
        print("Please provide --endpoint or ensure stacked_endpoint_info.json exists.")
        sys.exit(1)

def get_region(args):
    """Get the AWS region from args or environment."""
    if args.region:
        return args.region
    
    region = os.environ.get('AWS_REGION')
    if not region:
        print("Error: AWS region not specified. Use --region or set AWS_REGION env var.")
        sys.exit(1)
    
    return region

def make_predictions(endpoint_name, input_data, region):
    """Make predictions using the SageMaker endpoint."""
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    try:
        # Convert DataFrame to CSV string (what the LightGBM container expects)
        csv_data = input_data.to_csv(index=False)
        
        # Make prediction request
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=csv_data
        )
        
        # Parse response
        result = response['Body'].read().decode('utf-8')
        predictions = np.fromstring(result, sep=',')
        
        return predictions
    
    except ClientError as e:
        print(f"Error invoking endpoint: {e}")
        sys.exit(1)

def main():
    """Main function to make predictions with stacked model."""
    args = parse_args()
    endpoint_name = get_endpoint_name(args)
    region = get_region(args)
    
    print(f"Loading input data from {args.input}...")
    try:
        input_data = pd.read_csv(args.input)
        print(f"Loaded {len(input_data)} rows with {len(input_data.columns)} columns")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    print(f"Making predictions with endpoint {endpoint_name}...")
    predictions = make_predictions(endpoint_name, input_data, region)
    
    # Add predictions to the input data
    input_data['prediction'] = predictions
    
    print(f"Saving predictions to {args.output}...")
    input_data.to_csv(args.output, index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
