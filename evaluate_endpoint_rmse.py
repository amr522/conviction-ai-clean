#!/usr/bin/env python3
"""
evaluate_endpoint_rmse.py - Evaluate a SageMaker endpoint's performance using hold-out data

This script loads hold-out data, sends it to a SageMaker endpoint for prediction,
and calculates performance metrics (RMSE, MAE, R²).
"""

import argparse
import boto3
import json
import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a SageMaker endpoint using hold-out data')
    
    parser.add_argument('--endpoint-name', type=str, required=True,
                        help='Name of the SageMaker endpoint to evaluate')
    
    parser.add_argument('--holdout-csv', type=str, required=True,
                        help='Path to the CSV file containing hold-out data with target column')
    
    parser.add_argument('--target-col', type=str, required=True,
                        help='Name of the target column in the hold-out data')
    
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for sending predictions to the endpoint (default: 50)')
    
    parser.add_argument('--region', type=str, default=os.environ.get('AWS_REGION', 'us-east-1'),
                        help='AWS region for SageMaker endpoint (default: from env or us-east-1)')
    
    return parser.parse_args()

def invoke_endpoint_batch(endpoint_name, records, region):
    """
    Invoke a SageMaker endpoint for a batch of records.
    
    Args:
        endpoint_name: Name of the SageMaker endpoint
        records: List of records for inference
        region: AWS region
        
    Returns:
        List of predictions
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
        
        # Prepare payload
        payload_data = pd.DataFrame(records).to_csv(index=False)
        
        # Invoke endpoint
        logger.debug(f"Invoking endpoint for batch of {len(records)} records")
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Accept='text/csv',
            Body=payload_data
        )
        
        # Parse response
        response_body = response['Body'].read().decode('utf-8')
        
        # Try to parse response as list of numbers
        try:
            # First try to parse as JSON
            try:
                result = json.loads(response_body)
                if isinstance(result, list):
                    # Extract values if it's a list of dicts
                    if len(result) > 0 and isinstance(result[0], dict):
                        values = []
                        for p in result:
                            if isinstance(p, dict):
                                if 'predicted_label' in p:
                                    values.append(p['predicted_label'])
                                elif 'prediction' in p:
                                    values.append(p['prediction'])
                                else:
                                    values.append(list(p.values())[0])
                            else:
                                values.append(p)
                        return values
                    else:
                        return result
                else:
                    # Single prediction or dict
                    if isinstance(result, dict):
                        if 'predicted_label' in result:
                            return [result['predicted_label']]
                        elif 'prediction' in result:
                            return [result['prediction']]
                        else:
                            return [list(result.values())[0]]
                    else:
                        return [result]
            except json.JSONDecodeError:
                # If not JSON, try to parse as CSV
                predictions = []
                for line in response_body.strip().split('\n'):
                    if line.strip():
                        try:
                            predictions.append(float(line.strip()))
                        except ValueError:
                            # If not a number, skip
                            pass
                return predictions
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            logger.error(f"Response body: {response_body}")
            raise
        
    except Exception as e:
        logger.error(f"Error invoking endpoint: {str(e)}")
        raise

def evaluate_endpoint(args):
    """
    Evaluate a SageMaker endpoint using hold-out data.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load hold-out data
    logger.info(f"Loading hold-out data from {args.holdout_csv}")
    try:
        df = pd.read_csv(args.holdout_csv)
    except Exception as e:
        logger.error(f"Error loading hold-out data: {str(e)}")
        raise
    
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Check if target column exists
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in hold-out data")
    
    # Extract target values
    y_true = df[args.target_col].values
    
    # Remove target column from features
    X = df.drop(columns=[args.target_col])
    
    # Prepare for batched prediction
    logger.info(f"Sending data to endpoint {args.endpoint_name} in batches of {args.batch_size}")
    predictions = []
    
    # Process in batches
    for i in range(0, len(X), args.batch_size):
        batch = X.iloc[i:i+args.batch_size].to_dict('records')
        batch_predictions = invoke_endpoint_batch(args.endpoint_name, batch, args.region)
        predictions.extend(batch_predictions)
        logger.debug(f"Processed batch {i//args.batch_size + 1}/{(len(X) + args.batch_size - 1)//args.batch_size}")
    
    # Ensure predictions match the number of records
    if len(predictions) != len(X):
        logger.warning(f"Number of predictions ({len(predictions)}) doesn't match number of records ({len(X)})")
        # Adjust predictions to match number of records
        if len(predictions) < len(X):
            predictions.extend([None] * (len(X) - len(predictions)))
        else:
            predictions = predictions[:len(X)]
    
    # Calculate metrics
    y_pred = np.array(predictions, dtype=float)
    valid_indices = ~np.isnan(y_pred)
    
    if not any(valid_indices):
        logger.error("No valid predictions to evaluate")
        return {
            'rmse': None,
            'mae': None,
            'r2': None
        }
    
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    
    rmse = sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    r2 = r2_score(y_true_valid, y_pred_valid)
    
    # Add predictions to original dataframe
    df['prediction'] = predictions
    
    # Save augmented data
    output_file = 'predictions_with_labels.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Saved predictions with labels to {output_file}")
    
    # Return metrics
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        metrics = evaluate_endpoint(args)
        
        # Print metrics
        print(f"\nEvaluation Metrics:")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE:  {metrics['mae']:.6f}")
        print(f"R²:   {metrics['r2']:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error evaluating endpoint: {str(e)}")
        return False

if __name__ == "__main__":
    main()
