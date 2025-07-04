#!/usr/bin/env python3
"""
Deploy best model from HPO job to SageMaker endpoint
"""
import os
import argparse
import boto3
import tarfile
import tempfile
from datetime import datetime

class BestModelDeployer:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = 'hpo-bucket-773934887314'
        
    def create_inference_script(self, temp_dir):
        """Create inference.py script for XGBoost models"""
        inference_script = '''
import joblib
import numpy as np
import json
import os
import xgboost as xgb

def model_fn(model_dir):
    """Load the XGBoost model"""
    model_path = os.path.join(model_dir, 'xgboost-model')
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'text/csv':
        import io
        import pandas as pd
        input_data = pd.read_csv(io.StringIO(request_body), header=None)
        return xgb.DMatrix(input_data.values)
    elif request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return xgb.DMatrix(np.array(input_data['instances']))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    predictions = model.predict(input_data)
    return predictions.tolist()

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == 'application/json':
        return json.dumps({'predictions': prediction})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
        
        inference_path = os.path.join(temp_dir, 'inference.py')
        with open(inference_path, 'w') as f:
            f.write(inference_script)
        return inference_path
    
    def deploy_model(self, model_artifact_s3, endpoint_name, role_arn=None):
        """Deploy model to SageMaker endpoint"""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        if not role_arn:
            role_arn = 'arn:aws:iam::773934887314:role/SageMakerExecutionRole'
        
        model_name = f"best-model-{timestamp}"
        config_name = f"best-config-{timestamp}"
        
        try:
            response = self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1',
                    'ModelDataUrl': model_artifact_s3,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                    }
                },
                ExecutionRoleArn=role_arn
            )
            
            print(f"✅ Model created: {response['ModelArn']}")
            
            config_response = self.sagemaker.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.t2.medium',
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            print(f"✅ Endpoint config created: {config_response['EndpointConfigArn']}")
            
            try:
                self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
                endpoint_response = self.sagemaker.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=config_name
                )
                print(f"✅ Endpoint updated: {endpoint_response['EndpointArn']}")
            except self.sagemaker.exceptions.ClientError as e:
                if 'does not exist' in str(e) or 'Could not find endpoint' in str(e):
                    endpoint_response = self.sagemaker.create_endpoint(
                        EndpointName=endpoint_name,
                        EndpointConfigName=config_name
                    )
                    print(f"✅ Endpoint created: {endpoint_response['EndpointArn']}")
                else:
                    raise
            
            return {
                'endpoint_name': endpoint_name,
                'model_name': model_name,
                'config_name': config_name
            }
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Deploy best model to SageMaker')
    parser.add_argument('--model-artifact', type=str, required=True,
                        help='S3 URI to model artifact (model.tar.gz)')
    parser.add_argument('--endpoint-name', type=str, required=True,
                        help='Name for SageMaker endpoint')
    parser.add_argument('--role-arn', type=str,
                        help='SageMaker execution role ARN')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode without making actual deployments')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🧪 DRY RUN MODE - No SageMaker calls will be made")
        print(f"✅ DRY RUN: Would deploy model from: {args.model_artifact}")
        print(f"✅ DRY RUN: Would create/update endpoint: {args.endpoint_name}")
        print(f"✅ DRY RUN: Would use role: {args.role_arn or 'arn:aws:iam::773934887314:role/SageMakerExecutionRole'}")
        return
    
    deployer = BestModelDeployer()
    
    try:
        result = deployer.deploy_model(args.model_artifact, args.endpoint_name, args.role_arn)
        print("✅ Deployment completed successfully!")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
