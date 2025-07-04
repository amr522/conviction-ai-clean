#!/usr/bin/env python3
"""Deploy the best model from pinned HPO artifacts to SageMaker endpoint"""

import os
import sys
import json
import argparse
import tempfile
import tarfile
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aws_utils import AWSClientManager, safe_aws_operation, load_pinned_config, get_sagemaker_execution_role, format_resource_name, validate_iam_permissions

def create_xgboost_inference_script():
    """Create inference script for XGBoost model"""
    return '''
import json
import pickle
import numpy as np
import xgboost as xgb

def model_fn(model_dir):
    """Load the XGBoost model"""
    model_path = os.path.join(model_dir, "xgboost-model")
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == "text/csv":
        return np.array([float(x) for x in request_body.strip().split(",")])
    elif request_content_type == "application/json":
        input_data = json.loads(request_body)
        return np.array(input_data["instances"])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    dmatrix = xgb.DMatrix(input_data.reshape(1, -1))
    prediction = model.predict(dmatrix)
    return prediction[0]

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == "application/json":
        return json.dumps({"prediction": float(prediction)})
    else:
        return str(prediction)
'''

def deploy_xgboost_model(model_file, endpoint_name, dry_run=False):
    """Deploy XGBoost model to SageMaker endpoint with dry-run support"""
    print(f"🚀 Deploying XGBoost model to endpoint: {endpoint_name}")
    
    required_permissions = [
        "sagemaker:CreateModel",
        "sagemaker:CreateEndpointConfig", 
        "sagemaker:CreateEndpoint",
        "s3:PutObject",
        "s3:GetObject"
    ]
    validate_iam_permissions(required_permissions, dry_run)
    
    aws_manager = AWSClientManager()
    bucket = 'hpo-bucket-773934887314'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    if dry_run:
        print(f"🔍 [DRY-RUN] Would deploy model from: {model_file}")
        print(f"🔍 [DRY-RUN] Would upload to S3: s3://{bucket}/models/best-hpo/{timestamp}/model.tar.gz")
        print(f"🔍 [DRY-RUN] Would create SageMaker model: best-hpo-model-{timestamp}")
        print(f"🔍 [DRY-RUN] Would create endpoint config: best-hpo-config-{timestamp}")
        print(f"🔍 [DRY-RUN] Would create endpoint: {endpoint_name}")
        return {
            'dry_run': True,
            'endpoint_name': endpoint_name,
            'model_name': f"best-hpo-model-{timestamp}",
            'config_name': f"best-hpo-config-{timestamp}",
            'model_uri': f"s3://{bucket}/models/best-hpo/{timestamp}/model.tar.gz"
        }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(model_file, 'r:gz') as tar:
            tar.extractall(temp_dir, filter='data')
        
        inference_script = create_xgboost_inference_script()
        with open(os.path.join(temp_dir, 'inference.py'), 'w') as f:
            f.write(inference_script)
        
        model_tar_path = os.path.join(temp_dir, 'model.tar.gz')
        with tarfile.open(model_tar_path, 'w:gz') as tar:
            tar.add(temp_dir, arcname='.')
        
        s3_key = f"models/best-hpo/{timestamp}/model.tar.gz"
        safe_aws_operation(
            "S3 Upload",
            aws_manager.s3.upload_file,
            dry_run=False,
            Filename=model_tar_path,
            Bucket=bucket,
            Key=s3_key
        )
        model_uri = f"s3://{bucket}/{s3_key}"
        print(f"📤 Model uploaded to: {model_uri}")
        
        model_name = format_resource_name("best-hpo-model", timestamp)
        config_name = format_resource_name("best-hpo-config", timestamp)
        execution_role = get_sagemaker_execution_role()
        
        safe_aws_operation(
            "Create SageMaker Model",
            aws_manager.sagemaker.create_model,
            dry_run=False,
            ModelName=model_name,
            PrimaryContainer={
                'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3',
                'ModelDataUrl': model_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                }
            },
            ExecutionRoleArn=execution_role
        )
        
        safe_aws_operation(
            "Create Endpoint Configuration",
            aws_manager.sagemaker.create_endpoint_config,
            dry_run=False,
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
        
        safe_aws_operation(
            "Create Endpoint",
            aws_manager.sagemaker.create_endpoint,
            dry_run=False,
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        print(f"✅ Endpoint creation initiated: {endpoint_name}")
        print("⏳ Endpoint is being created... This may take several minutes.")
        
        return {
            'endpoint_name': endpoint_name,
            'model_name': model_name,
            'config_name': config_name,
            'model_uri': model_uri
        }

def extract_and_deploy_pinned_model(endpoint_name="conviction-best-model", dry_run=False):
    """Extract and deploy the pinned best model"""
    pinned_dir = "models/pinned_successful_hpo"
    model_file = None
    
    for file in os.listdir(pinned_dir):
        if file.endswith(".tar.gz") and "best_model" in file:
            model_file = os.path.join(pinned_dir, file)
            break
    
    if not model_file or not os.path.exists(model_file):
        print("❌ Best model artifact not found in pinned directory")
        return None
    
    config = load_pinned_config()
    
    print(f"🚀 Deploying best model from HPO job: {config['successful_hpo_job']}")
    print(f"📊 Model achieved validation AUC: {config['validation_auc']}")
    
    aws_manager = AWSClientManager()
    try:
        existing_endpoint = aws_manager.sagemaker.describe_endpoint(EndpointName=endpoint_name)
        if not dry_run:
            print(f"⚠️ Endpoint {endpoint_name} already exists with status: {existing_endpoint['EndpointStatus']}")
            response = input("Do you want to update the existing endpoint? (y/N): ")
            if response.lower() != 'y':
                print("❌ Deployment cancelled by user")
                return None
    except aws_manager.sagemaker.exceptions.ClientError:
        pass
    
    result = deploy_xgboost_model(model_file, endpoint_name, dry_run)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy best HPO model to SageMaker endpoint')
    parser.add_argument('--endpoint-name', default='conviction-best-model', help='Endpoint name')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deployed without creating resources')
    args = parser.parse_args()
    
    result = extract_and_deploy_pinned_model(args.endpoint_name, args.dry_run)
    if result:
        if args.dry_run:
            print("✅ Dry-run completed successfully!")
        else:
            print("✅ Best model deployment completed successfully!")
    else:
        print("❌ Deployment failed")
        sys.exit(1)
