#!/usr/bin/env python3
"""
Deploy ensemble model to AWS SageMaker endpoint
"""
import os
import argparse
import boto3
import joblib
import tarfile
import tempfile
from datetime import datetime
import json

class EnsembleDeployer:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = 'hpo-bucket-773934887314'
        
    def package_model(self, model_path, output_dir):
        """Package ensemble model for SageMaker deployment"""
        print(f"📦 Packaging model from {model_path}")
        
        ensemble_data = joblib.load(model_path)
        print(f"✅ Loaded ensemble with {ensemble_data['num_base_models']} base models")
        
        package_dir = os.path.join(output_dir, 'model_package')
        os.makedirs(package_dir, exist_ok=True)
        
        inference_script = f"""
import joblib
import numpy as np
import json
import os

def model_fn(model_dir):
    '''Load the ensemble model'''
    model_path = os.path.join(model_dir, 'ensemble_model.pkl')
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    '''Parse input data'''
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data['instances'])
    else:
        raise ValueError(f"Unsupported content type: {{request_content_type}}")

def predict_fn(input_data, model):
    '''Make predictions'''
    ensemble_model = model['ensemble_model']
    
    predictions = ensemble_model.predict_proba(input_data)[:, 1]
    return predictions.tolist()

def output_fn(prediction, content_type):
    '''Format output'''
    if content_type == 'application/json':
        return json.dumps({{'predictions': prediction}})
    else:
        raise ValueError(f"Unsupported content type: {{content_type}}")
"""
        
        with open(os.path.join(package_dir, 'inference.py'), 'w') as f:
            f.write(inference_script)
        
        import shutil
        shutil.copy(model_path, os.path.join(package_dir, 'ensemble_model.pkl'))
        
        model_tar_path = os.path.join(output_dir, 'model.tar.gz')
        with tarfile.open(model_tar_path, 'w:gz') as tar:
            tar.add(package_dir, arcname='.')
        
        print(f"✅ Model packaged to {model_tar_path}")
        return model_tar_path
    
    def upload_model(self, model_tar_path, s3_key):
        """Upload model package to S3"""
        print(f"📤 Uploading model to s3://{self.bucket}/{s3_key}")
        
        self.s3.upload_file(model_tar_path, self.bucket, s3_key)
        model_uri = f"s3://{self.bucket}/{s3_key}"
        
        print(f"✅ Model uploaded to {model_uri}")
        return model_uri
    
    def create_model(self, model_name, model_uri, role_arn):
        """Create SageMaker model"""
        print(f"🔧 Creating SageMaker model: {model_name}")
        
        try:
            response = self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
                    'ModelDataUrl': model_uri,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                    }
                },
                ExecutionRoleArn=role_arn
            )
            
            print(f"✅ Model created: {response['ModelArn']}")
            return response['ModelArn']
            
        except Exception as e:
            print(f"❌ Failed to create model: {e}")
            raise
    
    def create_endpoint_config(self, config_name, model_name, instance_type='ml.t2.medium'):
        """Create endpoint configuration"""
        print(f"⚙️ Creating endpoint config: {config_name}")
        
        try:
            response = self.sagemaker.create_endpoint_configuration(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': instance_type,
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            print(f"✅ Endpoint config created: {response['EndpointConfigArn']}")
            return response['EndpointConfigArn']
            
        except Exception as e:
            print(f"❌ Failed to create endpoint config: {e}")
            raise
    
    def create_endpoint(self, endpoint_name, config_name):
        """Create SageMaker endpoint"""
        print(f"🚀 Creating endpoint: {endpoint_name}")
        
        try:
            response = self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            
            print(f"✅ Endpoint creation initiated: {response['EndpointArn']}")
            print("⏳ Endpoint is being created... This may take several minutes.")
            
            return response['EndpointArn']
            
        except Exception as e:
            print(f"❌ Failed to create endpoint: {e}")
            raise
    
    def deploy(self, model_path, endpoint_name, role_arn=None):
        """Deploy ensemble model end-to-end"""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        if not role_arn:
            role_arn = 'arn:aws:iam::773934887314:role/SageMakerExecutionRole'
        
        model_name = f"ensemble-model-{timestamp}"
        config_name = f"ensemble-config-{timestamp}"
        s3_key = f"models/ensemble/{timestamp}/model.tar.gz"
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                model_tar_path = self.package_model(model_path, temp_dir)
                
                # Upload to S3
                model_uri = self.upload_model(model_tar_path, s3_key)
            
            model_arn = self.create_model(model_name, model_uri, role_arn)
            config_arn = self.create_endpoint_config(config_name, model_name)
            endpoint_arn = self.create_endpoint(endpoint_name, config_name)
            
            print(f"""
🎉 Deployment Summary:
   Model: {model_name}
   Config: {config_name}
   Endpoint: {endpoint_name}
   S3 URI: {model_uri}
   
📋 Next Steps:
   1. Monitor endpoint status: aws sagemaker describe-endpoint --endpoint-name {endpoint_name}
   2. Test endpoint: aws sagemaker-runtime invoke-endpoint --endpoint-name {endpoint_name} --body '...'
   3. Delete when done: aws sagemaker delete-endpoint --endpoint-name {endpoint_name}
""")
            
            return {
                'endpoint_name': endpoint_name,
                'model_name': model_name,
                'config_name': config_name,
                'model_uri': model_uri
            }
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Deploy ensemble model to SageMaker')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to ensemble model file')
    parser.add_argument('--endpoint-name', type=str, required=True,
                        help='Name for SageMaker endpoint')
    parser.add_argument('--role-arn', type=str,
                        help='SageMaker execution role ARN')
    parser.add_argument('--instance-type', type=str, default='ml.t2.medium',
                        help='Instance type for endpoint')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        exit(1)
    
    deployer = EnsembleDeployer()
    
    try:
        result = deployer.deploy(args.model_path, args.endpoint_name, args.role_arn)
        print("✅ Deployment completed successfully!")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
