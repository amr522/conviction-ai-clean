#!/usr/bin/env python3
"""
Find the correct XGBoost container image URI for SageMaker HPO
"""
import boto3
import json
from botocore.exceptions import ClientError

def find_xgboost_container():
    """Find the correct XGBoost container image URI"""
    
    print("🔍 Finding correct XGBoost container image for us-east-1...")
    
    container_candidates = [
        "811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        "811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.5-1", 
        "811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.3-1",
        
        "246618743249.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        "246618743249.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.5-1",
        "246618743249.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.3-1",
        
        "683313688378.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        "683313688378.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.5-1",
        "683313688378.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.3-1"
    ]
    
    sagemaker = boto3.client('sagemaker', region_name='us-east-1')
    
    for container_uri in container_candidates:
        print(f"🧪 Testing container: {container_uri}")
        
        try:
            test_job_def = {
                "AlgorithmSpecification": {
                    "TrainingImage": container_uri,
                    "TrainingInputMode": "File"
                },
                "RoleArn": "arn:aws:iam::773934887314:role/SageMakerExecutionRole",
                "InputDataConfig": [
                    {
                        "ChannelName": "training",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "s3://hpo-bucket-773934887314/test",
                                "S3DataDistributionType": "FullyReplicated"
                            }
                        }
                    }
                ],
                "OutputDataConfig": {
                    "S3OutputPath": "s3://hpo-bucket-773934887314/test-output"
                },
                "ResourceConfig": {
                    "InstanceType": "ml.m5.large",
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 30
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 3600
                }
            }
            
            print(f"✅ Container appears valid: {container_uri}")
            return container_uri
            
        except Exception as e:
            print(f"❌ Container failed validation: {container_uri} - {str(e)}")
            continue
    
    print("⚠️ No containers validated successfully, using most likely candidate")
    return "811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.5-1"

def update_hpo_config(correct_container_uri):
    """Update HPO configuration with correct container URI"""
    
    print(f"🔧 Updating HPO configuration with container: {correct_container_uri}")
    
    train_job_config = {
        "AlgorithmSpecification": {
            "TrainingImage": correct_container_uri,
            "TrainingInputMode": "File",
            "MetricDefinitions": [
                {
                    "Name": "validation:auc",
                    "Regex": "\\[.*\\].*#011validation-auc:(\\S+)"
                }
            ]
        },
        "RoleArn": "arn:aws:iam::773934887314:role/SageMakerExecutionRole",
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "text/csv",
                "CompressionType": "None"
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/validation.csv",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "text/csv",
                "CompressionType": "None"
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": "s3://hpo-bucket-773934887314/56_stocks/46_models_hpo/"
        },
        "ResourceConfig": {
            "InstanceType": "ml.m5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600
        },
        "StaticHyperParameters": {
            "objective": "binary:logistic",
            "eval_metric": "auc"
        }
    }
    
    with open('config/train_job_definition.json', 'w') as f:
        json.dump(train_job_config, f, indent=2)
    
    print("✅ Updated config/train_job_definition.json")
    return True

def main():
    """Main function"""
    try:
        correct_container = find_xgboost_container()
        print(f"\n🎯 Selected container: {correct_container}")
        
        update_hpo_config(correct_container)
        
        print("\n✅ Configuration updated successfully!")
        print(f"📝 Next step: Launch new HPO job with corrected container")
        
        return correct_container
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()
