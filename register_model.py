#!/usr/bin/env python3
"""
register_model.py - Register models in SageMaker Model Registry

This script enables:
1. Model versioning and tracking in SageMaker Model Registry
2. Model promotion through stages (Development, Staging, Production)
3. Model comparison based on metrics
"""

import os
import sys
import argparse
import logging
import json
import boto3
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_logs/model_registry.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def ensure_directory(directory):
    """Ensure a directory exists"""
    os.makedirs(directory, exist_ok=True)

def register_model(model_name, model_uri, description=None, tags=None):
    """Register a model in SageMaker Model Registry"""
    logger.info(f"Registering model {model_name} from {model_uri}")
    
    # Create SageMaker client
    sm_client = boto3.client('sagemaker')
    
    # Check if model package group exists
    try:
        sm_client.describe_model_package_group(
            ModelPackageGroupName=model_name
        )
        logger.info(f"Model package group {model_name} already exists")
    except sm_client.exceptions.ResourceNotFound:
        logger.info(f"Creating model package group {model_name}")
        sm_client.create_model_package_group(
            ModelPackageGroupName=model_name,
            ModelPackageGroupDescription=description or f"Model package group for {model_name}"
        )
    
    # Check if this model URI is already registered
    # This is a basic check to avoid duplicates with same model URI
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=10
        )
        for package in response.get("ModelPackageSummaryList", []):
            package_arn = package.get("ModelPackageArn")
            details = sm_client.describe_model_package(ModelPackageName=package_arn)
            if details.get("SourceUri") == model_uri:
                logger.info(f"Model with URI {model_uri} already exists as {package_arn}. Skipping registration.")
                return package_arn
    except Exception as e:
        # If there's an error checking, we'll proceed with registration
        logger.warning(f"Error checking for existing models: {e}")
    
    # Create model package
    create_args = {
        "ModelPackageGroupName": model_name,
        "ModelPackageDescription": description or f"Model version for {model_name}",
        "SourceUri": model_uri,
        "ApprovalStatus": "PendingManualApproval"  # Start in pending state
    }
    
    # Add tags if provided
    if tags:
        create_args["Tags"] = [{"Key": k, "Value": v} for k, v in tags.items()]
    
    try:
        response = sm_client.create_model_package(**create_args)
        model_package_arn = response["ModelPackageArn"]
        logger.info(f"Successfully registered model with ARN: {model_package_arn}")
        return model_package_arn
    except sm_client.exceptions.ResourceLimitExceeded as e:
        logger.warning(f"Resource limit exceeded: {e}")
        return None
    except sm_client.exceptions.ConflictException as e:
        logger.warning(f"Conflict creating model package: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        return None

def promote_model(model_name, version, stage):
    """Promote a model to a new stage (Development, Staging, Production)"""
    logger.info(f"Promoting model {model_name} version {version} to {stage}")
    
    # Create SageMaker client
    sm_client = boto3.client('sagemaker')
    
    # List model packages in the group
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=100
        )
        
        # Find the requested version
        model_packages = response.get("ModelPackageSummaryList", [])
        target_package = None
        
        for i, package in enumerate(model_packages):
            if str(i + 1) == version or package.get("ModelPackageVersion") == int(version):
                target_package = package
                break
        
        if not target_package:
            logger.error(f"Model version {version} not found for model {model_name}")
            return False
        
        # Update the model package approval status
        approval_status = "Approved" if stage.lower() in ["production", "staging"] else "Rejected"
        
        sm_client.update_model_package(
            ModelPackageName=target_package["ModelPackageArn"],
            ApprovalStatus=approval_status
        )
        
        # Add stage tag
        sm_client.add_tags(
            ResourceArn=target_package["ModelPackageArn"],
            Tags=[
                {
                    "Key": "Stage",
                    "Value": stage
                }
            ]
        )
        
        logger.info(f"Successfully promoted model {model_name} version {version} to {stage}")
        return True
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        return False

def list_registered_models(model_name=None):
    """List all registered models or versions of a specific model"""
    logger.info(f"Listing registered models{' for ' + model_name if model_name else ''}")
    
    # Create SageMaker client
    sm_client = boto3.client('sagemaker')
    
    try:
        if model_name:
            # List versions of the specified model
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_name,
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=100
            )
            
            model_packages = response.get("ModelPackageSummaryList", [])
            if not model_packages:
                logger.info(f"No model versions found for {model_name}")
                return []
            
            # Get details for each version
            versions = []
            for package in model_packages:
                # Get tags to determine stage
                tags_response = sm_client.list_tags(ResourceArn=package["ModelPackageArn"])
                tags = {tag["Key"]: tag["Value"] for tag in tags_response.get("Tags", [])}
                
                # Get model package details
                details = sm_client.describe_model_package(ModelPackageName=package["ModelPackageArn"])
                
                versions.append({
                    "Version": package.get("ModelPackageVersion", "Unknown"),
                    "ARN": package["ModelPackageArn"],
                    "CreationTime": package["CreationTime"].isoformat(),
                    "ApprovalStatus": details.get("ModelApprovalStatus", "Unknown"),
                    "Stage": tags.get("Stage", "Development")
                })
            
            logger.info(f"Found {len(versions)} versions for model {model_name}")
            return versions
        else:
            # List all model package groups
            response = sm_client.list_model_package_groups(
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=100
            )
            
            groups = response.get("ModelPackageGroupSummaryList", [])
            model_groups = []
            
            for group in groups:
                model_groups.append({
                    "Name": group["ModelPackageGroupName"],
                    "ARN": group["ModelPackageGroupArn"],
                    "CreationTime": group["CreationTime"].isoformat()
                })
            
            logger.info(f"Found {len(model_groups)} model package groups")
            return model_groups
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []

def compare_models(model_name, version1, version2):
    """Compare two versions of a model"""
    logger.info(f"Comparing model {model_name} versions {version1} and {version2}")
    
    # Create SageMaker client
    sm_client = boto3.client('sagemaker')
    
    try:
        # Get model versions
        models = list_registered_models(model_name)
        
        # Find the specified versions
        model1 = next((m for m in models if str(m["Version"]) == str(version1)), None)
        model2 = next((m for m in models if str(m["Version"]) == str(version2)), None)
        
        if not model1 or not model2:
            logger.error(f"One or both model versions not found")
            return None
        
        # Get model details
        details1 = sm_client.describe_model_package(ModelPackageName=model1["ARN"])
        details2 = sm_client.describe_model_package(ModelPackageName=model2["ARN"])
        
        # Extract metrics
        metrics1 = details1.get("ModelMetrics", {}).get("ModelQuality", {}).get("Statistics", {})
        metrics2 = details2.get("ModelMetrics", {}).get("ModelQuality", {}).get("Statistics", {})
        
        comparison = {
            "Version1": {
                "Version": version1,
                "ApprovalStatus": model1["ApprovalStatus"],
                "Stage": model1["Stage"],
                "CreationTime": model1["CreationTime"],
                "Metrics": metrics1
            },
            "Version2": {
                "Version": version2,
                "ApprovalStatus": model2["ApprovalStatus"],
                "Stage": model2["Stage"],
                "CreationTime": model2["CreationTime"],
                "Metrics": metrics2
            }
        }
        
        logger.info(f"Comparison complete")
        return comparison
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        return None

def setup_model_monitoring(model_name, version, baseline_uri, schedule="daily"):
    """Set up model monitoring for data quality and model quality"""
    logger.info(f"Setting up monitoring for model {model_name} version {version}")
    
    # Create SageMaker client
    sm_client = boto3.client('sagemaker')
    
    try:
        # Find the model package
        models = list_registered_models(model_name)
        model = next((m for m in models if str(m["Version"]) == str(version)), None)
        
        if not model:
            logger.error(f"Model version {version} not found for model {model_name}")
            return False
        
        # Get endpoint name associated with this model
        endpoints = sm_client.list_endpoints()
        model_endpoints = []
        
        for endpoint in endpoints.get("Endpoints", []):
            config = sm_client.describe_endpoint_config(
                EndpointConfigName=endpoint["EndpointConfigName"]
            )
            
            for production_variant in config.get("ProductionVariants", []):
                if model["ARN"] in production_variant.get("ModelName", ""):
                    model_endpoints.append(endpoint["EndpointName"])
        
        if not model_endpoints:
            logger.error(f"No endpoints found for model {model_name} version {version}")
            return False
        
        # Set up monitoring for each endpoint
        for endpoint_name in model_endpoints:
            # Create monitoring schedule
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            schedule_name = f"{model_name}-monitoring-{timestamp}"
            
            # Set schedule expression based on frequency
            if schedule.lower() == "hourly":
                schedule_expression = "cron(0 * ? * * *)"
            elif schedule.lower() == "daily":
                schedule_expression = "cron(0 0 ? * * *)"
            elif schedule.lower() == "weekly":
                schedule_expression = "cron(0 0 ? * 1 *)"
            else:
                schedule_expression = "cron(0 0 ? * * *)"  # Default to daily
            
            # Create monitoring schedule
            sm_client.create_monitoring_schedule(
                MonitoringScheduleName=schedule_name,
                MonitoringScheduleConfig={
                    "ScheduleConfig": {
                        "ScheduleExpression": schedule_expression
                    },
                    "MonitoringJobDefinition": {
                        "BaselineConfig": {
                            "ConstraintsResource": {
                                "S3Uri": f"{baseline_uri}/constraints.json"
                            },
                            "StatisticsResource": {
                                "S3Uri": f"{baseline_uri}/statistics.json"
                            }
                        },
                        "MonitoringInputs": [
                            {
                                "EndpointInput": {
                                    "EndpointName": endpoint_name,
                                    "LocalPath": "/opt/ml/processing/input"
                                }
                            }
                        ],
                        "MonitoringOutputConfig": {
                            "MonitoringOutputs": [
                                {
                                    "S3Output": {
                                        "S3Uri": f"s3://{os.environ.get('S3_BUCKET', 'hpo-bucket-773934887314')}/monitoring/{model_name}/{version}/",
                                        "LocalPath": "/opt/ml/processing/output"
                                    }
                                }
                            ]
                        },
                        "MonitoringResources": {
                            "ClusterConfig": {
                                "InstanceCount": 1,
                                "InstanceType": "ml.m5.xlarge",
                                "VolumeSizeInGB": 20
                            }
                        },
                        "RoleArn": os.environ.get("SAGEMAKER_ROLE_ARN"),
                        "StoppingCondition": {
                            "MaxRuntimeInSeconds": 3600
                        }
                    }
                },
                Tags=[
                    {
                        "Key": "Model",
                        "Value": model_name
                    },
                    {
                        "Key": "Version",
                        "Value": str(version)
                    }
                ]
            )
            
            logger.info(f"Successfully set up monitoring for endpoint {endpoint_name}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up monitoring: {e}")
        return False

def model_exists(model_name, version=None):
    """Check if a model package version already exists"""
    logger.info(f"Checking if model {model_name} version {version} exists")
    
    # Create SageMaker client
    sm_client = boto3.client('sagemaker')
    
    try:
        # First check if model package group exists
        try:
            sm_client.describe_model_package_group(
                ModelPackageGroupName=model_name
            )
        except sm_client.exceptions.ResourceNotFound:
            return False
        
        # If no specific version, just return True if group exists
        if version is None:
            return True
        
        # List model packages in the group
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=100
        )
        
        # Check if requested version exists
        model_packages = response.get("ModelPackageSummaryList", [])
        
        for i, package in enumerate(model_packages):
            package_version = package.get("ModelPackageVersion", i + 1)
            if str(package_version) == str(version):
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking if model exists: {e}")
        return False

def main():
    """Main function to parse arguments and execute commands"""
    parser = argparse.ArgumentParser(description="SageMaker Model Registry Management")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Register model command
    register_parser = subparsers.add_parser("register", help="Register a model")
    register_parser.add_argument("--model-name", required=True, help="Model package group name")
    register_parser.add_argument("--model-uri", required=True, help="S3 URI of the model artifacts")
    register_parser.add_argument("--description", help="Model description")
    register_parser.add_argument("--tags", help="JSON string of tags")
    
    # Promote model command
    promote_parser = subparsers.add_parser("promote", help="Promote a model to a new stage")
    promote_parser.add_argument("--model-name", required=True, help="Model package group name")
    promote_parser.add_argument("--version", required=True, help="Model version to promote")
    promote_parser.add_argument("--stage", required=True, choices=["Development", "Staging", "Production"], help="Stage to promote to")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List registered models")
    list_parser.add_argument("--model-name", help="Model package group name to list versions for")
    
    # Compare models command
    compare_parser = subparsers.add_parser("compare", help="Compare two model versions")
    compare_parser.add_argument("--model-name", required=True, help="Model package group name")
    compare_parser.add_argument("--version1", required=True, help="First model version to compare")
    compare_parser.add_argument("--version2", required=True, help="Second model version to compare")
    
    # Monitor model command
    monitor_parser = subparsers.add_parser("monitor", help="Set up model monitoring")
    monitor_parser.add_argument("--model-name", required=True, help="Model package group name")
    monitor_parser.add_argument("--version", required=True, help="Model version to monitor")
    monitor_parser.add_argument("--baseline-uri", required=True, help="S3 URI to baseline statistics and constraints")
    monitor_parser.add_argument("--schedule", default="daily", choices=["hourly", "daily", "weekly"], help="Monitoring schedule")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure the logs directory exists
    ensure_directory("pipeline_logs")
    
    # Execute command
    if args.command == "register":
        tags = json.loads(args.tags) if args.tags else None
        register_model(args.model_name, args.model_uri, args.description, tags)
    elif args.command == "promote":
        promote_model(args.model_name, args.version, args.stage)
    elif args.command == "list":
        models = list_registered_models(args.model_name)
        print(json.dumps(models, indent=2))
    elif args.command == "compare":
        comparison = compare_models(args.model_name, args.version1, args.version2)
        print(json.dumps(comparison, indent=2))
    elif args.command == "monitor":
        setup_model_monitoring(args.model_name, args.version, args.baseline_uri, args.schedule)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
