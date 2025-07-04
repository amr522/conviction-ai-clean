"""Shared AWS utilities for HPO pipeline scripts"""

import boto3
import logging
import json
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AWSClientManager:
    """Centralized AWS client management with error handling"""
    
    def __init__(self, region_name: str = 'us-east-1'):
        self.region_name = region_name
        self._clients = {}
    
    def get_client(self, service_name: str):
        """Get or create AWS client with error handling"""
        if service_name not in self._clients:
            try:
                self._clients[service_name] = boto3.client(service_name, region_name=self.region_name)
                logger.info(f"✅ Created {service_name} client for region {self.region_name}")
            except Exception as e:
                logger.error(f"❌ Failed to create {service_name} client: {e}")
                raise
        return self._clients[service_name]
    
    @property
    def sagemaker(self):
        return self.get_client('sagemaker')
    
    @property
    def s3(self):
        return self.get_client('s3')
    
    @property
    def cloudwatch(self):
        return self.get_client('cloudwatch')
    
    @property
    def sns(self):
        return self.get_client('sns')

def safe_aws_operation(operation_name: str, operation_func, dry_run: bool = False, **kwargs):
    """Execute AWS operation with error handling and dry-run support"""
    if dry_run:
        logger.info(f"🔍 [DRY-RUN] Would execute {operation_name} with params: {kwargs}")
        return {"dry_run": True, "operation": operation_name, "params": kwargs}
    
    try:
        logger.info(f"🔄 Executing {operation_name}...")
        result = operation_func(**kwargs)
        logger.info(f"✅ {operation_name} completed successfully")
        return result
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"❌ {operation_name} failed: {error_code} - {error_message}")
        raise
    except Exception as e:
        logger.error(f"❌ {operation_name} failed with unexpected error: {e}")
        raise

def load_pinned_config() -> Dict[str, Any]:
    """Load pinned HPO configuration"""
    config_path = "models/pinned_successful_hpo/hpo_config_pinned.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✅ Loaded pinned configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Pinned configuration not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in pinned configuration: {e}")
        raise

def get_aws_account_id() -> str:
    """Get current AWS account ID"""
    try:
        sts_client = boto3.client('sts')
        account_id = sts_client.get_caller_identity()['Account']
        logger.info(f"✅ Current AWS account: {account_id}")
        return account_id
    except Exception as e:
        logger.error(f"❌ Failed to get AWS account ID: {e}")
        raise

def validate_iam_permissions(required_permissions: list, dry_run: bool = False) -> bool:
    """Validate that current IAM role has required permissions"""
    if dry_run:
        logger.info(f"🔍 [DRY-RUN] Would validate IAM permissions: {required_permissions}")
        return True
    
    logger.info(f"📋 Required IAM permissions: {required_permissions}")
    logger.warning("⚠️ IAM permission validation not implemented - ensure role has required permissions")
    return True

def get_sagemaker_execution_role() -> str:
    """Get SageMaker execution role ARN"""
    account_id = get_aws_account_id()
    role_arn = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    logger.info(f"📋 Using SageMaker execution role: {role_arn}")
    return role_arn

def format_resource_name(base_name: str, timestamp: Optional[str] = None) -> str:
    """Format AWS resource name with consistent naming convention"""
    from datetime import datetime
    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    resource_name = f"{base_name}-{timestamp}"
    logger.debug(f"📝 Formatted resource name: {resource_name}")
    return resource_name
