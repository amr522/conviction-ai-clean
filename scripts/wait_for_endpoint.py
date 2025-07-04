#!/usr/bin/env python3
"""
Wait for SageMaker endpoint to reach InService status
"""
import boto3
import argparse
import time
import sys
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def wait_for_endpoint(endpoint_name: str, timeout_minutes: int = 30, expect_deleted: bool = False) -> bool:
    """Wait for endpoint to reach InService status or be deleted"""
    sagemaker = boto3.client('sagemaker')
    start_time = datetime.now()
    timeout = timedelta(minutes=timeout_minutes)
    
    if expect_deleted:
        logger.info(f"🔍 Waiting for endpoint {endpoint_name} to be deleted (timeout: {timeout_minutes}min)")
    else:
        logger.info(f"🔍 Waiting for endpoint {endpoint_name} (timeout: {timeout_minutes}min)")
    
    while datetime.now() - start_time < timeout:
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            if expect_deleted:
                logger.info(f"⏳ Endpoint {endpoint_name} still exists with status: {status}")
                time.sleep(60)
                continue
            
            if status == 'InService':
                logger.info(f"✅ Endpoint {endpoint_name} is InService!")
                return True
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown')
                logger.error(f"❌ Endpoint {endpoint_name} failed: {failure_reason}")
                return False
            else:
                logger.info(f"⏳ Endpoint {endpoint_name} status: {status}")
                time.sleep(60)
                
        except sagemaker.exceptions.ClientError as e:
            if 'ValidationException' in str(e) and expect_deleted:
                logger.info(f"✅ Endpoint {endpoint_name} has been deleted!")
                return True
            elif 'ValidationException' in str(e):
                logger.error(f"❌ Endpoint {endpoint_name} does not exist")
                return False
            else:
                logger.error(f"Error checking endpoint {endpoint_name}: {e}")
                time.sleep(60)
        except Exception as e:
            logger.error(f"Error checking endpoint {endpoint_name}: {e}")
            time.sleep(60)
    
    logger.error(f"⏰ Endpoint {endpoint_name} timeout after {timeout_minutes} minutes")
    return False

def main():
    parser = argparse.ArgumentParser(description='Wait for SageMaker endpoint to reach InService status')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of SageMaker endpoint to monitor')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Timeout in seconds (default: 1800)')
    parser.add_argument('--expect-deleted', action='store_true',
                        help='Wait for endpoint to be deleted instead of InService')
    
    args = parser.parse_args()
    timeout_minutes = args.timeout // 60
    
    success = wait_for_endpoint(args.name, timeout_minutes, args.expect_deleted)
    
    if success:
        if args.expect_deleted:
            print(f"✅ Endpoint {args.name} has been deleted")
        else:
            print(f"✅ Endpoint {args.name} is ready for use")
        sys.exit(0)
    else:
        if args.expect_deleted:
            print(f"❌ Endpoint {args.name} failed to be deleted")
        else:
            print(f"❌ Endpoint {args.name} failed to reach InService status")
        sys.exit(1)

if __name__ == "__main__":
    main()
