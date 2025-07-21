#!/usr/bin/env python3
"""
mock_cleanup_sagemaker_resources.py - Simulates cleaning up SageMaker resources for smoke testing

This script simulates successfully cleaning up SageMaker resources created during the pipeline
execution.
"""

import argparse
import logging
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mock cleanup of SageMaker resources for Conviction-AI')
    
    parser.add_argument('--prefix', type=str, default="conviction-automl",
                        help='Prefix for SageMaker resources to clean up')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform a dry run without actually deleting resources')
    
    return parser.parse_args()

def simulate_cleanup(args):
    """
    Simulate cleaning up SageMaker resources.
    
    This function:
    1. Finds resources matching the given prefix
    2. Lists the resources that would be deleted
    3. Deletes the resources if not in dry-run mode
    """
    try:
        logger.info(f"🔍 Searching for SageMaker resources with prefix: {args.prefix}")
        
        # Load endpoint info if it exists
        endpoint_info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "endpoint_info.json")
        endpoint_info = None
        
        if os.path.exists(endpoint_info_path):
            with open(endpoint_info_path, "r") as f:
                endpoint_info = json.load(f)
                logger.info(f"Found endpoint info: {endpoint_info}")
        
        # Create a list of mock resources to clean up
        resources = []
        
        # Add the endpoint from endpoint_info.json
        if endpoint_info:
            job_name = endpoint_info.get("job_name")
            endpoint_name = endpoint_info.get("endpoint_name")
            
            # Add AutoML job
            resources.append({
                "type": "AutoMLJob",
                "name": job_name,
                "creation_time": endpoint_info.get("created_at", datetime.now().isoformat())
            })
            
            # Add endpoint
            resources.append({
                "type": "Endpoint",
                "name": endpoint_name,
                "creation_time": endpoint_info.get("created_at", datetime.now().isoformat())
            })
            
            # Add endpoint config
            resources.append({
                "type": "EndpointConfig",
                "name": f"{job_name}-config",
                "creation_time": endpoint_info.get("created_at", datetime.now().isoformat())
            })
            
            # Add model
            resources.append({
                "type": "Model",
                "name": f"{job_name}-model",
                "creation_time": endpoint_info.get("created_at", datetime.now().isoformat())
            })
        
        # Add some additional mock resources
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_name = f"{args.prefix}-{timestamp}"
        
        # If no resources were added from endpoint_info.json, add some mock ones
        if not resources:
            resources.append({
                "type": "AutoMLJob",
                "name": f"{base_name}",
                "creation_time": datetime.now().isoformat()
            })
            
            resources.append({
                "type": "Endpoint",
                "name": f"{base_name}-endpoint",
                "creation_time": datetime.now().isoformat()
            })
            
            resources.append({
                "type": "EndpointConfig",
                "name": f"{base_name}-config",
                "creation_time": datetime.now().isoformat()
            })
            
            resources.append({
                "type": "Model",
                "name": f"{base_name}-model",
                "creation_time": datetime.now().isoformat()
            })
        
        # Display the resources that would be deleted
        logger.info(f"Found {len(resources)} resources to clean up:")
        for resource in resources:
            logger.info(f"  - {resource['type']}: {resource['name']} (created: {resource['creation_time']})")
        
        # Simulate deletion if not in dry-run mode
        if args.dry_run:
            logger.info("⚠️ DRY RUN: No resources were deleted")
        else:
            logger.info("🗑️ Deleting resources...")
            
            for resource in resources:
                logger.info(f"  - Deleting {resource['type']}: {resource['name']}...")
            
            logger.info("✅ All resources successfully deleted")
            
            # Delete the endpoint_info.json file if it exists
            if os.path.exists(endpoint_info_path):
                os.remove(endpoint_info_path)
                logger.info(f"Deleted {endpoint_info_path}")
        
        # Save cleanup information
        cleanup_info = {
            "dry_run": args.dry_run,
            "prefix": args.prefix,
            "resources_cleaned": [r for r in resources] if not args.dry_run else [],
            "resources_found": [r for r in resources],
            "timestamp": datetime.now().isoformat()
        }
        
        cleanup_info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleanup_info.json")
        
        with open(cleanup_info_path, "w") as f:
            json.dump(cleanup_info, f, indent=2)
        
        logger.info(f"Cleanup information saved to {cleanup_info_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in mock cleanup process: {str(e)}")
        return False

if __name__ == "__main__":
    args = parse_arguments()
    
    success = simulate_cleanup(args)
    if success:
        logger.info("Mock cleanup completed successfully")
        exit(0)  # Successful exit
    else:
        logger.error("Mock cleanup failed")
        exit(1)  # Error exit
