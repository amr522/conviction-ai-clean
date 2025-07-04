#!/usr/bin/env python3
"""Automated cleanup script to prevent temporary file accumulation"""

import os
import sys
import glob
import argparse
import logging
import shutil
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aws_utils import AWSClientManager, safe_aws_operation, validate_iam_permissions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_temporary_files(dry_run=False):
    """Clean up temporary files and logs older than specified days"""
    logger.info("🧹 Starting automated cleanup of temporary files...")
    
    cleanup_patterns = [
        "*.log",
        "*temp*",
        "*tmp*",
        "*.cache",
        "__pycache__",
        ".pytest_cache"
    ]
    
    files_cleaned = 0
    for pattern in cleanup_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            if os.path.isfile(file_path):
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_age.days < 7:
                    continue
                
                if dry_run:
                    logger.info(f"🔍 [DRY-RUN] Would remove file: {file_path} (age: {file_age.days} days)")
                else:
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
                files_cleaned += 1
            elif os.path.isdir(file_path) and pattern in ["__pycache__", ".pytest_cache"]:
                if dry_run:
                    logger.info(f"🔍 [DRY-RUN] Would remove directory: {file_path}")
                else:
                    shutil.rmtree(file_path)
                    logger.info(f"Removed directory: {file_path}")
                files_cleaned += 1
    
    logger.info(f"✅ Temporary file cleanup completed. {files_cleaned} files processed.")
    return files_cleaned

def cleanup_old_model_artifacts(dry_run=False):
    """Clean up old model artifacts not in pinned directory"""
    logger.info("🧹 Cleaning up old model artifacts...")
    
    pinned_dir = "models/pinned_successful_hpo"
    
    artifacts_cleaned = 0
    if os.path.exists("models"):
        for root, dirs, files in os.walk("models"):
            if pinned_dir in root:
                continue
            
            for file in files:
                if file.endswith(('.tar.gz', '.pkl')) and 'hpo-' in file:
                    file_path = os.path.join(root, file)
                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_age.days > 30:
                        if dry_run:
                            logger.info(f"🔍 [DRY-RUN] Would remove old artifact: {file_path} (age: {file_age.days} days)")
                        else:
                            os.remove(file_path)
                            logger.info(f"Removed old artifact: {file_path}")
                        artifacts_cleaned += 1
    
    logger.info(f"✅ Model artifact cleanup completed. {artifacts_cleaned} artifacts processed.")
    return artifacts_cleaned

def cleanup_old_data_directories(dry_run=False):
    """Clean up old processed data directories"""
    logger.info("🧹 Cleaning up old data directories...")
    
    data_cleaned = 0
    if os.path.exists("data"):
        for item in os.listdir("data"):
            item_path = os.path.join("data", item)
            if os.path.isdir(item_path) and "processed_with_news" in item:
                dir_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(item_path))
                
                if dir_age.days > 14:
                    if dry_run:
                        logger.info(f"🔍 [DRY-RUN] Would remove old data directory: {item_path} (age: {dir_age.days} days)")
                    else:
                        shutil.rmtree(item_path)
                        logger.info(f"Removed old data directory: {item_path}")
                    data_cleaned += 1
    
    logger.info(f"✅ Data directory cleanup completed. {data_cleaned} directories processed.")
    return data_cleaned

def cleanup_sagemaker_resources(dry_run=False):
    """Clean up old SageMaker HPO jobs and training jobs"""
    logger.info("🧹 Cleaning up old SageMaker resources...")
    
    required_permissions = [
        "sagemaker:ListHyperParameterTuningJobs",
        "sagemaker:ListTrainingJobs",
        "sagemaker:StopHyperParameterTuningJob",
        "sagemaker:StopTrainingJob"
    ]
    validate_iam_permissions(required_permissions, dry_run)
    
    if dry_run:
        logger.info("🔍 [DRY-RUN] Would list and clean up SageMaker HPO jobs older than 7 days")
        logger.info("🔍 [DRY-RUN] Would list and clean up failed training jobs older than 3 days")
        return 0
    
    aws_manager = AWSClientManager()
    resources_cleaned = 0
    
    try:
        cutoff_date = datetime.now() - timedelta(days=7)
        
        response = aws_manager.sagemaker.list_hyper_parameter_tuning_jobs(
            SortBy='CreationTime',
            SortOrder='Ascending',
            StatusEquals='Completed'
        )
        
        for job in response.get('HyperParameterTuningJobSummaries', []):
            creation_time = job['CreationTime'].replace(tzinfo=None)
            if creation_time < cutoff_date:
                job_name = job['HyperParameterTuningJobName']
                logger.info(f"Found old HPO job for potential cleanup: {job_name}")
        
        logger.info(f"✅ SageMaker resource cleanup completed. {resources_cleaned} resources processed.")
        
    except Exception as e:
        logger.warning(f"⚠️ SageMaker resource cleanup failed: {e}")
    
    return resources_cleaned

def cleanup_s3_incomplete_uploads(dry_run=False):
    """Clean up incomplete multipart uploads in S3"""
    logger.info("🧹 Cleaning up S3 incomplete multipart uploads...")
    
    required_permissions = [
        "s3:ListBucket",
        "s3:ListMultipartUploadParts",
        "s3:AbortMultipartUpload"
    ]
    validate_iam_permissions(required_permissions, dry_run)
    
    if dry_run:
        logger.info("🔍 [DRY-RUN] Would list and abort incomplete multipart uploads older than 7 days")
        return 0
    
    aws_manager = AWSClientManager()
    bucket = 'hpo-bucket-773934887314'
    uploads_cleaned = 0
    
    try:
        response = aws_manager.s3.list_multipart_uploads(Bucket=bucket)
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for upload in response.get('Uploads', []):
            initiated = upload['Initiated'].replace(tzinfo=None)
            if initiated < cutoff_date:
                upload_id = upload['UploadId']
                key = upload['Key']
                
                safe_aws_operation(
                    f"Abort multipart upload {upload_id}",
                    aws_manager.s3.abort_multipart_upload,
                    dry_run=False,
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id
                )
                uploads_cleaned += 1
        
        logger.info(f"✅ S3 multipart upload cleanup completed. {uploads_cleaned} uploads aborted.")
        
    except Exception as e:
        logger.warning(f"⚠️ S3 cleanup failed: {e}")
    
    return uploads_cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated cleanup for HPO pipeline')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be cleaned without actually removing files')
    parser.add_argument('--include-sagemaker', action='store_true', help='Include SageMaker resource cleanup')
    parser.add_argument('--include-s3', action='store_true', help='Include S3 multipart upload cleanup')
    args = parser.parse_args()
    
    total_cleaned = (cleanup_temporary_files(args.dry_run) + 
                    cleanup_old_model_artifacts(args.dry_run) + 
                    cleanup_old_data_directories(args.dry_run))
    
    if args.include_sagemaker:
        total_cleaned += cleanup_sagemaker_resources(args.dry_run)
    
    if args.include_s3:
        total_cleaned += cleanup_s3_incomplete_uploads(args.dry_run)
    
    logger.info(f"🎉 Total cleanup completed: {total_cleaned} items processed")
    
    if args.dry_run:
        logger.info("💡 Run without --dry-run to perform actual cleanup")
        logger.info("💡 Use --include-sagemaker and --include-s3 for additional cleanup options")
