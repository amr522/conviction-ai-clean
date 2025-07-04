#!/usr/bin/env python3
"""Automated cleanup script to prevent temporary file accumulation"""

import os
import sys
import glob
import argparse
import logging
import shutil
from datetime import datetime, timedelta

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
                    logger.info(f"Would remove: {file_path}")
                else:
                    os.remove(file_path)
                    logger.info(f"Removed: {file_path}")
                files_cleaned += 1
            elif os.path.isdir(file_path) and pattern in ["__pycache__", ".pytest_cache"]:
                if dry_run:
                    logger.info(f"Would remove directory: {file_path}")
                else:
                    shutil.rmtree(file_path)
                    logger.info(f"Removed directory: {file_path}")
                files_cleaned += 1
    
    logger.info(f"✅ Cleanup completed. {files_cleaned} files processed.")
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
                            logger.info(f"Would remove old artifact: {file_path}")
                        else:
                            os.remove(file_path)
                            logger.info(f"Removed old artifact: {file_path}")
                        artifacts_cleaned += 1
    
    logger.info(f"✅ Artifact cleanup completed. {artifacts_cleaned} artifacts processed.")
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
                        logger.info(f"Would remove old data directory: {item_path}")
                    else:
                        shutil.rmtree(item_path)
                        logger.info(f"Removed old data directory: {item_path}")
                    data_cleaned += 1
    
    logger.info(f"✅ Data directory cleanup completed. {data_cleaned} directories processed.")
    return data_cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated cleanup for HPO pipeline')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be cleaned without actually removing files')
    args = parser.parse_args()
    
    total_cleaned = (cleanup_temporary_files(args.dry_run) + 
                    cleanup_old_model_artifacts(args.dry_run) + 
                    cleanup_old_data_directories(args.dry_run))
    
    logger.info(f"🎉 Total cleanup completed: {total_cleaned} items processed")
