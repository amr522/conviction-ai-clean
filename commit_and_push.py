#!/usr/bin/env python3
"""
Commit and Push Changes for Both Part A and Part B
Automated git operations for leak-proof retrain and sentiment integration
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_git_command(cmd, check=True):
    """Run git command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def commit_part_a_changes():
    """Commit Part A changes to retrain/leak-proof-TA branch"""
    try:
        logger.info("Committing Part A changes to retrain/leak-proof-TA branch...")
        
        part_a_files = [
            "aws_xgb_hpo_launch.py",
            "aws_catboost_hpo_launch.py", 
            "aws_lgbm_hpo_launch.py",
            "lgbm_train.py",
            "train_price_gru.py",
            "launch_algorithms.py",
            "monitor_jobs.py",
            "execute_workflow.py",
            "verify_sagemaker.py",
            "omar.md",
            "ENHANCED_TRAINING_README.md",
            "NEXT_SESSION_PROMPT.md"
        ]
        
        for file in part_a_files:
            if os.path.exists(file):
                success, stdout, stderr = run_git_command(f"git add {file}")
                if not success:
                    logger.warning(f"Failed to add {file}: {stderr}")
        
        commit_msg = "feat: complete leak-proof retrain workflow with AWS resource fixes and algorithm automation"
        success, stdout, stderr = run_git_command(f'git commit -m "{commit_msg}"')
        
        if success:
            logger.info("✅ Part A changes committed successfully")
            
            success, stdout, stderr = run_git_command("git push origin retrain/leak-proof-TA")
            if success:
                logger.info("✅ Part A changes pushed to remote")
                return True
            else:
                logger.error(f"Failed to push Part A changes: {stderr}")
                return False
        else:
            logger.warning(f"Part A commit result: {stderr}")
            return True  # May be no changes to commit
            
    except Exception as e:
        logger.error(f"Failed to commit Part A changes: {e}")
        return False

def create_sentiment_branch():
    """Create and switch to feature/twitter-sentiment branch"""
    try:
        logger.info("Creating feature/twitter-sentiment branch...")
        
        success, stdout, stderr = run_git_command("git checkout -b feature/twitter-sentiment")
        
        if success:
            logger.info("✅ Created feature/twitter-sentiment branch")
            return True
        else:
            success, stdout, stderr = run_git_command("git checkout feature/twitter-sentiment")
            if success:
                logger.info("✅ Switched to existing feature/twitter-sentiment branch")
                return True
            else:
                logger.error(f"Failed to create/checkout sentiment branch: {stderr}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to create sentiment branch: {e}")
        return False

def commit_part_b_changes():
    """Commit Part B changes to feature/twitter-sentiment branch"""
    try:
        logger.info("Committing Part B changes to feature/twitter-sentiment branch...")
        
        part_b_files = [
            "scripts/twitter_stream_ingest.py",
            "score_tweets_finbert.py",
            "aws_utils.py",
            "create_intraday_features.py",
            "scripts/orchestrate_hpo_pipeline.py",
            "sentiment_integration_plan.md",
            "test_sentiment_pipeline.py",
            "omar.md",
            "ENHANCED_TRAINING_README.md"
        ]
        
        for file in part_b_files:
            if os.path.exists(file):
                success, stdout, stderr = run_git_command(f"git add {file}")
                if not success:
                    logger.warning(f"Failed to add {file}: {stderr}")
        
        commit_msg = "feat: implement Twitter sentiment integration (Phases 1-4) with FinBERT scoring and feature engineering"
        success, stdout, stderr = run_git_command(f'git commit -m "{commit_msg}"')
        
        if success:
            logger.info("✅ Part B changes committed successfully")
            
            success, stdout, stderr = run_git_command("git push origin feature/twitter-sentiment")
            if success:
                logger.info("✅ Part B changes pushed to remote")
                return True
            else:
                logger.error(f"Failed to push Part B changes: {stderr}")
                return False
        else:
            logger.warning(f"Part B commit result: {stderr}")
            return True  # May be no changes to commit
            
    except Exception as e:
        logger.error(f"Failed to commit Part B changes: {e}")
        return False

def main():
    """Execute git operations for both parts"""
    logger.info("🚀 Starting Git Operations for Part A and Part B")
    
    os.chdir("/home/ubuntu/repos/conviction-ai-clean")
    
    success_a = commit_part_a_changes()
    
    if create_sentiment_branch():
        success_b = commit_part_b_changes()
    else:
        success_b = False
    
    logger.info("=== GIT OPERATIONS SUMMARY ===")
    logger.info(f"Part A (retrain/leak-proof-TA): {'✅ SUCCESS' if success_a else '❌ FAILED'}")
    logger.info(f"Part B (feature/twitter-sentiment): {'✅ SUCCESS' if success_b else '❌ FAILED'}")
    
    if success_a and success_b:
        logger.info("🎉 All git operations completed successfully!")
        return True
    else:
        logger.warning("⚠️ Some git operations failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
