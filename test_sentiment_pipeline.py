#!/usr/bin/env python3
"""
Test Sentiment Pipeline Components
Comprehensive testing for Twitter sentiment integration
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_twitter_ingestion():
    """Test Twitter stream ingestion in dry-run mode"""
    try:
        logger.info("Testing Twitter stream ingestion...")
        result = subprocess.run([
            'python', 'scripts/twitter_stream_ingest.py',
            '--dry-run', '--duration', '10'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ Twitter ingestion test passed")
            return True
        else:
            logger.error(f"❌ Twitter ingestion test failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Twitter ingestion test error: {e}")
        return False

def test_finbert_scoring():
    """Test FinBERT sentiment scoring in test mode"""
    try:
        logger.info("Testing FinBERT sentiment scoring...")
        result = subprocess.run([
            'python', 'score_tweets_finbert.py',
            '--test-mode', '--symbol', 'AAPL'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ FinBERT scoring test passed")
            return True
        else:
            logger.error(f"❌ FinBERT scoring test failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ FinBERT scoring test error: {e}")
        return False

def test_feature_creation():
    """Test sentiment feature creation"""
    try:
        logger.info("Testing sentiment feature creation...")
        result = subprocess.run([
            'python', 'create_intraday_features.py',
            '--test-sentiment', '--symbol', 'AAPL'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ Feature creation test passed")
            return True
        else:
            logger.error(f"❌ Feature creation test failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Feature creation test error: {e}")
        return False

def test_orchestration():
    """Test orchestration with sentiment flag"""
    try:
        logger.info("Testing orchestration with sentiment flag...")
        result = subprocess.run([
            'python', 'scripts/orchestrate_hpo_pipeline.py',
            '--dry-run', '--twitter-sentiment'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ Orchestration test passed")
            return True
        else:
            logger.error(f"❌ Orchestration test failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Orchestration test error: {e}")
        return False

def main():
    """Run all sentiment pipeline tests"""
    logger.info("🧪 Starting Sentiment Pipeline Tests")
    
    tests = [
        ("Twitter Ingestion", test_twitter_ingestion),
        ("FinBERT Scoring", test_finbert_scoring),
        ("Feature Creation", test_feature_creation),
        ("Orchestration", test_orchestration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        results[test_name] = test_func()
    
    logger.info("=== TEST RESULTS ===")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 All sentiment pipeline tests passed!")
        return True
    else:
        logger.warning("⚠️ Some sentiment pipeline tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
