#!/usr/bin/env python3
"""
ML Ops CLI for Conviction AI HPO System

This CLI provides management interface for:
1. Dashboard generation and monitoring
2. Automated retraining system control
3. Performance metrics monitoring
4. System health checks
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlops_cli.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MLOpsCLI:
    def __init__(self):
        self.default_endpoint = 'conviction-ensemble-v4-1751650627'
        self.default_data_uri = 's3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv'
    
    def generate_dashboard(self, endpoint_name: str, output_file: Optional[str] = None, 
                         continuous: bool = False, interval: int = 5, dry_run: bool = False) -> bool:
        """Generate ML Ops dashboard"""
        cmd = [sys.executable, 'mlops_dashboard.py', '--endpoint-name', endpoint_name]
        
        if output_file:
            cmd.extend(['--output-file', output_file])
        
        if continuous:
            cmd.extend(['--continuous', '--interval', str(interval)])
        
        if dry_run:
            cmd.append('--dry-run')
        
        try:
            logger.info(f"Generating dashboard: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Dashboard generated successfully")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                logger.error(f"❌ Dashboard generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception generating dashboard: {e}")
            return False
    
    def setup_retraining(self, dry_run: bool = False) -> bool:
        """Set up automated retraining system"""
        cmd = [sys.executable, 'automated_retraining.py', '--setup']
        
        if dry_run:
            cmd.append('--dry-run')
        
        try:
            logger.info(f"Setting up automated retraining: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Automated retraining setup successful")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                logger.error(f"❌ Automated retraining setup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception setting up retraining: {e}")
            return False
    
    def check_retraining(self, endpoint_name: str, data_uri: str, 
                        auc_threshold: float = 0.50, max_data_age_days: int = 7, 
                        dry_run: bool = False) -> bool:
        """Run retraining check"""
        cmd = [
            sys.executable, 'automated_retraining.py', '--check',
            '--endpoint-name', endpoint_name,
            '--data-s3-uri', data_uri,
            '--auc-threshold', str(auc_threshold),
            '--max-data-age-days', str(max_data_age_days)
        ]
        
        if dry_run:
            cmd.append('--dry-run')
        
        try:
            logger.info(f"Running retraining check: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Retraining check completed")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                logger.error(f"❌ Retraining check failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception running retraining check: {e}")
            return False
    
    def setup_eventbridge(self, dry_run: bool = False) -> bool:
        """Set up EventBridge automated triggers"""
        cmd = [sys.executable, 'setup_eventbridge.py', '--setup']
        
        if dry_run:
            cmd.append('--dry-run')
        
        try:
            logger.info(f"Setting up EventBridge: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ EventBridge setup successful")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                logger.error(f"❌ EventBridge setup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception setting up EventBridge: {e}")
            return False
    
    def test_endpoint(self, endpoint_name: str, sample_count: int = 5, dry_run: bool = False) -> bool:
        """Test endpoint inference"""
        cmd = [
            sys.executable, 'sample_inference.py',
            '--endpoint-name', endpoint_name,
            '--sample-count', str(sample_count)
        ]
        
        if dry_run:
            cmd.append('--dry-run')
        
        try:
            logger.info(f"Testing endpoint: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Endpoint test successful")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                logger.error(f"❌ Endpoint test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception testing endpoint: {e}")
            return False
    
    def run_hpo_pipeline(self, data_uri: str, dry_run: bool = False) -> bool:
        """Run HPO pipeline with existing orchestration"""
        cmd = [
            sys.executable, 'scripts/orchestrate_hpo_pipeline.py',
            '--set-and-forget',
            '--input-data-s3', data_uri,
            '--auto-recover',
            '--auto-ensemble',
            '--notify'
        ]
        
        if dry_run:
            cmd.append('--dry-run')
        
        try:
            logger.info(f"Running HPO pipeline: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ HPO pipeline completed successfully")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                logger.error(f"❌ HPO pipeline failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception running HPO pipeline: {e}")
            return False
    
    def system_status(self, endpoint_name: str) -> Dict:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'endpoint_name': endpoint_name,
            'components': {}
        }
        
        logger.info(f"Checking system status for {endpoint_name}")
        
        status['components']['endpoint_test'] = self.test_endpoint(endpoint_name, dry_run=True)
        status['components']['dashboard_ready'] = os.path.exists('mlops_dashboard.py')
        status['components']['retraining_ready'] = os.path.exists('automated_retraining.py')
        status['components']['eventbridge_ready'] = os.path.exists('setup_eventbridge.py')
        status['components']['orchestration_ready'] = os.path.exists('scripts/orchestrate_hpo_pipeline.py')
        
        all_ready = all(status['components'].values())
        status['overall_status'] = 'READY' if all_ready else 'PARTIAL'
        
        logger.info(f"System status: {status['overall_status']}")
        for component, ready in status['components'].items():
            logger.info(f"  {component}: {'✅' if ready else '❌'}")
        
        return status

def main():
    parser = argparse.ArgumentParser(description='ML Ops CLI for Conviction AI HPO System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--endpoint-name', type=str, 
                        default='conviction-ensemble-v4-1751650627',
                        help='SageMaker endpoint name')
    parent_parser.add_argument('--data-uri', type=str,
                        default='s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv',
                        help='S3 URI for training data')
    parent_parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode without making actual changes')
    
    dashboard_parser = subparsers.add_parser('dashboard', help='Dashboard management', parents=[parent_parser])
    dashboard_parser.add_argument('--output-file', type=str, help='Output HTML file')
    dashboard_parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    dashboard_parser.add_argument('--interval', type=int, default=5, help='Monitoring interval (minutes)')
    
    retraining_parser = subparsers.add_parser('retraining', help='Retraining management', parents=[parent_parser])
    retraining_parser.add_argument('--setup', action='store_true', help='Set up automated retraining')
    retraining_parser.add_argument('--check', action='store_true', help='Run retraining check')
    retraining_parser.add_argument('--auc-threshold', type=float, default=0.50, help='AUC threshold')
    retraining_parser.add_argument('--max-data-age-days', type=int, default=7, help='Max data age (days)')
    
    eventbridge_parser = subparsers.add_parser('eventbridge', help='EventBridge management', parents=[parent_parser])
    eventbridge_parser.add_argument('--setup', action='store_true', help='Set up EventBridge triggers')
    
    test_parser = subparsers.add_parser('test', help='Testing commands', parents=[parent_parser])
    test_parser.add_argument('--endpoint', action='store_true', help='Test endpoint inference')
    test_parser.add_argument('--sample-count', type=int, default=5, help='Number of test samples')
    
    hpo_parser = subparsers.add_parser('hpo', help='HPO pipeline management', parents=[parent_parser])
    hpo_parser.add_argument('--run', action='store_true', help='Run HPO pipeline')
    
    status_parser = subparsers.add_parser('status', help='System status check', parents=[parent_parser])
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🧪 DRY RUN MODE - ML Ops CLI simulation")
    
    cli = MLOpsCLI()
    
    if args.command == 'dashboard':
        success = cli.generate_dashboard(
            args.endpoint_name, 
            args.output_file, 
            args.continuous, 
            args.interval, 
            args.dry_run
        )
        sys.exit(0 if success else 1)
    
    elif args.command == 'retraining':
        if args.setup:
            success = cli.setup_retraining(args.dry_run)
        elif args.check:
            success = cli.check_retraining(
                args.endpoint_name, 
                args.data_uri, 
                args.auc_threshold, 
                args.max_data_age_days, 
                args.dry_run
            )
        else:
            retraining_parser.print_help()
            sys.exit(1)
        
        sys.exit(0 if success else 1)
    
    elif args.command == 'eventbridge':
        if args.setup:
            success = cli.setup_eventbridge(args.dry_run)
            sys.exit(0 if success else 1)
        else:
            eventbridge_parser.print_help()
            sys.exit(1)
    
    elif args.command == 'test':
        if args.endpoint:
            success = cli.test_endpoint(args.endpoint_name, args.sample_count, args.dry_run)
            sys.exit(0 if success else 1)
        else:
            test_parser.print_help()
            sys.exit(1)
    
    elif args.command == 'hpo':
        if args.run:
            success = cli.run_hpo_pipeline(args.data_uri, args.dry_run)
            sys.exit(0 if success else 1)
        else:
            hpo_parser.print_help()
            sys.exit(1)
    
    elif args.command == 'status':
        status = cli.system_status(args.endpoint_name)
        print(f"\n=== ML Ops System Status ===")
        print(f"Timestamp: {status['timestamp']}")
        print(f"Endpoint: {status['endpoint_name']}")
        print(f"Overall Status: {status['overall_status']}")
        print(f"\nComponent Status:")
        for component, ready in status['components'].items():
            print(f"  {component}: {'✅ READY' if ready else '❌ NOT READY'}")
        
        sys.exit(0 if status['overall_status'] == 'READY' else 1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
