#!/usr/bin/env python3
"""Master orchestration script for HPO pipeline DAG execution"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aws_utils import AWSClientManager, safe_aws_operation, load_pinned_config, validate_iam_permissions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HPOPipelineOrchestrator:
    """Orchestrates the complete HPO pipeline as a DAG"""
    
    def __init__(self, dry_run: bool = False, email: str = "amr522@gmail.com"):
        self.dry_run = dry_run
        self.email = email
        self.aws_manager = AWSClientManager()
        self.pipeline_state = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'current_step': None,
            'hpo_job_name': None,
            'endpoint_name': None
        }
    
    def run_script(self, script_path: str, args: Optional[List[str]] = None, check_return_code: bool = True) -> bool:
        """Run a script with error handling"""
        if args is None:
            args = []
        
        if self.dry_run:
            args.append('--dry-run') if '--dry-run' not in args else None
        
        cmd = [script_path] + args
        logger.info(f"🔄 Executing: {' '.join(cmd)}")
        
        if self.dry_run:
            logger.info(f"🔍 [DRY-RUN] Would execute: {' '.join(cmd)}")
            return True
        
        try:
            result = subprocess.run(cmd, check=check_return_code, capture_output=True, text=True)
            logger.info(f"✅ Script completed: {script_path}")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Script failed: {script_path}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def step_1_setup_monitoring(self) -> bool:
        """Step 1: Setup CloudWatch monitoring infrastructure"""
        logger.info("1️⃣ Setting up CloudWatch monitoring...")
        self.pipeline_state['current_step'] = 'setup_monitoring'
        
        success = self.run_script('./scripts/setup_hpo_monitoring.sh', [self.email, str(self.dry_run).lower()])
        if success:
            self.pipeline_state['steps_completed'].append('setup_monitoring')
        return success
    
    def step_2_backup_dataset(self) -> bool:
        """Step 2: Backup successful dataset with versioning"""
        logger.info("2️⃣ Backing up successful dataset...")
        self.pipeline_state['current_step'] = 'backup_dataset'
        
        success = self.run_script('./scripts/backup_successful_dataset.sh', [str(self.dry_run).lower()])
        if success:
            self.pipeline_state['steps_completed'].append('backup_dataset')
        return success
    
    def step_3_launch_hpo(self, job_type: str = "full") -> bool:
        """Step 3: Launch HPO job"""
        logger.info("3️⃣ Launching HPO job...")
        self.pipeline_state['current_step'] = 'launch_hpo'
        
        if self.dry_run:
            logger.info(f"🔍 [DRY-RUN] Would launch HPO job with type: {job_type}")
            self.pipeline_state['hpo_job_name'] = f"hpo-{job_type}-dry-run-{int(time.time())}"
            self.pipeline_state['steps_completed'].append('launch_hpo')
            return True
        
        cmd = ['python', 'aws_hpo_launch.py', '--job-type', job_type]
        logger.info(f"🔄 Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'hpo-' in line and 'launched' in line.lower():
                    import re
                    match = re.search(r'hpo-[a-zA-Z0-9-]+', line)
                    if match:
                        self.pipeline_state['hpo_job_name'] = match.group()
                        break
            
            if not self.pipeline_state['hpo_job_name']:
                timestamp = int(time.time())
                self.pipeline_state['hpo_job_name'] = f"hpo-{job_type}-{timestamp}"
            
            logger.info(f"✅ HPO job launched: {self.pipeline_state['hpo_job_name']}")
            self.pipeline_state['steps_completed'].append('launch_hpo')
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ HPO job launch failed: {e.stderr}")
            return False
    
    def step_4_monitor_hpo(self, timeout_minutes: int = 180) -> bool:
        """Step 4: Monitor HPO job completion"""
        logger.info("4️⃣ Monitoring HPO job completion...")
        self.pipeline_state['current_step'] = 'monitor_hpo'
        
        if self.dry_run:
            logger.info(f"🔍 [DRY-RUN] Would monitor HPO job: {self.pipeline_state['hpo_job_name']}")
            logger.info(f"🔍 [DRY-RUN] Would wait up to {timeout_minutes} minutes for completion")
            self.pipeline_state['steps_completed'].append('monitor_hpo')
            return True
        
        job_name = self.pipeline_state['hpo_job_name']
        if not job_name:
            logger.error("❌ No HPO job name available for monitoring")
            return False
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            try:
                response = self.aws_manager.sagemaker.describe_hyper_parameter_tuning_job(
                    HyperParameterTuningJobName=job_name
                )
                
                status = response['HyperParameterTuningJobStatus']
                logger.info(f"📊 HPO job status: {status}")
                
                if status == 'Completed':
                    logger.info("✅ HPO job completed successfully")
                    self.pipeline_state['steps_completed'].append('monitor_hpo')
                    return True
                elif status == 'Failed':
                    logger.error("❌ HPO job failed")
                    return False
                elif status in ['Stopping', 'Stopped']:
                    logger.warning("⚠️ HPO job was stopped")
                    return False
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Error monitoring HPO job: {e}")
                return False
        
        logger.warning(f"⏰ HPO job monitoring timed out after {timeout_minutes} minutes")
        return False
    
    def step_5_deploy_best_model(self) -> bool:
        """Step 5: Deploy best model to endpoint"""
        logger.info("5️⃣ Deploying best model...")
        self.pipeline_state['current_step'] = 'deploy_model'
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        endpoint_name = f"conviction-hpo-{timestamp}"
        self.pipeline_state['endpoint_name'] = endpoint_name
        
        args = ['--endpoint-name', endpoint_name]
        if self.dry_run:
            args.append('--dry-run')
        
        success = self.run_script('python', ['scripts/deploy_best_model.py'] + args)
        if success:
            self.pipeline_state['steps_completed'].append('deploy_model')
        return success
    
    def step_6_cleanup_resources(self) -> bool:
        """Step 6: Cleanup old resources"""
        logger.info("6️⃣ Cleaning up old resources...")
        self.pipeline_state['current_step'] = 'cleanup'
        
        args = ['--include-sagemaker', '--include-s3']
        if self.dry_run:
            args.append('--dry-run')
        
        success = self.run_script('python', ['scripts/automated_cleanup.py'] + args)
        if success:
            self.pipeline_state['steps_completed'].append('cleanup')
        return success
    
    def save_pipeline_state(self):
        """Save pipeline state to file"""
        state_file = f"pipeline_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2)
        logger.info(f"💾 Pipeline state saved to: {state_file}")
    
    def execute_pipeline(self, job_type: str = "full", skip_hpo: bool = False) -> bool:
        """Execute the complete HPO pipeline DAG"""
        logger.info("🚀 Starting HPO Pipeline Orchestration")
        logger.info(f"📋 Configuration: dry_run={self.dry_run}, job_type={job_type}, email={self.email}")
        
        required_permissions = [
            "sagemaker:*",
            "s3:*",
            "cloudwatch:*",
            "sns:*",
            "cloudformation:*"
        ]
        validate_iam_permissions(required_permissions, self.dry_run)
        
        steps = [
            ("Setup Monitoring", self.step_1_setup_monitoring),
            ("Backup Dataset", self.step_2_backup_dataset),
        ]
        
        if not skip_hpo:
            steps.extend([
                ("Launch HPO", lambda: self.step_3_launch_hpo(job_type)),
                ("Monitor HPO", self.step_4_monitor_hpo),
            ])
        
        steps.extend([
            ("Deploy Model", self.step_5_deploy_best_model),
            ("Cleanup Resources", self.step_6_cleanup_resources),
        ])
        
        for step_name, step_func in steps:
            logger.info(f"🔄 Executing step: {step_name}")
            
            try:
                success = step_func()
                if not success:
                    logger.error(f"❌ Pipeline failed at step: {step_name}")
                    self.save_pipeline_state()
                    return False
                
                logger.info(f"✅ Step completed: {step_name}")
                
            except Exception as e:
                logger.error(f"❌ Step failed with exception: {step_name} - {e}")
                self.save_pipeline_state()
                return False
        
        self.pipeline_state['end_time'] = datetime.now().isoformat()
        self.pipeline_state['status'] = 'completed'
        self.save_pipeline_state()
        
        logger.info("🎉 HPO Pipeline completed successfully!")
        logger.info(f"📊 Steps completed: {len(self.pipeline_state['steps_completed'])}")
        logger.info(f"🔗 HPO Job: {self.pipeline_state.get('hpo_job_name', 'N/A')}")
        logger.info(f"🚀 Endpoint: {self.pipeline_state.get('endpoint_name', 'N/A')}")
        
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Orchestrate HPO pipeline execution')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be executed without running')
    parser.add_argument('--email', default='amr522@gmail.com', help='Email for notifications')
    parser.add_argument('--job-type', choices=['aapl', 'full'], default='full', help='HPO job type')
    parser.add_argument('--skip-hpo', action='store_true', help='Skip HPO launch and monitoring (use existing pinned model)')
    args = parser.parse_args()
    
    orchestrator = HPOPipelineOrchestrator(dry_run=args.dry_run, email=args.email)
    success = orchestrator.execute_pipeline(job_type=args.job_type, skip_hpo=args.skip_hpo)
    
    if not success:
        sys.exit(1)
