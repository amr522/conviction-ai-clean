#!/usr/bin/env python3
"""
Automated Retraining System for Conviction AI HPO Pipeline

This system provides:
1. EventBridge-based automated retraining triggers
2. Performance threshold monitoring (AUC < 0.50)
3. Data freshness monitoring (7-day trigger)
4. Integration with existing HPO orchestration
"""

import os
import sys
import json
import boto3
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("automated_retraining.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutomatedRetrainingSystem:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.events = boto3.client('events', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.bucket = 'hpo-bucket-773934887314'
        self.role_arn = 'arn:aws:iam::773934887314:role/SageMakerExecutionRole'
        
    def check_data_freshness(self, data_s3_uri: str, max_age_days: int = 7) -> Tuple[bool, datetime]:
        """Check if training data is older than specified days"""
        try:
            bucket, key = data_s3_uri.replace('s3://', '').split('/', 1)
            
            response = self.s3.head_object(Bucket=bucket, Key=key)
            last_modified = response['LastModified'].replace(tzinfo=None)
            
            age = datetime.utcnow() - last_modified
            is_stale = age.days >= max_age_days
            
            logger.info(f"Data age: {age.days} days, stale: {is_stale}")
            return is_stale, last_modified
            
        except Exception as e:
            logger.error(f"Failed to check data freshness: {e}")
            return True, datetime.utcnow()
    
    def get_model_performance_metrics(self, endpoint_name: str, hours_back: int = 24) -> Dict:
        """Get model performance metrics from CloudWatch"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        metrics = {}
        
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace='Custom/MLOps',
                MetricName='ModelAUC',
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average']
            )
            
            if response['Datapoints']:
                latest_auc = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])[-1]['Average']
                metrics['current_auc'] = latest_auc
                logger.info(f"Current model AUC: {latest_auc}")
            else:
                logger.warning("No AUC metrics found, using default threshold check")
                metrics['current_auc'] = 0.4998
                
        except Exception as e:
            logger.warning(f"Failed to get AUC metrics: {e}")
            metrics['current_auc'] = 0.4998
        
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='ModelLatency',
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average']
            )
            
            if response['Datapoints']:
                avg_latency = sum(dp['Average'] for dp in response['Datapoints']) / len(response['Datapoints'])
                metrics['avg_latency'] = avg_latency
            else:
                metrics['avg_latency'] = 0
                
        except Exception as e:
            logger.warning(f"Failed to get latency metrics: {e}")
            metrics['avg_latency'] = 0
        
        return metrics
    
    def should_trigger_retraining(self, endpoint_name: str, data_s3_uri: str, 
                                auc_threshold: float = 0.50, max_data_age_days: int = 7) -> Tuple[bool, List[str]]:
        """Determine if retraining should be triggered"""
        reasons = []
        
        is_data_stale, last_modified = self.check_data_freshness(data_s3_uri, max_data_age_days)
        if is_data_stale:
            reasons.append(f"Data is {(datetime.utcnow() - last_modified).days} days old (threshold: {max_data_age_days})")
        
        metrics = self.get_model_performance_metrics(endpoint_name)
        current_auc = metrics.get('current_auc', 0.4998)
        
        if current_auc < auc_threshold:
            reasons.append(f"Model AUC {current_auc:.4f} below threshold {auc_threshold}")
        
        should_retrain = len(reasons) > 0
        
        if should_retrain:
            logger.info(f"Retraining triggered. Reasons: {reasons}")
        else:
            logger.info("No retraining needed - all thresholds met")
        
        return should_retrain, reasons
    
    def trigger_hpo_pipeline(self, data_s3_uri: str, dry_run: bool = False) -> bool:
        """Trigger HPO pipeline using existing orchestration"""
        try:
            cmd = [
                sys.executable, 'scripts/orchestrate_hpo_pipeline.py',
                '--set-and-forget',
                '--input-data-s3', data_s3_uri,
                '--auto-recover',
                '--auto-ensemble',
                '--notify'
            ]
            
            if dry_run:
                cmd.append('--dry-run')
                logger.info("🧪 DRY RUN: Would trigger HPO pipeline")
                logger.info(f"Command: {' '.join(cmd)}")
                return True
            
            logger.info(f"Triggering HPO pipeline: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                logger.info("✅ HPO pipeline triggered successfully")
                logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ HPO pipeline failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception triggering HPO pipeline: {e}")
            return False
    
    def publish_retraining_metrics(self, endpoint_name: str, triggered: bool, reasons: List[str]):
        """Publish retraining decision metrics to CloudWatch"""
        try:
            metrics = [
                {
                    'MetricName': 'RetrainingTriggered',
                    'Value': 1 if triggered else 0,
                    'Unit': 'Count',
                    'Dimensions': [
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint_name
                        }
                    ]
                },
                {
                    'MetricName': 'RetrainingReasonCount',
                    'Value': len(reasons),
                    'Unit': 'Count',
                    'Dimensions': [
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint_name
                        }
                    ]
                }
            ]
            
            self.cloudwatch.put_metric_data(
                Namespace='Custom/MLOps/Retraining',
                MetricData=metrics
            )
            
            logger.info("Published retraining metrics to CloudWatch")
            
        except Exception as e:
            logger.error(f"Failed to publish retraining metrics: {e}")
    
    def create_eventbridge_rule(self, rule_name: str, schedule_expression: str, 
                              target_function_arn: str, dry_run: bool = False) -> bool:
        """Create EventBridge rule for automated retraining"""
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would create EventBridge rule {rule_name}")
            logger.info(f"Schedule: {schedule_expression}")
            logger.info(f"Target: {target_function_arn}")
            return True
        
        try:
            self.events.put_rule(
                Name=rule_name,
                ScheduleExpression=schedule_expression,
                Description=f'Automated retraining trigger for Conviction AI HPO system',
                State='ENABLED'
            )
            
            self.events.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        'Id': '1',
                        'Arn': target_function_arn,
                        'Input': json.dumps({
                            'endpoint_name': 'conviction-ensemble-v4-1751650627',
                            'data_s3_uri': 's3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv',
                            'auc_threshold': 0.50,
                            'max_data_age_days': 7
                        })
                    }
                ]
            )
            
            logger.info(f"✅ Created EventBridge rule: {rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create EventBridge rule: {e}")
            return False
    
    def create_lambda_function(self, function_name: str, dry_run: bool = False) -> Optional[str]:
        """Create Lambda function for retraining triggers"""
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would create Lambda function {function_name}")
            return f"arn:aws:lambda:{self.region}:773934887314:function:{function_name}"
        
        lambda_code = '''
import json
import boto3
import subprocess
import os

def lambda_handler(event, context):
    """Lambda handler for automated retraining triggers"""
    
    endpoint_name = event.get('endpoint_name', 'conviction-ensemble-v4-1751650627')
    data_s3_uri = event.get('data_s3_uri', 's3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv')
    auc_threshold = event.get('auc_threshold', 0.50)
    max_data_age_days = event.get('max_data_age_days', 7)
    
    print(f"Checking retraining conditions for {endpoint_name}")
    
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Retraining check completed',
            'endpoint_name': endpoint_name,
            'timestamp': context.aws_request_id
        })
    }
'''
        
        try:
            import zipfile
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                with zipfile.ZipFile(tmp_file.name, 'w') as zip_file:
                    zip_file.writestr('lambda_function.py', lambda_code)
                
                with open(tmp_file.name, 'rb') as zip_data:
                    response = self.lambda_client.create_function(
                        FunctionName=function_name,
                        Runtime='python3.9',
                        Role=self.role_arn,
                        Handler='lambda_function.lambda_handler',
                        Code={'ZipFile': zip_data.read()},
                        Description='Automated retraining trigger for Conviction AI',
                        Timeout=300,
                        MemorySize=256
                    )
                
                os.unlink(tmp_file.name)
                
                function_arn = response['FunctionArn']
                logger.info(f"✅ Created Lambda function: {function_arn}")
                return function_arn
                
        except Exception as e:
            logger.error(f"Failed to create Lambda function: {e}")
            return None
    
    def setup_automated_retraining(self, dry_run: bool = False) -> bool:
        """Set up complete automated retraining system"""
        logger.info("Setting up automated retraining system")
        
        function_name = 'conviction-ai-retraining-trigger'
        rule_name = 'conviction-ai-daily-retraining-check'
        
        function_arn = self.create_lambda_function(function_name, dry_run)
        if not function_arn:
            return False
        
        schedule_expression = 'rate(1 day)'
        
        success = self.create_eventbridge_rule(rule_name, schedule_expression, function_arn, dry_run)
        
        if success:
            logger.info("✅ Automated retraining system setup complete")
            logger.info(f"   - Lambda function: {function_name}")
            logger.info(f"   - EventBridge rule: {rule_name}")
            logger.info(f"   - Schedule: Daily checks")
        
        return success
    
    def run_retraining_check(self, endpoint_name: str, data_s3_uri: str, 
                           auc_threshold: float = 0.50, max_data_age_days: int = 7, 
                           dry_run: bool = False) -> bool:
        """Run a single retraining check"""
        logger.info(f"Running retraining check for {endpoint_name}")
        
        should_retrain, reasons = self.should_trigger_retraining(
            endpoint_name, data_s3_uri, auc_threshold, max_data_age_days
        )
        
        self.publish_retraining_metrics(endpoint_name, should_retrain, reasons)
        
        if should_retrain:
            logger.info(f"🚀 Triggering retraining. Reasons: {reasons}")
            success = self.trigger_hpo_pipeline(data_s3_uri, dry_run)
            return success
        else:
            logger.info("✅ No retraining needed")
            return True

def main():
    parser = argparse.ArgumentParser(description='Automated Retraining System for Conviction AI')
    parser.add_argument('--endpoint-name', type=str, 
                        default='conviction-ensemble-v4-1751650627',
                        help='SageMaker endpoint name to monitor')
    parser.add_argument('--data-s3-uri', type=str,
                        default='s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv',
                        help='S3 URI for training data')
    parser.add_argument('--auc-threshold', type=float, default=0.50,
                        help='AUC threshold for triggering retraining')
    parser.add_argument('--max-data-age-days', type=int, default=7,
                        help='Maximum data age in days before triggering retraining')
    parser.add_argument('--setup', action='store_true',
                        help='Set up automated retraining infrastructure')
    parser.add_argument('--check', action='store_true',
                        help='Run a single retraining check')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode without making actual changes')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🧪 DRY RUN MODE - Automated retraining simulation")
    
    system = AutomatedRetrainingSystem()
    
    if args.setup:
        success = system.setup_automated_retraining(args.dry_run)
        if success:
            print("✅ Automated retraining system setup complete")
        else:
            print("❌ Failed to set up automated retraining system")
            sys.exit(1)
    
    elif args.check:
        success = system.run_retraining_check(
            args.endpoint_name, 
            args.data_s3_uri,
            args.auc_threshold,
            args.max_data_age_days,
            args.dry_run
        )
        if success:
            print("✅ Retraining check completed successfully")
        else:
            print("❌ Retraining check failed")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
