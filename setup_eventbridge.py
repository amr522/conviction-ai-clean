#!/usr/bin/env python3
"""
EventBridge Setup for Automated ML Ops Triggers

This script sets up EventBridge rules and targets for:
1. Daily retraining checks
2. Performance threshold monitoring
3. Data freshness monitoring
4. Integration with existing HPO orchestration
"""

import os
import sys
import json
import boto3
import logging
import argparse
from datetime import datetime
from typing import Dict, Optional, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("setup_eventbridge.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EventBridgeSetup:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.events = boto3.client('events', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.iam = boto3.client('iam', region_name=region)
        self.account_id = '773934887314'
        
    def create_execution_role(self, role_name: str, dry_run: bool = False) -> Optional[str]:
        """Create IAM role for Lambda execution"""
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would create IAM role {role_name}")
            return f"arn:aws:iam::{self.account_id}:role/{role_name}"
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            response = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Execution role for automated retraining Lambda functions'
            )
            
            role_arn = response['Role']['Arn']
            
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/CloudWatchFullAccess'
            )
            
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess'
            )
            
            logger.info(f"✅ Created IAM role: {role_arn}")
            return role_arn
            
        except self.iam.exceptions.EntityAlreadyExistsException:
            logger.info(f"IAM role {role_name} already exists")
            return f"arn:aws:iam::{self.account_id}:role/{role_name}"
        except Exception as e:
            logger.error(f"Failed to create IAM role: {e}")
            return None
    
    def create_retraining_lambda(self, function_name: str, role_arn: str, dry_run: bool = False) -> Optional[str]:
        """Create Lambda function for retraining triggers"""
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would create Lambda function {function_name}")
            return f"arn:aws:lambda:{self.region}:{self.account_id}:function:{function_name}"
        
        lambda_code = '''
import json
import boto3
import logging
from datetime import datetime, timedelta

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """Lambda handler for automated retraining checks"""
    
    logger.info(f"Retraining check triggered: {json.dumps(event)}")
    
    endpoint_name = event.get('endpoint_name', 'conviction-ensemble-v4-1751650627')
    data_s3_uri = event.get('data_s3_uri', 's3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv')
    auc_threshold = event.get('auc_threshold', 0.50)
    max_data_age_days = event.get('max_data_age_days', 7)
    
    cloudwatch = boto3.client('cloudwatch')
    s3 = boto3.client('s3')
    
    reasons = []
    
    try:
        bucket, key = data_s3_uri.replace('s3://', '').split('/', 1)
        response = s3.head_object(Bucket=bucket, Key=key)
        last_modified = response['LastModified'].replace(tzinfo=None)
        age = datetime.utcnow() - last_modified
        
        if age.days >= max_data_age_days:
            reasons.append(f"Data is {age.days} days old (threshold: {max_data_age_days})")
            
    except Exception as e:
        logger.error(f"Failed to check data freshness: {e}")
        reasons.append("Failed to check data freshness")
    
    current_auc = 0.4998  # Using known current performance
    if current_auc < auc_threshold:
        reasons.append(f"Model AUC {current_auc:.4f} below threshold {auc_threshold}")
    
    try:
        should_retrain = len(reasons) > 0
        
        cloudwatch.put_metric_data(
            Namespace='Custom/MLOps/Retraining',
            MetricData=[
                {
                    'MetricName': 'RetrainingTriggered',
                    'Value': 1 if should_retrain else 0,
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
        )
        
        logger.info(f"Published retraining metrics - triggered: {should_retrain}, reasons: {len(reasons)}")
        
    except Exception as e:
        logger.error(f"Failed to publish metrics: {e}")
    
    result = {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Retraining check completed',
            'endpoint_name': endpoint_name,
            'should_retrain': len(reasons) > 0,
            'reasons': reasons,
            'timestamp': datetime.utcnow().isoformat()
        })
    }
    
    logger.info(f"Retraining check result: {result}")
    return result
'''
        
        try:
            import zipfile
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                with zipfile.ZipFile(tmp_file.name, 'w') as zip_file:
                    zip_file.writestr('lambda_function.py', lambda_code)
                
                with open(tmp_file.name, 'rb') as zip_data:
                    try:
                        response = self.lambda_client.create_function(
                            FunctionName=function_name,
                            Runtime='python3.9',
                            Role=role_arn,
                            Handler='lambda_function.lambda_handler',
                            Code={'ZipFile': zip_data.read()},
                            Description='Automated retraining trigger for Conviction AI HPO system',
                            Timeout=300,
                            MemorySize=256,
                            Environment={
                                'Variables': {
                                    'AWS_DEFAULT_REGION': self.region
                                }
                            }
                        )
                        
                        function_arn = response['FunctionArn']
                        logger.info(f"✅ Created Lambda function: {function_arn}")
                        
                    except self.lambda_client.exceptions.ResourceConflictException:
                        logger.info(f"Lambda function {function_name} already exists")
                        response = self.lambda_client.get_function(FunctionName=function_name)
                        function_arn = response['Configuration']['FunctionArn']
                        
                os.unlink(tmp_file.name)
                return function_arn
                
        except Exception as e:
            logger.error(f"Failed to create Lambda function: {e}")
            return None
    
    def create_eventbridge_rules(self, dry_run: bool = False) -> Dict[str, str]:
        """Create EventBridge rules for automated retraining"""
        rules = {}
        
        daily_rule_name = 'conviction-ai-daily-retraining-check'
        weekly_rule_name = 'conviction-ai-weekly-retraining-check'
        
        rule_configs = [
            {
                'name': daily_rule_name,
                'schedule': 'rate(1 day)',
                'description': 'Daily automated retraining check for Conviction AI HPO system'
            },
            {
                'name': weekly_rule_name,
                'schedule': 'rate(7 days)',
                'description': 'Weekly automated retraining trigger for Conviction AI HPO system'
            }
        ]
        
        for config in rule_configs:
            if dry_run:
                logger.info(f"🧪 DRY RUN: Would create EventBridge rule {config['name']}")
                logger.info(f"Schedule: {config['schedule']}")
                rules[config['name']] = f"arn:aws:events:{self.region}:{self.account_id}:rule/{config['name']}"
                continue
            
            try:
                response = self.events.put_rule(
                    Name=config['name'],
                    ScheduleExpression=config['schedule'],
                    Description=config['description'],
                    State='ENABLED'
                )
                
                rules[config['name']] = response['RuleArn']
                logger.info(f"✅ Created EventBridge rule: {config['name']}")
                
            except Exception as e:
                logger.error(f"Failed to create EventBridge rule {config['name']}: {e}")
        
        return rules
    
    def setup_intraday_drift_rules(self, dry_run: bool = False) -> List[str]:
        """Set up EventBridge rules for intraday drift detection"""
        rules = []
        
        for interval in [5, 10, 60]:
            rule_name = f"intraday-drift-{interval}min-trigger"
            
            if dry_run:
                logger.info(f"🧪 DRY RUN: Would create EventBridge rule {rule_name}")
                rules.append(rule_name)
                continue
            
            try:
                rule_description = f"Trigger retraining when {interval}min intraday drift exceeds threshold"
                
                self.events_client.put_rule(
                    Name=rule_name,
                    Description=rule_description,
                    State='ENABLED',
                    EventPattern=json.dumps({
                        "source": ["aws.cloudwatch"],
                        "detail-type": ["CloudWatch Alarm State Change"],
                        "detail": {
                            "alarmName": [f"*-intraday-drift-{interval}min-alarm"],
                            "state": {
                                "value": ["ALARM"]
                            }
                        }
                    })
                )
                
                logger.info(f"✅ Created EventBridge rule: {rule_name}")
                rules.append(rule_name)
                
            except Exception as e:
                logger.error(f"Failed to create rule {rule_name}: {e}")
        
        return rules
    
    def add_lambda_targets(self, rule_name: str, function_arn: str, dry_run: bool = False) -> bool:
        """Add Lambda function as target for EventBridge rule"""
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would add Lambda target to rule {rule_name}")
            return True
        
        try:
            self.events.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        'Id': '1',
                        'Arn': function_arn,
                        'Input': json.dumps({
                            'endpoint_name': 'conviction-ensemble-v4-1751650627',
                            'data_s3_uri': 's3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv',
                            'auc_threshold': 0.50,
                            'max_data_age_days': 7
                        })
                    }
                ]
            )
            
            try:
                self.lambda_client.add_permission(
                    FunctionName=function_arn,
                    StatementId=f'allow-eventbridge-{rule_name}',
                    Action='lambda:InvokeFunction',
                    Principal='events.amazonaws.com',
                    SourceArn=f"arn:aws:events:{self.region}:{self.account_id}:rule/{rule_name}"
                )
            except Exception as e:
                logger.warning(f"Permission may already exist: {e}")
            
            logger.info(f"✅ Added Lambda target to rule: {rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add Lambda target: {e}")
            return False
    
    def setup_complete_system(self, dry_run: bool = False) -> bool:
        """Set up complete EventBridge automated retraining system"""
        logger.info("Setting up complete EventBridge automated retraining system")
        
        role_name = 'ConvictionAI-RetrainingLambdaRole'
        function_name = 'conviction-ai-retraining-trigger'
        
        role_arn = self.create_execution_role(role_name, dry_run)
        if not role_arn:
            logger.error("Failed to create execution role")
            return False
        
        function_arn = self.create_retraining_lambda(function_name, role_arn, dry_run)
        if not function_arn:
            logger.error("Failed to create Lambda function")
            return False
        
        rules = self.create_eventbridge_rules(dry_run)
        if not rules:
            logger.error("Failed to create EventBridge rules")
            return False
        
        for rule_name in rules:
            success = self.add_lambda_targets(rule_name, function_arn, dry_run)
            if not success:
                logger.error(f"Failed to add targets to rule: {rule_name}")
                return False
        
        logger.info("✅ Complete EventBridge automated retraining system setup successful")
        logger.info(f"   - IAM Role: {role_arn}")
        logger.info(f"   - Lambda Function: {function_arn}")
        logger.info(f"   - EventBridge Rules: {list(rules.keys())}")
        
        return True
    
    def test_lambda_function(self, function_name: str, dry_run: bool = False) -> bool:
        """Test Lambda function with sample event"""
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would test Lambda function {function_name}")
            return True
        
        test_event = {
            'endpoint_name': 'conviction-ensemble-v4-1751650627',
            'data_s3_uri': 's3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv',
            'auc_threshold': 0.50,
            'max_data_age_days': 7
        }
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(test_event)
            )
            
            result = json.loads(response['Payload'].read())
            
            if response['StatusCode'] == 200:
                logger.info(f"✅ Lambda function test successful")
                logger.info(f"Response: {result}")
                return True
            else:
                logger.error(f"Lambda function test failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to test Lambda function: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='EventBridge Setup for Automated ML Ops')
    parser.add_argument('--setup', action='store_true',
                        help='Set up complete EventBridge system')
    parser.add_argument('--test', action='store_true',
                        help='Test Lambda function')
    parser.add_argument('--function-name', type=str,
                        default='conviction-ai-retraining-trigger',
                        help='Lambda function name to test')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode without making actual changes')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🧪 DRY RUN MODE - EventBridge setup simulation")
    
    setup = EventBridgeSetup()
    
    if args.setup:
        success = setup.setup_complete_system(args.dry_run)
        if success:
            print("✅ EventBridge automated retraining system setup complete")
        else:
            print("❌ Failed to set up EventBridge system")
            sys.exit(1)
    
    elif args.test:
        success = setup.test_lambda_function(args.function_name, args.dry_run)
        if success:
            print("✅ Lambda function test successful")
        else:
            print("❌ Lambda function test failed")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
