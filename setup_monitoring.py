#!/usr/bin/env python3
"""
setup_monitoring.py - Set up CloudWatch monitoring and alerts for ML pipeline

This script:
1. Creates CloudWatch alarms for model performance metrics
2. Sets up SNS notifications for pipeline failures and model degradation
3. Configures data drift detection and monitoring
"""

import os
import sys
import argparse
import logging
import json
import boto3
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_logs/monitoring.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def ensure_directory(directory):
    """Ensure a directory exists"""
    os.makedirs(directory, exist_ok=True)

def create_sns_topic(topic_name, email=None):
    """Create an SNS topic and subscribe an email address if provided"""
    logger.info(f"Creating SNS topic {topic_name}")
    
    # Create SNS client
    sns_client = boto3.client('sns')
    
    try:
        # Create the SNS topic
        response = sns_client.create_topic(Name=topic_name)
        topic_arn = response['TopicArn']
        
        logger.info(f"Successfully created SNS topic: {topic_arn}")
        
        # Subscribe email if provided
        if email:
            logger.info(f"Subscribing email {email} to topic")
            sns_client.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email
            )
            logger.info(f"Subscription request sent to {email}")
        
        return topic_arn
    except Exception as e:
        logger.error(f"Failed to create SNS topic: {e}")
        return None

def create_performance_alarm(metric_name, endpoint_name, topic_arn, threshold=0.5, period=3600, eval_periods=1):
    """Create a CloudWatch alarm for model performance metrics"""
    logger.info(f"Creating performance alarm for {metric_name} on endpoint {endpoint_name}")
    
    # Create CloudWatch client
    cw_client = boto3.client('cloudwatch')
    
    try:
        alarm_name = f"{endpoint_name}-{metric_name}-alarm"
        
        # Create the alarm
        cw_client.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='LessThanThreshold',
            EvaluationPeriods=eval_periods,
            MetricName=metric_name,
            Namespace='AWS/SageMaker',
            Period=period,
            Statistic='Average',
            Threshold=threshold,
            ActionsEnabled=True,
            AlarmActions=[topic_arn],
            AlarmDescription=f'Alarm when {metric_name} drops below {threshold}',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                }
            ]
        )
        
        logger.info(f"Successfully created alarm {alarm_name}")
        return alarm_name
    except Exception as e:
        logger.error(f"Failed to create performance alarm: {e}")
        return None

def create_pipeline_failure_alarm(pipeline_name, topic_arn, eval_periods=1):
    """Create a CloudWatch alarm for pipeline failures"""
    logger.info(f"Creating pipeline failure alarm for {pipeline_name}")
    
    # Create CloudWatch client
    cw_client = boto3.client('cloudwatch')
    
    try:
        alarm_name = f"{pipeline_name}-failure-alarm"
        
        # Create the alarm
        cw_client.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanOrEqualToThreshold',
            EvaluationPeriods=eval_periods,
            MetricName='FailedPipelineExecutionCount',
            Namespace='AWS/CodePipeline',
            Period=300,  # 5 minutes
            Statistic='Sum',
            Threshold=1,
            ActionsEnabled=True,
            AlarmActions=[topic_arn],
            AlarmDescription=f'Alarm when {pipeline_name} pipeline fails',
            Dimensions=[
                {
                    'Name': 'PipelineName',
                    'Value': pipeline_name
                }
            ]
        )
        
        logger.info(f"Successfully created pipeline failure alarm {alarm_name}")
        return alarm_name
    except Exception as e:
        logger.error(f"Failed to create pipeline failure alarm: {e}")
        return None

def create_data_drift_detection(endpoint_name, topic_arn, feature_baselines, threshold=0.1):
    """Create custom CloudWatch metrics and alarms for data drift detection"""
    logger.info(f"Setting up data drift detection for endpoint {endpoint_name}")
    
    # Create CloudWatch client
    cw_client = boto3.client('cloudwatch')
    
    alarms = []
    try:
        # Create alarms for each feature
        for feature, baseline in feature_baselines.items():
            alarm_name = f"{endpoint_name}-{feature}-drift-alarm"
            
            # Create the alarm
            cw_client.put_metric_alarm(
                AlarmName=alarm_name,
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=1,
                MetricName=f"{feature}_drift",
                Namespace='Custom/ModelMonitoring',
                Period=3600,  # 1 hour
                Statistic='Average',
                Threshold=threshold,
                ActionsEnabled=True,
                AlarmActions=[topic_arn],
                AlarmDescription=f'Alarm when {feature} drifts more than {threshold} from baseline',
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ]
            )
            
            logger.info(f"Successfully created data drift alarm {alarm_name}")
            alarms.append(alarm_name)
        
        return alarms
    except Exception as e:
        logger.error(f"Failed to create data drift detection: {e}")
        return []

def setup_sagemaker_monitoring(endpoint_name, topic_arn, baseline_uri, schedule="daily"):
    """Set up SageMaker Model Monitor for data quality"""
    logger.info(f"Setting up SageMaker monitoring for endpoint {endpoint_name}")
    
    # Create SageMaker client
    sm_client = boto3.client('sagemaker')
    
    try:
        # Create monitoring schedule
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        schedule_name = f"{endpoint_name}-monitoring-{timestamp}"
        
        # Set schedule expression based on frequency
        if schedule.lower() == "hourly":
            schedule_expression = "cron(0 * ? * * *)"
        elif schedule.lower() == "daily":
            schedule_expression = "cron(0 0 ? * * *)"
        elif schedule.lower() == "weekly":
            schedule_expression = "cron(0 0 ? * 1 *)"
        else:
            schedule_expression = "cron(0 0 ? * * *)"  # Default to daily
        
        # Create data quality monitoring schedule
        sm_client.create_monitoring_schedule(
            MonitoringScheduleName=schedule_name,
            MonitoringScheduleConfig={
                "ScheduleConfig": {
                    "ScheduleExpression": schedule_expression
                },
                "MonitoringJobDefinition": {
                    "BaselineConfig": {
                        "ConstraintsResource": {
                            "S3Uri": f"{baseline_uri}/constraints.json"
                        },
                        "StatisticsResource": {
                            "S3Uri": f"{baseline_uri}/statistics.json"
                        }
                    },
                    "MonitoringInputs": [
                        {
                            "EndpointInput": {
                                "EndpointName": endpoint_name,
                                "LocalPath": "/opt/ml/processing/input"
                            }
                        }
                    ],
                    "MonitoringOutputConfig": {
                        "MonitoringOutputs": [
                            {
                                "S3Output": {
                                    "S3Uri": f"s3://{os.environ.get('S3_BUCKET', 'hpo-bucket-773934887314')}/monitoring/{endpoint_name}/",
                                    "LocalPath": "/opt/ml/processing/output"
                                }
                            }
                        ]
                    },
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.xlarge",
                            "VolumeSizeInGB": 20
                        }
                    },
                    "RoleArn": os.environ.get("SAGEMAKER_ROLE_ARN"),
                    "StoppingCondition": {
                        "MaxRuntimeInSeconds": 3600
                    }
                }
            },
            Tags=[
                {
                    "Key": "Endpoint",
                    "Value": endpoint_name
                }
            ]
        )
        
        # Create CloudWatch alarm for monitoring failures
        cw_client = boto3.client('cloudwatch')
        alarm_name = f"{schedule_name}-failure-alarm"
        
        cw_client.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanOrEqualToThreshold',
            EvaluationPeriods=1,
            MetricName='MonitoringScheduleFailedExecution',
            Namespace='AWS/SageMaker',
            Period=300,  # 5 minutes
            Statistic='Sum',
            Threshold=1,
            ActionsEnabled=True,
            AlarmActions=[topic_arn],
            AlarmDescription=f'Alarm when monitoring for {endpoint_name} fails',
            Dimensions=[
                {
                    'Name': 'MonitoringScheduleName',
                    'Value': schedule_name
                }
            ]
        )
        
        logger.info(f"Successfully set up SageMaker monitoring for endpoint {endpoint_name}")
        return schedule_name
    except Exception as e:
        logger.error(f"Failed to set up SageMaker monitoring: {e}")
        return None

def setup_mlops_monitoring(endpoint_name, topic_arn, auc_threshold=0.50, data_age_threshold=7):
    """Set up ML Ops specific monitoring for dashboard and retraining triggers"""
    logger.info(f"Setting up ML Ops monitoring for endpoint {endpoint_name}")
    
    cw_client = boto3.client('cloudwatch')
    
    try:
        auc_alarm_name = f"{endpoint_name}-auc-retraining-alarm"
        
        cw_client.put_metric_alarm(
            AlarmName=auc_alarm_name,
            ComparisonOperator='LessThanThreshold',
            EvaluationPeriods=2,
            MetricName='ModelAUC',
            Namespace='Custom/MLOps',
            Period=3600,
            Statistic='Average',
            Threshold=auc_threshold,
            ActionsEnabled=True,
            AlarmActions=[topic_arn],
            AlarmDescription=f'Alarm when model AUC drops below {auc_threshold} for retraining trigger',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                }
            ]
        )
        
        # Create data freshness alarm
        data_freshness_alarm_name = f"{endpoint_name}-data-freshness-alarm"
        
        cw_client.put_metric_alarm(
            AlarmName=data_freshness_alarm_name,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=1,
            MetricName='DataAgeDays',
            Namespace='Custom/MLOps',
            Period=86400,
            Statistic='Maximum',
            Threshold=data_age_threshold,
            ActionsEnabled=True,
            AlarmActions=[topic_arn],
            AlarmDescription=f'Alarm when training data is older than {data_age_threshold} days',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                }
            ]
        )
        
        # Create dashboard health alarm
        dashboard_health_alarm_name = f"{endpoint_name}-dashboard-health-alarm"
        
        cw_client.put_metric_alarm(
            AlarmName=dashboard_health_alarm_name,
            ComparisonOperator='LessThanThreshold',
            EvaluationPeriods=3,
            MetricName='DashboardHealth',
            Namespace='Custom/MLOps',
            Period=300,
            Statistic='Average',
            Threshold=0.8,
            ActionsEnabled=True,
            AlarmActions=[topic_arn],
            AlarmDescription=f'Alarm when dashboard health score drops below 0.8',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                }
            ]
        )
        
        logger.info(f"Successfully set up ML Ops monitoring alarms:")
        logger.info(f"  - AUC threshold alarm: {auc_alarm_name}")
        logger.info(f"  - Data freshness alarm: {data_freshness_alarm_name}")
        logger.info(f"  - Dashboard health alarm: {dashboard_health_alarm_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up ML Ops monitoring: {e}")
        return False

def main():
    """Main function to parse arguments and execute commands"""
    parser = argparse.ArgumentParser(description="Set up monitoring and alerts for ML pipeline")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # SNS topic command
    sns_parser = subparsers.add_parser("create-sns-topic", help="Create an SNS topic for alerts")
    sns_parser.add_argument("--topic-name", required=True, help="Name of the SNS topic")
    sns_parser.add_argument("--email", help="Email address to subscribe to the topic")
    
    # Performance alarm command
    perf_parser = subparsers.add_parser("create-performance-alarm", help="Create a performance alarm")
    perf_parser.add_argument("--metric-name", required=True, help="Name of the metric to monitor")
    perf_parser.add_argument("--endpoint-name", required=True, help="Name of the SageMaker endpoint")
    perf_parser.add_argument("--topic-arn", required=True, help="ARN of the SNS topic for notifications")
    perf_parser.add_argument("--threshold", type=float, default=0.5, help="Alarm threshold value")
    
    # Pipeline failure alarm command
    pipe_parser = subparsers.add_parser("create-pipeline-alarm", help="Create a pipeline failure alarm")
    pipe_parser.add_argument("--pipeline-name", required=True, help="Name of the CodePipeline")
    pipe_parser.add_argument("--topic-arn", required=True, help="ARN of the SNS topic for notifications")
    
    # Data drift detection command
    drift_parser = subparsers.add_parser("create-drift-detection", help="Create data drift detection")
    drift_parser.add_argument("--endpoint-name", required=True, help="Name of the SageMaker endpoint")
    drift_parser.add_argument("--topic-arn", required=True, help="ARN of the SNS topic for notifications")
    drift_parser.add_argument("--baseline-file", required=True, help="JSON file with feature baselines")
    drift_parser.add_argument("--threshold", type=float, default=0.1, help="Drift threshold value")
    
    # SageMaker monitoring command
    monitor_parser = subparsers.add_parser("setup-sagemaker-monitoring", help="Set up SageMaker monitoring")
    monitor_parser.add_argument("--endpoint-name", required=True, help="Name of the SageMaker endpoint")
    monitor_parser.add_argument("--topic-arn", required=True, help="ARN of the SNS topic for notifications")
    monitor_parser.add_argument("--baseline-uri", required=True, help="S3 URI to baseline statistics and constraints")
    monitor_parser.add_argument("--schedule", default="daily", choices=["hourly", "daily", "weekly"], help="Monitoring schedule")
    
    # Setup MLOps dashboard monitoring command
    mlops_parser = subparsers.add_parser("mlops", help="Set up ML Ops dashboard and retraining monitoring")
    mlops_parser.add_argument("--endpoint-name", required=True, help="Name of the SageMaker endpoint")
    mlops_parser.add_argument("--topic-arn", required=True, help="ARN of the SNS topic for notifications")
    mlops_parser.add_argument("--auc-threshold", type=float, default=0.50, help="AUC threshold for retraining alerts")
    mlops_parser.add_argument("--data-age-threshold", type=int, default=7, help="Data age threshold in days")
    
    # Setup all monitoring command
    all_parser = subparsers.add_parser("setup-all", help="Set up all monitoring components")
    all_parser.add_argument("--endpoint-name", required=True, help="Name of the SageMaker endpoint")
    all_parser.add_argument("--email", required=True, help="Email address for notifications")
    all_parser.add_argument("--baseline-uri", required=True, help="S3 URI to baseline statistics and constraints")
    all_parser.add_argument("--baseline-file", help="JSON file with feature baselines")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure the logs directory exists
    ensure_directory("pipeline_logs")
    
    # Execute command
    if args.command == "create-sns-topic":
        create_sns_topic(args.topic_name, args.email)
    elif args.command == "create-performance-alarm":
        create_performance_alarm(args.metric_name, args.endpoint_name, args.topic_arn, args.threshold)
    elif args.command == "create-pipeline-alarm":
        create_pipeline_failure_alarm(args.pipeline_name, args.topic_arn)
    elif args.command == "create-drift-detection":
        with open(args.baseline_file, 'r') as f:
            baselines = json.load(f)
        create_data_drift_detection(args.endpoint_name, args.topic_arn, baselines, args.threshold)
    elif args.command == "setup-sagemaker-monitoring":
        setup_sagemaker_monitoring(args.endpoint_name, args.topic_arn, args.baseline_uri, args.schedule)
    elif args.command == "setup-all":
        # Create SNS topic
        topic_name = f"{args.endpoint_name}-alerts"
        topic_arn = create_sns_topic(topic_name, args.email)
        
        if topic_arn:
            # Create performance alarm
            create_performance_alarm("InvocationsPerInstance", args.endpoint_name, topic_arn)
            create_performance_alarm("ModelLatency", args.endpoint_name, topic_arn, threshold=100)  # 100ms
            
            # Create data drift detection if baseline file provided
            if args.baseline_file:
                with open(args.baseline_file, 'r') as f:
                    baselines = json.load(f)
                create_data_drift_detection(args.endpoint_name, topic_arn, baselines)
            
            # Set up SageMaker monitoring
            setup_sagemaker_monitoring(args.endpoint_name, topic_arn, args.baseline_uri)
            
            logger.info(f"Successfully set up all monitoring for endpoint {args.endpoint_name}")
        else:
            logger.error("Failed to create SNS topic, monitoring setup aborted")
    elif args.command == "mlops":
        success = setup_mlops_monitoring(args.endpoint_name, args.topic_arn, args.auc_threshold, args.data_age_threshold)
        if success:
            print(f"✅ ML Ops monitoring setup complete for {args.endpoint_name}")
        else:
            print("❌ Failed to set up ML Ops monitoring")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
