#!/bin/bash
set -euo pipefail


EMAIL=${1:-"amr522@gmail.com"}
DRY_RUN=${2:-"false"}
STACK_NAME="hpo-monitoring-stack"

echo "🔧 Setting up CloudWatch monitoring for HPO pipeline..."

cd "$(dirname "$0")/.."

REQUIRED_PERMISSIONS=(
    "cloudformation:CreateStack"
    "cloudformation:UpdateStack" 
    "cloudformation:DescribeStacks"
    "sns:CreateTopic"
    "sns:Subscribe"
    "cloudwatch:PutMetricAlarm"
)

if [[ "$DRY_RUN" == "true" ]]; then
    echo "🔍 [DRY-RUN] Would validate IAM permissions: ${REQUIRED_PERMISSIONS[*]}"
    echo "🔍 [DRY-RUN] Would deploy CloudFormation stack: $STACK_NAME"
    echo "🔍 [DRY-RUN] Would create SNS subscription for: $EMAIL"
    echo "🔍 [DRY-RUN] Would create CloudWatch alarms for HPO monitoring"
    exit 0
fi

if aws cloudformation describe-stacks --stack-name "$STACK_NAME" >/dev/null 2>&1; then
    echo "📋 Stack $STACK_NAME already exists, updating..."
    OPERATION="update-stack"
else
    echo "🆕 Creating new stack: $STACK_NAME"
    OPERATION="create-stack"
fi

aws cloudformation "$OPERATION" \
    --stack-name "$STACK_NAME" \
    --template-body file://cloudformation/hpo-monitoring.yaml \
    --parameters ParameterKey=NotificationEmail,ParameterValue="$EMAIL" \
    --capabilities CAPABILITY_IAM

echo "⏳ Waiting for stack operation to complete..."
if [[ "$OPERATION" == "create-stack" ]]; then
    aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME"
else
    aws cloudformation wait stack-update-complete --stack-name "$STACK_NAME"
fi

SNS_TOPIC_ARN=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query 'Stacks[0].Outputs[?OutputKey==`SNSTopicArn`].OutputValue' \
    --output text)

echo "✅ CloudFormation stack deployed successfully"
echo "📧 SNS Topic ARN: $SNS_TOPIC_ARN"
echo "📧 Email subscription sent to: $EMAIL"
echo "⚠️ Please check your email and confirm the SNS subscription"

echo "🔍 Verifying CloudWatch alarms..."
aws cloudwatch describe-alarms \
    --alarm-names "HPO-Pipeline-Training-Job-Failures" "HPO-Pipeline-Job-Failures" "HPO-Pipeline-Job-Completion" \
    --query 'MetricAlarms[].{Name:AlarmName,State:StateValue}' \
    --output table

echo "✅ HPO monitoring setup completed successfully"
