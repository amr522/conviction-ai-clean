#!/bin/bash
set -euo pipefail

EMAIL=${1:-"amr522@gmail.com"}

echo "🔧 Setting up CloudWatch monitoring for HPO pipeline..."

cd "$(dirname "$0")/.."

TOPIC_ARN=$(python setup_monitoring.py create-sns-topic --topic-name "hpo-pipeline-alerts" --email "$EMAIL" 2>&1 | grep "arn:aws:sns" | tail -1 || echo "")

if [[ -z "$TOPIC_ARN" ]]; then
    echo "❌ Failed to create SNS topic, creating manually..."
    TOPIC_ARN=$(aws sns create-topic --name "hpo-pipeline-alerts" --query 'TopicArn' --output text)
    if [[ -n "$EMAIL" ]]; then
        aws sns subscribe --topic-arn "$TOPIC_ARN" --protocol email --notification-endpoint "$EMAIL"
        echo "📧 Subscription request sent to $EMAIL"
    fi
fi

echo "✅ SNS Topic: $TOPIC_ARN"

aws cloudwatch put-metric-alarm \
    --alarm-name "HPO-Training-Job-Failures" \
    --alarm-description "Alert when HPO training jobs fail" \
    --metric-name "TrainingJobsFailed" \
    --namespace "AWS/SageMaker" \
    --statistic "Sum" \
    --period 300 \
    --threshold 1 \
    --comparison-operator "GreaterThanOrEqualToThreshold" \
    --evaluation-periods 1 \
    --alarm-actions "$TOPIC_ARN"

aws cloudwatch put-metric-alarm \
    --alarm-name "HPO-Job-Failures" \
    --alarm-description "Alert when HPO jobs fail" \
    --metric-name "HyperParameterTuningJobsFailed" \
    --namespace "AWS/SageMaker" \
    --statistic "Sum" \
    --period 300 \
    --threshold 1 \
    --comparison-operator "GreaterThanOrEqualToThreshold" \
    --evaluation-periods 1 \
    --alarm-actions "$TOPIC_ARN"

echo "✅ CloudWatch alarms created for HPO monitoring"
