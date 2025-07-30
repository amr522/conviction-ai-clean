#!/usr/bin/env bash
set -euo pipefail

# Check AWS Config compliance status
REGION=${AWS_REGION:-"us-east-1"}
PROJECT_NAME=${1:-"conviction-ai"}

echo "🔍 Checking AWS Config compliance status"
echo "Region: $REGION"
echo "Project: $PROJECT_NAME"

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if Config is enabled
echo "📋 Checking Config service status..."
if aws configservice describe-configuration-recorders --region "$REGION" --query 'ConfigurationRecorders[0].name' --output text 2>/dev/null | grep -q "$PROJECT_NAME"; then
    echo "✅ Config recorder is active"
else
    echo "❌ Config recorder not found or inactive"
    echo "Run: terraform apply to enable Config"
    exit 1
fi

# Get compliance summary
echo ""
echo "📊 Compliance Summary:"
echo "======================"

# Function to check rule compliance
check_rule_compliance() {
    local rule_name=$1
    local display_name=$2

    if aws configservice get-compliance-details-by-config-rule \
        --config-rule-name "$rule_name" \
        --region "$REGION" \
        --query 'EvaluationResults[0].ComplianceType' \
        --output text 2>/dev/null | grep -q "COMPLIANT"; then
        echo "✅ $display_name: COMPLIANT"
    else
        echo "❌ $display_name: NON-COMPLIANT"
    fi
}

# Check each compliance rule
check_rule_compliance "${PROJECT_NAME}-s3-encryption-enabled" "S3 Encryption"
check_rule_compliance "${PROJECT_NAME}-s3-public-read-prohibited" "S3 Public Read"
check_rule_compliance "${PROJECT_NAME}-s3-public-write-prohibited" "S3 Public Write"
check_rule_compliance "${PROJECT_NAME}-eks-version-compliance" "EKS Version"
check_rule_compliance "${PROJECT_NAME}-eks-endpoint-access-public-disabled" "EKS Endpoint Access"
check_rule_compliance "${PROJECT_NAME}-iam-password-policy" "IAM Password Policy"
check_rule_compliance "${PROJECT_NAME}-root-access-key-check" "Root Access Key"

echo ""
echo "📈 Detailed Compliance Report:"
echo "=============================="

# Get detailed compliance for all rules
aws configservice get-compliance-summary-by-config-rule \
    --region "$REGION" \
    --query 'ComplianceSummary' \
    --output table 2>/dev/null || echo "No compliance data available"

echo ""
echo "🔧 Non-Compliant Resources:"
echo "=========================="

# List non-compliant resources
aws configservice get-compliance-details-by-config-rule \
    --config-rule-name "${PROJECT_NAME}-s3-encryption-enabled" \
    --compliance-types NON_COMPLIANT \
    --region "$REGION" \
    --query 'EvaluationResults[*].[EvaluationResultIdentifier.EvaluationResultQualifier.ResourceId,ComplianceType]' \
    --output table 2>/dev/null || echo "No non-compliant S3 buckets found"

echo ""
echo "📊 Config Dashboard URLs:"
echo "========================"
echo "Config Console: https://$REGION.console.aws.amazon.com/config/home?region=$REGION"
echo "Compliance Dashboard: https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards:name=${PROJECT_NAME}-config-compliance"

echo ""
echo "🔄 Remediation Status:"
echo "====================="

# Check remediation executions
aws configservice describe-remediation-executions \
    --config-rule-name "${PROJECT_NAME}-s3-encryption-enabled" \
    --region "$REGION" \
    --query 'RemediationExecutions[*].[ResourceKey.resourceId,State,StepDetails[0].State]' \
    --output table 2>/dev/null || echo "No remediation executions found"

echo ""
echo "💡 Next Steps:"
echo "============="
echo "1. Review non-compliant resources above"
echo "2. Check Config console for detailed findings"
echo "3. Remediation will run automatically for supported rules"
echo "4. Manual remediation may be required for some violations"
