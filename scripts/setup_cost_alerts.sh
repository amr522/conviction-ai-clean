#!/usr/bin/env bash
set -euo pipefail

# Setup AWS cost management and budget alerts
BUDGET_EMAIL=${1:-""}
MONTHLY_BUDGET=${2:-500}
SLACK_WEBHOOK=${3:-""}

if [ -z "$BUDGET_EMAIL" ]; then
    echo "❌ Budget email is required"
    echo "Usage: $0 <budget_email> [monthly_budget] [slack_webhook]"
    echo "Example: $0 admin@company.com 500 https://hooks.slack.com/services/..."
    exit 1
fi

echo "💰 Setting up AWS cost management"
echo "Budget Email: $BUDGET_EMAIL"
echo "Monthly Budget: \$${MONTHLY_BUDGET}"
echo "Slack Webhook: ${SLACK_WEBHOOK:-"Not provided"}"

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Update Terraform variables
TFVARS_FILE="infra/terraform/terraform.tfvars"

if [ ! -f "$TFVARS_FILE" ]; then
    echo "📝 Creating terraform.tfvars from example..."
    cp infra/terraform/terraform.tfvars.example "$TFVARS_FILE"
fi

# Update budget configuration
echo "📝 Updating Terraform variables..."

# Update budget email
if grep -q "budget_email" "$TFVARS_FILE"; then
    sed -i.bak "s/budget_email = .*/budget_email = \"$BUDGET_EMAIL\"/" "$TFVARS_FILE"
else
    echo "budget_email = \"$BUDGET_EMAIL\"" >> "$TFVARS_FILE"
fi

# Update monthly budget
if grep -q "monthly_budget_amount" "$TFVARS_FILE"; then
    sed -i.bak "s/monthly_budget_amount = .*/monthly_budget_amount = $MONTHLY_BUDGET/" "$TFVARS_FILE"
else
    echo "monthly_budget_amount = $MONTHLY_BUDGET" >> "$TFVARS_FILE"
fi

# Update Slack webhook if provided
if [ -n "$SLACK_WEBHOOK" ]; then
    if grep -q "slack_budget_webhook_url" "$TFVARS_FILE"; then
        sed -i.bak "s|slack_budget_webhook_url = .*|slack_budget_webhook_url = \"$SLACK_WEBHOOK\"|" "$TFVARS_FILE"
    else
        echo "slack_budget_webhook_url = \"$SLACK_WEBHOOK\"" >> "$TFVARS_FILE"
    fi
fi

# Enable cost management
if grep -q "enable_cost_management" "$TFVARS_FILE"; then
    sed -i.bak "s/enable_cost_management = .*/enable_cost_management = true/" "$TFVARS_FILE"
else
    echo "enable_cost_management = true" >> "$TFVARS_FILE"
fi

# Clean up backup files
rm -f "$TFVARS_FILE.bak"

echo "✅ Terraform variables updated"

# Plan and apply changes
echo "📊 Planning Terraform changes..."
cd infra/terraform

terraform init
terraform plan -target=aws_budgets_budget.monthly -target=aws_sns_topic.budget_alerts

echo ""
echo "🚀 Ready to apply cost management configuration!"
echo ""
echo "Next steps:"
echo "1. Review the Terraform plan above"
echo "2. Apply changes: cd infra/terraform && terraform apply"
echo "3. Confirm email subscription in your inbox"
echo "4. Test Slack integration (if configured)"
echo ""
echo "📊 After deployment, you can view:"
echo "- AWS Budgets: https://console.aws.amazon.com/billing/home#/budgets"
echo "- Cost Explorer: https://console.aws.amazon.com/ce/home"
echo "- CloudWatch Cost Dashboard: Check Terraform outputs"

cd - > /dev/null