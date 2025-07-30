#!/usr/bin/env bash
set -euo pipefail

# Deploy AWS infrastructure with Terraform
TERRAFORM_DIR="infra/terraform"
ACTION=${1:-"plan"}
ENVIRONMENT=${2:-"dev"}

echo "🏗️ Deploying AWS infrastructure for Conviction AI"
echo "Action: $ACTION"
echo "Environment: $ENVIRONMENT"

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform not found. Please install Terraform first."
    echo "Visit: https://developer.hashicorp.com/terraform/downloads"
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Navigate to Terraform directory
cd "$TERRAFORM_DIR"

# Initialize Terraform if not already done
if [ ! -d ".terraform" ]; then
    echo "📋 Initializing Terraform..."
    terraform init
fi

# Validate Terraform configuration
echo "🔍 Validating Terraform configuration..."
terraform validate

# Format Terraform files
echo "📝 Formatting Terraform files..."
terraform fmt -recursive

# Create terraform.tfvars if it doesn't exist
if [ ! -f "terraform.tfvars" ]; then
    echo "⚠️ terraform.tfvars not found. Creating from example..."
    cp terraform.tfvars.example terraform.tfvars
    echo "📝 Please edit terraform.tfvars with your specific values before proceeding."
    exit 1
fi

# Execute Terraform action
case $ACTION in
    "plan")
        echo "📊 Running Terraform plan..."
        terraform plan -var="environment=$ENVIRONMENT"
        ;;
    "apply")
        echo "🚀 Applying Terraform configuration..."
        terraform apply -var="environment=$ENVIRONMENT" -auto-approve

        # Output important information
        echo ""
        echo "✅ Infrastructure deployment completed!"
        echo ""
        echo "📋 Important outputs:"
        terraform output
        echo ""
        echo "🔧 Configure kubectl:"
        echo "$(terraform output -raw kubectl_config_command)"
        echo ""
        echo "📊 CloudWatch Dashboard:"
        if [ "$(terraform output -raw cloudwatch_dashboard_url)" != "null" ]; then
            echo "$(terraform output -raw cloudwatch_dashboard_url)"
        else
            echo "Monitoring disabled"
        fi
        ;;
    "destroy")
        echo "🗑️ Destroying infrastructure..."
        echo "⚠️ This will delete all resources. Are you sure? (y/N)"
        read -r confirmation
        if [[ $confirmation =~ ^[Yy]$ ]]; then
            terraform destroy -var="environment=$ENVIRONMENT" -auto-approve
            echo "✅ Infrastructure destroyed"
        else
            echo "❌ Destruction cancelled"
        fi
        ;;
    "output")
        echo "📋 Terraform outputs:"
        terraform output
        ;;
    *)
        echo "❌ Unknown action: $ACTION"
        echo "Usage: $0 [plan|apply|destroy|output] [environment]"
        exit 1
        ;;
esac

cd - > /dev/null

echo "🎉 Terraform operation completed!"
