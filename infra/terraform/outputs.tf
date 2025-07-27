output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "eks_cluster_arn" {
  description = "ARN of the EKS cluster"
  value       = module.eks.cluster_arn
}

output "eks_cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "eks_oidc_provider_arn" {
  description = "ARN of the OIDC Provider for EKS"
  value       = module.eks.oidc_provider_arn
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.arn
}

output "glue_database_name" {
  description = "Name of the Glue catalog database"
  value       = aws_glue_catalog_database.conviction_ai.name
}

output "glue_job_name" {
  description = "Name of the Glue ETL job"
  value       = aws_glue_job.etl_job.name
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "lambda_execution_role_arn" {
  description = "ARN of the Lambda execution role"
  value       = aws_iam_role.lambda_execution_role.arn
}

output "step_functions_role_arn" {
  description = "ARN of the Step Functions execution role"
  value       = aws_iam_role.step_functions_role.arn
}

output "eks_service_account_role_arn" {
  description = "ARN of the EKS service account role"
  value       = aws_iam_role.eks_service_account_role.arn
}

output "cloudwatch_dashboard_url" {
  description = "URL of the CloudWatch dashboard"
  value       = var.enable_monitoring ? "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${var.project_name}-pipeline-dashboard" : null
}

output "sns_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = var.enable_monitoring ? aws_sns_topic.alerts[0].arn : null
}

# Kubectl configuration command
output "kubectl_config_command" {
  description = "Command to configure kubectl for the EKS cluster"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

# Environment variables for applications
output "environment_variables" {
  description = "Environment variables for ML pipeline applications"
  value = {
    AWS_REGION           = var.aws_region
    S3_BUCKET_NAME       = aws_s3_bucket.ml_artifacts.bucket
    GLUE_DATABASE        = aws_glue_catalog_database.conviction_ai.name
    EKS_CLUSTER_NAME     = module.eks.cluster_name
    SAGEMAKER_ROLE_ARN   = aws_iam_role.sagemaker_execution_role.arn
    ENVIRONMENT          = var.environment
  }
}

# Cost management outputs
output "budget_sns_topic_arn" {
  description = "ARN of the SNS topic for budget alerts"
  value       = var.enable_cost_management ? aws_sns_topic.budget_alerts[0].arn : null
}

output "monthly_budget_name" {
  description = "Name of the monthly budget"
  value       = var.enable_cost_management ? aws_budgets_budget.monthly[0].name : null
}

output "daily_budget_name" {
  description = "Name of the daily budget"
  value       = var.enable_cost_management ? aws_budgets_budget.daily[0].name : null
}

output "cost_anomaly_detector_arn" {
  description = "ARN of the cost anomaly detector"
  value       = var.enable_cost_management ? aws_ce_anomaly_detector.service_monitor[0].arn : null
}

output "cost_dashboard_url" {
  description = "URL of the cost monitoring dashboard"
  value       = var.enable_cost_management && var.enable_monitoring ? "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${var.project_name}-cost-monitoring" : null
}

# AWS Config outputs
output "config_recorder_name" {
  description = "Name of the AWS Config recorder"
  value       = var.enable_config ? aws_config_configuration_recorder.main[0].name : null
}

output "config_delivery_channel_name" {
  description = "Name of the AWS Config delivery channel"
  value       = var.enable_config ? aws_config_delivery_channel.main[0].name : null
}

output "config_rules" {
  description = "List of AWS Config rule names"
  value = var.enable_config ? [
    aws_config_config_rule.s3_encryption[0].name,
    aws_config_config_rule.s3_public_read[0].name,
    aws_config_config_rule.s3_public_write[0].name,
    aws_config_config_rule.eks_version[0].name,
    aws_config_config_rule.eks_endpoint_access[0].name,
    aws_config_config_rule.iam_password_policy[0].name,
    aws_config_config_rule.root_access_key[0].name
  ] : []
}

output "config_compliance_dashboard_url" {
  description = "URL of the Config compliance dashboard"
  value       = var.enable_config && var.enable_monitoring ? "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${var.project_name}-config-compliance" : null
}
