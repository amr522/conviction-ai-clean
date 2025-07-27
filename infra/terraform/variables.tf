variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "conviction-ai"
}

variable "tfstate_bucket" {
  description = "S3 bucket for Terraform state"
  type        = string
}

variable "s3_bucket_name" {
  description = "S3 bucket for ML artifacts and data"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "conviction-ai-cluster"
}

variable "cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "glue_database_name" {
  description = "AWS Glue catalog database name"
  type        = string
  default     = "conviction_ai"
}

variable "enable_gpu_nodes" {
  description = "Enable GPU node group for ML training"
  type        = bool
  default     = true
}

variable "cpu_node_instance_type" {
  description = "Instance type for CPU nodes"
  type        = string
  default     = "m6i.large"
}

variable "gpu_node_instance_type" {
  description = "Instance type for GPU nodes"
  type        = string
  default     = "g4dn.xlarge"
}

variable "cpu_node_desired_capacity" {
  description = "Desired capacity for CPU node group"
  type        = number
  default     = 2
}

variable "gpu_node_desired_capacity" {
  description = "Desired capacity for GPU node group"
  type        = number
  default     = 1
}

variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring and logging"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

variable "monthly_budget_amount" {
  description = "Monthly cost budget in USD"
  type        = number
  default     = 500
}

variable "budget_email" {
  description = "Email address to receive budget alerts"
  type        = string
}

variable "slack_budget_webhook_url" {
  description = "Slack webhook URL for budget alerts"
  type        = string
  default     = ""
}

variable "enable_cost_management" {
  description = "Enable AWS cost management and budgets"
  type        = bool
  default     = true
}

variable "enable_config" {
  description = "Enable AWS Config for compliance monitoring"
  type        = bool
  default     = true
}

variable "desired_eks_version" {
  description = "Desired EKS cluster version for compliance"
  type        = string
  default     = "1.28"
}