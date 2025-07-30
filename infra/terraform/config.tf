# AWS Config for compliance monitoring and auto-remediation

# IAM role for AWS Config
resource "aws_iam_role" "config" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-config-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# Attach AWS managed policy for Config
resource "aws_iam_role_policy_attachment" "config" {
  count = var.enable_config ? 1 : 0

  role       = aws_iam_role.config[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/ConfigRole"
}

# Additional policy for Config to access S3 bucket
resource "aws_iam_role_policy" "config_s3" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-config-s3-policy"
  role = aws_iam_role.config[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketAcl",
          "s3:GetBucketLocation",
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.ml_artifacts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.ml_artifacts.arn}/config/*"
      }
    ]
  })
}

# AWS Config configuration recorder
resource "aws_config_configuration_recorder" "main" {
  count = var.enable_config ? 1 : 0

  name     = "${var.project_name}-recorder"
  role_arn = aws_iam_role.config[0].arn

  recording_group {
    all_supported                 = true
    include_global_resource_types = true
  }

  depends_on = [aws_config_delivery_channel.main]
}

# AWS Config delivery channel
resource "aws_config_delivery_channel" "main" {
  count = var.enable_config ? 1 : 0

  name           = "${var.project_name}-channel"
  s3_bucket_name = aws_s3_bucket.ml_artifacts.bucket
  s3_key_prefix  = "config"

  snapshot_delivery_properties {
    delivery_frequency = "TwentyFour_Hours"
  }
}

# Start the configuration recorder
resource "aws_config_configuration_recorder_status" "main" {
  count = var.enable_config ? 1 : 0

  name       = aws_config_configuration_recorder.main[0].name
  is_enabled = true
  depends_on = [aws_config_delivery_channel.main]
}

# Config rule: S3 bucket server-side encryption enabled
resource "aws_config_config_rule" "s3_encryption" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-s3-encryption-enabled"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_SERVER_SIDE_ENCRYPTION_ENABLED"
  }

  depends_on = [aws_config_configuration_recorder.main]

  tags = var.tags
}

# Config rule: S3 bucket public read prohibited
resource "aws_config_config_rule" "s3_public_read" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-s3-public-read-prohibited"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_PUBLIC_READ_PROHIBITED"
  }

  depends_on = [aws_config_configuration_recorder.main]

  tags = var.tags
}

# Config rule: S3 bucket public write prohibited
resource "aws_config_config_rule" "s3_public_write" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-s3-public-write-prohibited"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_PUBLIC_WRITE_PROHIBITED"
  }

  depends_on = [aws_config_configuration_recorder.main]

  tags = var.tags
}

# Config rule: EKS cluster version compliance
resource "aws_config_config_rule" "eks_version" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-eks-version-compliance"

  source {
    owner             = "AWS"
    source_identifier = "EKS_CLUSTER_SUPPORTED_VERSION"
  }

  input_parameters = jsonencode({
    desiredVersion = var.desired_eks_version
  })

  depends_on = [aws_config_configuration_recorder.main]

  tags = var.tags
}

# Config rule: EKS endpoint access public disabled
resource "aws_config_config_rule" "eks_endpoint_access" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-eks-endpoint-access-public-disabled"

  source {
    owner             = "AWS"
    source_identifier = "EKS_ENDPOINT_NO_PUBLIC_ACCESS"
  }

  depends_on = [aws_config_configuration_recorder.main]

  tags = var.tags
}

# Config rule: IAM password policy
resource "aws_config_config_rule" "iam_password_policy" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-iam-password-policy"

  source {
    owner             = "AWS"
    source_identifier = "IAM_PASSWORD_POLICY"
  }

  input_parameters = jsonencode({
    RequireUppercaseCharacters = "true"
    RequireLowercaseCharacters = "true"
    RequireSymbols            = "true"
    RequireNumbers            = "true"
    MinimumPasswordLength     = "14"
    PasswordReusePrevention   = "24"
    MaxPasswordAge            = "90"
  })

  depends_on = [aws_config_configuration_recorder.main]

  tags = var.tags
}

# Config rule: Root access key check
resource "aws_config_config_rule" "root_access_key" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-root-access-key-check"

  source {
    owner             = "AWS"
    source_identifier = "ROOT_ACCESS_KEY_CHECK"
  }

  depends_on = [aws_config_configuration_recorder.main]

  tags = var.tags
}

# IAM role for Config remediation
resource "aws_iam_role" "config_remediation" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-config-remediation-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ssm.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# Policy for Config remediation
resource "aws_iam_role_policy" "config_remediation" {
  count = var.enable_config ? 1 : 0

  name = "${var.project_name}-config-remediation-policy"
  role = aws_iam_role.config_remediation[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutEncryptionConfiguration",
          "s3:PutBucketPublicAccessBlock",
          "s3:GetBucketPublicAccessBlock",
          "s3:GetEncryptionConfiguration"
        ]
        Resource = [
          aws_s3_bucket.ml_artifacts.arn,
          "${aws_s3_bucket.ml_artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "config:PutEvaluations",
          "config:GetComplianceDetailsByConfigRule",
          "config:GetComplianceDetailsByResource"
        ]
        Resource = "*"
      }
    ]
  })
}

# Remediation configuration for S3 encryption
resource "aws_config_remediation_configuration" "s3_encryption" {
  count = var.enable_config ? 1 : 0

  config_rule_name = aws_config_config_rule.s3_encryption[0].name

  resource_type    = "AWS::S3::Bucket"
  target_type      = "SSM_DOCUMENT"
  target_id        = "AWS-PublishSNSNotification"
  target_version   = "1"

  parameter {
    name           = "AutomationAssumeRole"
    static_value   = aws_iam_role.config_remediation[0].arn
  }

  parameter {
    name           = "TopicArn"
    static_value   = var.enable_cost_management ? aws_sns_topic.budget_alerts[0].arn : ""
  }

  parameter {
    name           = "Message"
    static_value   = "S3 bucket encryption compliance violation detected and remediation attempted"
  }

  automatic                = true
  maximum_automatic_attempts = 3
}

# CloudWatch dashboard for Config compliance
resource "aws_cloudwatch_dashboard" "config_compliance" {
  count = var.enable_config && var.enable_monitoring ? 1 : 0

  dashboard_name = "${var.project_name}-config-compliance"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/Config", "ComplianceByConfigRule", "ConfigRuleName", aws_config_config_rule.s3_encryption[0].name, "ComplianceType", "COMPLIANT"],
            [".", ".", ".", ".", ".", "NON_COMPLIANT"],
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "S3 Encryption Compliance"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/Config", "ComplianceByConfigRule", "ConfigRuleName", aws_config_config_rule.eks_version[0].name, "ComplianceType", "COMPLIANT"],
            [".", ".", ".", ".", ".", "NON_COMPLIANT"],
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "EKS Version Compliance"
          period  = 300
        }
      }
    ]
  })
}
