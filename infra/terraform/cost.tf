# AWS Cost Management and Budgets

# SNS Topic for budget alerts
resource "aws_sns_topic" "budget_alerts" {
  count = var.enable_cost_management ? 1 : 0

  name = "${var.project_name}-budget-alerts"

  tags = merge(var.tags, {
    Purpose = "cost-management"
  })
}

# SNS Topic Policy
resource "aws_sns_topic_policy" "budget_alerts" {
  count = var.enable_cost_management ? 1 : 0

  arn = aws_sns_topic.budget_alerts[0].arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "budgets.amazonaws.com"
        }
        Action   = "SNS:Publish"
        Resource = aws_sns_topic.budget_alerts[0].arn
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })
}

# Email subscription for budget alerts
resource "aws_sns_topic_subscription" "budget_email" {
  count = var.enable_cost_management ? 1 : 0

  topic_arn = aws_sns_topic.budget_alerts[0].arn
  protocol  = "email"
  endpoint  = var.budget_email
}

# Slack webhook subscription (if provided)
resource "aws_sns_topic_subscription" "budget_slack" {
  count = var.enable_cost_management && var.slack_budget_webhook_url != "" ? 1 : 0

  topic_arn = aws_sns_topic.budget_alerts[0].arn
  protocol  = "https"
  endpoint  = var.slack_budget_webhook_url
}

# Monthly budget for the project
resource "aws_budgets_budget" "monthly" {
  count = var.enable_cost_management ? 1 : 0

  name         = "${var.project_name}-monthly-budget"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_amount
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  time_period_start = "2025-01-01_00:00"

  cost_filters = {
    Service = [
      "Amazon Elastic Compute Cloud - Compute",
      "Amazon Simple Storage Service",
      "AWS Glue",
      "Amazon SageMaker",
      "Amazon Elastic Kubernetes Service",
      "Amazon Elastic Container Service for Kubernetes",
      "Amazon CloudWatch"
    ]
    TagKey = ["Project"]
    TagValue = [var.project_name]
  }

  # Alert at 80% of budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts[0].arn]
  }

  # Alert at 90% of budget (forecasted)
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 90
    threshold_type            = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts[0].arn]
  }

  # Alert at 100% of budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts[0].arn]
  }

  # Alert at 110% of budget (overspend)
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 110
    threshold_type            = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts[0].arn]
  }

  depends_on = [aws_sns_topic_policy.budget_alerts]
}

# Daily budget for fine-grained monitoring
resource "aws_budgets_budget" "daily" {
  count = var.enable_cost_management ? 1 : 0

  name         = "${var.project_name}-daily-budget"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_amount / 30  # Approximate daily budget
  limit_unit   = "USD"
  time_unit    = "DAILY"
  time_period_start = "2025-01-01_00:00"

  cost_filters = {
    Service = [
      "Amazon Elastic Compute Cloud - Compute",
      "Amazon Simple Storage Service",
      "AWS Glue",
      "Amazon SageMaker",
      "Amazon Elastic Kubernetes Service"
    ]
    TagKey = ["Project"]
    TagValue = [var.project_name]
  }

  # Alert at 150% of daily budget (spike detection)
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 150
    threshold_type            = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts[0].arn]
  }

  depends_on = [aws_sns_topic_policy.budget_alerts]
}

# Cost anomaly detection
resource "aws_ce_anomaly_detector" "service_monitor" {
  count = var.enable_cost_management ? 1 : 0

  name         = "${var.project_name}-anomaly-detector"
  monitor_type = "DIMENSIONAL"

  specification = jsonencode({
    Dimension = "SERVICE"
    MatchOptions = ["EQUALS"]
    Values = [
      "Amazon Elastic Compute Cloud - Compute",
      "Amazon SageMaker",
      "AWS Glue"
    ]
  })

  tags = var.tags
}

# Cost anomaly subscription
resource "aws_ce_anomaly_subscription" "service_monitor" {
  count = var.enable_cost_management ? 1 : 0

  name      = "${var.project_name}-anomaly-subscription"
  frequency = "DAILY"

  monitor_arn_list = [
    aws_ce_anomaly_detector.service_monitor[0].arn
  ]

  subscriber {
    type    = "EMAIL"
    address = var.budget_email
  }

  threshold_expression {
    and {
      dimension {
        key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
        values        = ["100"]
        match_options = ["GREATER_THAN_OR_EQUAL"]
      }
    }
  }

  tags = var.tags
}

# CloudWatch dashboard for cost monitoring
resource "aws_cloudwatch_dashboard" "cost_monitoring" {
  count = var.enable_cost_management && var.enable_monitoring ? 1 : 0

  dashboard_name = "${var.project_name}-cost-monitoring"

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
            ["AWS/Billing", "EstimatedCharges", "Currency", "USD"],
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1"  # Billing metrics are only in us-east-1
          title   = "Estimated Monthly Charges"
          period  = 86400  # Daily
          stat    = "Maximum"
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
            ["AWS/Billing", "EstimatedCharges", "Currency", "USD", "ServiceName", "AmazonEC2"],
            [".", ".", ".", ".", ".", "AmazonS3"],
            [".", ".", ".", ".", ".", "AWSGlue"],
            [".", ".", ".", ".", ".", "AmazonSageMaker"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-1"
          title   = "Estimated Charges by Service"
          period  = 86400
          stat    = "Maximum"
        }
      }
    ]
  })
}
