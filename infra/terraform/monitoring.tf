# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "eks_cluster" {
  count = var.enable_monitoring ? 1 : 0

  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = 7

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "glue_jobs" {
  count = var.enable_monitoring ? 1 : 0

  name              = "/aws/glue/jobs/${var.project_name}"
  retention_in_days = 14

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "lambda_functions" {
  count = var.enable_monitoring ? 1 : 0

  name              = "/aws/lambda/${var.project_name}"
  retention_in_days = 14

  tags = var.tags
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "ml_pipeline" {
  count = var.enable_monitoring ? 1 : 0

  dashboard_name = "${var.project_name}-pipeline-dashboard"

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
            ["AWS/EKS", "cluster_failed_request_count", "ClusterName", var.cluster_name],
            [".", "cluster_request_total", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "EKS Cluster Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/Glue", "glue.driver.aggregate.numCompletedTasks", "JobName", aws_glue_job.etl_job.name],
            [".", "glue.driver.aggregate.numFailedTasks", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Glue Job Metrics"
          period  = 300
        }
      },
      {
        type   = "log"
        x      = 0
        y      = 12
        width  = 24
        height = 6

        properties = {
          query   = "SOURCE '/aws/glue/jobs/${var.project_name}' | fields @timestamp, @message | sort @timestamp desc | limit 100"
          region  = var.aws_region
          title   = "Recent Glue Job Logs"
        }
      }
    ]
  })
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "glue_job_failures" {
  count = var.enable_monitoring ? 1 : 0

  alarm_name          = "${var.project_name}-glue-job-failures"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "glue.driver.aggregate.numFailedTasks"
  namespace           = "AWS/Glue"
  period              = "300"
  statistic           = "Sum"
  threshold           = "0"
  alarm_description   = "This metric monitors glue job failures"
  alarm_actions       = [aws_sns_topic.alerts[0].arn]

  dimensions = {
    JobName = aws_glue_job.etl_job.name
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "eks_cluster_failures" {
  count = var.enable_monitoring ? 1 : 0

  alarm_name          = "${var.project_name}-eks-cluster-failures"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "cluster_failed_request_count"
  namespace           = "AWS/EKS"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "This metric monitors EKS cluster failures"
  alarm_actions       = [aws_sns_topic.alerts[0].arn]

  dimensions = {
    ClusterName = var.cluster_name
  }

  tags = var.tags
}

# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  count = var.enable_monitoring ? 1 : 0

  name = "${var.project_name}-alerts"

  tags = var.tags
}

resource "aws_sns_topic_policy" "alerts" {
  count = var.enable_monitoring ? 1 : 0

  arn = aws_sns_topic.alerts[0].arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "cloudwatch.amazonaws.com"
        }
        Action   = "SNS:Publish"
        Resource = aws_sns_topic.alerts[0].arn
      }
    ]
  })
}

# EventBridge rule for scheduled pipeline execution
resource "aws_cloudwatch_event_rule" "daily_pipeline" {
  count = var.enable_monitoring ? 1 : 0

  name                = "${var.project_name}-daily-pipeline"
  description         = "Trigger daily ML pipeline execution"
  schedule_expression = "cron(0 2 * * ? *)"  # Daily at 2 AM UTC

  tags = var.tags
}

resource "aws_cloudwatch_event_target" "lambda_target" {
  count = var.enable_monitoring ? 1 : 0

  rule      = aws_cloudwatch_event_rule.daily_pipeline[0].name
  target_id = "TriggerLambda"
  arn       = "arn:aws:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:${var.project_name}-pipeline-trigger"
}
