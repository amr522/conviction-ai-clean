# Terraform CI/CD Workflows

This document describes the automated Terraform workflows for managing AWS infrastructure.

## Workflow Overview

### 1. Terraform Plan (`terraform-plan.yml`)
- **Trigger**: Pull requests that modify `infra/terraform/**`
- **Purpose**: Validate and plan infrastructure changes
- **Environment**: Development
- **Permissions**: Read-only with PR comment access

### 2. Terraform Apply (`terraform-apply.yml`)
- **Trigger**: Push to `main` branch with Terraform changes
- **Purpose**: Apply approved infrastructure changes
- **Environment**: Production (with manual approval)
- **Permissions**: Full AWS access for resource creation

### 3. Terraform Destroy (`terraform-destroy.yml`)
- **Trigger**: Manual workflow dispatch
- **Purpose**: Destroy infrastructure for cleanup
- **Environment**: Configurable (dev/staging/prod)
- **Permissions**: Full AWS access with confirmation required

## Required Secrets and Variables

### Repository Secrets
```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Repository Variables
```bash
AWS_REGION=us-east-1
TFSTATE_BUCKET=conviction-ai-terraform-state
S3_BUCKET_NAME=conviction-ai-ml-artifacts
```

## Workflow Process

### Pull Request Flow
1. **Developer creates PR** with Terraform changes
2. **Terraform Plan workflow** automatically runs:
   - Format check (`terraform fmt`)
   - Initialization (`terraform init`)
   - Validation (`terraform validate`)
   - Plan generation (`terraform plan`)
3. **Plan results** posted as PR comment
4. **Code review** by infrastructure team (via CODEOWNERS)
5. **PR approval** required before merge

### Main Branch Flow
1. **PR merged** to main branch
2. **Terraform Apply workflow** triggered:
   - Requires manual approval in production environment
   - Runs fresh plan and apply
   - Uploads outputs as artifacts
   - Sends Slack notifications

### Destroy Flow
1. **Manual trigger** via GitHub Actions UI
2. **Confirmation required**: Must type "destroy"
3. **Environment selection**: dev/staging/prod
4. **Plan and apply destroy** operation
5. **Slack notification** on completion

## Branch Protection Rules

Configure the following branch protection rules for `main`:

```yaml
Required status checks:
  - Terraform Plan
  - Security Scan
  - Tests

Required reviews:
  - 2 approvals required
  - Dismiss stale reviews
  - Require review from CODEOWNERS

Restrictions:
  - Restrict pushes to admins only
  - Allow force pushes: false
  - Allow deletions: false
```

## Environment Protection

### Production Environment
- **Required reviewers**: Infrastructure team
- **Wait timer**: 5 minutes
- **Deployment branches**: main only

### Development Environment
- **Required reviewers**: None
- **Wait timer**: None
- **Deployment branches**: Any

## Artifact Management

### Plan Artifacts
- **Retention**: 5 days
- **Naming**: `tfplan-{PR_NUMBER}`
- **Usage**: Review planned changes before apply

### Output Artifacts
- **Retention**: 30 days
- **Naming**: `terraform-outputs`
- **Usage**: Reference infrastructure outputs

## Monitoring and Alerts

### Slack Notifications
- **Success**: Infrastructure deployment completed
- **Failure**: Deployment failed with error details
- **Destroy**: Infrastructure destruction status

### CloudWatch Integration
- Terraform state changes logged to CloudWatch
- Infrastructure metrics available in dashboards
- Automated alerts for resource failures

## Troubleshooting

### Common Issues

**Plan Failures:**
- Check AWS credentials and permissions
- Verify Terraform state bucket access
- Review variable configuration

**Apply Failures:**
- Check resource limits and quotas
- Verify IAM permissions for resource creation
- Review dependency conflicts

**State Lock Issues:**
- Check for concurrent Terraform operations
- Manually unlock state if needed:
  ```bash
  terraform force-unlock LOCK_ID
  ```

### Manual Operations

**Emergency Apply:**
```bash
# Local emergency apply
cd infra/terraform
terraform init
terraform plan -var-file="prod.tfvars"
terraform apply
```

**State Management:**
```bash
# View state
terraform state list

# Import existing resource
terraform import aws_s3_bucket.example bucket-name

# Remove resource from state
terraform state rm aws_s3_bucket.example
```

## Security Considerations

### Least Privilege Access
- Terraform service account has minimal required permissions
- Separate roles for different environments
- Regular access review and rotation

### State Security
- Terraform state stored in encrypted S3 bucket
- State locking with DynamoDB
- Versioning enabled for rollback capability

### Secrets Management
- AWS credentials stored as GitHub secrets
- No hardcoded secrets in Terraform code
- Regular credential rotation

## Best Practices

### Code Organization
- Modular Terraform code with clear separation
- Consistent naming conventions
- Comprehensive variable documentation

### Change Management
- Small, incremental changes
- Thorough testing in development environment
- Rollback plan for each deployment

### Documentation
- Keep Terraform code well-documented
- Update README for infrastructure changes
- Maintain architecture diagrams