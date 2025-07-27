# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the Conviction-AI ETL pipeline as a scheduled CronJob.

## Prerequisites

- Kubernetes cluster (EKS, GKE, or on-premises)
- kubectl configured to access your cluster
- Docker registry access for the vol-pipeline image

## Deployment Steps

### 1. Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### 2. Create Persistent Volume Claims

```bash
kubectl apply -f k8s/pvcs.yaml
```

### 3. Create Secrets

First, create the actual secrets (replace template values):

```bash
# Create AWS credentials secret
kubectl create secret generic aws-credentials \
  --from-literal=access-key-id=YOUR_ACCESS_KEY \
  --from-literal=secret-access-key=YOUR_SECRET_KEY \
  -n data-processing

# Create Slack webhook secret (optional)
kubectl create secret generic slack-webhook \
  --from-literal=url=YOUR_SLACK_WEBHOOK_URL \
  -n data-processing
```

### 4. Update Image Registry

Edit `k8s/cronjob-vol-pipeline.yaml` and replace:
```yaml
image: your-docker-registry/vol-pipeline:latest
```

With your actual registry:
```yaml
image: your-registry.com/vol-pipeline:latest
```

### 5. Deploy CronJob

```bash
kubectl apply -f k8s/cronjob-vol-pipeline.yaml
```

## Configuration

### Schedule

The CronJob runs daily at 18:00 UTC. To change the schedule, modify the `schedule` field:

```yaml
spec:
  schedule: "0 18 * * *"  # 18:00 UTC daily
```

### Resource Limits

Adjust CPU and memory limits based on your data size:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Storage

PVC sizes can be adjusted in `k8s/pvcs.yaml`:

```yaml
resources:
  requests:
    storage: 100Gi  # Adjust as needed
```

## Monitoring

### Check CronJob Status

```bash
kubectl get cronjobs -n data-processing
kubectl describe cronjob vol-pipeline-daily -n data-processing
```

### View Job History

```bash
kubectl get jobs -n data-processing
kubectl logs -l app=vol-pipeline -n data-processing
```

### Manual Trigger

```bash
kubectl create job --from=cronjob/vol-pipeline-daily manual-run-$(date +%s) -n data-processing
```

## Troubleshooting

### Common Issues

1. **PVC not binding**: Check storage class availability
2. **Image pull errors**: Verify registry credentials
3. **AWS permissions**: Ensure IAM role has required permissions
4. **Memory issues**: Increase resource limits

### Debug Commands

```bash
# Check pod logs
kubectl logs -l app=vol-pipeline -n data-processing --tail=100

# Describe failed jobs
kubectl describe job JOB_NAME -n data-processing

# Check PVC status
kubectl get pvc -n data-processing
```
