# Canary Deployment Guide

This guide explains how to set up and use Argo Rollouts for canary deployments of the Conviction AI inference service.

## Prerequisites

1. **Kubernetes cluster** with Argo Rollouts installed
2. **Prometheus** for metrics collection and analysis
3. **Helm 3.0+** for chart deployment

## Setup

### 1. Install Argo Rollouts

```bash
# Create namespace
kubectl create namespace argo-rollouts

# Install Argo Rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

# Verify installation
kubectl get pods -n argo-rollouts
```

### 2. Install Argo Rollouts CLI (Optional)

```bash
# macOS
brew install argoproj/tap/kubectl-argo-rollouts

# Linux
curl -LO https://github.com/argoproj/argo-rollouts/releases/latest/download/kubectl-argo-rollouts-linux-amd64
chmod +x kubectl-argo-rollouts-linux-amd64
sudo mv kubectl-argo-rollouts-linux-amd64 /usr/local/bin/kubectl-argo-rollouts
```

### 3. Deploy with Canary Enabled

```bash
# Enable canary deployments
helm upgrade --install conviction-ai-pipeline charts/conviction-ai-pipeline \
  --set rollout.enabled=true \
  --set rollout.canary.analysis.enabled=true \
  --set rollout.canary.analysis.prometheus.address=http://prometheus:9090
```

## Canary Configuration

### Default Canary Steps

The default configuration implements a 3-step canary deployment:

1. **10% traffic** → pause 1 minute
2. **50% traffic** → pause 2 minutes
3. **100% traffic** (full promotion)

### Custom Canary Steps

```yaml
rollout:
  enabled: true
  canary:
    steps:
      - weight: 5
        pause: "30s"
      - weight: 25
        pause: "2m"
      - weight: 75
        pause: "5m"
      - weight: 100
```

### Analysis Configuration

Automated promotion/rollback based on metrics:

```yaml
rollout:
  canary:
    analysis:
      enabled: true
      prometheus:
        address: http://prometheus:9090
```

**Metrics monitored:**
- **Success Rate**: ≥95% (HTTP 2xx/3xx responses)
- **Average Latency**: ≤500ms (95th percentile)
- **Error Rate**: ≤5% (HTTP 5xx responses)

## Deployment Workflow

### 1. Deploy New Version

```bash
# Update image tag to trigger rollout
helm upgrade conviction-ai-pipeline charts/conviction-ai-pipeline \
  --set rollout.enabled=true \
  --set inference.image.tag=v1.2.0
```

### 2. Monitor Rollout

```bash
# Watch rollout progress
kubectl argo rollouts get rollout conviction-ai-pipeline-inference --watch

# View detailed status
kubectl argo rollouts status conviction-ai-pipeline-inference
```

### 3. Manual Control

```bash
# Promote to next step
kubectl argo rollouts promote conviction-ai-pipeline-inference

# Abort rollout (rollback to stable)
kubectl argo rollouts abort conviction-ai-pipeline-inference

# Restart rollout
kubectl argo rollouts restart conviction-ai-pipeline-inference
```

## Monitoring and Observability

### Rollout Status

```bash
# Get current rollout status
kubectl get rollout conviction-ai-pipeline-inference

# Describe rollout for detailed information
kubectl describe rollout conviction-ai-pipeline-inference
```

### Analysis Results

```bash
# View analysis runs
kubectl get analysisrun

# Describe specific analysis
kubectl describe analysisrun <analysis-run-name>

# View analysis logs
kubectl logs -l app.kubernetes.io/name=argo-rollouts
```

### Prometheus Metrics

Key metrics for canary analysis:

```promql
# Success rate
sum(rate(predictions_total{service="conviction-ai-pipeline-inference", status_code!~"5.."}[2m])) /
sum(rate(predictions_total{service="conviction-ai-pipeline-inference"}[2m]))

# 95th percentile latency
histogram_quantile(0.95,
  sum(rate(prediction_latency_seconds_bucket{service="conviction-ai-pipeline-inference"}[2m])) by (le)
)

# Error rate
sum(rate(errors_total{service="conviction-ai-pipeline-inference"}[2m])) /
sum(rate(predictions_total{service="conviction-ai-pipeline-inference"}[2m]))
```

## Traffic Routing

### Service Architecture

- **Active Service**: `conviction-ai-pipeline-inference` (receives all traffic)
- **Stable Service**: `conviction-ai-pipeline-inference-stable` (stable version)
- **Canary Service**: `conviction-ai-pipeline-inference-canary` (new version)

### Traffic Split

During canary deployment, traffic is automatically split:

```
┌─────────────────┐    10%     ┌─────────────────┐
│   Load Balancer │ ────────── │ Canary Pods     │
│                 │            │ (v1.2.0)        │
└─────────────────┘    90%     └─────────────────┘
         │                     ┌─────────────────┐
         └─────────────────────│ Stable Pods     │
                               │ (v1.1.0)        │
                               └─────────────────┘
```

## Troubleshooting

### Common Issues

#### 1. Rollout Stuck in Progressing State

```bash
# Check analysis results
kubectl get analysisrun
kubectl describe analysisrun <name>

# Check Prometheus connectivity
kubectl exec -it <rollout-pod> -- curl http://prometheus:9090/api/v1/query?query=up
```

#### 2. Analysis Failing

```bash
# Verify metrics are available
kubectl port-forward svc/prometheus 9090:9090
# Open http://localhost:9090 and test queries

# Check analysis template
kubectl describe analysistemplate conviction-ai-pipeline-success-rate
```

#### 3. Manual Intervention Required

```bash
# Skip analysis and promote manually
kubectl argo rollouts promote conviction-ai-pipeline-inference --skip-current-step

# Abort and rollback
kubectl argo rollouts abort conviction-ai-pipeline-inference
```

### Debug Commands

```bash
# View rollout events
kubectl get events --field-selector involvedObject.name=conviction-ai-pipeline-inference

# Check replica sets
kubectl get rs -l app.kubernetes.io/name=conviction-ai-pipeline

# View pod logs
kubectl logs -l app.kubernetes.io/name=conviction-ai-pipeline,component=inference
```

## Best Practices

### 1. Gradual Traffic Increase

Start with small traffic percentages and increase gradually:

```yaml
steps:
  - weight: 5    # Start small
  - weight: 25   # Increase gradually
  - weight: 50   # Half traffic
  - weight: 100  # Full promotion
```

### 2. Appropriate Pause Durations

Allow sufficient time for metrics collection:

```yaml
steps:
  - weight: 10
    pause: "2m"   # Minimum 2 minutes for reliable metrics
  - weight: 50
    pause: "5m"   # Longer pause for higher traffic
```

### 3. Conservative Success Criteria

Set conservative thresholds to avoid false positives:

```yaml
metrics:
  - name: success-rate
    successCondition: result[0] >= 0.98  # 98% success rate
    failureLimit: 2                      # Allow 2 failures before abort
```

### 4. Monitor Business Metrics

In addition to technical metrics, monitor business-specific metrics:

- Prediction accuracy
- Feature store latency
- Model inference time
- User satisfaction scores

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Deploy Canary
  run: |
    helm upgrade conviction-ai-pipeline charts/conviction-ai-pipeline \
      --set rollout.enabled=true \
      --set inference.image.tag=${{ github.sha }}

    # Wait for rollout to complete
    kubectl argo rollouts status conviction-ai-pipeline-inference \
      --timeout=600s
```

### Automated Rollback

```yaml
- name: Check Rollout Status
  run: |
    if ! kubectl argo rollouts status conviction-ai-pipeline-inference --timeout=300s; then
      echo "Rollout failed, aborting..."
      kubectl argo rollouts abort conviction-ai-pipeline-inference
      exit 1
    fi
```

This guide provides a comprehensive approach to implementing canary deployments for the Conviction AI inference service using Argo Rollouts with automated analysis and monitoring.
