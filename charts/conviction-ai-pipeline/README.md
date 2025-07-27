# Conviction AI Pipeline Helm Chart

This Helm chart deploys the Conviction AI ETL and ML training pipeline on Kubernetes with GPU support, automated scheduling, Argo Rollouts canary deployments, and comprehensive configuration options.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- GPU nodes with NVIDIA drivers (for training workloads)
- Persistent storage support
- Argo Rollouts CRD installed (for canary deployments)
- Prometheus (for automated canary analysis)

## Installation

### Add Helm Repository

```bash
helm repo add conviction-ai https://your-chart-repo/
helm repo update
```

### Install Argo Rollouts (if not already installed)

```bash
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
```

### Install Chart

```bash
# Basic installation
helm install conviction-ai-pipeline conviction-ai/conviction-ai-pipeline

# With canary deployments enabled
helm install conviction-ai-pipeline conviction-ai/conviction-ai-pipeline \
  --set rollout.enabled=true

# With custom values
helm install conviction-ai-pipeline conviction-ai/conviction-ai-pipeline \
  --set runDate=2025-07-25 \
  --set nTrials=100 \
  --set nJobs=24 \
  --set image.tag=v1.2.3 \
  --set rollout.enabled=true
```

### Create Secrets

Before installation, create the required secrets:

```bash
kubectl create secret generic pipeline-secrets \
  --from-literal=aws-access-key-id=YOUR_ACCESS_KEY \
  --from-literal=aws-secret-access-key=YOUR_SECRET_KEY \
  --from-literal=s3-bucket=your-bucket-name \
  --from-literal=slack-webhook-url=https://hooks.slack.com/services/... \
  --from-literal=mlflow-tracking-uri=http://mlflow-server:5000
```

## Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Docker image repository | `docker.io/conviction-ai/conviction-ai-pipeline` |
| `image.tag` | Docker image tag | `Chart.AppVersion` |
| `runDate` | Processing date (YYYY-MM-DD) | Current date |
| `nTrials` | Number of hyperparameter trials | `50` |
| `nJobs` | Number of parallel jobs | `8` |
| `resources.limits.nvidia.com/gpu` | GPU limit | `1` |
| `persistence.enabled` | Enable persistent storage | `true` |
| `persistence.size` | Storage size | `50Gi` |
| `autoscaling.enabled` | Enable horizontal pod autoscaling | `true` |
| `autoscaling.minReplicas` | Minimum number of replicas | `1` |
| `autoscaling.maxReplicas` | Maximum number of replicas | `5` |
| `autoscaling.cpuUtilizationPercentage` | CPU target for scaling | `80` |
| `autoscaling.gpuUtilizationPercentage` | GPU target for scaling | `75` |
| `prometheus.enabled` | Enable Prometheus monitoring | `true` |
| `prometheus.scrapeInterval` | Metrics scrape interval | `30s` |
| `grafana.dashboard.enabled` | Enable Grafana dashboard | `true` |
| `grafana.dashboard.title` | Dashboard title | `Conviction AI Pipeline` |
| `rollout.enabled` | Enable Argo Rollouts canary deployments | `false` |
| `rollout.canary.steps` | Canary deployment steps | `[{weight: 10, pause: "1m"}, {weight: 50, pause: "2m"}, {weight: 100}]` |
| `rollout.canary.analysis.enabled` | Enable automated canary analysis | `true` |
| `rollout.canary.analysis.prometheus.address` | Prometheus server address | `http://prometheus:9090` |
| `inference.replicaCount` | Number of inference service replicas | `2` |
| `inference.image.repository` | Inference service image repository | `docker.io/conviction-ai/conviction-ai-inference` |
| `inference.service.type` | Inference service type | `ClusterIP` |
| `inference.service.port` | Inference service port | `80` |

### GPU Configuration

For GPU-enabled training:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    cpu: "2"
    memory: "4Gi"

nodeSelector:
  nvidia.com/gpu.present: "true"

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### Backfill Configuration

Enable automated backfill jobs:

```yaml
backfill:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  startDate: "2025-01-01"
  endDate: "2025-01-31"
  workers: 16
```

### Canary Deployment Configuration

Enable canary deployments with Argo Rollouts:

```yaml
rollout:
  enabled: true
  canary:
    steps:
      - weight: 10
        pause: "1m"
      - weight: 50
        pause: "2m"
      - weight: 100
    analysis:
      enabled: true
      prometheus:
        address: http://prometheus:9090
```

## Usage Examples

### Daily ETL Processing

```bash
helm upgrade --install conviction-ai-pipeline . \
  --set runDate=2025-07-25 \
  --set nTrials=50 \
  --set nJobs=16
```

### GPU Training Workload

```bash
helm upgrade --install conviction-ai-pipeline . \
  --set resources.limits.nvidia\.com/gpu=2 \
  --set nTrials=100 \
  --set nodeSelector.nvidia\.com/gpu\.present=true
```

### High-Performance Configuration

```bash
helm upgrade --install conviction-ai-pipeline . \
  --set resources.limits.cpu=8 \
  --set resources.limits.memory=16Gi \
  --set resources.limits.nvidia\.com/gpu=2 \
  --set nJobs=32 \
  --set nTrials=200
```

### Autoscaling Configuration

```bash
# Enable autoscaling with custom thresholds
helm upgrade --install conviction-ai-pipeline . \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=2 \
  --set autoscaling.maxReplicas=10 \
  --set autoscaling.cpuUtilizationPercentage=75 \
  --set autoscaling.gpuUtilizationPercentage=70

# Enable backfill autoscaling (alternative to CronJob)
helm upgrade --install conviction-ai-pipeline . \
  --set backfill.enabled=true \
  --set autoscaling.backfill.enabled=true \
  --set autoscaling.backfill.maxReplicas=5
```

### Canary Deployment Examples

```bash
# Enable canary deployments
helm upgrade --install conviction-ai-pipeline . \
  --set rollout.enabled=true

# Custom canary configuration
helm upgrade --install conviction-ai-pipeline . \
  --set rollout.enabled=true \
  --set rollout.canary.analysis.enabled=true \
  --set rollout.canary.analysis.prometheus.address=http://prometheus:9090

# Monitor rollout status
kubectl argo rollouts get rollout conviction-ai-pipeline-inference --watch

# Promote canary manually
kubectl argo rollouts promote conviction-ai-pipeline-inference

# Abort rollout
kubectl argo rollouts abort conviction-ai-pipeline-inference
```

### Monitoring Configuration

```bash
# Enable Prometheus & Grafana monitoring
helm upgrade --install conviction-ai-pipeline . \
  --set prometheus.enabled=true \
  --set grafana.dashboard.enabled=true \
  --set grafana.dashboard.title="Conviction AI Pipeline"

# Custom scrape intervals
helm upgrade --install conviction-ai-pipeline . \
  --set prometheus.enabled=true \
  --set prometheus.scrapeInterval=15s \
  --set prometheus.scrapeTimeout=5s
```

## Components

### Deployment
- Main ETL and training pipeline
- GPU support for ML workloads
- Persistent storage for data, models, and metrics
- Health checks and monitoring

### CronJob (Backfill)
- Automated historical data processing
- Configurable schedule and date ranges
- Parallel processing with worker configuration

### HorizontalPodAutoscaler (HPA)
- CPU and GPU-based autoscaling for main deployment
- Configurable min/max replicas and utilization thresholds
- Optional backfill deployment autoscaling
- Stabilization windows for smooth scaling behavior

### Monitoring Stack
- **ServiceMonitor**: Prometheus Operator integration for metrics scraping
- **GrafanaDashboard**: Pre-configured dashboard with pipeline metrics
- **Metrics Endpoint**: Exposed on port 9090 with pipeline-specific metrics
- **Real-time Monitoring**: ETL duration, training progress, resource utilization

### Rollout (Canary Deployments)
- Argo Rollouts integration for inference service
- Automated canary analysis with Prometheus metrics
- Traffic splitting between stable and canary versions
- Automatic promotion/rollback based on success criteria

### Job (Training)
- On-demand training job execution
- Hyperparameter optimization
- Model artifact storage

### Storage
- Data PVC: Raw and processed data storage
- Models PVC: Trained model artifacts
- Metrics PVC: Training metrics and logs

## Monitoring

### Health Checks
- Liveness probe: Process monitoring
- Readiness probe: Application availability

### Logging
All components log to stdout/stderr for Kubernetes log aggregation.

### Metrics
Integration with MLflow for experiment tracking and model registry.

## Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   kubectl describe nodes | grep nvidia.com/gpu
   ```

2. **Storage Issues**
   ```bash
   kubectl get pvc
   kubectl describe pvc conviction-ai-pipeline-data
   ```

3. **Secret Configuration**
   ```bash
   kubectl get secret pipeline-secrets -o yaml
   ```

4. **Argo Rollouts Not Installed**
   ```bash
   kubectl get crd rollouts.argoproj.io
   kubectl apply -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
   ```

5. **Canary Analysis Failing**
   ```bash
   kubectl describe analysisrun
   kubectl logs -l app.kubernetes.io/name=argo-rollouts
   ```

### Debug Commands

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/name=conviction-ai-pipeline

# View logs
kubectl logs -l component=pipeline -f

# Describe deployment/rollout
kubectl describe deployment conviction-ai-pipeline
kubectl describe rollout conviction-ai-pipeline-inference

# Check rollout status
kubectl argo rollouts get rollout conviction-ai-pipeline-inference

# View analysis results
kubectl get analysisrun
kubectl describe analysisrun

# Run test
helm test conviction-ai-pipeline
```

## Uninstallation

```bash
helm uninstall conviction-ai-pipeline
kubectl delete pvc -l app.kubernetes.io/name=conviction-ai-pipeline
kubectl delete secret pipeline-secrets
```

## Development

### Local Testing

```bash
# Lint chart
helm lint charts/conviction-ai-pipeline

# Template rendering
helm template conviction-ai-pipeline charts/conviction-ai-pipeline

# Dry run
helm install --dry-run --debug conviction-ai-pipeline charts/conviction-ai-pipeline
```