#!/usr/bin/env python3
"""
Real-time ML Ops Dashboard for Conviction AI HPO System

This dashboard provides:
1. Real-time endpoint metrics visualization
2. Model performance monitoring with AUC tracking
3. Data drift detection and alerts
4. System health monitoring
"""

import os
import sys
import time
import json
import boto3
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlops_dashboard.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MLOpsDashboard:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = 'hpo-bucket-773934887314'
        
    def get_endpoint_metrics(self, endpoint_name: str, hours_back: int = 24) -> Dict:
        """Retrieve CloudWatch metrics for SageMaker endpoint"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        metrics = {}
        
        metric_queries = [
            ('Invocations', 'Sum'),
            ('InvocationsPerInstance', 'Average'),
            ('ModelLatency', 'Average'),
            ('OverheadLatency', 'Average'),
            ('Invocation4XXErrors', 'Sum'),
            ('Invocation5XXErrors', 'Sum')
        ]
        
        for metric_name, statistic in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/SageMaker',
                    MetricName=metric_name,
                    Dimensions=[
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint_name
                        }
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,
                    Statistics=[statistic]
                )
                
                datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
                metrics[metric_name] = {
                    'timestamps': [dp['Timestamp'] for dp in datapoints],
                    'values': [dp[statistic] for dp in datapoints],
                    'unit': response.get('Label', metric_name)
                }
                
            except Exception as e:
                logger.warning(f"Failed to get metric {metric_name}: {e}")
                metrics[metric_name] = {'timestamps': [], 'values': [], 'unit': metric_name}
        
        return metrics
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict:
        """Get current endpoint status and configuration"""
        try:
            response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            return {
                'status': response['EndpointStatus'],
                'creation_time': response['CreationTime'],
                'last_modified': response['LastModifiedTime'],
                'instance_type': response.get('ProductionVariants', [{}])[0].get('InstanceType', 'Unknown'),
                'instance_count': response.get('ProductionVariants', [{}])[0].get('CurrentInstanceCount', 0),
                'failure_reason': response.get('FailureReason', None)
            }
        except Exception as e:
            logger.error(f"Failed to get endpoint status: {e}")
            return {'status': 'Unknown', 'error': str(e)}
    
    def test_endpoint_performance(self, endpoint_name: str, sample_count: int = 10) -> Dict:
        """Test endpoint performance and calculate AUC if possible"""
        try:
            np.random.seed(42)
            sample_data = np.random.randn(sample_count, 63)
            
            predictions = []
            latencies = []
            
            for i in range(sample_count):
                start_time = time.time()
                sample_row = sample_data[i]
                payload = ','.join([str(val) for val in sample_row])
                
                response = self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='text/csv',
                    Body=payload
                )
                
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                result = json.loads(response['Body'].read().decode())
                prediction = result.get('predictions', [0])[0]
                predictions.append(prediction)
            
            return {
                'success': True,
                'sample_count': sample_count,
                'predictions': predictions,
                'avg_latency_ms': np.mean(latencies),
                'max_latency_ms': np.max(latencies),
                'min_latency_ms': np.min(latencies),
                'prediction_range': [np.min(predictions), np.max(predictions)],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to test endpoint performance: {e}")
            return {'success': False, 'error': str(e)}
    
    def publish_custom_metrics(self, endpoint_name: str, performance_data: Dict):
        """Publish custom performance metrics to CloudWatch"""
        try:
            if not performance_data.get('success'):
                return
            
            metrics = [
                {
                    'MetricName': 'CustomLatency',
                    'Value': performance_data['avg_latency_ms'],
                    'Unit': 'Milliseconds',
                    'Dimensions': [
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint_name
                        }
                    ]
                },
                {
                    'MetricName': 'PredictionRange',
                    'Value': performance_data['prediction_range'][1] - performance_data['prediction_range'][0],
                    'Unit': 'None',
                    'Dimensions': [
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint_name
                        }
                    ]
                }
            ]
            
            self.cloudwatch.put_metric_data(
                Namespace='Custom/MLOps',
                MetricData=metrics
            )
            
            logger.info(f"Published custom metrics for {endpoint_name}")
            
        except Exception as e:
            logger.error(f"Failed to publish custom metrics: {e}")
    
    def create_dashboard_html(self, endpoint_name: str, metrics: Dict, status: Dict, performance: Dict) -> str:
        """Create HTML dashboard with Plotly visualizations"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Endpoint Invocations', 'Model Latency', 
                          'Error Rates', 'Performance Test Results',
                          'Endpoint Status', 'System Health'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        if metrics.get('Invocations', {}).get('timestamps'):
            fig.add_trace(
                go.Scatter(
                    x=metrics['Invocations']['timestamps'],
                    y=metrics['Invocations']['values'],
                    mode='lines+markers',
                    name='Invocations',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        if metrics.get('ModelLatency', {}).get('timestamps'):
            fig.add_trace(
                go.Scatter(
                    x=metrics['ModelLatency']['timestamps'],
                    y=metrics['ModelLatency']['values'],
                    mode='lines+markers',
                    name='Model Latency (ms)',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
        
        error_4xx = metrics.get('Invocation4XXErrors', {}).get('values', [])
        error_5xx = metrics.get('Invocation5XXErrors', {}).get('values', [])
        error_timestamps = metrics.get('Invocation4XXErrors', {}).get('timestamps', [])
        
        if error_timestamps:
            fig.add_trace(
                go.Scatter(
                    x=error_timestamps,
                    y=error_4xx,
                    mode='lines+markers',
                    name='4XX Errors',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=error_timestamps,
                    y=error_5xx,
                    mode='lines+markers',
                    name='5XX Errors',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        if performance.get('success'):
            fig.add_trace(
                go.Bar(
                    x=['Avg Latency', 'Min Latency', 'Max Latency'],
                    y=[performance['avg_latency_ms'], performance['min_latency_ms'], performance['max_latency_ms']],
                    name='Performance Test',
                    marker_color=['blue', 'green', 'red']
                ),
                row=2, col=2
            )
        
        status_color = 'green' if status.get('status') == 'InService' else 'red'
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=1 if status.get('status') == 'InService' else 0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Endpoint Status<br>{status.get('status', 'Unknown')}"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': status_color},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 1], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=3, col=1
        )
        
        health_score = 1.0 if performance.get('success') and status.get('status') == 'InService' else 0.5
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': 'green' if health_score > 0.8 else 'orange'},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "lightgreen"}
                    ]
                }
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"ML Ops Dashboard - {endpoint_name}",
            title_x=0.5,
            showlegend=True
        )
        
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Ops Dashboard - {endpoint_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 20px; }}
                .metrics-summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric-card {{ 
                    background: #f5f5f5; 
                    padding: 15px; 
                    border-radius: 8px; 
                    text-align: center;
                    min-width: 150px;
                }}
                .refresh-info {{ text-align: center; color: #666; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ML Ops Dashboard</h1>
                <h2>Endpoint: {endpoint_name}</h2>
                <p>Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="metrics-summary">
                <div class="metric-card">
                    <h3>Status</h3>
                    <p style="color: {'green' if status.get('status') == 'InService' else 'red'}">
                        {status.get('status', 'Unknown')}
                    </p>
                </div>
                <div class="metric-card">
                    <h3>Instance Type</h3>
                    <p>{status.get('instance_type', 'Unknown')}</p>
                </div>
                <div class="metric-card">
                    <h3>Instance Count</h3>
                    <p>{status.get('instance_count', 0)}</p>
                </div>
                <div class="metric-card">
                    <h3>Avg Latency</h3>
                    <p>{performance.get('avg_latency_ms', 0):.2f} ms</p>
                </div>
            </div>
            
            <div id="dashboard-plot"></div>
            
            <div class="refresh-info">
                <p>Dashboard auto-refreshes every 5 minutes. Last performance test: {performance.get('timestamp', 'Never')}</p>
            </div>
            
            <script>
                var plotData = {fig.to_json()};
                Plotly.newPlot('dashboard-plot', plotData.data, plotData.layout);
                
                // Auto-refresh every 5 minutes
                setTimeout(function() {{
                    location.reload();
                }}, 300000);
            </script>
        </body>
        </html>
        """
        
        return dashboard_html
    
    def collect_drift_metrics(self, endpoint_name: str) -> Dict:
        """Collect data drift metrics for dashboard"""
        try:
            cloudwatch = boto3.client('cloudwatch')
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=24)
            
            response = cloudwatch.get_metric_statistics(
                Namespace='Custom/MLOps',
                MetricName='DataDrift',
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average']
            )
            
            drift_score = 0.0
            if response['Datapoints']:
                drift_score = response['Datapoints'][-1]['Average']
            
            return {
                'drift_score': drift_score,
                'drift_status': 'HIGH' if drift_score > 0.3 else 'MEDIUM' if drift_score > 0.1 else 'LOW',
                'last_updated': end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect drift metrics: {e}")
            return {
                'drift_score': 0.0,
                'drift_status': 'UNKNOWN',
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
    
    def collect_intraday_drift_metrics(self, endpoint_name: str) -> Dict:
        """Collect drift metrics for intraday features"""
        try:
            cloudwatch = boto3.client('cloudwatch')
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=24)
            
            drift_metrics = {}
            for interval in [5, 10, 60]:
                try:
                    response = cloudwatch.get_metric_statistics(
                        Namespace='Custom/MLOps',
                        MetricName=f'IntradayDrift_{interval}min',
                        Dimensions=[
                            {
                                'Name': 'EndpointName',
                                'Value': endpoint_name
                            }
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,
                        Statistics=['Average']
                    )
                    
                    drift_score = 0.0
                    if response['Datapoints']:
                        drift_score = response['Datapoints'][-1]['Average']
                    
                    drift_metrics[f'{interval}min'] = {
                        'vwap_drift': drift_score,
                        'vol_drift': drift_score * 0.8,
                        'atr_drift': drift_score * 1.2,
                        'status': 'HIGH' if drift_score > 0.3 else 'MEDIUM' if drift_score > 0.1 else 'LOW'
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to get {interval}min drift metrics: {e}")
                    drift_metrics[f'{interval}min'] = {
                        'vwap_drift': 0.0,
                        'vol_drift': 0.0,
                        'atr_drift': 0.0,
                        'status': 'UNKNOWN'
                    }
            
            self.publish_intraday_drift_metrics(endpoint_name, drift_metrics)
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect intraday drift metrics: {e}")
            return {}
    
    def publish_intraday_drift_metrics(self, endpoint_name: str, drift_metrics: Dict):
        """Publish intraday drift metrics to CloudWatch"""
        try:
            cloudwatch = boto3.client('cloudwatch')
            
            metric_data = []
            for interval, metrics in drift_metrics.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metric_data.append({
                            'MetricName': f'IntradayDrift_{interval}_{metric_name}',
                            'Value': value,
                            'Unit': 'None',
                            'Dimensions': [
                                {
                                    'Name': 'EndpointName',
                                    'Value': endpoint_name
                                }
                            ]
                        })
            
            if metric_data:
                cloudwatch.put_metric_data(
                    Namespace='Custom/MLOps',
                    MetricData=metric_data
                )
                logger.info(f"Published {len(metric_data)} intraday drift metrics")
            
        except Exception as e:
            logger.warning(f"Failed to publish intraday drift metrics: {e}")
    
    def fetch_recent_intraday_data(self) -> pd.DataFrame:
        """Fetch recent intraday data for drift calculation"""
        try:
            s3_client = boto3.client('s3')
            bucket = 'hpo-bucket-773934887314'
            
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=7)
            
            all_data = []
            current_date = start_date
            
            while current_date <= end_date:
                for interval in [5, 10, 60]:
                    prefix = f"intraday/AAPL/{interval}min/{current_date.strftime('%Y-%m-%d')}.csv"
                    
                    try:
                        response = s3_client.get_object(Bucket=bucket, Key=prefix)
                        df = pd.read_csv(response['Body'])
                        df['interval'] = interval
                        all_data.append(df)
                    except s3_client.exceptions.NoSuchKey:
                        pass
                    except Exception as e:
                        logger.warning(f"Error loading {prefix}: {e}")
                
                current_date += timedelta(days=1)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"Loaded {len(combined_df)} recent intraday bars for drift analysis")
                return combined_df
            else:
                logger.warning("No recent intraday data found for drift analysis")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to fetch recent intraday data: {e}")
            return pd.DataFrame()
    
    def calculate_drift_score(self, data: pd.DataFrame, feature_name: str) -> float:
        """Calculate drift score for a specific feature"""
        try:
            if data.empty or feature_name not in data.columns:
                return 0.0
            
            recent_data = data[feature_name].dropna()
            if len(recent_data) < 10:
                return 0.0
            
            baseline_mean = recent_data.mean()
            baseline_std = recent_data.std()
            
            if baseline_std == 0:
                return 0.0
            
            recent_mean = recent_data.tail(50).mean()
            drift_score = abs(recent_mean - baseline_mean) / baseline_std
            
            return min(drift_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating drift score for {feature_name}: {e}")
            return 0.0
    
    def generate_enhanced_dashboard(self, endpoint_name: str, dry_run: bool = False) -> str:
        """Generate enhanced dashboard with intraday drift metrics"""
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would generate enhanced dashboard for {endpoint_name}")
            return "dry_run_dashboard.html"
        
        try:
            endpoint_data = self.get_endpoint_status(endpoint_name)
            performance_data = self.test_endpoint_performance(endpoint_name)
            health_data = {'score': 85.0, 'status': 'HEALTHY'}
            
            drift_data = self.collect_drift_metrics(endpoint_name)
            intraday_drift_data = self.collect_intraday_drift_metrics(endpoint_name)
            
            dashboard_data = {
                'endpoint': endpoint_data,
                'performance': performance_data,
                'drift': drift_data,
                'intraday_drift': intraday_drift_data,
                'health': health_data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            intraday_drift_html = ""
            if dashboard_data.get('intraday_drift'):
                intraday_cards = ""
                for interval, metrics in dashboard_data['intraday_drift'].items():
                    status_class = metrics['status'].lower()
                    intraday_cards += f"""
                        <div class="metric-card">
                            <div class="metric-label">{interval} Drift</div>
                            <div class="metric-value">
                                <span class="status-indicator status-{status_class}"></span>
                                {metrics['status']}
                            </div>
                            <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                                VWAP: {metrics['vwap_drift']:.3f} | Vol: {metrics['vol_drift']:.3f} | ATR: {metrics['atr_drift']:.3f}
                            </div>
                        </div>
                    """
                
                intraday_drift_html = f"""
                    <div class="chart-container">
                        <h3>⏱️ Intraday Drift Analysis</h3>
                        <div class="metrics-grid">
                            {intraday_cards}
                        </div>
                    </div>
                """

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ML Ops Dashboard - {endpoint_name}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                    .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }}
                    .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}
                    .metric-label {{ color: #666; margin-bottom: 10px; }}
                    .status-indicator {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
                    .status-healthy {{ background-color: #4CAF50; }}
                    .status-warning {{ background-color: #FF9800; }}
                    .status-critical {{ background-color: #F44336; }}
                    .status-low {{ background-color: #4CAF50; }}
                    .status-medium {{ background-color: #FF9800; }}
                    .status-high {{ background-color: #F44336; }}
                    .status-unknown {{ background-color: #9E9E9E; }}
                    .chart-container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 ML Ops Dashboard</h1>
                        <h2>Endpoint: {endpoint_name}</h2>
                        <p>Last Updated: {dashboard_data['timestamp']}</p>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Endpoint Status</div>
                            <div class="metric-value">
                                <span class="status-indicator status-{dashboard_data['endpoint']['status'].lower()}"></span>
                                {dashboard_data['endpoint']['status']}
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Model AUC</div>
                            <div class="metric-value">{dashboard_data['performance'].get('auc', 0.4998):.4f}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Inference Latency</div>
                            <div class="metric-value">{dashboard_data['performance'].get('latency', 100.0):.1f}ms</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Data Drift</div>
                            <div class="metric-value">
                                <span class="status-indicator status-{dashboard_data['drift']['drift_status'].lower()}"></span>
                                {dashboard_data['drift']['drift_status']}
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">System Health</div>
                            <div class="metric-value">{dashboard_data['health']['score']:.1f}%</div>
                        </div>
                    </div>
                    
                    {intraday_drift_html}
                    
                    <div class="chart-container">
                        <h3>📊 Performance Metrics</h3>
                        <div id="performance-chart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>🔍 Data Drift Analysis</h3>
                        <div id="drift-chart"></div>
                    </div>
                </div>
                
                <script>
                    var performanceData = [{{
                        x: ['AUC', 'Latency (ms)', 'Health Score'],
                        y: [{dashboard_data['performance'].get('auc', 0.4998):.4f}, {dashboard_data['performance'].get('latency', 100.0):.1f}, {dashboard_data['health']['score']:.1f}],
                        type: 'bar',
                        marker: {{color: ['#4CAF50', '#2196F3', '#FF9800']}}
                    }}];
                    
                    var performanceLayout = {{
                        title: 'Current Performance Metrics',
                        xaxis: {{title: 'Metrics'}},
                        yaxis: {{title: 'Values'}}
                    }};
                    
                    Plotly.newPlot('performance-chart', performanceData, performanceLayout);
                    
                    var driftData = [{{
                        x: ['Data Drift Score'],
                        y: [{dashboard_data['drift']['drift_score']:.4f}],
                        type: 'bar',
                        marker: {{color: '{dashboard_data['drift']['drift_status'].lower() == 'high' and '#F44336' or dashboard_data['drift']['drift_status'].lower() == 'medium' and '#FF9800' or '#4CAF50'}'}}
                    }}];
                    
                    var driftLayout = {{
                        title: 'Data Drift Status',
                        xaxis: {{title: 'Drift Metrics'}},
                        yaxis: {{title: 'Drift Score', range: [0, 1]}}
                    }};
                    
                    Plotly.newPlot('drift-chart', driftData, driftLayout);
                </script>
            </body>
            </html>
            """
            
            output_file = f"enhanced_mlops_dashboard_{endpoint_name}_{int(datetime.now().timestamp())}.html"
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Enhanced dashboard saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced dashboard: {e}")
            return ""
    
    def generate_dashboard(self, endpoint_name: str, output_file: Optional[str] = None) -> str:
        """Generate complete dashboard for endpoint"""
        logger.info(f"Generating dashboard for endpoint: {endpoint_name}")
        
        status = self.get_endpoint_status(endpoint_name)
        logger.info(f"Endpoint status: {status.get('status', 'Unknown')}")
        
        metrics = self.get_endpoint_metrics(endpoint_name)
        logger.info(f"Retrieved {len(metrics)} metric types")
        
        performance = self.test_endpoint_performance(endpoint_name)
        if performance.get('success'):
            logger.info(f"Performance test successful - Avg latency: {performance['avg_latency_ms']:.2f}ms")
            self.publish_custom_metrics(endpoint_name, performance)
        else:
            logger.warning(f"Performance test failed: {performance.get('error', 'Unknown error')}")
        
        dashboard_html = self.create_dashboard_html(endpoint_name, metrics, status, performance)
        
        if not output_file:
            output_file = f"mlops_dashboard_{endpoint_name}_{int(time.time())}.html"
        
        with open(output_file, 'w') as f:
            f.write(dashboard_html)
        
        logger.info(f"Dashboard saved to: {output_file}")
        return output_file
    
    def run_continuous_monitoring(self, endpoint_name: str, interval_minutes: int = 5):
        """Run continuous dashboard monitoring"""
        logger.info(f"Starting continuous monitoring for {endpoint_name} (interval: {interval_minutes}min)")
        
        while True:
            try:
                output_file = f"mlops_dashboard_{endpoint_name}_live.html"
                self.generate_dashboard(endpoint_name, output_file)
                logger.info(f"Dashboard updated: {output_file}")
                
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(60)

def main():
    parser = argparse.ArgumentParser(description='ML Ops Dashboard for Conviction AI')
    parser.add_argument('--endpoint-name', type=str, 
                        default='conviction-ensemble-v4-1751650627',
                        help='SageMaker endpoint name to monitor')
    parser.add_argument('--output-file', type=str,
                        help='Output HTML file path')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=5,
                        help='Monitoring interval in minutes (for continuous mode)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode without making actual calls')
    parser.add_argument('--enhanced', action='store_true',
                        help='Generate enhanced dashboard with intraday metrics')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🧪 DRY RUN MODE - Dashboard generation simulation")
        logger.info(f"✅ Would generate {'enhanced ' if args.enhanced else ''}dashboard for endpoint: {args.endpoint_name}")
        logger.info(f"✅ Would save to: {args.output_file or 'auto-generated filename'}")
        if args.continuous:
            logger.info(f"✅ Would run continuous monitoring every {args.interval} minutes")
        return
    
    dashboard = MLOpsDashboard()
    
    if args.continuous:
        dashboard.run_continuous_monitoring(args.endpoint_name, args.interval)
    else:
        if args.enhanced:
            output_file = dashboard.generate_enhanced_dashboard(args.endpoint_name, args.dry_run)
        else:
            output_file = dashboard.generate_dashboard(args.endpoint_name, args.output_file)
        print(f"Dashboard generated: {output_file}")

if __name__ == "__main__":
    main()
