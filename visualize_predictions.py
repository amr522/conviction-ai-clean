#!/usr/bin/env python3
"""
visualize_predictions.py - Create visualizations for model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Load predictions
df = pd.read_csv('predictions_with_labels.csv')

# Create directory for figures
import os
os.makedirs('evaluation_figures', exist_ok=True)

# Calculate metrics
rmse = sqrt(mean_squared_error(df['return'], df['prediction']))
mae = mean_absolute_error(df['return'], df['prediction'])
r2 = r2_score(df['return'], df['prediction'])

# 1. Scatter plot of Predicted vs Actual
plt.figure(figsize=(10, 8))
sns.scatterplot(x='return', y='prediction', data=df, alpha=0.6)
plt.plot([-0.4, 0.8], [-0.4, 0.8], 'r--')  # Perfect prediction line
plt.title(f'Predicted vs Actual Returns\nRMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}')
plt.xlabel('Actual Return')
plt.ylabel('Predicted Return')
plt.xlim(-0.4, 0.8)
plt.ylim(-0.4, 0.8)
plt.savefig('evaluation_figures/predicted_vs_actual.png', dpi=300, bbox_inches='tight')

# 2. Histogram of residuals
plt.figure(figsize=(10, 8))
residuals = df['prediction'] - df['return']
sns.histplot(residuals, kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Histogram of Residuals (Predicted - Actual)')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.savefig('evaluation_figures/residuals_histogram.png', dpi=300, bbox_inches='tight')

# 3. Residuals vs Actual Values
plt.figure(figsize=(10, 8))
sns.scatterplot(x='return', y=residuals, data=df, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Actual Values')
plt.xlabel('Actual Return')
plt.ylabel('Residual (Predicted - Actual)')
plt.savefig('evaluation_figures/residuals_vs_actual.png', dpi=300, bbox_inches='tight')

# 4. Time Series of Actual vs Predicted
plt.figure(figsize=(15, 8))
# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
plt.plot(df['timestamp'], df['return'], label='Actual', alpha=0.7)
plt.plot(df['timestamp'], df['prediction'], label='Predicted', alpha=0.7)
plt.title('Time Series of Actual vs Predicted Returns')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('evaluation_figures/time_series.png', dpi=300, bbox_inches='tight')

# 5. Error Distribution by Date
plt.figure(figsize=(15, 8))
df['abs_error'] = np.abs(residuals)
df['month'] = df['timestamp'].dt.to_period('M')
monthly_error = df.groupby('month')['abs_error'].mean().reset_index()
monthly_error['month'] = monthly_error['month'].astype(str)
plt.bar(monthly_error['month'], monthly_error['abs_error'])
plt.title('Average Absolute Error by Month')
plt.xlabel('Month')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('evaluation_figures/error_by_month.png', dpi=300, bbox_inches='tight')

print("Visualizations created in 'evaluation_figures' directory")
print(f"Evaluation Metrics:")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"R²:   {r2:.6f}")
