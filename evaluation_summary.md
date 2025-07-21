# Model Evaluation Summary

## Endpoint
- **Endpoint Name:** conviction-ai-endpoint-20250721045048
- **Holdout Dataset:** data/holdout_2023_clean.csv (623 records from 2023-01-01 onward)
- **Target Column:** return

## Performance Metrics
- **RMSE:** 0.108840
- **MAE:** 0.070254
- **R²:** -0.776685

## Analysis
The negative R² value indicates that the model performs worse than simply using the mean of the target values as predictions. This suggests the model needs improvement.

### Statistics Comparison
| Statistic | Predictions | Actual Returns |
|-----------|-------------|----------------|
| Min       | -0.224243   | -0.357539      |
| Max       | 0.699265    | 0.740391       |
| Mean      | 0.001009    | 0.002591       |
| Std Dev   | 0.065514    | 0.081721       |

### Observations
1. The model's predictions have a smaller range and standard deviation than the actual values, suggesting it tends to make more conservative predictions.
2. The negative R² value indicates poor predictive performance, which is concerning for a financial prediction model.
3. The model appears to struggle with capturing the full volatility of the returns.

### Next Steps
1. Retrain the model with different hyperparameters or algorithms
2. Consider feature engineering to better capture relevant patterns in the data
3. Evaluate whether additional data sources could improve model performance
4. Investigate whether the training data differs significantly from the holdout data (data drift)

### Visualizations
Visualizations have been saved to the `evaluation_figures` directory:
- Predicted vs Actual scatter plot
- Residuals histogram
- Residuals vs Actual values
- Time series comparison
- Error by month analysis

