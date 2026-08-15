# Model Evaluation Metrics

## XGBoost (Held-Out Test Set)

### Performance Comparison

| Metric | Default (0.50 Threshold) | Calibrated (0.9790 Threshold) |
|--------|--------------------------|---------------------------------------------|
| **Precision** | 0.0927 | **0.8258** |
| **Recall** | 0.8839 | **0.5725** |
| **F1-Score** | 0.1679 | **0.6762** |
| **ROC-AUC** | 0.9796 | **0.9796** |
| **PR-AUC** | 0.7019 | **0.7019** |

### Confusion Matrix (Calibrated Threshold)
```
[[553315, 259], [917, 1228]]
```

### Feature Importance (Top Features)
| Feature | Importance |
|---------|------------|
| `amt` | 0.4103 |
| `amt_log` | 0.2039 |
| `benford_score` | 0.1203 |
| `hour` | 0.0976 |
| `category_code` | 0.0670 |
| `dayofweek` | 0.0468 |
| `city_pop` | 0.0145 |
| `distance` | 0.0123 |
