# Model Evaluation Metrics

## XGBoost (Test Set)
| Metric | Value |
|--------|-------|
| precision | 0.0773 |
| recall | 0.8699 |
| f1 | 0.1419 |
| roc_auc | 0.9768 |
| pr_auc | 0.6616 |

### Confusion Matrix
```
[[531287, 22287], [279, 1866]]
```

### Feature Importance (Top 5)
| Feature | Importance |
|---------|------------|
| amt | 0.4098 |
| amt_log | 0.2017 |
| benford_score | 0.1231 |
| hour | 0.0933 |
| category_code | 0.0691 |
