# EDA Report — Sparkov Fraud Simulation Dataset

## Dataset Shape
| Split | Rows | Columns |
|-------|------|---------|
| Train | 1,296,675 | 15 |
| Test  | 555,719 | 15 |

## Class Distribution (is_fraud)
| Split | Legit (0) | Fraud (1) | Fraud % |
|-------|-----------|-----------|---------|
| Train | 1,289,169 | 7,506 | 0.5789% |
| Test  | 553,574  | 2,145  | 0.3860% |

> The fraud rate is **< 1 %** — severe class imbalance that requires
> special handling during modelling (SMOTE / class weights).

## Transaction Amount Distribution
![Amount Distribution](amount_distribution.png)

## Fraud Rate by Merchant Category
![Fraud by Category](fraud_by_category.png)

## Fraud Rate by Hour of Day
![Fraud by Hour](fraud_by_hour.png)

## Missing Values & Duplicates
- No missing values detected after initial cleaning.
- Duplicate rows (if any) were removed during preprocessing.

## Key Observations
1. Fraud transactions cluster at higher dollar amounts.
2. Certain merchant categories (e.g. `grocery_pos`, `shopping_net`) show
   elevated fraud rates.
3. Late-night / early-morning hours exhibit higher fraud incidence.

---
*Data source: Sparkov fraud-simulation dataset (synthetic). This is NOT real
transaction data.*
