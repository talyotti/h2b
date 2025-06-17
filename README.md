# H-1B Visa Outcome Analysis & Prediction

**R • tidymodels • rpart • ranger • xgboost**

## 1. Introduction

This project investigates how machine-learning models can automate preliminary decision-making in the U.S. H-1B visa approval process by predicting case outcomes (“Confirmed” vs. “Denied”) using historical application data (2011–2018). We exclude potentially discriminatory attributes (age, gender, race) to promote fairness and focus on model interpretability and performance.

## 2. Data

- **Source:** “H1B Non-immigrant Labour Visa” dataset from Kaggle  
- **Original size:** ~3,000,000 records (2011–2018)  
- **Subset used:** 500,000 records (80,000 Denied + 420,000 Confirmed)  
- **Split:**  
  - Training: 300,000  
  - Validation: 100,000  
  - Test: 100,000  

## 3. Preprocessing

1. **Feature Removal:** drop `decision_date`, `emp_country`, `wage_to`, `pw_level`  
2. **Outcome Cleaning:** recode “CW” → “C”, remove “W”, drop rows with missing `case_status`  
3. **Recipe Pipeline (tidymodels):**  
   - Convert nominal predictors to factors  
   - Extract `case_year`, `case_month`, `case_wday` from `case_submitted`  
   - Lump rare factor levels (<1%) into “other”  
   - Median imputation for numeric, mode for nominal, “unknown” for flagged missing  
   - Normalize all numeric predictors  
   - Remove near-perfectly correlated predictors (|r|>0.9)

## 4. Models

We fit five classifiers:

1. **Baseline Logistic Regression**  
2. **Mini-Batch Gradient-Descent Logistic Regression**  
3. **Interpretable Decision Tree** (depth = 3)  
4. **High-Dimensional Lasso Logistic Regression** (α=1, λ=1e-6)  
5. **Stacked Ensemble** (XGBoost, Random Forest, GLM ⟶ XGBoost meta-learner)

Each model’s probability threshold was tuned on the validation set to maximize accuracy.

## 5. Results on Test Set

| Metric                   | Baseline | Grad-Desc | Decision Tree | Lasso    | Ensemble  |
|--------------------------|---------:|----------:|--------------:|---------:|----------:|
| **Accuracy**             |   0.84110 |    0.85109 |       0.84407 |   0.85200 |   0.86758 |
| **Sensitivity (Recall)** |   0.99623 |    0.98147 |       0.97558 |   0.98000 |   0.97053 |
| **Specificity**          |   0.03141 |    0.16366 |       0.15766 |   0.18000 |   0.32474 |
| **Precision**            |   0.84297 |    0.86087 |       0.85805 |   0.86200 |   0.88343 |
| **F1 Score**             |   0.91322 |    0.91722 |       0.91305 |   0.91700 |   0.92493 |
| **ROC AUC**              |   0.68592 |    0.77602 |       0.66482 |   0.78200 |   0.83280 |

The stacked ensemble achieves the best overall balance, boosting specificity to 0.3247 and AUC to 0.8328.
