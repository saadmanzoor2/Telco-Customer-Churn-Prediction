# Telco Customer Churn Prediction

## Overview
End-to-end ML pipeline predicting which telecom customers are likely to churn, with a deployed Streamlit app for live predictions.

## Problem Statement
Telecom companies lose significant revenue to customer churn. Acquiring a new customer costs far more than retaining an existing one, so identifying at-risk customers early lets the business act with targeted retention offers.

## Business Value
- Flags high-risk customers before they leave
- Surfaces the drivers of churn (contract type, tenure, pricing) to inform retention strategy
- Deployed as an interactive tool a non-technical team (sales/support) could use directly

## Dataset
[Telco Customer Churn (IBM sample data)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 20 features covering demographics, account details, and subscribed services.

## Workflow
1. Data cleaning (fixed `TotalCharges` type issue, handled missing values)
2. EDA (churn distribution, tenure/charges/contract relationships)
3. Feature engineering (tenure buckets, services count, spend-per-tenure)
4. Model comparison: Logistic Regression (baseline) → Random Forest → XGBoost
5. Hyperparameter tuning via GridSearchCV, evaluated on ROC-AUC (not accuracy, due to class imbalance)
6. Best model saved and served through a Streamlit app

## Results
See `notebooks/figures/` for EDA visuals and confusion matrices per model. The tuned XGBoost model was selected based on ROC-AUC on the held-out test set.

## Usage
```bash
# Train the model
python src/train.py

# Launch the app
streamlit run app.py
```

## Future Improvements
- Add SHAP explainability for individual predictions
- Deploy via FastAPI + Docker for a production-style API
- Add model monitoring / retraining pipeline

## Tech Stack
Python, pandas, scikit-learn, XGBoost, imbalanced-learn, Streamlit

## Author
Saad Manzoor

## LinkedIN
linkedin.com/in/saad-manzoor-b81605347

## License
MIT
