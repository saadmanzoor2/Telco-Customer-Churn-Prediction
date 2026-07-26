"""
Telco Customer Churn Prediction - Full Training Pipeline
Run from project root: python src/train.py
Requires: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

RAW_PATH = "/home/saad-s-linux/Documents/Telco Customer Churn Prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_PATH = "data/processed/cleaned_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("notebooks/figures", exist_ok=True)


# ----------------------------------------------------------------------
# 1. LOAD + CLEAN
# ----------------------------------------------------------------------
def load_and_clean():
    # If eda.py already saved a cleaned version, reuse it - saves recomputation
    if os.path.exists(PROCESSED_PATH):
        print(f"Loading pre-cleaned data from {PROCESSED_PATH}")
        df = pd.read_csv(PROCESSED_PATH)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    else:
        print(f"No processed file found, cleaning from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)

    # Always fill remaining NaNs (0-tenure customers with no billing history yet)
    # regardless of which branch loaded the data
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Drop ID column, not a feature
    df = df.drop(columns=["customerID"])

    # Target -> binary (skip if already converted)
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


# ----------------------------------------------------------------------
# 2. EDA (saves plots instead of showing, since this runs as a script)
# ----------------------------------------------------------------------
def run_eda(df):
    sns.set_style("whitegrid")

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Churn Distribution (0=Stayed, 1=Churned)")
    plt.savefig("notebooks/figures/churn_balance.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df, x="Churn", y="tenure")
    plt.title("Tenure vs Churn")
    plt.savefig("notebooks/figures/tenure_vs_churn.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly Charges vs Churn")
    plt.savefig("notebooks/figures/charges_vs_churn.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Contract Type vs Churn")
    plt.savefig("notebooks/figures/contract_vs_churn.png", bbox_inches="tight")
    plt.close()

    print("EDA figures saved to notebooks/figures/")
    print(df["Churn"].value_counts(normalize=True))


# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Tenure buckets - captures nonlinearity a raw number might miss
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"]
    )

    # Average monthly spend so far - proxy for engagement/value
    df["avg_charge_per_tenure"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Count of subscribed add-on services - more services = more "stickiness"
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(row == "Yes"), axis=1
    )

    return df


# ----------------------------------------------------------------------
# 4. BUILD PREPROCESSING + MODEL PIPELINE
# ----------------------------------------------------------------------
def build_pipeline(X, model):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return pipe


# ----------------------------------------------------------------------
# 5. TRAIN + EVALUATE MULTIPLE MODELS
# ----------------------------------------------------------------------
def evaluate_model(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"notebooks/figures/confusion_{name.replace(' ', '_')}.png", bbox_inches="tight")
    plt.close()

    return auc


def main():
    print("Loading data...")
    df = load_and_clean()

    print("Running EDA...")
    run_eda(df)

    print("Engineering features...")
    df = engineer_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # --- Baseline: Logistic Regression ---
    log_pipe = build_pipeline(X_train, LogisticRegression(max_iter=1000, class_weight="balanced"))
    log_pipe.fit(X_train, y_train)
    results["Logistic Regression"] = (log_pipe, evaluate_model("Logistic_Regression", log_pipe, X_test, y_test))

    # --- Random Forest ---
    rf_pipe = build_pipeline(X_train, RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ))
    rf_pipe.fit(X_train, y_train)
    results["Random Forest"] = (rf_pipe, evaluate_model("Random_Forest", rf_pipe, X_test, y_test))

    # --- XGBoost with hyperparameter tuning ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_pipe = build_pipeline(X_train, XGBClassifier(
        eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=42
    ))

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1],
    }

    print("\nRunning GridSearchCV for XGBoost (this may take a minute)...")
    grid = GridSearchCV(xgb_pipe, param_grid, scoring="roc_auc", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")

    results["XGBoost (tuned)"] = (best_xgb, evaluate_model("XGBoost_tuned", best_xgb, X_test, y_test))

    # --- Pick best model by ROC-AUC ---
    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"\nBest model: {best_name} (ROC-AUC: {results[best_name][1]:.4f})")

    joblib.dump(best_model, f"{MODEL_DIR}/churn_model.pkl")
    print(f"Saved best model to {MODEL_DIR}/churn_model.pkl")

    # Save feature columns for the Streamlit app
    joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_columns.pkl")


if __name__ == "__main__":
    main()