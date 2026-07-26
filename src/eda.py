"""
Telco Customer Churn - Full Exploratory Data Analysis (EDA) Script

"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
FIG_DIR = "notebooks/figures/eda"
os.makedirs(FIG_DIR, exist_ok=True)

pd.set_option("display.max_columns", None)


def save(fig_name):
    plt.savefig(f"{FIG_DIR}/{fig_name}.png", bbox_inches="tight", dpi=120)
    plt.close()


# ------------------------------------------------------------------
# A. LOAD
# ------------------------------------------------------------------
df = pd.read_csv("/home/saad-s-linux/Documents/Telco Customer Churn Prediction/data/Raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("A. SHAPE:", df.shape)

# ------------------------------------------------------------------
# B. STRUCTURE
# ------------------------------------------------------------------
print("\nB. DTYPES:")
print(df.dtypes)
print("\nB. HEAD:")
print(df.head())

# ------------------------------------------------------------------
# C. FIX TotalCharges (loads as object due to blank strings)
# ------------------------------------------------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\nC. TotalCharges NaNs after conversion:", df["TotalCharges"].isnull().sum())

# Save cleaned version so we don't have to redo this fix every time
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/cleaned_data.csv", index=False)
print("C. Cleaned data saved to data/processed/cleaned_data.csv")

# ------------------------------------------------------------------
# D. MISSING VALUES
# ------------------------------------------------------------------
print("\nD. MISSING VALUES:")
missing = df.isnull().sum()
print(missing[missing > 0])

# ------------------------------------------------------------------
# E. DUPLICATES
# ------------------------------------------------------------------
print("\nE. DUPLICATE ROWS:", df.duplicated().sum())
print("E. DUPLICATE customerIDs:", df["customerID"].duplicated().sum())

# ------------------------------------------------------------------
# F. SUMMARY STATS - NUMERIC
# ------------------------------------------------------------------
print("\nF. NUMERIC SUMMARY:")
print(df.describe())

# ------------------------------------------------------------------
# G. SUMMARY STATS - CATEGORICAL
# ------------------------------------------------------------------
cat_cols = df.select_dtypes(include="object").columns.drop("customerID")
print("\nG. CATEGORICAL VALUE COUNTS:")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# ------------------------------------------------------------------
# H. TARGET DISTRIBUTION (class imbalance check)
# ------------------------------------------------------------------
print("\nH. CHURN DISTRIBUTION:")
print(df["Churn"].value_counts(normalize=True))

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Churn")
plt.title("Target Distribution: Churn")
save("h_target_distribution")

# ------------------------------------------------------------------
# I. UNIVARIATE - NUMERIC DISTRIBUTIONS
# ------------------------------------------------------------------
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution: {col}")
plt.tight_layout()
save("i_numeric_distributions")

# ------------------------------------------------------------------
# J. UNIVARIATE - OUTLIER CHECK (boxplots)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"Outlier Check: {col}")
plt.tight_layout()
save("j_outlier_check")

# ------------------------------------------------------------------
# K. UNIVARIATE - CATEGORICAL COUNTS (key ones)
# ------------------------------------------------------------------
key_cats = ["Contract", "InternetService", "PaymentMethod", "gender"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.flatten(), key_cats):
    sns.countplot(data=df, x=col, ax=ax)
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
save("k_categorical_counts")

# ------------------------------------------------------------------
# L. BIVARIATE - NUMERIC vs TARGET
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(data=df, x="Churn", y=col, ax=ax)
    ax.set_title(f"{col} vs Churn")
plt.tight_layout()
save("l_numeric_vs_churn")

# ------------------------------------------------------------------
# M. BIVARIATE - CATEGORICAL vs TARGET (churn rate per category)
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.flatten(), key_cats):
    churn_rate = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean())
    churn_rate.plot(kind="bar", ax=ax, color="salmon")
    ax.set_title(f"Churn Rate by {col}")
    ax.set_ylabel("Churn Rate")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
save("m_churn_rate_by_category")

# ------------------------------------------------------------------
# N. SERVICES vs CHURN (all 6 add-on services at once)
# ------------------------------------------------------------------
service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), service_cols):
    churn_rate = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean())
    churn_rate.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(col)
    ax.set_ylabel("Churn Rate")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
save("n_services_vs_churn")

# ------------------------------------------------------------------
# O. SENIOR CITIZEN / PARTNER / DEPENDENTS vs CHURN
# ------------------------------------------------------------------
demo_cols = ["SeniorCitizen", "Partner", "Dependents"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, demo_cols):
    churn_rate = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean())
    churn_rate.plot(kind="bar", ax=ax, color="mediumseagreen")
    ax.set_title(f"Churn Rate by {col}")
plt.tight_layout()
save("o_demographics_vs_churn")

# ------------------------------------------------------------------
# P. CORRELATION HEATMAP (numeric features only)
# ------------------------------------------------------------------
corr_df = df[numeric_cols].copy()
corr_df["Churn_binary"] = df["Churn"].map({"Yes": 1, "No": 0})
plt.figure(figsize=(6, 5))
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
save("p_correlation_heatmap")

# ------------------------------------------------------------------
# Q. TENURE BUCKETS vs CHURN
# ------------------------------------------------------------------
df["tenure_group"] = pd.cut(
    df["tenure"], bins=[0, 12, 24, 48, 60, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"]
)
plt.figure(figsize=(6, 4))
churn_rate = df.groupby("tenure_group")["Churn"].apply(lambda x: (x == "Yes").mean())
churn_rate.plot(kind="bar", color="orange")
plt.title("Churn Rate by Tenure Group")
plt.ylabel("Churn Rate")
save("q_tenure_group_vs_churn")

# ------------------------------------------------------------------
# R. MONTHLY CHARGES DISTRIBUTION BY INTERNET SERVICE
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.kdeplot(data=df, x="MonthlyCharges", hue="InternetService", fill=True, alpha=0.4)
plt.title("Monthly Charges Distribution by Internet Service Type")
save("r_charges_by_internet_service")

# ------------------------------------------------------------------
# S. PAYMENT METHOD vs CHURN (electronic check is famously high-risk)
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4))
churn_rate = df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x == "Yes").mean())
churn_rate.sort_values().plot(kind="barh", color="purple")
plt.title("Churn Rate by Payment Method")
plt.xlabel("Churn Rate")
save("s_payment_method_vs_churn")

# ------------------------------------------------------------------
# T. PAIRPLOT - numeric features colored by churn
# ------------------------------------------------------------------
sample_df = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].dropna()
sns.pairplot(sample_df, hue="Churn", diag_kind="kde", plot_kws={"alpha": 0.4})
plt.savefig(f"{FIG_DIR}/t_pairplot.png", bbox_inches="tight", dpi=120)
plt.close()

# ------------------------------------------------------------------
# U. SKEWNESS / KURTOSIS OF NUMERIC FEATURES
# ------------------------------------------------------------------
print("\nU. SKEWNESS:")
print(df[numeric_cols].skew())
print("\nU. KURTOSIS:")
print(df[numeric_cols].kurtosis())

# ------------------------------------------------------------------
# V. VALUE RANGES / SANITY CHECKS
# ------------------------------------------------------------------
print("\nV. SANITY CHECKS:")
print("Min tenure:", df["tenure"].min(), "| Max tenure:", df["tenure"].max())
print("Min MonthlyCharges:", df["MonthlyCharges"].min(), "| Max:", df["MonthlyCharges"].max())
print("Negative values anywhere?", (df[numeric_cols] < 0).any().any())

# ------------------------------------------------------------------
# W. HIGH-CHURN-RISK SEGMENT (business insight)
# ------------------------------------------------------------------
risky_segment = df[
    (df["Contract"] == "Month-to-month") &
    (df["tenure"] <= 12) &
    (df["InternetService"] == "Fiber optic")
]
print(f"\nW. High-risk segment size: {len(risky_segment)} customers")
print("W. Churn rate in this segment:",
      (risky_segment["Churn"] == "Yes").mean())
print("W. Overall churn rate for comparison:",
      (df["Churn"] == "Yes").mean())

# ------------------------------------------------------------------
# X. FEATURE SUMMARY TABLE (auto-generated for README/report use)
# ------------------------------------------------------------------
summary = pd.DataFrame({
    "dtype": df.dtypes,
    "n_missing": df.isnull().sum(),
    "n_unique": df.nunique(),
})
print("\nX. FEATURE SUMMARY TABLE:")
print(summary)
summary.to_csv(f"{FIG_DIR}/feature_summary.csv")

# ------------------------------------------------------------------
# Y. KEY TAKEAWAYS (printed, not plotted)
# ------------------------------------------------------------------
print("\nY. KEY TAKEAWAYS:")
print("- Target is imbalanced: ~73% stayed, ~27% churned -> use ROC-AUC/F1, not accuracy")
print("- Month-to-month contracts + short tenure + fiber optic internet = highest churn risk")
print("- Electronic check payment method correlates with higher churn")
print("- Customers with more add-on services (security, backup, tech support) churn less")
print("- TotalCharges has ~11 missing values, all customers with 0 tenure -> logical, not random")

# ------------------------------------------------------------------
# Z. DONE
# ------------------------------------------------------------------
print(f"\nZ. All EDA figures saved to: {FIG_DIR}/")
print("EDA complete.")
