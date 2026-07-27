import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("="*50)
print("First 5 Rows")
print("="*50)
print(df.head())

print("\nDataset Shape :", df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types")
print(df.dtypes)

# ======================================
# Missing Values
# ======================================

print("\nMissing Values")
print(df.isnull().sum())

# ======================================
# Remove Duplicates
# ======================================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

df.drop_duplicates(inplace=True)

# ======================================
# TotalCharges Cleaning
# ======================================

if "TotalCharges" in df.columns:

    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df.to_csv("dataset/customer_churn_cleaned_for_eda.csv", index=False)
    

# ======================================
# Label Encoding
# ======================================

encoder = LabelEncoder()

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:

    df[col] = encoder.fit_transform(df[col])

print("\nLabel Encoding Completed.")

# ======================================
# Feature Scaling
# ======================================

scaler = StandardScaler()

numerical_columns = df.drop("Churn", axis=1).columns

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("Feature Scaling Completed.")

# ======================================
# Save Processed Dataset
# ======================================

df.to_csv("dataset/customer_churn_processed.csv", index=False)

print("\nProcessed dataset saved successfully!")

print("\nFinal Dataset Shape :", df.shape)