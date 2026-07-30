"""
Data Preprocessing — Customer Churn

Cleans the raw dataset, encodes categoricals, scales numerics, and saves
the fitted encoders/scaler so app.py can apply the *exact same* transform
to new user input at prediction time.

Run:
    python preprocess.py

Outputs:
    dataset/customer_churn_processed.csv
    model/label_encoders.pkl   (dict of {column: fitted LabelEncoder})
    model/scaler.pkl           (fitted StandardScaler)
    model/feature_columns.pkl  (list of column names/order used for training)
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_PATH = "dataset/customer_churn_processed.csv"


def run_preprocessing():
    df = pd.read_csv(RAW_PATH)
    print("Loaded:", df.shape)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop customerID — pure identifier, no predictive value
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Fix TotalCharges (blank strings for tenure=0 customers)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Label encode all categorical columns — save each encoder so app.py
    # can transform raw user input the same way at prediction time
    categorical_columns = df.select_dtypes(include="object").columns
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    print("Label encoded columns:", list(categorical_columns))

    # Scale all features except the target
    feature_columns = df.drop("Churn", axis=1).columns.tolist()
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    print("Feature scaling completed.")

    # Save cleaned dataset
    df.to_csv(CLEAN_PATH, index=False)

    # Save the encoders, scaler, and feature column order — app.py needs
    # all three to preprocess a brand-new customer's raw input identically
    joblib.dump(encoders, "model/label_encoders.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(feature_columns, "model/feature_columns.pkl")

    print(f"\nSaved processed dataset -> {CLEAN_PATH}  {df.shape}")
    print("Saved model/label_encoders.pkl, model/scaler.pkl, model/feature_columns.pkl")


if __name__ == "__main__":
    run_preprocessing()
