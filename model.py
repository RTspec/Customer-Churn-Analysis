import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# Load Dataset
df = pd.read_csv("dataset/customer_churn_processed.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

encoder = LabelEncoder()

encoder = LabelEncoder()

categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "Contract",
    "Churn"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print("Label Encoding Done")


# ==========================
# Features and Target
# ==========================

X = df[
    [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract"
    ]
]

y = df["Churn"]


# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training Model...")


# ==========================
# Logistic Regression Model
# ==========================

model = LogisticRegression(max_iter=1000)


# Train Model
model.fit(X_train, y_train)

print("Model Trained Successfully!")


# ==========================
# Prediction
# ==========================

y_pred = model.predict(X_test)


# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n===== MODEL PERFORMANCE =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))


# ==========================
# Save Model
# ==========================

with open("model/lr_churn_model.pkl", "wb") as f:
    pickle.dump(model, f)


# Save Encoder
with open("encoders/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


print("\nModel and Encoder Saved Successfully!")