"""
Model Training — Customer Churn

Trains three classifiers on the preprocessed dataset and saves each one.
app.py loads all three and combines their predictions into a single
majority-vote result — the user never picks a model.

Run:
    python train_model.py

Outputs:
    model/logistic_regression.pkl
    model/decision_tree.pkl
    model/random_forest.pkl
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "dataset/customer_churn_processed.csv"


def train_and_save():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"\n{'=' * 50}")
        print(f"{name}  —  Accuracy: {acc:.4f}")
        print(f"{'=' * 50}")
        print(classification_report(y_test, preds))

        joblib.dump(model, f"model/{name}.pkl")
        print(f"Saved -> model/{name}.pkl")


if __name__ == "__main__":
    train_and_save()
