"""IN6227 Assignment 1 - Variant 1
End-to-end comparison of Logistic Regression and Random Forest.

Usage:
    python classification_workflow.py

Expected files in the same directory:
    train.csv
    test.csv
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
TARGET = "label"
RANDOM_STATE = 42


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"Recall   : {recall_score(y_test, pred):.4f}")
    print(f"F1       : {f1_score(y_test, pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, prob):.4f}")
    print("Confusion matrix [rows=true, cols=predicted]:")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["no", "yes"], digits=4))
    return model


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # The target cannot be used for supervised learning when it is missing.
    train = train.dropna(subset=[TARGET]).copy()
    test = test.dropna(subset=[TARGET]).copy()

    X_train = train.drop(columns=TARGET)
    y_train = train[TARGET].map({"no": 0, "yes": 1})
    X_test = test.drop(columns=TARGET)
    y_test = test[TARGET].map({"no": 0, "yes": 1})

    print(f"Training rows used: {len(train)}")
    print(f"Test rows used:     {len(test)}")
    print("Training class distribution:")
    print(train[TARGET].value_counts(normalize=True).rename("proportion"))
    print("Missing feature values in training data:")
    print(X_train.isna().sum()[X_train.isna().sum() > 0])

    lr = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    rf = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    evaluate("Logistic Regression", lr, X_train, y_train, X_test, y_test)
    evaluate("Random Forest", rf, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
