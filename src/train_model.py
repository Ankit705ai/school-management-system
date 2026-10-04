"""Train and compare beginner-friendly classification models."""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).parents[1] / "data"
TARGET = "at_risk"
FORBIDDEN_FEATURES = {"student_id", "final_exam_score", TARGET}
MODEL_PATH = DATA_DIR / "random_forest_model.pkl"


def evaluate_model(name, model, x_train, y_train, x_test, y_test) -> None:
    """Fit one model and print its test-set evaluation metrics."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.3f}")
    print("Confusion matrix [rows = actual, columns = predicted]:")
    print(confusion_matrix(y_test, predictions))


def main() -> None:
    train_data = pd.read_csv(DATA_DIR / "train.csv")
    test_data = pd.read_csv(DATA_DIR / "test.csv")

    if TARGET not in train_data.columns or TARGET not in test_data.columns:
        raise ValueError("Both datasets must contain the at_risk target")

    feature_columns = [column for column in train_data.columns if column != TARGET]
    if FORBIDDEN_FEATURES.intersection(feature_columns):
        raise ValueError("student_id and final_exam_score must not be model features")
    if feature_columns != [column for column in test_data.columns if column != TARGET]:
        raise ValueError("Train and test feature columns must match")

    x_train = train_data[feature_columns]
    y_train = train_data[TARGET]
    x_test = test_data[feature_columns]
    y_test = test_data[TARGET]

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    print(f"Training rows: {x_train.shape[0]}")
    print(f"Test rows: {x_test.shape[0]}")
    print(f"Features used: {len(feature_columns)}")
    print("Excluded: student_id, final_exam_score")

    for name, model in models.items():
        evaluate_model(name, model, x_train, y_train, x_test, y_test)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {"model": models["Random Forest"], "feature_columns": feature_columns},
            model_file,
        )
    print(f"\nSaved Random Forest model to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()