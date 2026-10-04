"""Prepare model features without training a model or changing the CSV files."""

from pathlib import Path

import pandas as pd


CLEANED_PATH = Path(__file__).parents[1] / "data" / "students_cleaned.csv"
TARGET = "at_risk"
EXCLUDED_FEATURES = {"student_id", "final_exam_score", TARGET}


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return encoded input features X and the target y for a later data split."""
    if TARGET not in data.columns:
        raise ValueError("The dataset must contain the at_risk target")
    if "final_exam_score" not in data.columns:
        raise ValueError("The dataset must contain final_exam_score so it can be excluded")
    if data[TARGET].isna().any() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("at_risk must contain only non-missing values 0 or 1")

    feature_columns = [column for column in data.columns if column not in EXCLUDED_FEATURES]
    features = pd.get_dummies(data[feature_columns], dtype=int)
    target = data[TARGET].astype("int64")

    if "final_exam_score" in features.columns:
        raise ValueError("final_exam_score must not be included in the features")
    if features.isna().any().any():
        raise ValueError("Features must not contain missing values")

    return features.reset_index(drop=True), target.reset_index(drop=True)


def main() -> None:
    data = pd.read_csv(CLEANED_PATH)
    features, target = prepare_features(data)

    print(f"Input rows: {len(data)}")
    print(f"Prepared feature shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target: {TARGET}")
    print("Excluded from features: student_id, final_exam_score")
    print("Ready for a later train/test split; no model was trained.")


if __name__ == "__main__":
    main()