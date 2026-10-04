"""Create stratified training and test datasets from cleaned student data."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import CLEANED_PATH, TARGET, prepare_features


DATA_DIR = Path(__file__).parents[1] / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def main() -> None:
    data = pd.read_csv(CLEANED_PATH)
    features, target = prepare_features(data)

    if "final_exam_score" in features.columns:
        raise ValueError("final_exam_score must not be used as a feature")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    train_data = x_train.copy()
    train_data[TARGET] = y_train
    test_data = x_test.copy()
    test_data[TARGET] = y_test

    train_data.to_csv(TRAIN_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)

    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    print("Train at_risk distribution:")
    print(train_data[TARGET].value_counts(normalize=True).sort_index().round(4))
    print("Test at_risk distribution:")
    print(test_data[TARGET].value_counts(normalize=True).sort_index().round(4))
    print(f"Saved train data to: {TRAIN_PATH.resolve()}")
    print(f"Saved test data to: {TEST_PATH.resolve()}")


if __name__ == "__main__":
    main()