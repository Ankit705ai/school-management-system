"""Clean the raw student dataset without changing the original CSV."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent
RAW_PATH = DATA_DIR / "students.csv"
CLEANED_PATH = DATA_DIR / "students_cleaned.csv"

NUMERIC_RANGES = {
    "age": (15, 22),
    "study_hours_per_week": (0, 40),
    "attendance_rate": (0, 100),
    "previous_exam_score": (0, 100),
    "assignment_score": (0, 100),
    "final_exam_score": (0, 100),
}

CATEGORY_VALUES = {
    "gender": ["Female", "Male", "Non-binary"],
    "program": ["STEM", "Humanities", "Commerce", "Arts"],
    "parental_education": ["High School", "Bachelor", "Master"],
    "internet_access": ["Yes", "No"],
    "extracurricular_activity": ["Yes", "No"],
    "academic_support": ["Yes", "No"],
}


def clean_dataset() -> tuple[pd.DataFrame, int, int]:
    """Read, validate, and clean the raw data."""
    raw_data = pd.read_csv(RAW_PATH)
    cleaned_data = raw_data.drop_duplicates().copy()
    removed_duplicates = len(raw_data) - len(cleaned_data)
    original_missing = int(raw_data.isna().sum().sum())

    required_columns = [
        "student_id",
        "age",
        "gender",
        "program",
        "study_hours_per_week",
        "attendance_rate",
        "previous_exam_score",
        "assignment_score",
        "final_exam_score",
        "parental_education",
        "internet_access",
        "extracurricular_activity",
        "academic_support",
        "at_risk",
    ]
    if set(cleaned_data.columns) != set(required_columns):
        raise ValueError("The raw dataset does not have the expected columns")

    # Convert numeric columns and turn values outside documented ranges into missing values.
    for column, (lower, upper) in NUMERIC_RANGES.items():
        cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors="coerce")
        invalid = cleaned_data[column].notna() & ~cleaned_data[column].between(lower, upper)
        cleaned_data.loc[invalid, column] = np.nan

    # Turn unknown categories into missing values before filling them with the mode.
    for column, valid_values in CATEGORY_VALUES.items():
        invalid = cleaned_data[column].notna() & ~cleaned_data[column].isin(valid_values)
        cleaned_data.loc[invalid, column] = np.nan

    # Validate the existing target; do not recalculate it from final_exam_score.
    cleaned_data["at_risk"] = pd.to_numeric(cleaned_data["at_risk"], errors="coerce")
    if cleaned_data["at_risk"].isna().any() or not cleaned_data["at_risk"].isin([0, 1]).all():
        raise ValueError("at_risk must contain only non-missing values 0 or 1")
    cleaned_data["at_risk"] = cleaned_data["at_risk"].astype("int64")

    for column in NUMERIC_RANGES:
        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].median())

    for column in CATEGORY_VALUES:
        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].mode().iloc[0])

    cleaned_data["student_id"] = cleaned_data["student_id"].astype("string")
    cleaned_data["age"] = cleaned_data["age"].round().astype("int64")
    return cleaned_data, original_missing, removed_duplicates


def main() -> None:
    raw_data = pd.read_csv(RAW_PATH)
    cleaned_data, original_missing, removed_duplicates = clean_dataset()
    cleaned_data.to_csv(CLEANED_PATH, index=False)

    # Re-read the output so the validation checks the file that was actually saved.
    saved_data = pd.read_csv(CLEANED_PATH)
    if saved_data.isna().sum().sum() != 0:
        raise ValueError("The cleaned dataset still contains missing values")
    if saved_data.duplicated().any():
        raise ValueError("The cleaned dataset still contains duplicate rows")
    if not saved_data["at_risk"].isin([0, 1]).all():
        raise ValueError("The cleaned target contains values other than 0 and 1")

    print(f"Original rows: {len(raw_data)}")
    print(f"Cleaned rows: {len(saved_data)}")
    print(f"Removed duplicates: {removed_duplicates}")
    print(f"Original missing values: {original_missing}")
    print(f"Missing values remaining: {int(saved_data.isna().sum().sum())}")
    print(f"Columns: {saved_data.columns.tolist()}")
    print("at_risk counts:")
    print(saved_data["at_risk"].value_counts().sort_index().to_string())
    print(f"Saved to: {CLEANED_PATH.resolve()}")


if __name__ == "__main__":
    main()