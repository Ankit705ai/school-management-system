"""Generate a reproducible synthetic student-performance dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
STUDENT_COUNT = 5_000


def generate_dataset() -> pd.DataFrame:
    """Create student records with related academic variables and a risk label."""
    rng = np.random.default_rng(SEED)

    # Generate background characteristics first so later variables can depend on them.
    ages = rng.integers(15, 23, size=STUDENT_COUNT)
    genders = rng.choice(["Female", "Male", "Non-binary"], size=STUDENT_COUNT, p=[0.48, 0.48, 0.04])
    programs = rng.choice(
        ["STEM", "Humanities", "Commerce", "Arts"],
        size=STUDENT_COUNT,
        p=[0.35, 0.25, 0.25, 0.15],
    )
    parental_education = rng.choice(
        ["None", "High School", "Bachelor", "Master"],
        size=STUDENT_COUNT,
        p=[0.08, 0.42, 0.35, 0.15],
    )
    internet_access = rng.choice(["Yes", "No"], size=STUDENT_COUNT, p=[0.88, 0.12])
    extracurricular_activity = rng.choice(["Yes", "No"], size=STUDENT_COUNT, p=[0.42, 0.58])
    academic_support = rng.choice(["Yes", "No"], size=STUDENT_COUNT, p=[0.28, 0.72])

    # Study time is influenced slightly by access and extracurricular commitments.
    study_hours = rng.normal(12, 4.5, size=STUDENT_COUNT)
    study_hours += np.where(internet_access == "Yes", 1.0, -1.0)
    study_hours += np.where(extracurricular_activity == "Yes", -1.2, 0.0)
    study_hours = np.clip(study_hours, 0, 40).round(1)

    # Attendance and prior results share some underlying student engagement.
    attendance_rate = 82 + (study_hours - 12) * 0.45 + rng.normal(0, 8, STUDENT_COUNT)
    attendance_rate += np.where(academic_support == "Yes", 2.0, 0.0)
    attendance_rate = np.clip(attendance_rate, 45, 100).round(1)

    previous_exam_score = (
        48
        + attendance_rate * 0.22
        + study_hours * 0.75
        + rng.normal(0, 10, STUDENT_COUNT)
    )
    previous_exam_score = np.clip(previous_exam_score, 0, 100).round(1)

    assignment_score = (
        25
        + previous_exam_score * 0.45
        + study_hours * 1.1
        + rng.normal(0, 8, STUDENT_COUNT)
    )
    assignment_score = np.clip(assignment_score, 0, 100).round(1)

    # The final score combines ongoing performance with noise, rather than being random.
    final_exam_score = (
        0.30 * previous_exam_score
        + 0.30 * assignment_score
        + 0.20 * attendance_rate
        + 0.20 * (study_hours / 40 * 100)
        + rng.normal(0, 7, STUDENT_COUNT)
    )
    final_exam_score = np.clip(final_exam_score, 0, 100).round(1)

    # Create the label from pre-final-exam signals, before missing values are added.
    risk_signals = (
        (attendance_rate < 70).astype(int)
        + (study_hours < 8).astype(int)
        + (previous_exam_score < 55).astype(int)
        + (assignment_score < 60).astype(int)
        + (internet_access == "No").astype(int)
    )
    at_risk = (risk_signals >= 2).astype(int)

    dataset = pd.DataFrame(
        {
            "student_id": [f"STU{number:05d}" for number in range(1, STUDENT_COUNT + 1)],
            "age": ages,
            "gender": genders,
            "program": programs,
            "study_hours_per_week": study_hours,
            "attendance_rate": attendance_rate,
            "previous_exam_score": previous_exam_score,
            "assignment_score": assignment_score,
            "final_exam_score": final_exam_score,
            "parental_education": parental_education,
            "internet_access": internet_access,
            "extracurricular_activity": extracurricular_activity,
            "academic_support": academic_support,
            "at_risk": at_risk,
        }
    )

    # Missingness is added only to input columns, leaving the identifier and target usable.
    missingness = {
        "study_hours_per_week": 0.015,
        "attendance_rate": 0.012,
        "previous_exam_score": 0.018,
        "assignment_score": 0.015,
        "parental_education": 0.01,
        "internet_access": 0.008,
    }
    for column, fraction in missingness.items():
        missing_indices = rng.choice(STUDENT_COUNT, size=int(STUDENT_COUNT * fraction), replace=False)
        dataset.loc[missing_indices, column] = np.nan

    # Replace ten unique rows with copies of the first ten, keeping 5,000 total rows.
    duplicate_rows = dataset.iloc[:10].copy()
    dataset = pd.concat([dataset.iloc[:-10], duplicate_rows], ignore_index=True)
    return dataset


def main() -> None:
    dataset = generate_dataset()
    output_path = Path(__file__).with_name("students.csv")
    dataset.to_csv(output_path, index=False)
    print(f"Generated {len(dataset):,} rows at {output_path}")


if __name__ == "__main__":
    main()