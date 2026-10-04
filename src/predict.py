"""Predict academic risk for one student using the saved Random Forest model."""

import argparse
import pickle
from pathlib import Path

import pandas as pd


MODEL_PATH = Path(__file__).parents[1] / "data" / "random_forest_model.pkl"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether one student is at risk.")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--gender", choices=["Female", "Male", "Non-binary"], required=True)
    parser.add_argument("--program", choices=["STEM", "Humanities", "Commerce", "Arts"], required=True)
    parser.add_argument("--study-hours-per-week", type=float, required=True)
    parser.add_argument("--attendance-rate", type=float, required=True)
    parser.add_argument("--previous-exam-score", type=float, required=True)
    parser.add_argument("--assignment-score", type=float, required=True)
    parser.add_argument("--parental-education", choices=["High School", "Bachelor", "Master"], required=True)
    parser.add_argument("--internet-access", choices=["Yes", "No"], required=True)
    parser.add_argument("--extracurricular-activity", choices=["Yes", "No"], required=True)
    parser.add_argument("--academic-support", choices=["Yes", "No"], required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    with MODEL_PATH.open("rb") as model_file:
        saved_model = pickle.load(model_file)

    student = pd.DataFrame(
        [
            {
                "age": arguments.age,
                "gender": arguments.gender,
                "program": arguments.program,
                "study_hours_per_week": arguments.study_hours_per_week,
                "attendance_rate": arguments.attendance_rate,
                "previous_exam_score": arguments.previous_exam_score,
                "assignment_score": arguments.assignment_score,
                "parental_education": arguments.parental_education,
                "internet_access": arguments.internet_access,
                "extracurricular_activity": arguments.extracurricular_activity,
                "academic_support": arguments.academic_support,
            }
        ]
    )
    prepared_student = pd.get_dummies(student, dtype=int).reindex(
        columns=saved_model["feature_columns"],
        fill_value=0,
    )

    prediction = int(saved_model["model"].predict(prepared_student)[0])
    print(f"Predicted at_risk: {prediction}")
    print("1 means at risk; 0 means not at risk.")


if __name__ == "__main__":
    main()