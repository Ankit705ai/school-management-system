# Data Directory

This project uses a reproducible synthetic dataset for learning Data Science workflows. It is generated locally by `generate_dataset.py`; no data is downloaded automatically.

## Dataset Generation

Run the following command from the project root:

```text
python data/generate_dataset.py
```

This creates `data/students.csv` with 5,000 rows. The generator uses a fixed random seed, includes relationships between academic variables, intentionally inserts missing values, and adds a small number of duplicate rows for later data-cleaning practice.

## Data Dictionary

| Column | Type | Expected values or range | Why it is useful |
| --- | --- | --- | --- |
| `student_id` | string | `STU00001`-style identifier | Identifies records and helps detect duplicate rows. It should not be used as a predictive feature. |
| `age` | integer | 15-22 years | Provides basic student context and may capture differences in academic stage. |
| `gender` | category | `Female`, `Male`, `Non-binary` | Allows subgroup analysis and fairness checks. It should be handled carefully and not assumed to cause performance. |
| `program` | category | `STEM`, `Humanities`, `Commerce`, `Arts` | Represents the academic program, whose workload and assessment patterns may differ. |
| `study_hours_per_week` | float | 0-40 hours | Measures study effort and may be associated with assessment performance. |
| `attendance_rate` | float | 0-100 percent | Captures participation and is often an early signal of disengagement. |
| `previous_exam_score` | float | 0-100 points | Represents prior achievement and provides a strong baseline for later performance. |
| `assignment_score` | float | 0-100 points | Captures ongoing coursework performance rather than relying only on one exam. |
| `final_exam_score` | float | 0-100 points | Measures the final academic outcome used to define the synthetic risk label. It may be excluded from future prediction features to avoid target leakage. |
| `parental_education` | category | `None`, `High School`, `Bachelor`, `Master` | Provides socioeconomic and educational-context information for analysis. |
| `internet_access` | category | `Yes`, `No` | Represents access to an important study resource. |
| `extracurricular_activity` | category | `Yes`, `No` | Captures time commitments and engagement outside coursework. |
| `academic_support` | category | `Yes`, `No` | Indicates whether the student receives additional academic help. |
| `at_risk` | integer target | `0` or `1` | Synthetic outcome indicating whether the student meets the academic-risk rule below. |

## Target Definition

`at_risk` is determined from complete pre-final-exam values, before missing values are inserted:

```text
risk_signals = count of:
	attendance_rate < 70
	study_hours_per_week < 8
	previous_exam_score < 55
	assignment_score < 60
	internet_access == No

at_risk = 1 when risk_signals >= 2
at_risk = 0 otherwise
```

This is a teaching label, not a validated educational policy. `final_exam_score` is retained for later analysis, but it does not determine `at_risk` and should not be treated as part of the target-definition logic.

## Data Quality Practice

Missing values are intentionally inserted into several input columns. Ten complete records are also duplicated, so the generated file contains 5,000 rows but fewer than 5,000 unique student records. The target column and identifier are kept complete so that the quality issues can be investigated without making the dataset unusable.
