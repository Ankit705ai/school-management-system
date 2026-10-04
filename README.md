# Student Performance & Academic Risk Prediction System

## Project Objective

Build a practical Data Science project that studies factors associated with student performance and develops an interpretable system for identifying students who may be at academic risk. The project is intended for learning and portfolio demonstration, so every stage will document assumptions, data quality decisions, and limitations.

## Problem Statement

Educational teams often need to identify students who may benefit from support before their performance declines substantially. Given student demographic, attendance, study, assessment, and support information, this project will investigate which variables are associated with academic outcomes and later build a model that estimates academic risk.

The model will support prioritization of follow-up, not replace educator judgment or make high-stakes decisions automatically.

## Planned Data Science Workflow

1. **Data Gathering**: Select a documented dataset, record its source and license, and place the local data file in `data/`.
2. **Data Cleaning**: Inspect data types, missing values, duplicates, invalid values, and inconsistent categories.
3. **Exploratory Data Analysis**: Understand distributions and relationships using summary statistics and visualizations.
4. **Statistics**: Use appropriate descriptive and inferential methods to assess important patterns.
5. **SQL Analysis**: Practice answering analytical questions with SQL after documenting the table schema.
6. **Feature Engineering**: Create meaningful predictors without leaking future information into the model.
7. **Machine Learning**: Train baseline and improved classification models for academic risk.
8. **Evaluation**: Compare suitable metrics, inspect errors, and check whether results are useful and responsible.
9. **Dashboard**: Present validated findings and model outputs in a clear decision-support interface.

## Expected Dataset Columns

The exact schema will depend on the selected dataset. Candidate columns may include:

- `student_id`: anonymized student identifier
- `age`: student age
- `gender`: recorded gender category, if available
- `school` or `program`: school or academic program
- `study_time`: regular study-time measure
- `attendance_rate`: attendance percentage or count
- `previous_grade`: prior academic result
- `midterm_score` and `final_score`: assessment results, if available
- `assignments_completed`: assignment completion measure
- `parental_education`: parental education category, if available
- `internet_access`: access indicator, if available
- `extracurricular_activity`: participation indicator, if available
- `support_received`: academic support indicator, if available
- `academic_risk`: target label to be defined from the dataset and project rules

These are planning assumptions only. They must be checked against the real dataset before writing cleaning or modeling code.

## Future ML Objective

After the data has been understood and cleaned, build a baseline model to classify whether a student is at academic risk. Define the target carefully, avoid target leakage, establish a simple baseline, and evaluate with metrics that reflect the cost of missed at-risk students as well as false alerts. Model interpretation and fairness checks will be part of the evaluation.

## Current Stage

Only the initial project structure has been created. No dataset has been downloaded, no results have been invented, and no machine-learning model has been implemented yet.

Start with `notebooks/01_eda.ipynb` after selecting and documenting a real dataset.
