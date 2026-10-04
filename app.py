"""School-friendly dashboard for single and batch student-risk predictions."""

import base64
import json
import io
import logging
import os
import pickle
import random
import re
import sqlite3
import unicodedata
import uuid
from datetime import date
from functools import wraps
from pathlib import Path

import pandas as pd
from flask import Flask, Response, redirect, render_template, render_template_string, request, send_file, session, url_for
from PIL import Image
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "data" / "random_forest_model.pkl"
DATA_PATH = ROOT / "data" / "students_cleaned.csv"
with MODEL_PATH.open("rb") as model_file:
    saved_model = pickle.load(model_file)
students = pd.read_csv(DATA_PATH)
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "student-risk-local-development-key")
logger = logging.getLogger(__name__)
latest_batch_csv = None
GAME_DB_PATH = ROOT / "data" / "student_game.db"
GAME_QUESTIONS = [
    {"prompt": "2, 4, 8, 16, ?", "options": ["24", "32", "30", "20"], "answer": "32"},
    {"prompt": "Which number comes next? 3, 6, 9, 12, ?", "options": ["14", "15", "16", "18"], "answer": "15"},
    {"prompt": "Which item is different?", "options": ["Triangle", "Square", "Circle", "Cube"], "answer": "Cube"},
    {"prompt": "Remember this sequence: 7 - 2 - 9. What was the middle number?", "options": ["2", "7", "9", "5"], "answer": "2"},
    {"prompt": "If all birds have wings and a robin is a bird, what must be true?", "options": ["A robin has wings", "A robin can swim", "All wings are birds", "Nothing"], "answer": "A robin has wings"},
    {"prompt": "Which number is the odd one out?", "options": ["10", "20", "30", "35"], "answer": "35"},
    {"prompt": "5, 10, 20, 40, ?", "options": ["45", "60", "80", "100"], "answer": "80"},
    {"prompt": "Which word does not belong?", "options": ["Apple", "Pear", "Carrot", "Banana"], "answer": "Carrot"},
    {"prompt": "If today is Monday, what day is it in two days?", "options": ["Tuesday", "Wednesday", "Thursday", "Sunday"], "answer": "Wednesday"},
    {"prompt": "Which shape has no corners?", "options": ["Circle", "Square", "Triangle", "Rectangle"], "answer": "Circle"},
]

INPUT_COLUMNS = ["age", "gender", "program", "study_hours_per_week", "attendance_rate", "previous_exam_score", "assignment_score", "parental_education", "internet_access", "extracurricular_activity", "academic_support"]
SCHOOL_CLASSES = ["PG", "LKG", "UKG"] + [f"Class {number}" for number in range(1, 13)]
FEE_MONTHS = ["April", "May", "June", "July", "August", "September", "October", "November", "December", "January", "February", "March"]
FEE_STATUSES = ["Pending", "Paid", "Free"]
MAX_RECEIPT_BYTES = 5 * 1024 * 1024
FEE_RECEIPT_STAGING = ROOT / "data" / "fee_receipt_staging"
MAX_STUDENT_PHOTO_BYTES = 5 * 1024 * 1024
STUDENT_PHOTO_STORAGE = ROOT / "data" / "student_photo_storage"
MAX_CALENDAR_PHOTO_BYTES = 5 * 1024 * 1024
CALENDAR_PHOTO_STORAGE = ROOT / "data" / "calendar_photo_storage"
MAX_MEMORY_PHOTO_BYTES = 5 * 1024 * 1024
MEMORY_PHOTO_STORAGE = ROOT / "data" / "school_memory_storage"
MEMORY_PHOTO_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_REPORT_FILE_BYTES = 10 * 1024 * 1024
REPORT_FILE_STORAGE = ROOT / "data" / "report_file_storage"
REPORT_ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx", "jpg", "jpeg", "png"}
CATEGORICAL_COLUMNS = {
    "gender": ["Female", "Male", "Non-binary"],
    "program": ["STEM", "Business", "Humanities", "Commerce", "Arts"],
    "parental_education": ["High School", "Bachelor", "Master"],
    "internet_access": ["Yes", "No"],
    "extracurricular_activity": ["Yes", "No"],
    "academic_support": ["Yes", "No"],
}
NUMERIC_LIMITS = {"age": (15, 22), "study_hours_per_week": (0, 40), "attendance_rate": (0, 100), "previous_exam_score": (0, 100), "assignment_score": (0, 100)}
MODEL_METRICS = {"Logistic Regression": ["0.936", "0.760", "0.724", "0.742"], "Random Forest": ["0.988", "1.000", "0.906", "0.950"]}


def prepare_input(frame):
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    values = frame[INPUT_COLUMNS].copy()
    for column, (low, high) in NUMERIC_LIMITS.items():
        values[column] = pd.to_numeric(values[column], errors="coerce")
        if values[column].isna().any() or not values[column].between(low, high).all():
            raise ValueError(f"{column} must contain values from {low} to {high}")
    for column, options in CATEGORICAL_COLUMNS.items():
        values[column] = values[column].astype("string").str.strip()
        if values[column].isna().any() or not values[column].isin(options).all():
            raise ValueError(f"{column} contains an unsupported value")
    return pd.get_dummies(values, dtype=int).reindex(columns=saved_model["feature_columns"], fill_value=0)


def make_prediction(frame):
    encoded = prepare_input(frame)
    model = saved_model["model"]
    labels = model.predict(encoded).astype(int)
    probabilities = model.predict_proba(encoded)[:, 1] if hasattr(model, "predict_proba") else None
    return labels, probabilities


def suggestions(values, prediction):
    result = []
    if float(values["attendance_rate"]) < 70: result.append("Improve attendance and discuss barriers to regular participation.")
    if float(values["study_hours_per_week"]) < 8: result.append("Build a consistent weekly study routine.")
    if float(values["previous_exam_score"]) < 55: result.append("Use targeted revision on topics missed in the previous exam.")
    if float(values["assignment_score"]) < 60: result.append("Review assignment feedback and improve completion habits.")
    if values["academic_support"] == "No": result.append("Consider tutoring, teacher check-ins, or another academic support option.")
    if values["internet_access"] == "No": result.append("Look for school or offline learning resources.")
    if prediction == 1 and not result: result.append("Review these indicators with an educator and agree on a support plan.")
    return result[:4] or ["Continue monitoring progress and maintaining current study habits."]


def single_explanation(values):
    factors = []
    if float(values["attendance_rate"]) < 70: factors.append("low attendance")
    if float(values["study_hours_per_week"]) < 8: factors.append("low study time")
    if float(values["previous_exam_score"]) < 55: factors.append("a low previous score")
    if float(values["assignment_score"]) < 60: factors.append("a low assignment score")
    return "No warning threshold was triggered by the entered indicators." if not factors else "The result is supported by " + ", ".join(factors) + "."


def dashboard_insights():
    by_program = students.groupby("program")["at_risk"].mean().sort_values(ascending=False)
    by_support = students.groupby("academic_support")["at_risk"].mean()
    columns = ["attendance_rate", "study_hours_per_week", "previous_exam_score", "assignment_score", "final_exam_score"]
    correlations = students[columns].corr()["final_exam_score"].drop("final_exam_score")
    strongest = correlations.abs().idxmax()
    return [f"{by_program.index[0]} has the highest observed risk rate ({by_program.iloc[0] * 100:.1f}%).", f"Observed risk was {by_support.get('Yes', 0) * 100:.1f}% with academic support and {by_support.get('No', 0) * 100:.1f}% without it.", f"{strongest.replace('_', ' ').title()} had the strongest association with final exam score (r = {correlations[strongest]:.2f})."]


def game_connection():
    connection = sqlite3.connect(GAME_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_game_database():
    with game_connection() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS game_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                must_change_password INTEGER NOT NULL DEFAULT 0,
                managed_student_id INTEGER UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS game_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                score INTEGER NOT NULL,
                total_questions INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                level TEXT NOT NULL,
                played_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES game_users (id)
            );
            CREATE TABLE IF NOT EXISTS schools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS school_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                upload_id TEXT NOT NULL,
                row_number INTEGER NOT NULL,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (school_id) REFERENCES schools (id)
            );
            CREATE TABLE IF NOT EXISTS managed_students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                roll_number TEXT NOT NULL,
                uid_number TEXT NOT NULL,
                student_name TEXT NOT NULL,
                father_name TEXT NOT NULL,
                mother_name TEXT NOT NULL,
                parent_mobile TEXT NOT NULL,
                second_mobile TEXT,
                address TEXT NOT NULL,
                transport_student INTEGER NOT NULL DEFAULT 0,
                active_student INTEGER NOT NULL DEFAULT 1,
                siblings_json TEXT NOT NULL DEFAULT '[]',
                admission_date TEXT,
                photo_path TEXT,
                section_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (school_id, uid_number),
                UNIQUE (school_id, class_name, roll_number),
                FOREIGN KEY (school_id) REFERENCES schools (id)
            );
            CREATE TABLE IF NOT EXISTS student_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                section_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (school_id, class_name, section_name),
                FOREIGN KEY (school_id) REFERENCES schools (id)
            );
            CREATE TABLE IF NOT EXISTS admission_uid_sequence (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_uid INTEGER NOT NULL,
                uid_width INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS student_attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                student_id INTEGER NOT NULL,
                attendance_date TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('Present', 'Absent')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (school_id, class_name, student_id, attendance_date),
                FOREIGN KEY (school_id) REFERENCES schools (id),
                FOREIGN KEY (student_id) REFERENCES managed_students (id)
            );
            CREATE TABLE IF NOT EXISTS class_notices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                notice_date TEXT NOT NULL,
                event_date TEXT,
                priority TEXT NOT NULL CHECK (priority IN ('Normal', 'Important')) DEFAULT 'Normal',
                status TEXT NOT NULL CHECK (status IN ('Published', 'Archived')) DEFAULT 'Published',
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (school_id) REFERENCES schools (id)
            );
            CREATE TABLE IF NOT EXISTS student_fee_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                student_id INTEGER NOT NULL,
                fee_month TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('Paid', 'Pending', 'Free')),
                locked INTEGER NOT NULL DEFAULT 0,
                receipt_filename TEXT,
                receipt_data BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (school_id, class_name, student_id, fee_month),
                FOREIGN KEY (school_id) REFERENCES schools (id),
                FOREIGN KEY (student_id) REFERENCES managed_students (id)
            );
            CREATE TABLE IF NOT EXISTS school_academic_calendar (
                school_id INTEGER PRIMARY KEY,
                photo1_filename TEXT,
                photo2_filename TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (school_id) REFERENCES schools (id)
            );
            CREATE TABLE IF NOT EXISTS school_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                title TEXT,
                description TEXT,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_extension TEXT NOT NULL,
                uploaded_by TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY (school_id) REFERENCES schools (id)
            );
            CREATE TABLE IF NOT EXISTS examination_subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                subject_name TEXT NOT NULL,
                subject_code TEXT,
                max_marks REAL NOT NULL DEFAULT 100,
                pass_marks REAL NOT NULL DEFAULT 33,
                subject_order INTEGER NOT NULL DEFAULT 1,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(school_id, class_name, subject_name),
                FOREIGN KEY (school_id) REFERENCES schools (id)
            );
            CREATE TABLE IF NOT EXISTS school_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                report_name TEXT NOT NULL,
                description TEXT,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_extension TEXT NOT NULL,
                uploaded_by TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY (school_id) REFERENCES schools (id)
            );
            CREATE TABLE IF NOT EXISTS examinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                exam_name TEXT NOT NULL,
                academic_session TEXT NOT NULL,
                class_name TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                status TEXT NOT NULL CHECK (status IN ('Draft', 'Published', 'Completed')) DEFAULT 'Draft',
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(school_id, class_name, exam_name, academic_session),
                FOREIGN KEY (school_id) REFERENCES schools (id)
            );
            CREATE TABLE IF NOT EXISTS student_exam_marks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school_id INTEGER NOT NULL,
                exam_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                subject_id INTEGER NOT NULL,
                obtained_marks REAL,
                is_absent INTEGER NOT NULL DEFAULT 0,
                remarks TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(school_id, exam_id, student_id, subject_id),
                FOREIGN KEY (school_id) REFERENCES schools (id),
                FOREIGN KEY (exam_id) REFERENCES examinations (id),
                FOREIGN KEY (student_id) REFERENCES managed_students (id),
                FOREIGN KEY (subject_id) REFERENCES examination_subjects (id)
            );
        """)
        existing_uids = connection.execute("SELECT uid_number FROM managed_students").fetchall()
        numeric_uids = [
            match
            for row in existing_uids
            for match in [re.fullmatch(r"UID([0-9]+)", str(row["uid_number"] or ""), re.IGNORECASE)]
            if match
        ]
        highest_existing = max((int(match.group(1)) for match in numeric_uids), default=0)
        existing_width = max((len(match.group(1)) for match in numeric_uids), default=4)
        sequence = connection.execute(
            "SELECT last_uid, uid_width FROM admission_uid_sequence WHERE id = 1"
        ).fetchone()
        if sequence is None:
            connection.execute(
                "INSERT INTO admission_uid_sequence (id, last_uid, uid_width) VALUES (1, ?, ?)",
                (highest_existing, max(4, existing_width)),
            )
        else:
            connection.execute(
                "UPDATE admission_uid_sequence SET last_uid = MAX(last_uid, ?), "
                "uid_width = MAX(uid_width, ?) WHERE id = 1",
                (highest_existing, existing_width),
            )
        connection.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_duplicate_student_uid
            BEFORE INSERT ON managed_students
            WHEN EXISTS (SELECT 1 FROM managed_students WHERE uid_number = NEW.uid_number)
            BEGIN
                SELECT RAISE(ABORT, 'duplicate student UID');
            END
        """)
        connection.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_duplicate_student_uid_update
            BEFORE UPDATE OF uid_number ON managed_students
            WHEN EXISTS (
                SELECT 1 FROM managed_students
                WHERE uid_number = NEW.uid_number AND id <> OLD.id
            )
            BEGIN
                SELECT RAISE(ABORT, 'duplicate student UID');
            END
        """)
        duplicate_uid_values = [
            row["uid_number"]
            for row in connection.execute(
                "SELECT uid_number FROM managed_students GROUP BY uid_number HAVING COUNT(*) > 1"
            ).fetchall()
        ]
        excluded_values = ", ".join(
            "'" + value.replace("'", "''") + "'" for value in duplicate_uid_values
        ) or "NULL"
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_managed_students_uid_global "
            f"ON managed_students(uid_number) WHERE uid_number NOT IN ({excluded_values})"
        )
        user_columns = {row[1] for row in connection.execute("PRAGMA table_info(game_users)")}
        if "school_id" not in user_columns:
            connection.execute("ALTER TABLE game_users ADD COLUMN school_id INTEGER REFERENCES schools(id)")
        if "must_change_password" not in user_columns:
            connection.execute("ALTER TABLE game_users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0")
        if "managed_student_id" not in user_columns:
            connection.execute("ALTER TABLE game_users ADD COLUMN managed_student_id INTEGER REFERENCES managed_students(id)")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_game_users_managed_student ON game_users(managed_student_id) WHERE managed_student_id IS NOT NULL")
        student_columns = {row[1] for row in connection.execute("PRAGMA table_info(managed_students)")}
        if "active_student" not in student_columns:
            connection.execute("ALTER TABLE managed_students ADD COLUMN active_student INTEGER NOT NULL DEFAULT 1")
        if "admission_date" not in student_columns:
            connection.execute("ALTER TABLE managed_students ADD COLUMN admission_date TEXT")
        if "photo_path" not in student_columns:
            connection.execute("ALTER TABLE managed_students ADD COLUMN photo_path TEXT")
        if "section_id" not in student_columns:
            connection.execute("ALTER TABLE managed_students ADD COLUMN section_id INTEGER REFERENCES student_sections(id)")
        ensure_fee_schema(connection)
        normalize_managed_roll_numbers(connection)


def ensure_fee_schema(connection):
    table = connection.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'student_fee_status'").fetchone()
    columns = {row[1] for row in connection.execute("PRAGMA table_info(student_fee_status)")}
    if table and {"locked", "receipt_filename", "receipt_data"}.issubset(columns) and "'Free'" in (table[0] or ""):
        return
    connection.execute("ALTER TABLE student_fee_status RENAME TO student_fee_status_legacy")
    connection.execute("""
        CREATE TABLE student_fee_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            school_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            student_id INTEGER NOT NULL,
            fee_month TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('Paid', 'Pending', 'Free')),
            locked INTEGER NOT NULL DEFAULT 0,
            receipt_filename TEXT,
            receipt_data BLOB,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (school_id, class_name, student_id, fee_month),
            FOREIGN KEY (school_id) REFERENCES schools (id),
            FOREIGN KEY (student_id) REFERENCES managed_students (id)
        )
    """)
    connection.execute("INSERT INTO student_fee_status (id, school_id, class_name, student_id, fee_month, status, created_at, updated_at) SELECT id, school_id, class_name, student_id, fee_month, status, created_at, updated_at FROM student_fee_status_legacy")
    connection.execute("DROP TABLE student_fee_status_legacy")


def normalize_managed_roll_numbers(connection):
    rows = connection.execute("SELECT id, school_id, class_name, section_id, student_name FROM managed_students ORDER BY school_id, class_name, section_id, id").fetchall()
    grouped = {}
    for row in rows:
        grouped.setdefault((row["school_id"], row["class_name"], row["section_id"]), []).append(row)
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=lambda row: (str(row["student_name"] or "").casefold(), row["id"]))
        for index, row in enumerate(ordered, start=1):
            connection.execute("UPDATE managed_students SET roll_number = ?, updated_at = datetime('now') WHERE id = ?", (str(index), row["id"]))


def refresh_class_roll_numbers(connection, school_id, class_name, section_id=None):
    if section_id is None:
        rows = connection.execute(
            "SELECT id, student_name FROM managed_students WHERE school_id = ? AND class_name = ? AND section_id IS NULL ORDER BY LOWER(student_name), student_name, id",
            (school_id, class_name),
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT id, student_name FROM managed_students WHERE school_id = ? AND class_name = ? AND section_id = ? ORDER BY LOWER(student_name), student_name, id",
            (school_id, class_name, section_id),
        ).fetchall()
    for index, row in enumerate(rows, start=1):
        connection.execute("UPDATE managed_students SET roll_number = ?, updated_at = datetime('now') WHERE id = ?", (str(index), row["id"]))


def school_required(view):
    @wraps(view)
    def guarded(*args, **kwargs):
        if not session.get("school_id"):
            return redirect(url_for("school_login"))
        return view(*args, **kwargs)
    return guarded


def current_school():
    school_id = session.get("school_id")
    if not school_id:
        return None
    with game_connection() as connection:
        return connection.execute("SELECT * FROM schools WHERE id = ?", (school_id,)).fetchone()


def school_records(school_id):
    with game_connection() as connection:
        rows = connection.execute("SELECT record_json FROM school_predictions WHERE school_id = ? ORDER BY id", (school_id,)).fetchall()
        return pd.DataFrame([json.loads(row[0]) for row in rows if row[0] is not None])


def school_attendance_rate(school_id):
    with game_connection() as connection:
        attendance = connection.execute("SELECT COUNT(*) AS total, COALESCE(SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END), 0) AS present FROM student_attendance WHERE school_id = ?", (school_id,)).fetchone()
    return round(attendance["present"] / attendance["total"] * 100, 1) if attendance["total"] else 0


def current_user():
    user_id = session.get("game_user_id")
    if not user_id:
        return None
    with game_connection() as connection:
        user = connection.execute(
            "SELECT game_users.*, managed_students.school_id AS managed_school_id, managed_students.active_student "
            "FROM game_users LEFT JOIN managed_students ON managed_students.id = game_users.managed_student_id "
            "WHERE game_users.id = ? AND (game_users.managed_student_id IS NULL OR "
            "(game_users.school_id = managed_students.school_id AND managed_students.active_student = 1))",
            (user_id,),
        ).fetchone()
        return user if user else None


def student_required(view):
    @wraps(view)
    def guarded(*args, **kwargs):
        user = current_user()
        if user is None or user["managed_student_id"] is None:
            session["game_error"] = "Student account not found. Please contact the school administration."
            return redirect(url_for("login"))
        if user["must_change_password"]:
            return redirect(url_for("change_password"))
        return view(*args, **kwargs)
    return guarded


def game_level(accuracy):
    if accuracy < 40: return "Beginner"
    if accuracy < 60: return "Developing"
    if accuracy < 75: return "Good"
    if accuracy < 90: return "Strong"
    return "Excellent"


def score_summary(user_id):
    with game_connection() as connection:
        scores = connection.execute("SELECT * FROM game_scores WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()
    if not scores:
        return {"games": 0, "best": "No score yet", "latest": "No score yet", "average": "No score yet", "history": []}
    best = max(scores, key=lambda row: row["score"] / row["total_questions"])
    latest = scores[0]
    average = sum(row["score"] / row["total_questions"] * 100 for row in scores) / len(scores)
    return {"games": len(scores), "best": f"{best['score']}/{best['total_questions']}", "latest": f"{latest['score']}/{latest['total_questions']}", "average": f"{average:.0f}%", "history": [dict(score) for score in scores]}


initialize_game_database()


TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Student Support Dashboard</title><style>
:root{--navy:#173442;--teal:#176b87;--orange:#d9784a;--ink:#203040;--muted:#657783;--paper:#fff;--line:#d8e2e6;--bg:#eef3f4}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Arial,sans-serif}.top{background:var(--navy);color:#fff;padding:22px max(22px,5vw);display:flex;justify-content:space-between;align-items:center}.brand{font-size:21px;font-weight:bold}.brand small{display:block;color:#a9c7cb;font-size:11px;font-weight:normal}.top a{color:#d6e8ea}.main{max-width:1180px;margin:auto;padding:34px 22px 60px}.hero h1{font-size:34px;margin:5px 0}.hero p,.muted{color:var(--muted)}.choice{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin:26px 0 34px}.choice-card{background:var(--paper);border:1px solid var(--line);border-radius:9px;padding:30px;text-decoration:none;box-shadow:0 12px 28px #20304012}.choice-card .icon{font-size:31px}.choice-card h2{margin:14px 0 3px;font-size:22px}.choice-card p{color:var(--muted);margin:0}.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}.stat,.panel{background:var(--paper);border:1px solid var(--line);border-radius:7px}.stat{padding:15px}.stat span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase}.stat strong{font-size:24px}.section-head{display:flex;justify-content:space-between;align-items:center;margin:30px 0 14px}.section-head h2{margin:0}.back{color:var(--teal);font-weight:bold;text-decoration:none}.charts{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}.panel{padding:17px}.panel img{display:block;width:100%}.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.insight{background:#fff;border-left:4px solid var(--teal);padding:15px}.model-table,.results{width:100%;border-collapse:collapse}.model-table th,.model-table td,.results th,.results td{text-align:left;padding:10px;border-bottom:1px solid var(--line)}.selected{background:#e6f3ee}.form{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.form label{display:grid;gap:6px;font-weight:bold;font-size:12px}.form input,.form select,.upload input{font:inherit;padding:10px;border:1px solid #bdcbd1;border-radius:4px;background:#fff}.button{display:inline-block;background:var(--teal);color:#fff;border:0;border-radius:4px;padding:11px 16px;font-weight:bold;cursor:pointer;text-decoration:none}.form .button{grid-column:1}.result{margin-top:20px;padding:18px;border-radius:6px;background:#e5f4eb;color:#176b3a}.result.risk{background:#fff0e9;color:#a4472b}.result h2{margin:0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.metric{background:#fff;padding:14px;border-radius:5px}.metric b{display:block;font-size:22px}.suggestions{margin-bottom:0}.upload{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.error{background:#fde8e7;color:#a32924;padding:12px;border-radius:5px;margin-top:14px}.filter{display:flex;gap:8px;margin:16px 0;flex-wrap:wrap}.filter a:not(.button){padding:7px 13px;border:1px solid var(--line);border-radius:20px}.filter a.active{background:var(--teal);color:#fff}.table-wrap{overflow:auto}@media(max-width:900px){.stats{grid-template-columns:repeat(3,1fr)}.charts{grid-template-columns:1fr}.insights{grid-template-columns:1fr}}@media(max-width:620px){.choice,.stats,.form,.metrics{grid-template-columns:1fr}.top{display:block}.top a{display:block;margin-top:8px}}
</style></head><body><header class="top"><div class="brand">Student Support Dashboard<small>Performance insights for practical support</small></div>{% if view != 'home' %}<a href="/?view=home">Back to Dashboard</a>{% endif %}</header><main class="main">
{% if view == 'home' %}<div class="hero"><div class="muted">Student Performance &amp; Academic Risk Prediction System</div><h1>Make the next support conversation more informed.</h1><p>Explore the class picture or check one student using existing academic indicators.</p></div><div class="choice"><a class="choice-card" href="/?view=single"><div class="icon">👤</div><h2>Single Student</h2><p>Check one student's performance and risk</p></a><a class="choice-card" href="/?view=batch"><div class="icon">📄</div><h2>Upload CSV</h2><p>Predict risk for multiple students</p></a></div><div class="section-head"><h2>Class overview</h2><span class="muted">{{ total_students }} cleaned records</span></div><div class="stats"><div class="stat"><span>Total Students</span><strong>{{ total_students }}</strong></div><div class="stat"><span>Students At Risk</span><strong>{{ at_risk_students }}</strong></div><div class="stat"><span>Students Not At Risk</span><strong>{{ low_risk_students }}</strong></div><div class="stat"><span>Risk Rate</span><strong>{{ risk_rate }}%</strong></div><div class="stat"><span>Average Attendance</span><strong>{{ avg_attendance }}%</strong></div><div class="stat"><span>Average Study Hours</span><strong>{{ avg_study }}</strong></div></div><div class="section-head"><h2>Simple class signals</h2></div><div class="charts">{% for chart in charts %}<div class="panel"><img src="{{ chart }}" alt="Class overview chart"></div>{% endfor %}</div><div class="section-head"><h2>Key insights</h2><span class="muted">Observed associations, not causes</span></div><div class="insights">{% for insight in insights %}<div class="insight">{{ insight }}</div>{% endfor %}</div><div class="section-head"><h2>Model comparison</h2></div><div class="panel"><table class="model-table"><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th></tr>{% for name, metrics in model_metrics.items() %}<tr class="{% if name == 'Random Forest' %}selected{% endif %}"><td>{{ name }}{% if name == 'Random Forest' %} <b>(Selected)</b>{% endif %}</td>{% for metric in metrics %}<td>{{ metric }}</td>{% endfor %}</tr>{% endfor %}</table></div>
{% elif view == 'single' %}<div class="section-head student-history-heading"><div><span class="eyebrow">Student support</span><h2>Student History</h2></div><a class="back" href="/?view=home">Back to Dashboard</a></div><section class="panel student-lookup"><h3>Find a student</h3><p class="muted">Search existing students in your school to view their profile, class record, and fee history.</p><form method="get" class="student-search-form"><input type="hidden" name="view" value="single"><input name="student_query" value="{{ student_query }}" placeholder="Enter UID, student name, or class" aria-label="Student search"><button class="button" type="submit">Search</button></form></section>{% if student_query and not student_profile and not student_matches %}<div class="panel student-empty"><h3>No student found</h3><p class="muted">Try a UID number, student name, or class from your school records.</p></div>{% endif %}{% if student_matches and not student_profile %}<section class="panel student-matches"><h3>Matching students</h3><div class="student-match-list">{% for student in student_matches %}<a class="student-match" href="/?view=single&amp;student_query={{ student_query|urlencode }}&amp;student_id={{ student.id }}"><span><b>{{ student.student_name }}</b><small>{{ student.class_name }} · Roll No. {{ student.roll_number }}</small></span><span>{{ student.uid_number }}</span></a>{% endfor %}</div></section>{% endif %}{% if student_profile %}<section class="profile-hero"><div><span class="eyebrow">Student profile</span><h1>{{ student_profile.student_name }}</h1><p>{{ student_profile.class_name }} · Roll No. {{ student_profile.roll_number }} · UID {{ student_profile.uid_number }}</p></div><a class="button secondary" href="/?view=single{% if student_query %}&amp;student_query={{ student_query|urlencode }}{% endif %}">Search again</a></section><div class="profile-grid"><section class="panel profile-section"><h3>Student Information</h3><dl class="profile-details"><div><dt>Student Name</dt><dd>{{ student_profile.student_name }}</dd></div><div><dt>UID Number</dt><dd>{{ student_profile.uid_number }}</dd></div><div><dt>Roll Number</dt><dd>{{ student_profile.roll_number }}</dd></div><div><dt>Current Class</dt><dd>{{ student_profile.class_name }}</dd></div><div><dt>Father Name</dt><dd>{{ student_profile.father_name }}</dd></div><div><dt>Mother Name</dt><dd>{{ student_profile.mother_name }}</dd></div><div><dt>Parent Mobile Number</dt><dd>{{ student_profile.parent_mobile }}</dd></div><div><dt>Second Mobile Number</dt><dd>{{ student_profile.second_mobile or 'Not provided' }}</dd></div><div class="wide"><dt>Complete Address</dt><dd>{{ student_profile.address }}</dd></div></dl></section><section class="panel profile-section"><h3>Academic &amp; Class History</h3><dl class="profile-details"><div><dt>Current Class</dt><dd>{{ student_profile.class_name }}</dd></div><div><dt>Student Record Created</dt><dd>{{ student_profile.created_at }}</dd></div><div><dt>Last Updated</dt><dd>{{ student_profile.updated_at }}</dd></div><div><dt>Transport</dt><dd>{{ 'Yes' if student_profile.transport_student else 'No' }}</dd></div>{% if student_profile.siblings %}<div class="wide"><dt>Siblings at School</dt><dd>{% for sibling in student_profile.siblings %}{{ sibling.name }}{% if sibling.class_name %} ({{ sibling.class_name }}){% endif %}{% if not loop.last %}, {% endif %}{% endfor %}</dd></div>{% endif %}</dl><p class="muted profile-note">Previous class and academic/session history are not available in the current student records.</p></section></div><section class="panel profile-section fee-history"><div class="profile-section-heading"><div><h3>Fee History</h3><p class="muted">Monthly fee records from Fee Management.</p></div><a class="back" href="/management/fees?class={{ student_profile.class_name|urlencode }}">Open Fee Management</a></div><div class="table-wrap"><table class="results"><thead><tr><th>Month</th><th>Fee Status</th><th>Receipt</th><th>Record Updated</th></tr></thead><tbody>{% for fee in student_fees %}<tr><td>{{ fee.fee_month }}</td><td><span class="fee-status-badge {{ fee.status|lower }}">{{ fee.status }}</span></td><td>{% if fee.receipt_preview %}<img class="profile-receipt" src="data:image/png;base64,{{ fee.receipt_preview }}" alt="Receipt for {{ fee.fee_month }}">{% else %}<span class="receipt-empty">No receipt on file</span>{% endif %}</td><td>{{ fee.updated_at or '—' }}</td></tr>{% endfor %}</tbody></table></div></section>{% if student_exam_history %}<section class="panel profile-section exam-history"><div class="profile-section-heading"><div><h3>Examination &amp; Result History</h3><p class="muted">Academic performance records from Examination Management.</p></div><a class="back" href="/management/examinations/results?class={{ student_profile.class_name|urlencode }}">Open Examination Results</a></div><div class="table-wrap"><table class="results"><thead><tr><th>Examination</th><th>Session</th><th>Total Marks</th><th>Percentage</th><th>Grade</th><th>Result</th></tr></thead><tbody>{% for exam in student_exam_history %}<tr><td><b>{{ exam.exam_name }}</b></td><td>{{ exam.academic_session }}</td><td>{{ exam.total_obtained }} / {{ exam.total_max }}</td><td>{{ exam.percentage }}%</td><td><span class="fee-status-badge locked">{{ exam.grade }}</span></td><td><span class="fee-status-badge {% if exam.is_pass %}locked{% else %}pending{% endif %}">{{ 'PASSED' if exam.is_pass else 'FAILED' }}</span></td></tr>{% endfor %}</tbody></table></div></section>{% endif %}{% endif %}</div>
{% else %}<div class="section-head"><h2>CSV Upload &amp; Batch Prediction</h2><a class="back" href="/?view=home">Back to Dashboard</a></div><div class="panel"><p class="muted">Upload a CSV with the 11 input columns. An optional student_id column is preserved in results.</p><p class="muted">Required: {{ input_columns|join(', ') }}</p><form method="post" enctype="multipart/form-data" class="upload"><input type="file" name="file" accept=".csv" required><button class="button" type="submit">Upload and Predict</button></form>{% if error %}<div class="error">{{ error }}</div>{% endif %}</div>{% if batch %}<div class="section-head"><h2>Batch Results</h2></div><div class="stats"><div class="stat"><span>Total Students</span><strong>{{ batch.total }}</strong></div><div class="stat"><span>At Risk</span><strong>{{ batch.at_risk }}</strong></div><div class="stat"><span>Low Risk</span><strong>{{ batch.low_risk }}</strong></div><div class="stat"><span>Risk Rate</span><strong>{{ batch.rate }}%</strong></div></div><div class="filter"><a class="{% if filter == 'all' %}active{% endif %}" href="/?view=batch&filter=all">All</a><a class="{% if filter == 'risk' %}active{% endif %}" href="/?view=batch&filter=risk">At Risk</a><a class="{% if filter == 'low' %}active{% endif %}" href="/?view=batch&filter=low">Low Risk</a><a class="button" href="/download-results">Download Results CSV</a></div><div class="panel table-wrap"><table class="results"><tr>{% for column in batch.columns %}<th>{{ column }}</th>{% endfor %}</tr>{% for row in batch.rows %}<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}</table></div>{% endif %}{% endif %}</main></body></html>
"""

TEMPLATE = TEMPLATE.replace('<div class="section-head"><h2>Simple class signals</h2></div><div class="charts">{% for chart in charts %}<div class="panel"><img src="{{ chart }}" alt="Class overview chart"></div>{% endfor %}</div>', "")
TEMPLATE = TEMPLATE.replace(".charts{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}", "")
TEMPLATE = TEMPLATE.replace(
    "</style>",
    """
<style>
:root{--navy:#0b1118;--teal:#49b6c9;--orange:#e58a5d;--ink:#edf4f7;--muted:#95a6b5;--paper:#151e28;--line:#2b3b4a;--bg:#090e14}
body{background:var(--bg);color:var(--ink)}.top{background:#0d151e;border-bottom:1px solid #263544}.top a{color:#a9dce3}.brand small,.muted,.hero p{color:var(--muted)}.choice-card,.stat,.panel,.insight,.metric,.model-table,.results{background:var(--paper);border-color:var(--line);box-shadow:0 14px 35px #0000002b}.choice-card:hover{border-color:var(--teal);box-shadow:0 18px 40px #00000045}.choice-card h2,.section-head h2{color:var(--ink)}.choice-card p{color:var(--muted)}.stat span{color:var(--muted)}.back{color:var(--teal)}.form input,.form select,.upload input{background:#0f1720;color:var(--ink);border-color:#3b4d5d}.form input:focus,.form select:focus,.upload input:focus{outline:2px solid #49b6c955;border-color:var(--teal)}.button{background:var(--teal);color:#071117}.button:hover{background:#70cfda}.result{background:#123226;color:#b7f0ce}.result.risk{background:#3a211c;color:#ffc8b3}.metric{color:var(--ink)}.model-table th,.model-table td,.results th,.results td{border-color:var(--line)}.selected{background:#17382e}.insight{border-left-color:var(--teal)}.upload{background:#101923;border:1px dashed #526879;border-radius:7px;padding:20px}.upload input::file-selector-button{background:#263746;color:#e9f2f5;border:1px solid #526879;border-radius:4px;padding:9px;margin-right:10px;cursor:pointer}.error{background:#3b211f;color:#ffc2b5}.filter a:not(.button){border-color:var(--line);color:var(--muted)}.filter a.active{background:var(--teal);color:#071117}.selected b{color:#8ee2d0}
</style>
""",
)
TEMPLATE = TEMPLATE.replace("Make the next support conversation more informed.", "Student Risk Prediction")
TEMPLATE = TEMPLATE.replace('<div class="brand">Student Support Dashboard<small>Performance insights for practical support</small></div>{% if view != \'home\' %}', '<div class="brand">Student Support Dashboard<small>Performance insights for practical support</small></div><span>{{ school_name }} &nbsp; <a href="/school-logout">School Logout</a></span>{% if view != \'home\' %}')
TEMPLATE = TEMPLATE.replace("Explore the class picture or check one student using existing academic indicators.", "Identify students who may need additional academic support.")
TEMPLATE = TEMPLATE.replace(
    "Upload a CSV with the 11 input columns. An optional student_id column is preserved in results.",
    "Upload a CSV with the 11 input columns. If final_exam_score is included, it will be ignored because it is a later outcome and is not used for prediction. An optional student_id column is preserved in results.",
)
TEMPLATE = TEMPLATE.replace(
    "<table class=\"results\"><tr>{% for column in batch.columns %}<th>{{ column }}</th>{% endfor %}</tr>{% for row in batch.rows %}<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}</table>",
    "<table class=\"results\"><tr><th>Details</th>{% for column in batch.columns %}<th>{{ column }}</th>{% endfor %}</tr>{% for row in batch.rows %}<tr><td><a class=\"button detail-button\" href=\"/student-detail?index={{ row.index }}\">View Details</a></td>{% for cell in row.values %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}</table>",
)
TEMPLATE = TEMPLATE.replace(
    "{% else %}<div class=\"section-head\"><h2>CSV Upload &amp; Batch Prediction</h2>",
    """{% elif view == 'detail' %}<div class="section-head"><h2>Student Performance Detail</h2><a class="back" href="/?view=batch">Back to Results</a></div><div class="panel detail-panel"><div class="detail-heading"><div><span class="muted">Selected student</span><h1>{{ detail.student_id }}</h1></div><span class="status-pill {% if detail.prediction == 1 %}risk-pill{% endif %}">{{ 'At Risk' if detail.prediction == 1 else 'Low Risk' }}</span></div><p class="muted">{{ detail.explanation }}</p><div class="detail-grid">{% for label, value in detail.profile %}<div class="metric"><span>{{ label }}</span><b>{{ value }}</b></div>{% endfor %}</div><div class="indicator-list">{% for indicator in detail.indicators %}<div class="indicator"><div><span>{{ indicator.label }}</span><b>{{ indicator.value }}</b></div><div class="bar"><i style="width: {{ indicator.percent }}%"></i></div></div>{% endfor %}</div>{% if detail.later_outcome is not none %}<div class="later-outcome"><span>Later Outcome: final_exam_score</span><b>{{ detail.later_outcome }}</b><small>Shown for context only; never used for prediction or suggestions.</small></div>{% endif %}<h3>Practical next steps</h3><ul class="suggestions">{% for item in detail.suggestions %}<li>{{ item }}</li>{% endfor %}</ul></div>
{% else %}<div class="section-head"><h2>CSV Upload &amp; Batch Prediction</h2>""",
)
TEMPLATE = TEMPLATE.replace("{{ row.index }}", "{{ row['index'] }}")
TEMPLATE = TEMPLATE.replace("row.values", "row['values']")
TEMPLATE = TEMPLATE.replace("/student-detail?index={{ row['index'] }}", "/?view=detail&amp;index={{ row['index'] }}")
TEMPLATE = TEMPLATE.replace("</style>", "<style>:root{--navy:#173442;--teal:#176b87;--orange:#d9784a;--ink:#203040;--muted:#657783;--paper:#fff;--line:#d8e2e6;--bg:#eef3f4}body{background:var(--bg);color:var(--ink)}.top{background:var(--navy)}.top a,.back{color:#d6e8ea}.muted,.hero p{color:var(--muted)}.choice-card,.stat,.panel,.insight,.metric,.model-table,.results{background:#fff;border-color:var(--line);box-shadow:0 12px 28px #20304012}.choice-card:hover{border-color:var(--teal)}.form input,.form select,.upload input{background:#fff;color:var(--ink);border-color:#bdcbd1}.button{background:var(--teal);color:#fff}.result{background:#e5f4eb;color:#176b3a}.result.risk{background:#fff0e9;color:#a4472b}.selected{background:#e6f3ee}.insight{border-left-color:var(--teal)}.upload{background:#fff;border-color:#bdcbd1}.filter a:not(.button){border-color:var(--line);color:var(--muted)}.filter a.active{background:var(--teal);color:#fff}</style>")
TEMPLATE = TEMPLATE.replace("</style>", "<style>.detail-heading{display:flex;align-items:center;justify-content:space-between;gap:18px}.detail-heading h1{margin:3px 0 0}.detail-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}.status-pill{background:#e5f4eb;color:#176b3a;padding:10px 14px;border-radius:20px;font-weight:bold}.risk-pill{background:#fff0e9;color:#a4472b}.indicator-list{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:22px 0}.indicator span{display:block;color:var(--muted);font-size:12px}.indicator b{display:block;margin:3px 0 7px}.bar{height:8px;background:#e7edf0;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;background:var(--teal);border-radius:8px}.later-outcome{display:grid;grid-template-columns:1fr auto;gap:4px 12px;background:#f4f6f7;border:1px solid var(--line);padding:13px;border-radius:5px;margin:20px 0}.later-outcome span{font-weight:bold}.later-outcome small{grid-column:1/-1;color:var(--muted)}.detail-button{padding:5px 8px;font-size:11px;white-space:nowrap}.results .button{color:#fff}@media(max-width:620px){.detail-grid,.indicator-list{grid-template-columns:1fr}.detail-heading{align-items:flex-start;flex-direction:column}}</style>")

FEE_ANALYSIS_BLOCK = """{% else %}<div class="section-head fee-analysis-heading"><div><span class="eyebrow">School finance</span><h2>Fee Analysis</h2></div><a class="back" href="/?view=home">Back to Dashboard</a></div><section class="panel fee-selector-panel"><div class="fee-selector-copy"><h3>Select Class &amp; Month</h3><p>View and analyze monthly fee collection for existing students.</p></div><form class="form fee-analysis-filters" method="get"><input type="hidden" name="view" value="batch"><label>Class<select name="class" onchange="this.form.submit()"><option value="">Select class</option>{% for option in classes %}<option value="{{ option }}" {% if selected_class == option %}selected{% endif %}>{{ option }}</option>{% endfor %}</select></label><label>Month<select name="month" onchange="this.form.submit()"><option value="">Select month</option>{% for option in fee_months %}<option value="{{ option }}" {% if selected_month == option %}selected{% endif %}>{{ option }}</option>{% endfor %}</select></label></form></section>{% if selected_class and selected_month %}<div class="stats fee-analysis-stats"><div class="stat fee-stat total"><span>Total Students</span><strong>{{ fee_summary.total }}</strong></div><div class="stat fee-stat paid"><span>Paid</span><strong>{{ fee_summary.paid }}</strong></div><div class="stat fee-stat pending"><span>Pending</span><strong>{{ fee_summary.pending }}</strong></div><div class="stat fee-stat free"><span>Free</span><strong>{{ fee_summary.free }}</strong></div></div><section class="panel fee-table-panel"><div class="fee-table-heading"><div><h3>Fee Collection Records</h3><p>Showing {{ fee_summary.total }} student{% if fee_summary.total != 1 %}s{% endif %} for {{ selected_class }} — {{ selected_month }}</p></div><a class="button" href="/management/fees/download?class={{ selected_class|urlencode }}&amp;month={{ selected_month|urlencode }}">Download Excel</a></div><div class="fee-table-controls"><label class="fee-search"><span>Search student</span><input type="search" placeholder="Search name, roll no. or UID" data-fee-search></label><label class="fee-status-filter"><span>Fee Status</span><select data-fee-status-filter><option value="">All statuses</option><option value="Paid">Paid</option><option value="Pending">Pending</option><option value="Free">Free</option></select></label></div><div class="table-wrap"><table class="results fee-analysis-table"><thead><tr><th>Roll No</th><th>Student Name</th><th>UID Number</th><th>Fee Status</th><th>Receipt</th></tr></thead><tbody>{% for student in fee_students %}<tr data-fee-row data-status="{{ student.status }}"><td>{{ student.roll_number }}</td><td>{{ student.student_name }}</td><td>{{ student.uid_number }}</td><td><span class="fee-status-badge {{ student.status|lower }}">{{ student.status }}</span></td><td>{% if student.receipt_data %}<a class="receipt-link" href="/management/fees?class={{ selected_class|urlencode }}">View receipt</a>{% else %}<span class="receipt-empty">—</span>{% endif %}</td></tr>{% else %}<tr><td colspan="5">No students are registered in {{ selected_class }} yet.</td></tr>{% endfor %}</tbody></table></div></section><script>(function(){const search=document.querySelector('[data-fee-search]');const filter=document.querySelector('[data-fee-status-filter]');const rows=document.querySelectorAll('[data-fee-row]');function applyFilters(){const query=(search.value||'').toLowerCase().trim();const status=filter.value;rows.forEach(function(row){const matchesText=row.textContent.toLowerCase().includes(query);const matchesStatus=!status||row.dataset.status===status;row.hidden=!(matchesText&&matchesStatus);});}search?.addEventListener('input',applyFilters);filter?.addEventListener('change',applyFilters);})();</script>{% else %}<div class="panel fee-empty-state"><h3>Choose a class and month</h3><p class="muted">Select both filters above to view fee records and collection totals.</p></div>{% endif %}{% endif %}"""
TEMPLATE = TEMPLATE.replace(
    """{% else %}<div class="section-head"><h2>CSV Upload &amp; Batch Prediction</h2><a class="back" href="/?view=home">Back to Dashboard</a></div><div class="panel"><p class="muted">Upload a CSV with the 11 input columns. If final_exam_score is included, it will be ignored because it is a later outcome and is not used for prediction. An optional student_id column is preserved in results.</p><p class="muted">Required: {{ input_columns|join(', ') }}</p><form method="post" enctype="multipart/form-data" class="upload"><input type="file" name="file" accept=".csv" required><button class="button" type="submit">Upload and Predict</button></form>{% if error %}<div class="error">{{ error }}</div>{% endif %}</div>{% if batch %}<div class="section-head"><h2>Batch Results</h2></div><div class="stats"><div class="stat"><span>Total Students</span><strong>{{ batch.total }}</strong></div><div class="stat"><span>At Risk</span><strong>{{ batch.at_risk }}</strong></div><div class="stat"><span>Low Risk</span><strong>{{ batch.low_risk }}</strong></div><div class="stat"><span>Risk Rate</span><strong>{{ batch.rate }}%</strong></div></div><div class="filter"><a class="{% if filter == 'all' %}active{% endif %}" href="/?view=batch&filter=all">All</a><a class="{% if filter == 'risk' %}active{% endif %}" href="/?view=batch&filter=risk">At Risk</a><a class="{% if filter == 'low' %}active{% endif %}" href="/?view=batch&filter=low">Low Risk</a><a class="button" href="/download-results">Download Results CSV</a></div><div class="panel table-wrap"><table class="results"><tr><th>Details</th>{% for column in batch.columns %}<th>{{ column }}</th>{% endfor %}</tr>{% for row in batch.rows %}<tr><td><a class="button detail-button" href="/?view=detail&amp;index={{ row['index'] }}">View Details</a></td>{% for cell in row['values'] %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}</table></div>{% endif %}{% endif %}""",
    FEE_ANALYSIS_BLOCK,
)
TEMPLATE = TEMPLATE.replace("</style>", "<style>.fee-analysis-heading{align-items:end}.fee-analysis-heading h2{margin:4px 0 0}.fee-selector-panel{display:grid;grid-template-columns:minmax(220px,1fr) minmax(460px,1.35fr);gap:28px;align-items:end;padding:26px}.fee-selector-copy h3,.fee-table-heading h3{margin:0;font-size:21px}.fee-selector-copy p,.fee-table-heading p{margin:6px 0 0;color:var(--muted)}.fee-analysis-filters{grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:end}.fee-analysis-filters label,.fee-table-controls label{color:var(--muted);font-size:12px;font-weight:bold}.fee-analysis-filters select,.fee-table-controls input,.fee-table-controls select{width:100%;margin-top:7px;padding:13px 14px;border:1px solid var(--line);border-radius:10px;background:#0e1017;color:var(--ink);font:inherit;outline:0}.fee-analysis-filters select{font-size:15px;font-weight:bold}.fee-analysis-filters select:hover,.fee-table-controls input:hover,.fee-table-controls select:hover{border-color:#f39a4b99}.fee-analysis-filters select:focus,.fee-table-controls input:focus,.fee-table-controls select:focus{border-color:var(--amber);box-shadow:0 0 0 3px #f39a4b1c}.fee-analysis-stats{grid-template-columns:repeat(4,1fr);margin:22px 0}.fee-stat{position:relative;overflow:hidden;padding:19px;border-top:3px solid var(--line)}.fee-stat:before{content:'';position:absolute;top:0;left:0;width:48px;height:3px;background:var(--amber)}.fee-stat.paid:before{background:var(--mint)}.fee-stat.pending:before{background:var(--amber-light)}.fee-stat.free:before{background:var(--sky)}.fee-stat strong{margin-top:4px}.fee-table-panel{padding:0;overflow:hidden}.fee-table-heading{display:flex;justify-content:space-between;gap:18px;align-items:center;padding:24px 24px 18px}.fee-table-controls{display:grid;grid-template-columns:minmax(240px,1fr) minmax(180px,220px);gap:14px;padding:0 24px 20px;border-bottom:1px solid var(--line)}.fee-analysis-table{min-width:760px}.fee-analysis-table th{padding:13px 16px;background:#ffffff05}.fee-analysis-table td{padding:14px 16px}.fee-analysis-table tbody tr:hover{background:#ffffff08}.fee-status-badge{display:inline-flex;align-items:center;padding:5px 10px;border-radius:999px;font-size:11px;font-weight:bold}.fee-status-badge.paid{background:#92c9ae20;color:var(--mint)}.fee-status-badge.pending{background:#ffc47720;color:var(--amber-light)}.fee-status-badge.free{background:#88b8d020;color:var(--sky)}.receipt-link{color:var(--amber-light);font-weight:bold;text-decoration:none}.receipt-link:hover{text-decoration:underline}.receipt-link:focus-visible{outline:3px solid #ffc47766;outline-offset:3px;border-radius:3px}.receipt-empty{color:var(--muted)}.fee-empty-state{padding:28px}.fee-empty-state h3{margin:0 0 5px}@media(max-width:760px){.fee-selector-panel{grid-template-columns:1fr}.fee-analysis-stats{grid-template-columns:repeat(2,1fr)}.fee-table-heading{align-items:flex-start;flex-direction:column}.fee-table-controls{grid-template-columns:1fr}}@media(max-width:480px){.fee-analysis-filters,.fee-analysis-stats{grid-template-columns:1fr}.fee-selector-panel,.fee-table-heading,.fee-table-controls{padding-left:17px;padding-right:17px}}</style>")
TEMPLATE = TEMPLATE.replace("</style>", "<style>.student-history-heading{align-items:end}.student-history-heading h2{margin:4px 0 0}.student-lookup{padding:25px}.student-lookup h3,.student-matches h3,.profile-section h3{margin:0}.student-search-form{display:flex;gap:10px;margin-top:18px}.student-search-form input{flex:1;min-width:0;padding:13px 14px;border:1px solid var(--line);border-radius:10px;background:#0e1017;color:var(--ink);font:inherit}.student-search-form input:focus{outline:0;border-color:var(--amber);box-shadow:0 0 0 3px #f39a4b1c}.student-empty{margin-top:18px}.student-empty h3{margin:0 0 5px}.student-match-list{display:grid;gap:9px;margin-top:15px}.student-match{display:flex;justify-content:space-between;gap:20px;align-items:center;padding:14px 15px;border:1px solid var(--line);border-radius:11px;background:#ffffff05;color:var(--ink);text-decoration:none}.student-match:hover{border-color:#f39a4b99;background:#ffffff0a}.student-match:focus-visible{outline:3px solid #ffc47766;outline-offset:3px}.student-match small{display:block;margin-top:2px;color:var(--muted)}.student-match>span:last-child{color:var(--amber-light);font-size:12px}.profile-hero{display:flex;justify-content:space-between;gap:20px;align-items:center;margin:23px 0 14px;padding:24px;border:1px solid var(--line);border-radius:17px;background:linear-gradient(135deg,#1d202a,#15161d)}.profile-hero h1{margin:5px 0;font-size:30px}.profile-hero p{margin:0;color:var(--muted)}.profile-grid{display:grid;grid-template-columns:1.35fr 1fr;gap:16px}.profile-section{padding:22px}.profile-details{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:18px 0 0}.profile-details div{min-width:0}.profile-details .wide{grid-column:1/-1}.profile-details dt{color:var(--muted);font-size:11px;font-weight:bold;text-transform:uppercase;letter-spacing:.05em}.profile-details dd{margin:4px 0 0;overflow-wrap:anywhere}.profile-note{margin:19px 0 0;font-size:12px}.fee-history{margin-top:16px}.profile-section-heading{display:flex;align-items:start;justify-content:space-between;gap:16px;margin-bottom:17px}.profile-section-heading p{margin:4px 0 0}.profile-receipt{display:block;width:74px;height:48px;object-fit:cover;border:1px solid var(--line);border-radius:6px}@media(max-width:760px){.profile-grid{grid-template-columns:1fr}.profile-hero{align-items:flex-start;flex-direction:column}.student-match{align-items:flex-start;flex-direction:column;gap:4px}}@media(max-width:520px){.student-search-form{flex-direction:column}.student-search-form .button{width:100%;text-align:center}.profile-details{grid-template-columns:1fr}.profile-details .wide{grid-column:auto}.profile-section-heading{align-items:flex-start;flex-direction:column}}</style>")

STUDENT_HISTORY_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Check a Student | Student Support Dashboard</title><style>:root{--ink:#f7f1e8;--muted:#9ca0ab;--paper:#171921;--line:#30333e;--amber:#f39a4b;--light:#ffc477;--mint:#92c9ae;--sky:#88b8d0}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 80% 0%,#382317 0,#15151d 30%,#0b0c12 70%);color:var(--ink);font:14px/1.5 "Trebuchet MS","Segoe UI",sans-serif}.top{display:flex;justify-content:space-between;align-items:center;padding:20px max(22px,5vw);border-bottom:1px solid #ffffff12;background:#0d0e14cc}.top a,.back{color:var(--light);font-weight:bold;text-decoration:none}.main{max-width:1120px;margin:auto;padding:42px max(22px,5vw) 70px}.heading{display:flex;justify-content:space-between;align-items:end;gap:18px;margin-bottom:20px}.eyebrow{color:var(--light);font-size:11px;font-weight:bold;letter-spacing:.13em;text-transform:uppercase}.heading h1{margin:7px 0 0;font-size:clamp(30px,5vw,46px)}.panel{padding:23px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 14px 35px #0004}.muted{color:var(--muted)}.search{display:flex;gap:10px;margin-top:17px}.search input{flex:1;min-width:0;padding:13px 14px;border:1px solid var(--line);border-radius:10px;background:#0e1017;color:var(--ink);font:inherit}.search input:focus{outline:0;border-color:var(--amber);box-shadow:0 0 0 3px #f39a4b1c}.button{display:inline-block;border:0;border-radius:10px;padding:12px 16px;background:linear-gradient(135deg,var(--amber),#e87551);color:#1a120e;font:inherit;font-weight:bold;cursor:pointer;text-decoration:none}.matches,.history{margin-top:18px}.matches h2,.profile-card h2{margin:0 0 14px;font-size:20px}.match{display:flex;justify-content:space-between;gap:18px;align-items:center;padding:14px;border:1px solid var(--line);border-radius:11px;background:#ffffff05;color:var(--ink);text-decoration:none;margin-top:9px}.match:hover{border-color:#f39a4b99;background:#ffffff0a}.match small{display:block;color:var(--muted)}.profile-hero{display:flex;justify-content:space-between;gap:18px;align-items:center;margin:22px 0 16px;padding:23px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(135deg,#1d202a,#15161d)}.profile-hero h2{margin:5px 0}.grid{display:grid;grid-template-columns:1.3fr 1fr;gap:16px}.details{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px;margin:0}.details .wide{grid-column:1/-1}.details dt{color:var(--muted);font-size:11px;font-weight:bold;text-transform:uppercase}.details dd{margin:4px 0 0;overflow-wrap:anywhere}.table-wrap{overflow:auto}.fees{width:100%;border-collapse:collapse;min-width:620px}.fees th,.fees td{text-align:left;padding:11px;border-bottom:1px solid #ffffff12}.fees th{color:var(--muted);font-size:11px;text-transform:uppercase}.badge{display:inline-block;padding:5px 10px;border-radius:999px;font-size:11px;font-weight:bold}.paid{background:#92c9ae20;color:var(--mint)}.pending{background:#ffc47720;color:var(--light)}.free{background:#88b8d020;color:var(--sky)}.receipt{width:70px;height:45px;object-fit:cover;border:1px solid var(--line);border-radius:6px}@media(max-width:700px){.top,.heading,.profile-hero{align-items:flex-start;flex-direction:column}.grid{grid-template-columns:1fr}.search{flex-direction:column}.search .button{text-align:center}.details{grid-template-columns:1fr}.details .wide{grid-column:auto}}</style></head><body><header class="top"><strong>Student Support Dashboard</strong><span>{{ school_name }} &nbsp; <a href="/school-logout">School Logout</a></span></header><main class="main"><div class="heading"><div><span class="eyebrow">Student support</span><h1>Check a Student</h1></div><a class="back" href="/?view=home">&#8592; Back to Dashboard</a></div><section class="panel"><p class="muted">Find an existing student profile, class information, and verified fee history.</p><form class="search" method="get"><input name="q" value="{{ student_query }}" placeholder="Enter UID, Student Name, or Class" aria-label="Student search"><button class="button" type="submit">Search</button></form></section>{% if student_query and not student_profile and not student_matches %}<section class="panel history"><h2>No student found</h2><p class="muted">Try a UID number, student name, or class from your school records.</p></section>{% endif %}{% if student_matches and not student_profile %}<section class="panel matches"><h2>Matching Students</h2>{% for student in student_matches %}<a class="match" href="/student-history?q={{ student_query|urlencode }}&amp;student_id={{ student.id }}"><span><b>{{ student.student_name }}</b><small>{{ student.class_name }} · Roll No. {{ student.roll_number }}</small></span><span>{{ student.uid_number }}</span></a>{% endfor %}</section>{% endif %}{% if student_profile %}<section class="profile-hero"><div><span class="eyebrow">Student profile</span><h2>{{ student_profile.student_name }}</h2><span class="muted">{{ student_profile.class_name }} · Roll No. {{ student_profile.roll_number }} · {{ student_profile.uid_number }}</span></div><a class="button" href="/student-history{% if student_query %}?q={{ student_query|urlencode }}{% endif %}">Search again</a></section><div class="grid"><section class="panel profile-card"><h2>Student Profile</h2><dl class="details"><div><dt>Student Name</dt><dd>{{ student_profile.student_name }}</dd></div><div><dt>UID Number</dt><dd>{{ student_profile.uid_number }}</dd></div><div><dt>Roll Number</dt><dd>{{ student_profile.roll_number }}</dd></div><div><dt>Current Class</dt><dd>{{ student_profile.class_name }}</dd></div><div><dt>Father Name</dt><dd>{{ student_profile.father_name }}</dd></div><div><dt>Mother Name</dt><dd>{{ student_profile.mother_name }}</dd></div><div><dt>Parent Mobile</dt><dd>{{ student_profile.parent_mobile }}</dd></div><div><dt>Second Mobile</dt><dd>{{ student_profile.second_mobile or 'Not provided' }}</dd></div><div class="wide"><dt>Complete Address</dt><dd>{{ student_profile.address }}</dd></div></dl></section><section class="panel profile-card"><h2>Class / Academic Information</h2><dl class="details"><div><dt>Current Class</dt><dd>{{ student_profile.class_name }}</dd></div><div><dt>Student Record Created</dt><dd>{{ student_profile.created_at }}</dd></div><div><dt>Last Updated</dt><dd>{{ student_profile.updated_at }}</dd></div><div><dt>Transport</dt><dd>{{ 'Yes' if student_profile.transport_student else 'No' }}</dd></div></dl><p class="muted">Previous class/session information is not stored in the current records.</p></section></div><section class="panel history"><h2>Fee History</h2><div class="table-wrap"><table class="fees"><thead><tr><th>Month</th><th>Status</th><th>Receipt / Photo</th><th>Updated</th></tr></thead><tbody>{% for fee in student_fees %}<tr><td>{{ fee.fee_month }}</td><td><span class="badge {{ fee.status|lower }}">{{ fee.status }}</span></td><td>{% if fee.receipt_preview %}<img class="receipt" src="data:image/png;base64,{{ fee.receipt_preview }}" alt="Receipt for {{ fee.fee_month }}">{% else %}<span class="muted">No receipt on file</span>{% endif %}</td><td>{{ fee.updated_at or '—' }}</td></tr>{% endfor %}</tbody></table></div></section>{% endif %}</main></body></html>"""

STUDENT_HISTORY_TEMPLATE = STUDENT_HISTORY_TEMPLATE.replace(
    '<section class="profile-hero"><div><span class="eyebrow">Student profile</span><h2>{{ student_profile.student_name }}</h2><span class="muted">{{ student_profile.class_name }} · Roll No. {{ student_profile.roll_number }} · {{ student_profile.uid_number }}</span>',
    '<section class="profile-hero"><div><span class="eyebrow">Student profile</span><h2>{{ student_profile.student_name }}</h2><span class="muted">{{ student_profile.class_name }} · Roll No. {{ student_profile.roll_number }} · {{ student_profile.uid_number }}{% if student_profile.admission_date %} · Admission date {{ student_profile.admission_date }}{% endif %}</span>',
)
STUDENT_HISTORY_TEMPLATE = STUDENT_HISTORY_TEMPLATE.replace(
    '</dl></section><section class="panel profile-card"><h2>Class / Academic Information</h2>',
    '</dl><div class="student-photo-block"><span class="student-photo-label">Student Photo</span>{% if student_profile.photo_url %}<img class="student-photo" src="{{ student_profile.photo_url }}" alt="Student photo">{% else %}<div class="student-photo-placeholder">No photo available</div>{% endif %}</div></section><section class="panel profile-card"><h2>Class / Academic Information</h2>',
)
STUDENT_HISTORY_TEMPLATE = STUDENT_HISTORY_TEMPLATE.replace(
    "</style>",
    "<style>.student-photo-block{display:grid;gap:7px;margin-top:20px}.student-photo-label{color:var(--muted);font-size:11px;font-weight:bold;text-transform:uppercase;letter-spacing:.05em}.student-photo{display:block;width:112px;height:145px;object-fit:cover;border-radius:8px;border:1px solid #ffffff2b}.student-photo-placeholder{display:grid;place-items:center;width:112px;height:145px;border:1px dashed #ffffff35;border-radius:8px;color:var(--muted);font-size:12px;text-align:center;padding:10px}@media(max-width:620px){.student-photo,.student-photo-placeholder{width:min(112px,35vw);height:min(145px,45vw)}}</style>",
    1,
)

modern_home = """
{% if view == 'home' %}<section class="welcome"><div class="welcome-copy"><span class="eyebrow">Student support / {{ school_name }}</span><h1>Make every support conversation count.</h1><p>See where your students are thriving, spot who may need a little more support, and choose the next useful action.</p><div class="welcome-actions"><a class="button" href="/?view=single"><span aria-hidden="true">+</span> Check a student</a><a class="text-action" href="/?view=batch">Upload a class list <span aria-hidden="true">&#8594;</span></a></div></div><div class="welcome-art" aria-hidden="true"><div class="art-glow"></div><div class="art-orbit orbit-one"></div><div class="art-orbit orbit-two"></div><div class="art-note note-one"><span>✓</span> Attendance</div><div class="art-note note-two"><span>↗</span> Progress</div><div class="art-core">✦</div></div></section><section class="section-block"><div class="section-heading"><div><span class="eyebrow">Class overview</span><h2>Today at a glance</h2></div><span class="record-count">{{ total_students }} cleaned records</span></div><div class="overview-grid"><article class="risk-overview"><div class="card-topline"><span class="card-kicker">Support radar</span><span class="live-dot">Live view</span></div><div class="risk-copy"><div><h3>Students needing attention</h3><p>Prioritize a calm, practical follow-up with students showing several warning signs.</p></div><strong>{{ at_risk_students }}</strong></div><div class="risk-meter"><span style="width: {{ risk_rate }}%"></span></div><div class="meter-caption"><span>{{ risk_rate }}% of this class</span><span>{{ low_risk_students }} currently steady</span></div><a class="card-link" href="/?view=batch&amp;filter=risk">Review at-risk students <span aria-hidden="true">&#8594;</span></a></article><div class="stat-stack"><article class="support-stat"><span class="stat-icon orange">◎</span><div><span>Class size</span><strong>{{ total_students }}</strong><small>students in view</small></div></article><article class="support-stat"><span class="stat-icon blue">◒</span><div><span>Average attendance</span><strong>{{ avg_attendance }}<em>%</em></strong><small>across the class</small></div></article><article class="support-stat"><span class="stat-icon green">↗</span><div><span>Average study time</span><strong>{{ avg_study }}<em> hrs</em></strong><small>each week</small></div></article></div></div></section><section class="section-block"><div class="section-heading"><div><span class="eyebrow">Student wellbeing signals</span><h2>Support starts with context</h2></div><span class="section-note">Simple indicators, human decisions</span></div><div class="context-grid"><article class="context-card"><div class="context-icon">◒</div><div><h3>Academic progress</h3><p>Use previous scores and assignment results to guide a focused check-in.</p></div><div class="context-line"><span>Progress snapshot</span><span class="line-dots"><i></i><i></i><i></i><i></i><i></i></span></div></article><article class="context-card"><div class="context-icon warm">◷</div><div><h3>Attendance rhythm</h3><p>Consistent participation is a useful starting point for understanding barriers.</p></div><div class="context-line"><span>Daily presence matters</span><span class="line-arrow">&#8599;</span></div></article><article class="context-card"><div class="context-icon green">✦</div><div><h3>Study habits</h3><p>Small, realistic routines can make the next week feel more manageable.</p></div><div class="context-line"><span>Build momentum</span><span class="line-arrow">&#8594;</span></div></article></div></section><section class="section-block"><div class="section-heading"><div><span class="eyebrow">Support notes</span><h2>What to explore next</h2></div><span class="section-note">Observed patterns, not causes</span></div><div class="insight-grid">{% for insight in insights %}<article class="insight-card"><span class="insight-mark">✦</span><p>{{ insight }}</p><span class="insight-action">Open a conversation</span></article>{% endfor %}</div></section><section class="section-block model-section"><div class="section-heading"><div><span class="eyebrow">Prediction support</span><h2>Confidence in the system</h2></div><span class="section-note">For staff context only</span></div><div class="model-strip"><div><strong>Random Forest</strong><span class="model-selected">Selected model</span></div>{% for name, metrics in model_metrics.items() %}{% if name == 'Random Forest' %}<div class="model-score"><span>Accuracy</span><strong>{{ metrics[0] }}</strong></div><div class="model-score"><span>Precision</span><strong>{{ metrics[1] }}</strong></div><div class="model-score"><span>Recall</span><strong>{{ metrics[2] }}</strong></div>{% endif %}{% endfor %}<a class="text-action" href="/?view=single">Make a prediction <span aria-hidden="true">&#8594;</span></a></div></section><div class="home-game-slot"></div>
{% elif view == 'single' %}"""
TEMPLATE = TEMPLATE.replace(
    '<section class="profile-hero"><div><span class="eyebrow">Student profile</span><h1>{{ student_profile.student_name }}</h1>',
    '<section class="profile-hero">{% if student_profile.photo_url %}<img class="student-photo" src="{{ student_profile.photo_url }}" alt="Student photo">{% endif %}<div><span class="eyebrow">Student profile</span><h1>{{ student_profile.student_name }}</h1>',
).replace(
    '<p>{{ student_profile.class_name }} · Roll No. {{ student_profile.roll_number }} · UID {{ student_profile.uid_number }}</p>',
    '<p>{{ student_profile.class_name }} · Roll No. {{ student_profile.roll_number }} · UID {{ student_profile.uid_number }}{% if student_profile.admission_date %} · Admission date {{ student_profile.admission_date }}{% endif %}</p>',
)
STUDENT_HISTORY_TEMPLATE = STUDENT_HISTORY_TEMPLATE.replace(
    '<section class="profile-hero"><div><span class="eyebrow">Student profile</span><h2>{{ student_profile.student_name }}</h2>',
    '<section class="profile-hero">{% if student_profile.photo_url %}<img class="student-photo" src="{{ student_profile.photo_url }}" alt="Student photo">{% endif %}<div><span class="eyebrow">Student profile</span><h2>{{ student_profile.student_name }}</h2>',
).replace(
    '<span class="muted">{{ student_profile.class_name }} · Roll No. {{ student_profile.roll_number }} · {{ student_profile.uid_number }}</span>',
    '<span class="muted">{{ student_profile.class_name }} · Roll No. {{ student_profile.roll_number }} · {{ student_profile.uid_number }}{% if student_profile.admission_date %} · Admission date {{ student_profile.admission_date }}{% endif %}</span>',
)
TEMPLATE = TEMPLATE.replace("</style>", "<style>.student-photo{width:76px;height:76px;object-fit:cover;border-radius:14px;border:1px solid var(--line);margin-right:16px;vertical-align:middle}.profile-hero>div{min-width:0}</style>", 1)
modern_home = modern_home.replace("Upload a class list", "Fee Analysis")
TEMPLATE = TEMPLATE.replace("{% if view == 'home' %}", modern_home, 1)
TEMPLATE = TEMPLATE.replace('href="/?view=single"><span aria-hidden="true">+</span> Check a student', 'href="/student-history"><span aria-hidden="true">+</span> Check a student')
overview_start = TEMPLATE.find('<div class="overview-grid">')
overview_end = TEMPLATE.find('</div></section><section class="section-block"><div class="section-heading"><div><span class="eyebrow">Student wellbeing signals', overview_start)
if overview_start != -1 and overview_end != -1:
    STUDENT_OVERVIEW_BLOCK = '<div class="overview-grid student-overview-grid"><article class="student-overview-card"><div class="card-topline"><span class="card-kicker">Student overview</span><span class="student-icon" aria-hidden="true">◎</span></div><h3>Student Overview</h3><div class="student-counts"><div><strong>{{ total_students_ever }}</strong><span>Total Students Ever</span><small>Registered student records</small></div><div><strong>{{ active_students }}</strong><span>Active Students</span><small>Currently enrolled students</small></div></div></article><article class="active-students-card"><div class="card-topline"><span class="card-kicker">Current enrollment</span><span class="live-dot">Live count</span></div><div class="active-heading"><div><h3>Active Students</h3><p>Currently enrolled students</p></div><strong>{{ active_students }}</strong></div><div class="active-progress" role="progressbar" aria-label="Percentage of active students" aria-valuenow="{{ active_percentage }}" aria-valuemin="0" aria-valuemax="100"><span style="width: {{ active_percentage }}%"></span></div><div class="active-caption"><span>{{ active_percentage }}% of total students</span><span>{{ total_students_ever }} total</span></div></article>'
    TEMPLATE = TEMPLATE[:overview_start] + STUDENT_OVERVIEW_BLOCK + TEMPLATE[overview_end:]
classes_start = TEMPLATE.find('<section class="section-block"><div class="section-heading"><div><span class="eyebrow">Student wellbeing signals')
classes_end = TEMPLATE.find('<section class="section-block model-section">', classes_start)
if classes_start != -1 and classes_end != -1:
    CLASSES_BLOCK = '''<section class="section-block classes-section"><div class="section-heading"><div><span class="eyebrow">Student information</span><h2>Classes</h2><p class="classes-subtitle">Select a class to view student information</p></div></div><div class="classes-grid"><div class="class-card add-student-card"><span class="class-card-icon" aria-hidden="true">+</span><span class="class-card-name">Add Student</span><label class="class-card-select">Select a class<select id="admission-class" aria-label="Select a class for admission">{% for class_name in classes %}<option value="{{ class_name }}">{{ class_name }}</option>{% endfor %}</select></label><a id="admission-form-link" class="class-card-action" href="/students/{{ classes[0]|urlencode }}?open_add=1#add-student">Open form <span aria-hidden="true">&#8594;</span></a></div>{% for class_name in classes %}<a class="class-card" href="/students/{{ class_name|urlencode }}" aria-label="View {{ class_name }} student information"><span class="class-card-icon" aria-hidden="true">✦</span><span class="class-card-name">{{ class_name }}</span><span class="class-card-action">View Class <span aria-hidden="true">&#8594;</span></span></a>{% endfor %}</div></section><script>(function(){const classSelect=document.getElementById('admission-class');const formLink=document.getElementById('admission-form-link');classSelect?.addEventListener('change',function(){formLink.href='/students/'+encodeURIComponent(classSelect.value)+'?open_add=1#add-student';});})();</script>'''
    TEMPLATE = TEMPLATE[:classes_start] + CLASSES_BLOCK + TEMPLATE[classes_end:]
model_start = TEMPLATE.find('<section class="section-block model-section">')
model_end = TEMPLATE.find('</section>', model_start) + len('</section>') if model_start != -1 else -1
if model_start != -1 and model_end > model_start:
    PASSWORD_RESET_BLOCK = '''<section class="section-block account-support-section"><div class="section-heading"><div><span class="eyebrow">ACCOUNT SUPPORT</span><h2>Reset Student Password</h2><p class="account-support-subtitle">Find and verify the student before resetting their password.</p></div></div><div class="panel password-reset-card">{% if password_reset_step == 'verify' and password_reset_student %}<div class="verification-profile"><div>{% if password_reset_student.photo_path %}<img class="reset-student-photo" src="{{ url_for('student_photo', student_id=password_reset_student.id) }}" alt="Student photo">{% else %}<div class="reset-student-photo photo-placeholder">No photo</div>{% endif %}</div><div><div class="profile-details"><div><span>Student Name</span><strong>{{ password_reset_student.student_name }}</strong></div><div><span>UID / Student ID</span><strong>{{ password_reset_student.uid_number }} / {{ password_reset_student.id }}</strong></div><div><span>Class</span><strong>{{ password_reset_student.class_name }}</strong></div><div><span>Roll Number</span><strong>{{ password_reset_student.roll_number }}</strong></div><div><span>Father's Name</span><strong>{{ password_reset_student.father_name }}</strong></div><div><span>Mother's Name</span><strong>{{ password_reset_student.mother_name }}</strong></div><div><span>Parent Mobile Number</span><strong>{{ password_reset_student.parent_mobile }}</strong></div><div><span>Account Status</span><strong>{{ password_reset_student.account_status }}</strong></div><div class="wide"><span>Address</span><strong>{{ password_reset_student.address }}</strong></div></div></div></div><div class="reset-actions"><form method="post" action="{{ url_for('search_another_student') }}"><button class="button secondary" type="submit">&#8592; Search Another Student</button></form><form method="post" action="{{ url_for('confirm_student_password_reset') }}"><button class="button" type="submit">Confirm Student &#8594;</button></form></div>{% elif password_reset_step == 'reset' and password_reset_student %}<div class="reset-heading"><h3>Reset Password</h3><p>Confirm that this is the correct student, then create a new password.</p></div><div class="reset-summary">{% if password_reset_student.photo_path %}<img class="reset-summary-photo" src="{{ url_for('student_photo', student_id=password_reset_student.id) }}" alt="Student photo">{% endif %}<div><strong>{{ password_reset_student.student_name }}</strong><span>{{ password_reset_student.uid_number }} · {{ password_reset_student.class_name }} · Roll {{ password_reset_student.roll_number }}</span></div></div><form method="post" action="{{ url_for('reset_student_password') }}" class="form password-reset-form"><label>New Password<input name="new_password" type="password" minlength="6" autocomplete="new-password" required></label><label>Confirm New Password<input name="confirm_password" type="password" minlength="6" autocomplete="new-password" required></label><button class="button" type="submit">Reset Password</button></form>{% else %}<form method="post" action="{{ url_for('find_student_for_password_reset') }}" class="form password-reset-form search-password-reset-form"><label>Student UID / Student ID<input name="student_identifier" autocomplete="off" required></label><button class="button" type="submit">Find Student &#8594;</button></form>{% endif %}</div></section>'''
    TEMPLATE = TEMPLATE[:model_start] + PASSWORD_RESET_BLOCK + TEMPLATE[model_end:]
TEMPLATE = TEMPLATE.replace("</style>", "<style>.password-reset-card{background:linear-gradient(145deg,#1a1c25,#15161d)!important;border-color:var(--line)!important}.password-reset-form{align-items:end}.password-reset-form label{color:#d8d1cc;font-weight:bold}.password-reset-form input{margin-top:7px}.search-password-reset-form{grid-template-columns:minmax(0,1fr) auto}.search-password-reset-form .button{height:43px}.verification-profile{display:grid;grid-template-columns:132px minmax(0,1fr);gap:24px}.reset-student-photo{display:block;width:112px;height:145px;object-fit:cover;border-radius:10px;border:1px solid var(--line)}.reset-student-photo.photo-placeholder{display:grid;place-items:center;color:var(--muted);font-size:12px;text-align:center}.profile-details{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.profile-details div{min-width:0}.profile-details .wide{grid-column:1/-1}.profile-details span{display:block;color:var(--muted);font-size:11px;font-weight:bold;text-transform:uppercase}.profile-details strong{display:block;margin-top:4px;overflow-wrap:anywhere}.reset-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:25px}.reset-actions form{margin:0}.reset-heading h3{margin:0;font-size:22px}.reset-heading p{margin:6px 0 20px;color:var(--muted)}.reset-summary{display:flex;align-items:center;gap:14px;margin-bottom:20px;padding:12px;border:1px solid var(--line);border-radius:12px;background:#ffffff05}.reset-summary-photo{width:54px;height:68px;object-fit:cover;border-radius:7px;border:1px solid var(--line)}.reset-summary strong,.reset-summary span{display:block}.reset-summary span{margin-top:4px;color:var(--muted);font-size:12px}.password-reset-form .button{align-self:end;min-height:43px}@media(max-width:720px){.verification-profile{grid-template-columns:1fr}.profile-details{grid-template-columns:1fr}.search-password-reset-form{grid-template-columns:1fr}.reset-actions{justify-content:stretch;flex-direction:column}.reset-actions .button{width:100%}}</style>", 1)
TEMPLATE = TEMPLATE.replace("</head>", """<style>
:root{--ink:#f7f1e8;--muted:#9ca0ab;--paper:#171921;--paper-soft:#1d202a;--line:#30333e;--navy:#0b0c12;--amber:#f39a4b;--amber-light:#ffc477;--coral:#e87551;--mint:#92c9ae;--sky:#88b8d0;--shadow:0 20px 55px #0008}body{background:radial-gradient(circle at 80% 0%,#382317 0,#15151d 30%,#0b0c12 70%);color:var(--ink);font-family:"Trebuchet MS","Segoe UI",sans-serif;letter-spacing:.01em}.top{background:#0d0e14cc;border-bottom:1px solid #ffffff12;padding:18px max(22px,5vw);backdrop-filter:blur(16px);position:relative;z-index:2}.brand{font-size:18px;letter-spacing:.01em}.brand small{color:#aaa8b1}.top>span{display:flex;align-items:center;gap:14px;color:#bdb8b0;font-size:12px}.top a{color:#ffc17b;text-decoration:none}.main{max-width:1240px;padding:42px max(22px,5vw) 80px}.welcome{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(280px,.8fr);gap:30px;align-items:center;min-height:330px;padding:44px 48px;border:1px solid #ffffff12;border-radius:28px;background:linear-gradient(120deg,#171923e8,#17151acf 60%,#2a1a16e6);box-shadow:var(--shadow);overflow:hidden;position:relative}.welcome:after{content:"";position:absolute;inset:auto -10% -70% 35%;height:260px;background:#e27b3d2b;filter:blur(55px);border-radius:50%}.eyebrow{color:var(--amber-light);font-size:11px;text-transform:uppercase;letter-spacing:.13em;font-weight:bold}.welcome h1{font-size:clamp(34px,4.3vw,58px);line-height:1.04;letter-spacing:-.03em;max-width:650px;margin:14px 0}.welcome p{color:#c2bec0;max-width:560px;font-size:16px}.welcome-actions{display:flex;align-items:center;gap:22px;margin-top:28px}.button{background:linear-gradient(135deg,var(--amber),var(--coral));border:0;border-radius:12px;box-shadow:0 10px 24px #e57c3b3d;color:#17120f;padding:12px 17px;font-weight:bold;text-decoration:none;transition:transform .2s,box-shadow .2s}.button:hover{background:linear-gradient(135deg,#ffc477,#ef8961);box-shadow:0 14px 28px #e57c3b55;transform:translateY(-2px)}.button span{font-size:18px;vertical-align:-1px}.text-action,.card-link{color:#ffc477;font-weight:bold;text-decoration:none}.text-action span,.card-link span{padding-left:7px}.welcome-art{height:240px;position:relative;z-index:1}.art-glow{position:absolute;width:210px;height:210px;right:25px;top:10px;border-radius:50%;background:radial-gradient(circle,#f19a4b55,#d9633a14 55%,transparent 70%);filter:blur(2px)}.art-orbit{position:absolute;border:1px solid #f4ad6c55;border-radius:50%;transform:rotate(-22deg)}.orbit-one{width:250px;height:100px;right:-4px;top:66px}.orbit-two{width:190px;height:75px;right:26px;top:80px;transform:rotate(58deg)}.art-core{position:absolute;right:113px;top:81px;width:68px;height:68px;display:grid;place-items:center;border-radius:22px;background:linear-gradient(145deg,#ffbe6b,#d85d40);color:#251711;font-size:30px;box-shadow:0 12px 30px #e37e4e66;transform:rotate(8deg)}.art-note{position:absolute;padding:9px 12px;border:1px solid #ffffff1c;border-radius:10px;background:#22222bd9;box-shadow:0 8px 22px #0005;color:#e8ddd2;font-size:12px}.art-note span{color:var(--amber-light);font-size:16px;margin-right:5px}.note-one{right:4px;top:25px}.note-two{right:20px;bottom:25px}.section-block{margin-top:48px}.section-heading{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:17px}.section-heading h2{font-size:25px;letter-spacing:-.02em;margin:5px 0 0}.record-count,.section-note{color:var(--muted);font-size:12px}.overview-grid{display:grid;grid-template-columns:1.35fr .65fr;gap:16px}.risk-overview,.support-stat,.context-card,.insight-card,.model-strip{border:1px solid var(--line);background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 14px 35px #0003}.risk-overview{border-radius:20px;padding:25px}.card-topline,.meter-caption{display:flex;justify-content:space-between;align-items:center}.card-kicker{color:#d2c8bc;font-size:12px;text-transform:uppercase;letter-spacing:.1em}.live-dot{color:var(--mint);font-size:11px}.live-dot:before{content:"";display:inline-block;width:6px;height:6px;background:var(--mint);border-radius:50%;margin:0 6px 1px 0;box-shadow:0 0 10px var(--mint)}.risk-copy{display:flex;justify-content:space-between;gap:18px;align-items:end;margin:30px 0 22px}.risk-copy h3{font-size:21px;margin:0 0 7px}.risk-copy p{color:var(--muted);font-size:13px;max-width:420px;margin:0}.risk-copy>strong{color:var(--amber-light);font-size:58px;line-height:.9}.risk-meter{height:11px;border-radius:20px;background:#2b2d35;overflow:hidden}.risk-meter span{display:block;height:100%;min-width:2%;border-radius:20px;background:linear-gradient(90deg,var(--mint),var(--amber),var(--coral));box-shadow:0 0 16px #ef914b88}.meter-caption{font-size:11px;color:var(--muted);margin:9px 0 22px}.card-link{font-size:13px}.stat-stack{display:grid;gap:10px}.support-stat{display:flex;align-items:center;gap:13px;padding:17px;border-radius:16px}.stat-icon{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:#f09a4b1c;color:var(--amber-light);font-size:21px}.stat-icon.blue{background:#88b8d01c;color:var(--sky)}.stat-icon.green{background:#92c9ae1c;color:var(--mint)}.support-stat span:not(.stat-icon){display:block;color:var(--muted);font-size:11px}.support-stat strong{display:block;font-size:24px;margin-top:1px}.support-stat em{font-style:normal;font-size:12px;color:var(--muted);margin-left:2px}.support-stat small{color:#777b87;font-size:11px}.context-grid,.insight-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.context-card{min-height:190px;padding:21px;border-radius:18px;display:flex;flex-direction:column;gap:12px}.context-icon{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:#88b8d01c;color:var(--sky);font-size:21px}.context-icon.warm{background:#f39a4b1c;color:var(--amber-light)}.context-icon.green{background:#92c9ae1c;color:var(--mint)}.context-card h3{font-size:17px;margin:0 0 5px}.context-card p{color:var(--muted);font-size:13px;line-height:1.55;margin:0}.context-line{display:flex;justify-content:space-between;align-items:center;margin-top:auto;padding-top:12px;border-top:1px solid #ffffff0e;color:#858995;font-size:11px}.line-dots{display:flex;gap:4px}.line-dots i{width:7px;height:7px;background:var(--sky);border-radius:50%}.line-dots i:nth-child(4),.line-dots i:nth-child(5){background:#424651}.line-arrow{color:var(--amber-light);font-size:17px}.insight-card{border-radius:16px;padding:18px}.insight-mark{color:var(--amber-light);font-size:18px}.insight-card p{color:#d3ced0;font-size:13px;line-height:1.55;min-height:62px;margin:10px 0 15px}.insight-action{color:#888c97;font-size:11px}.model-strip{display:flex;align-items:center;gap:28px;border-radius:16px;padding:17px 20px}.model-strip>div:first-child{margin-right:auto}.model-strip strong{display:block;font-size:14px}.model-selected{color:var(--mint);font-size:11px}.model-score{border-left:1px solid var(--line);padding-left:22px}.model-score span{display:block;color:var(--muted);font-size:11px}.model-score strong{font-size:20px;color:#f2e6d8;margin-top:3px}.home-game-slot{margin-top:48px}.game-home,.account-panel{border-color:var(--line)!important;background:linear-gradient(145deg,#1a1c25,#15161d)!important}.section-head h2,.section-heading h2{color:var(--ink)}.stats,.choice{gap:14px}.stat,.panel{border-radius:16px}.insights{gap:14px}.error{border-radius:10px}@media(max-width:850px){.welcome{grid-template-columns:1fr;padding:32px}.welcome-art{position:absolute;right:-35px;bottom:-20px;opacity:.6;transform:scale(.8)}.overview-grid{grid-template-columns:1fr}.context-grid,.insight-grid{grid-template-columns:1fr 1fr}.model-strip{flex-wrap:wrap;gap:16px}.model-strip>div:first-child{width:100%}.model-score{padding-left:15px}.model-strip .text-action{margin-left:auto}}@media(max-width:620px){.top{display:block}.top>span{margin-top:10px}.main{padding-top:24px}.welcome{min-height:450px;padding:28px 23px}.welcome h1{font-size:38px}.welcome-actions{align-items:flex-start;flex-direction:column;gap:16px}.section-heading{align-items:flex-start;flex-direction:column;gap:7px}.context-grid,.insight-grid{grid-template-columns:1fr}.risk-copy>strong{font-size:46px}.risk-copy{align-items:start}.model-strip{align-items:flex-start;flex-direction:column}.model-score{border-left:0;border-top:1px solid var(--line);padding:10px 0 0;width:100%}.model-strip .text-action{margin-left:0}.game-home,.account-panel{grid-template-columns:1fr!important}}
:root{--ink:#f7f1e8;--muted:#9ca0ab;--paper:#171921;--paper-soft:#1d202a;--line:#30333e;--navy:#0b0c12;--amber:#f39a4b;--amber-light:#ffc477;--coral:#e87551;--mint:#92c9ae;--sky:#88b8d0;--shadow:0 20px 55px #0008}body{background:radial-gradient(circle at 80% 0%,#382317 0,#15151d 30%,#0b0c12 70%);color:var(--ink);font-family:"Trebuchet MS","Segoe UI",sans-serif;letter-spacing:.01em}.top{background:#0d0e14cc;border-bottom:1px solid #ffffff12;padding:18px max(22px,5vw);backdrop-filter:blur(16px);position:relative;z-index:2}.brand{font-size:18px;letter-spacing:.01em}.brand small{color:#aaa8b1}.top>span{display:flex;align-items:center;gap:14px;color:#bdb8b0;font-size:12px}.top a{color:#ffc17b;text-decoration:none}.main{max-width:1240px;padding:42px max(22px,5vw) 80px}.welcome{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(280px,.8fr);gap:30px;align-items:center;min-height:330px;padding:44px 48px;border:1px solid #ffffff12;border-radius:28px;background:linear-gradient(120deg,#171923e8,#17151acf 60%,#2a1a16e6);box-shadow:var(--shadow);overflow:hidden;position:relative}.welcome:after{content:"";position:absolute;inset:auto -10% -70% 35%;height:260px;background:#e27b3d2b;filter:blur(55px);border-radius:50%}.eyebrow{color:var(--amber-light);font-size:11px;text-transform:uppercase;letter-spacing:.13em;font-weight:bold}.welcome h1{font-size:clamp(34px,4.3vw,58px);line-height:1.04;letter-spacing:-.03em;max-width:650px;margin:14px 0}.welcome p{color:#c2bec0;max-width:560px;font-size:16px}.welcome-actions{display:flex;align-items:center;gap:22px;margin-top:28px}.button{background:linear-gradient(135deg,var(--amber),var(--coral));border:0;border-radius:12px;box-shadow:0 10px 24px #e57c3b3d;color:#17120f;padding:12px 17px;font-weight:bold;text-decoration:none;transition:transform .2s,box-shadow .2s}.button:hover{background:linear-gradient(135deg,#ffc477,#ef8961);box-shadow:0 14px 28px #e57c3b55;transform:translateY(-2px)}.button span{font-size:18px;vertical-align:-1px}.text-action,.card-link{color:#ffc477;font-weight:bold;text-decoration:none}.text-action span,.card-link span{padding-left:7px}.welcome-art{height:240px;position:relative;z-index:1}.art-glow{position:absolute;width:210px;height:210px;right:25px;top:10px;border-radius:50%;background:radial-gradient(circle,#f19a4b55,#d9633a14 55%,transparent 70%);filter:blur(2px)}.art-orbit{position:absolute;border:1px solid #f4ad6c55;border-radius:50%;transform:rotate(-22deg)}.orbit-one{width:250px;height:100px;right:-4px;top:66px}.orbit-two{width:190px;height:75px;right:26px;top:80px;transform:rotate(58deg)}.art-core{position:absolute;right:113px;top:81px;width:68px;height:68px;display:grid;place-items:center;border-radius:22px;background:linear-gradient(145deg,#ffbe6b,#d85d40);color:#251711;font-size:30px;box-shadow:0 12px 30px #e37e4e66;transform:rotate(8deg)}.art-note{position:absolute;padding:9px 12px;border:1px solid #ffffff1c;border-radius:10px;background:#22222bd9;box-shadow:0 8px 22px #0005;color:#e8ddd2;font-size:12px}.art-note span{color:var(--amber-light);font-size:16px;margin-right:5px}.note-one{right:4px;top:25px}.note-two{right:20px;bottom:25px}.section-block{margin-top:48px}.section-heading{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:17px}.section-heading h2{font-size:25px;letter-spacing:-.02em;margin:5px 0 0}.record-count,.section-note{color:var(--muted);font-size:12px}.overview-grid{display:grid;grid-template-columns:1.35fr .65fr;gap:16px}.risk-overview,.support-stat,.context-card,.insight-card,.model-strip{border:1px solid var(--line);background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 14px 35px #0003}.risk-overview{border-radius:20px;padding:25px}.card-topline,.meter-caption{display:flex;justify-content:space-between;align-items:center}.card-kicker{color:#d2c8bc;font-size:12px;text-transform:uppercase;letter-spacing:.1em}.live-dot{color:var(--mint);font-size:11px}.live-dot:before{content:"";display:inline-block;width:6px;height:6px;background:var(--mint);border-radius:50%;margin:0 6px 1px 0;box-shadow:0 0 10px var(--mint)}.risk-copy{display:flex;justify-content:space-between;gap:18px;align-items:end;margin:30px 0 22px}.risk-copy h3{font-size:21px;margin:0 0 7px}.risk-copy p{color:var(--muted);font-size:13px;max-width:420px;margin:0}.risk-copy>strong{color:var(--amber-light);font-size:58px;line-height:.9}.risk-meter{height:11px;border-radius:20px;background:#2b2d35;overflow:hidden}.risk-meter span{display:block;height:100%;min-width:2%;border-radius:20px;background:linear-gradient(90deg,var(--mint),var(--amber),var(--coral));box-shadow:0 0 16px #ef914b88}.meter-caption{font-size:11px;color:var(--muted);margin:9px 0 22px}.card-link{font-size:13px}.stat-stack{display:grid;gap:10px}.support-stat{display:flex;align-items:center;gap:13px;padding:17px;border-radius:16px}.stat-icon{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:#f09a4b1c;color:var(--amber-light);font-size:21px}.stat-icon.blue{background:#88b8d01c;color:var(--sky)}.stat-icon.green{background:#92c9ae1c;color:var(--mint)}.support-stat span:not(.stat-icon){display:block;color:var(--muted);font-size:11px}.support-stat strong{display:block;font-size:24px;margin-top:1px}.support-stat em{font-style:normal;font-size:12px;color:var(--muted);margin-left:2px}.support-stat small{color:#777b87;font-size:11px}.context-grid,.insight-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.context-card{min-height:190px;padding:21px;border-radius:18px;display:flex;flex-direction:column;gap:12px}.context-icon{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:#88b8d01c;color:var(--sky);font-size:21px}.context-icon.warm{background:#f39a4b1c;color:var(--amber-light)}.context-icon.green{background:#92c9ae1c;color:var(--mint)}.context-card h3{font-size:17px;margin:0 0 5px}.context-card p{color:var(--muted);font-size:13px;line-height:1.55;margin:0}.context-line{display:flex;justify-content:space-between;align-items:center;margin-top:auto;padding-top:12px;border-top:1px solid #ffffff0e;color:#858995;font-size:11px}.line-dots{display:flex;gap:4px}.line-dots i{width:7px;height:7px;background:var(--sky);border-radius:50%}.line-dots i:nth-child(4),.line-dots i:nth-child(5){background:#424651}.line-arrow{color:var(--amber-light);font-size:17px}.insight-card{border-radius:16px;padding:18px}.insight-mark{color:var(--amber-light);font-size:18px}.insight-card p{color:#d3ced0;font-size:13px;line-height:1.55;min-height:62px;margin:10px 0 15px}.insight-action{color:#888c97;font-size:11px}.model-strip{display:flex;align-items:center;gap:28px;border-radius:16px;padding:17px 20px}.model-strip>div:first-child{margin-right:auto}.model-strip strong{display:block;font-size:14px}.model-selected{color:var(--mint);font-size:11px}.model-score{border-left:1px solid var(--line);padding-left:22px}.model-score span{display:block;color:var(--muted);font-size:11px}.model-score strong{font-size:20px;color:#f2e6d8;margin-top:3px}.home-game-slot{margin-top:48px}.game-home,.account-panel{border-color:var(--line)!important;background:linear-gradient(145deg,#1a1c25,#15161d)!important}.section-head h2,.section-heading h2{color:var(--ink)}.stats,.choice{gap:14px}.stat,.panel{border-radius:16px}.insights{gap:14px}.error{border-radius:10px}@media(max-width:850px){.welcome{grid-template-columns:1fr;padding:32px}.welcome-art{position:absolute;right:-35px;bottom:-20px;opacity:.6;transform:scale(.8)}.overview-grid{grid-template-columns:1fr}.context-grid,.insight-grid{grid-template-columns:1fr 1fr}.model-strip{flex-wrap:wrap;gap:16px}.model-strip>div:first-child{width:100%}.model-score{padding-left:15px}.model-strip .text-action{margin-left:auto}}@media(max-width:620px){.top{display:block}.top>span{margin-top:10px}.main{padding-top:24px}.welcome{min-height:450px;padding:28px 23px}.welcome h1{font-size:38px}.welcome-actions{align-items:flex-start;flex-direction:column;gap:16px}.section-heading{align-items:flex-start;flex-direction:column;gap:7px}.context-grid,.insight-grid{grid-template-columns:1fr}.risk-copy>strong{font-size:46px}.risk-copy{align-items:start}.model-strip{align-items:flex-start;flex-direction:column}.model-score{border-left:0;border-top:1px solid var(--line);padding:10px 0 0;width:100%}.model-strip .text-action{margin-left:0}.game-home,.account-panel{grid-template-columns:1fr!important}}
</style></head>""", 1)
TEMPLATE = TEMPLATE.replace("</style>", "<style>.student-overview-grid{grid-template-columns:1fr 1fr}.student-overview-card,.active-students-card{min-height:238px;border:1px solid var(--line);border-radius:20px;padding:25px;background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 14px 35px #0003}.student-overview-card h3,.active-students-card h3{font-size:21px;margin:22px 0 0}.student-icon{display:grid;place-items:center;width:31px;height:31px;border-radius:10px;background:#f39a4b1c;color:var(--amber-light);font-size:19px}.student-counts{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:28px}.student-counts>div+div{border-left:1px solid var(--line);padding-left:18px}.student-counts strong{display:block;color:var(--amber-light);font-size:42px;line-height:1}.student-counts span{display:block;margin-top:9px;color:#e3d8cf;font-size:12px;font-weight:bold}.student-counts small{display:block;margin-top:4px;color:var(--muted);font-size:11px}.active-heading{display:flex;align-items:end;justify-content:space-between;gap:16px;margin:28px 0 23px}.active-heading h3{margin:0}.active-heading p{color:var(--muted);font-size:12px;margin:7px 0 0}.active-heading>strong{color:var(--mint);font-size:52px;line-height:.9}.active-progress{height:12px;border-radius:20px;background:#2b2d35;overflow:hidden}.active-progress span{display:block;height:100%;min-width:0;border-radius:20px;background:linear-gradient(90deg,var(--mint),var(--amber));box-shadow:0 0 14px #92c9ae66;transition:width .5s ease}.active-caption{display:flex;justify-content:space-between;margin-top:10px;color:var(--muted);font-size:11px}@media(max-width:850px){.student-overview-grid{grid-template-columns:1fr}}@media(max-width:620px){.student-overview-card,.active-students-card{padding:21px}.student-counts{gap:12px}.student-counts>div+div{padding-left:12px}.student-counts strong{font-size:34px}}</style>")
TEMPLATE = TEMPLATE.replace("</style>", "<style>.classes-subtitle{margin:7px 0 0;color:var(--muted);font-size:13px}.classes-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:13px}.class-card{min-height:142px;display:flex;flex-direction:column;align-items:flex-start;justify-content:space-between;padding:18px;border:1px solid var(--line);border-radius:17px;background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 12px 28px #0003;color:var(--ink);text-decoration:none;transition:transform .2s,border-color .2s,box-shadow .2s}.class-card:hover,.class-card:focus-visible{border-color:#f39a4b99;box-shadow:0 16px 32px #0005,0 0 20px #f39a4b12;transform:translateY(-4px)}.class-card:focus-visible{outline:3px solid #ffc47766;outline-offset:3px}.class-card-icon{display:grid;place-items:center;width:30px;height:30px;border-radius:10px;background:#f39a4b1c;color:var(--amber-light);font-size:15px}.class-card-name{font-size:20px;font-weight:bold;letter-spacing:-.02em}.class-card-select{display:grid;gap:7px;width:100%;margin-top:8px;color:var(--muted);font-size:11px}.class-card-select select{width:100%;padding:8px 9px;border:1px solid var(--line);border-radius:9px;background:#0e1017;color:var(--ink);font:inherit}.class-card-action{width:100%;padding-top:10px;border:0;background:none;color:var(--muted);font-size:11px;text-align:left;cursor:pointer}.class-card-action span{padding-left:5px;color:var(--amber-light);font-size:15px}.add-student-card{cursor:default}.add-student-card:hover,.add-student-card:focus-visible{border-color:#ffc47766}.class-card-action:hover{color:var(--amber-light)}@media(max-width:1050px){.classes-grid{grid-template-columns:repeat(4,minmax(0,1fr))}}@media(max-width:760px){.classes-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:520px){.classes-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.class-card{min-height:125px;padding:14px}.class-card-name{font-size:17px}} </style>")

home_game_start = TEMPLATE.find('<div class="home-game-slot"></div>')
home_game_end = home_game_start + len('<div class="home-game-slot"></div>')
if home_game_start != -1 and home_game_end != -1:
    MANAGEMENT_HOME_BLOCK = """<section class="section-block management-section"><div class="section-heading"><div><span class="eyebrow">School operations</span><h2>School Management</h2><p class="management-subtitle">Keep essential school workflows organized in one place.</p></div></div><div class="management-grid"><a class="management-card" href="/management/fees"><span class="management-icon fee-icon" aria-hidden="true">$</span><span class="management-card-title">Fee Management</span><span class="management-description">Manage student fees, payments, pending dues and fee history.</span><span class="management-action">Open <span aria-hidden="true">&#8594;</span></span></a><a class="management-card" href="/management/calendar"><span class="management-icon calendar-icon" aria-hidden="true">&#9633;</span><span class="management-card-title">Academic Calendar</span><span class="management-description">Manage academic events, holidays, examinations and important school dates.</span><span class="management-action">Open <span aria-hidden="true">&#8594;</span></span></a><a class="management-card" href="/management/examinations"><span class="management-icon exam-icon" aria-hidden="true">&#9998;</span><span class="management-card-title">Examination</span><span class="management-description">Manage exams, subjects, marks, results and student performance.</span><span class="management-action">Open <span aria-hidden="true">&#8594;</span></span></a><a class="management-card" href="/management/reports"><span class="management-icon report-icon" aria-hidden="true">&#8599;</span><span class="management-card-title">Reports</span><span class="management-description">View attendance, academic, examination, fee and student performance reports.</span><span class="management-action">View <span aria-hidden="true">&#8594;</span></span></a></div></section>"""
    MANAGEMENT_HOME_BLOCK = MANAGEMENT_HOME_BLOCK.replace(
        "</div></section>",
        "</div></section><section class=\"section-block memories-section\"><div class=\"section-heading\"><div><span class=\"eyebrow\">School life</span><h2>School Memories</h2><p class=\"memories-subtitle\">Keep a visual record of moments from your school community.</p></div><a class=\"button memory-upload-button\" href=\"/?memory_add=1#school-memories\">+ Upload Photo</a></div>{% if memory_message %}<div class=\"message success\">{{ memory_message }}</div>{% endif %}{% if memory_error %}<div class=\"message error\">{{ memory_error }}</div>{% endif %}{% if memory_form_open %}<form id=\"memory-form\" class=\"memory-form\" method=\"post\" action=\"/school-memories\" enctype=\"multipart/form-data\"><label>Photo<input type=\"file\" name=\"memory_photo\" accept=\".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp\" required></label><label>Title <span>(optional)</span><input name=\"title\" maxlength=\"120\"></label><label>Description <span>(optional)</span><textarea name=\"description\" maxlength=\"500\"></textarea><div class=\"memory-form-actions\"><button class=\"button\" type=\"submit\">Save Photo</button><a class=\"button secondary\" href=\"/?view=home#school-memories\">Cancel</a></div></form>{% endif %}{% if memory_photos %}<div class=\"memory-slider\" id=\"school-memories\"><button class=\"memory-nav previous\" type=\"button\" aria-label=\"Previous memory\">&#8592;</button><div class=\"memory-track\">{% for photo in memory_photos %}<article class=\"memory-slide{% if loop.first %} is-active{% endif %}\" data-memory-slide><img src=\"{{ url_for('school_memory_photo', memory_id=photo.id) }}\" alt=\"{{ photo.title or 'School memory' }}\" data-memory-image><div class=\"memory-caption\"><strong>{{ photo.title or 'School memory' }}</strong>{% if photo.description %}<p>{{ photo.description }}</p>{% endif %}<small>{{ photo.uploaded_at }} · {{ photo.uploaded_by }}</small>{% if can_delete_memories %}<form method=\"post\" action=\"{{ url_for('delete_school_memory', memory_id=photo.id) }}\"><button class=\"memory-delete\" type=\"submit\">Delete</button></form>{% endif %}</div></article>{% endfor %}</div><button class=\"memory-nav next\" type=\"button\" aria-label=\"Next memory\">&#8594;</button></div><div class=\"memory-lightbox\" hidden><button type=\"button\" class=\"memory-lightbox-close\" aria-label=\"Close preview\">&times;</button><img alt=\"Expanded school memory\"></div><script>(function(){const slides=[...document.querySelectorAll('[data-memory-slide]')];let index=0;const show=function(next){if(!slides.length)return;index=(next+slides.length)%slides.length;slides.forEach((slide,i)=>slide.classList.toggle('is-active',i===index));};document.querySelector('.memory-nav.previous')?.addEventListener('click',()=>show(index-1));document.querySelector('.memory-nav.next')?.addEventListener('click',()=>show(index+1));const lightbox=document.querySelector('.memory-lightbox');const lightboxImage=lightbox?.querySelector('img');document.querySelectorAll('[data-memory-image]').forEach(image=>image.addEventListener('click',()=>{lightboxImage.src=image.src;lightbox.hidden=false;}));document.querySelector('.memory-lightbox-close')?.addEventListener('click',()=>{lightbox.hidden=true;});lightbox?.addEventListener('click',event=>{if(event.target===lightbox)lightbox.hidden=true;});})();</script></section>",
    )
    MANAGEMENT_HOME_BLOCK = MANAGEMENT_HOME_BLOCK.replace(
        "</script></section>",
        "</script>{% else %}<div class=\"memory-empty\"><p>No school memories uploaded yet.</p><a class=\"button\" href=\"/?memory_add=1#memory-form\">Upload Photo</a></div>{% endif %}</section>",
    )
    TEMPLATE = TEMPLATE[:home_game_start] + MANAGEMENT_HOME_BLOCK + TEMPLATE[home_game_end:]

TEMPLATE = TEMPLATE.replace("</style>", "<style>.game-home,.account-panel{display:grid;gap:16px;align-items:center}.game-home{grid-template-columns:1fr auto auto auto}.game-stats{display:flex;gap:18px}.game-stats span{color:var(--muted);font-size:12px}.game-stats b{display:block;color:var(--ink);font-size:18px}.account-panel{grid-template-columns:1fr 2fr}.account-forms{display:grid;grid-template-columns:1fr 1fr;gap:12px}.account-forms form{display:grid;gap:8px}.account-forms input{font:inherit;padding:10px;border:1px solid #bdcbd1;border-radius:4px}.secondary{background:#657783}.game-disclaimer{color:var(--muted);font-size:12px}.game-page{max-width:760px;margin:auto}.game-question{font-size:24px;margin:18px 0}.game-options{display:grid;gap:10px}.game-options button{background:#f4f6f7;border:1px solid var(--line);padding:13px;text-align:left;border-radius:5px;cursor:pointer;font:inherit}.game-options button:hover{border-color:var(--teal);background:#eaf6f8}.history{width:100%;border-collapse:collapse}.history th,.history td{text-align:left;padding:10px;border-bottom:1px solid var(--line)}.score-bars{display:flex;align-items:end;gap:8px;height:130px;padding:15px 0}.score-bar{flex:1;background:var(--teal);min-height:4px;border-radius:4px 4px 0 0;position:relative}.score-bar span{position:absolute;bottom:-22px;width:100%;text-align:center;font-size:11px;color:var(--muted)}@media(max-width:850px){.game-home,.account-panel,.account-forms{grid-template-columns:1fr}.game-stats{flex-wrap:wrap}}</style>")

TEMPLATE = TEMPLATE.replace("</style>", "<style>/* Fee Analysis visual alignment with Student History */.fee-analysis-heading h2,.fee-selector-copy h3,.fee-table-heading h3,.fee-empty-state h3{color:var(--ink)}.fee-selector-panel,.fee-table-panel,.fee-empty-state,.fee-analysis-stats .fee-stat{background:linear-gradient(145deg,#1a1c25ef,#15161def)!important;border:1px solid #ffffff17!important;box-shadow:0 20px 55px #0008!important}.fee-selector-panel,.fee-table-panel,.fee-empty-state{border-radius:18px!important}.fee-selector-panel{backdrop-filter:blur(12px)}.fee-selector-copy p,.fee-table-heading p,.fee-empty-state p{color:var(--muted)!important}.fee-analysis-filters label,.fee-table-controls label{color:#d8d1cc}.fee-analysis-filters select,.fee-table-controls input,.fee-table-controls select{background:#0e1017!important;color:var(--ink)!important;border-color:var(--line)!important}.fee-analysis-filters select:hover,.fee-table-controls input:hover,.fee-table-controls select:hover{border-color:#f39a4b99!important}.fee-analysis-stats .fee-stat{border-radius:16px!important;color:var(--ink)}.fee-analysis-stats .fee-stat span{color:var(--muted)}.fee-analysis-stats .fee-stat strong{color:var(--amber-light)}.fee-analysis-stats .fee-stat.paid strong{color:var(--mint)}.fee-analysis-stats .fee-stat.free strong{color:var(--sky)}.fee-table-panel{overflow:hidden}.fee-table-heading,.fee-table-controls{border-color:#ffffff12!important}.fee-analysis-table{color:var(--ink)}.fee-analysis-table th{background:#ffffff08!important;color:var(--amber-light)!important;border-color:#ffffff12!important}.fee-analysis-table td{color:#e8e2dd!important;border-color:#ffffff12!important}.fee-analysis-table tbody tr:hover{background:#ffffff0b!important}.fee-table-panel .button{background:linear-gradient(135deg,var(--amber),var(--coral))!important;color:#1a120e!important;box-shadow:0 10px 24px #e57c3b3d}.fee-table-panel .button:hover{background:linear-gradient(135deg,#ffc477,#ef8961)!important;box-shadow:0 14px 28px #e57c3b55}.fee-status-badge.pending{color:var(--amber-light)!important}.receipt-link{color:var(--amber-light)!important}.fee-empty-state{margin-top:22px}</style>")

TEMPLATE = TEMPLATE.replace(
    "</body></html>",
    """<script>(function(){const form=document.getElementById("memory-form");if(!form)return;const input=form.querySelector('input[name="memory_photo"]');const submit=form.querySelector('button[type="submit"]');const maxBytes=5*1024*1024;const maxSide=2500;const setMessage=function(text,isError){let message=form.querySelector("[data-memory-upload-message]");if(!message){message=document.createElement("small");message.setAttribute("data-memory-upload-message","");message.style.display="block";message.style.marginTop="8px";form.querySelector(".memory-form-actions").before(message);}message.textContent=text;message.style.color=isError?"#ff9d8a":"#ffc477";};const blobFromCanvas=function(canvas,quality){return new Promise(function(resolve,reject){if(!canvas.toBlob){reject(new Error("Your browser cannot compress this image."));return;}canvas.toBlob(function(blob){if(blob)resolve(blob);else reject(new Error("The image could not be compressed."));},"image/jpeg",quality);});};const compress=function(file){return new Promise(function(resolve,reject){const url=URL.createObjectURL(file);const image=new Image();image.onload=async function(){try{const scale=Math.min(1,maxSide/Math.max(image.naturalWidth,image.naturalHeight));const canvas=document.createElement("canvas");canvas.width=Math.max(1,Math.round(image.naturalWidth*scale));canvas.height=Math.max(1,Math.round(image.naturalHeight*scale));const context=canvas.getContext("2d");if(!context)throw new Error("Your browser cannot prepare this image.");context.drawImage(image,0,0,canvas.width,canvas.height);const qualities=[0.92,0.86,0.8,0.74,0.68,0.62,0.56,0.5,0.44,0.38];for(const quality of qualities){const blob=await blobFromCanvas(canvas,quality);if(blob.size<=maxBytes){resolve(new File([blob],(file.name.replace(/\\.[^.]+$/,"")||"school-memory")+".jpg",{type:"image/jpeg",lastModified:Date.now()}));return;}}throw new Error("This image could not be reduced below 5 MB.");}catch(error){reject(error);}finally{URL.revokeObjectURL(url);}};image.onerror=function(){URL.revokeObjectURL(url);reject(new Error("The selected file is not a readable image."));};image.src=url;});};form.addEventListener("submit",async function(event){if(form.dataset.compressed==="true")return;event.preventDefault();if(!input.files||!input.files.length){setMessage("Please select a photo.",true);return;}const original=input.files[0];submit.disabled=true;input.disabled=true;setMessage("Compressing image...",false);try{const compressed=await compress(original);const transfer=new DataTransfer();transfer.items.add(compressed);input.files=transfer.files;form.dataset.compressed="true";form.submit();}catch(error){input.value="";setMessage(error.message||"The image could not be compressed. Please choose another photo.",true);submit.disabled=false;input.disabled=false;}});})();</script></body></html>""",
    1,
)

MANAGEMENT_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ title }} | Student Support Dashboard</title><style>:root{--ink:#f7f1e8;--muted:#9ca0ab;--line:#30333e;--amber:#f39a4b;--light:#ffc477;--coral:#e87551}*{box-sizing:border-box}body{min-height:100vh;margin:0;background:radial-gradient(circle at 80% 0%,#382317 0,#15151d 30%,#0b0c12 70%);color:var(--ink);font:14px/1.5 "Trebuchet MS","Segoe UI",sans-serif}.top{display:flex;justify-content:space-between;align-items:center;padding:20px max(22px,5vw);border-bottom:1px solid #ffffff12;background:#0d0e14cc}.top a{color:var(--light);text-decoration:none}.main{max-width:900px;margin:auto;padding:70px 22px}.panel{padding:38px;border:1px solid var(--line);border-radius:22px;background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 20px 55px #0005}.eyebrow{color:var(--light);font-size:11px;font-weight:bold;letter-spacing:.13em;text-transform:uppercase}.panel h1{margin:12px 0;font-size:clamp(30px,5vw,48px)}.panel p{max-width:620px;color:var(--muted);font-size:15px}.class-list{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:28px}.class-item{display:flex;align-items:center;justify-content:space-between;min-height:62px;padding:15px 17px;border:1px solid var(--line);border-radius:14px;background:linear-gradient(145deg,#1d202a,#15161d);color:var(--ink);text-decoration:none;font-weight:bold;transition:transform .2s,border-color .2s,box-shadow .2s}.class-item:hover,.class-item:focus-visible{border-color:#f39a4b99;box-shadow:0 12px 24px #0004;transform:translateY(-2px)}.class-item:focus-visible{outline:3px solid #ffc47766;outline-offset:3px}.class-item span{color:var(--light);font-size:18px}.actions{display:flex;gap:12px;align-items:center;margin-top:24px}.button{display:inline-block;border:0;border-radius:10px;padding:11px 16px;background:linear-gradient(135deg,var(--amber),var(--coral));color:#1a120e;font:inherit;font-weight:bold;cursor:pointer;text-decoration:none}.button.secondary{background:#282b35;color:var(--light);text-decoration:none}@media(max-width:650px){.class-list{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){.main{padding-top:35px}.panel{padding:26px}.class-list{grid-template-columns:1fr}}</style></head><body><header class="top"><strong>Student Support Dashboard</strong><span>{{ school_name }} &nbsp; <a href="/school-logout">School Logout</a></span></header><main class="main"><section class="panel"><span class="eyebrow">School management</span><h1>{{ title }}</h1><p>{{ description }}</p>{% if title == 'Fee Management' %}<div class="class-list" aria-label="Classes">{% for class_name in classes %}<a class="class-item" href="#"><span>{{ class_name }}</span><span aria-hidden="true">&#8594;</span></a>{% endfor %}</div><div class="actions"><a class="button secondary" href="/management/fees/download">Download Receipt Excel</a></div>{% else %}<p>This workspace is ready for school-managed records. No management records are available yet.</p>{% endif %}<a class="back" href="/?view=home">&#8592; Back to Dashboard</a></section></main></body></html>
"""
MANAGEMENT_TEMPLATE = MANAGEMENT_TEMPLATE.replace('<a class="class-item" href="#">', '<a class="class-item" href="/management/fees?class={{ class_name|urlencode }}">')
MANAGEMENT_TEMPLATE = MANAGEMENT_TEMPLATE.replace(
    "</style></head>",
    "<style>.report-toolbar{display:flex;justify-content:space-between;align-items:center;gap:14px}.report-form{display:grid;gap:14px;margin-top:18px;padding:18px;border:1px solid var(--line);border-radius:14px;background:#0e1017}.report-form label{display:grid;gap:6px;color:#d8d1cc;font-size:12px;font-weight:bold}.report-form input,.report-form textarea{width:100%;padding:10px 11px;border:1px solid var(--line);border-radius:9px;background:#151b24;color:var(--ink);font:inherit}.report-form textarea{min-height:72px;resize:vertical}.report-form input[type=file]::file-selector-button{background:#282b35;color:var(--light);border:1px solid var(--line);border-radius:6px;padding:7px 10px;margin-right:10px;cursor:pointer}.report-list{display:grid;gap:10px;margin-top:18px}.report-item{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:center;padding:15px;border:1px solid var(--line);border-radius:12px;background:#ffffff05}.report-item strong,.report-item small{display:block}.report-item small{color:var(--muted);margin-top:4px}.report-actions{display:flex;gap:12px;align-items:center}.report-actions form{margin:0}.report-link{color:var(--light);text-decoration:none;font-weight:bold}.report-delete{border:0;background:none;color:#ffb7a5;cursor:pointer;font:inherit;padding:0}@media(max-width:620px){.report-toolbar,.report-item{display:block}.report-actions{margin-top:12px;flex-wrap:wrap}}</style></head>",
    1,
)
MANAGEMENT_TEMPLATE = MANAGEMENT_TEMPLATE.replace(
    "<a class=\"button\" href=\"/management/reports?add=1#report-form\">+ Add Report</a></div>{% if report_form_open %}",
    "<a class=\"button\" href=\"/management/reports?add=1#report-form\">+ Add Report</a></div>{% if message %}<div class=\"message\">{{ message }}</div>{% endif %}{% if error %}<div class=\"error\">{{ error }}</div>{% endif %}{% if report_form_open %}",
)
MANAGEMENT_TEMPLATE = MANAGEMENT_TEMPLATE.replace(
    "{% else %}<p>This workspace is ready for school-managed records. No management records are available yet.</p>",
    """{% elif title == 'Reports' %}<section class=\"panel\"><div class=\"report-toolbar\"><div><h2>Reports &amp; Documents</h2><p>Store school reports and documents for authorized management users.</p></div><a class=\"button\" href=\"/management/reports?add=1#report-form\">+ Add Report</a></div>{% if report_form_open %}<form id=\"report-form\" class=\"report-form\" method=\"post\" action=\"/management/reports\" enctype=\"multipart/form-data\"><label for=\"report_name\">Report name<input id=\"report_name\" name=\"report_name\" maxlength=\"120\" required></label><label for=\"report_file\">File<input id=\"report_file\" name=\"report_file\" type=\"file\" accept=\".pdf,.docx,.xlsx,.jpg,.jpeg,.png\" required><small>Allowed: PDF, DOCX, XLSX, JPG, JPEG, PNG. Maximum size: 10 MB.</small></label><label for=\"report_description\">Description <span>(optional)</span><textarea id=\"report_description\" name=\"description\" maxlength=\"500\"></textarea></label><div class=\"actions\"><button class=\"button\" type=\"submit\">Add Report</button><a class=\"button secondary\" href=\"/management/reports\">Cancel</a></div></form>{% endif %}</section><section class=\"panel\"><h2>Saved Reports</h2>{% if reports %}<div class=\"report-list\">{% for report in reports %}<article class=\"report-item\"><div><strong>{{ report.report_name }}</strong><small>{{ report.file_extension|upper }} · Uploaded {{ report.uploaded_at }} · By {{ report.uploaded_by }}{% if report.description %}<br>{{ report.description }}{% endif %}</small></div><div class=\"report-actions\"><a class=\"report-link\" href=\"{{ url_for('download_report_file', report_id=report.id) }}\">View / Download</a><form method=\"post\" action=\"{{ url_for('delete_report_file', report_id=report.id) }}\" onsubmit=\"return confirm('Delete this report?');\"><button class=\"report-delete\" type=\"submit\">Delete</button></form></div></article>{% endfor %}</div>{% else %}<p>This school has no saved reports yet.</p>{% endif %}</section>{% else %}<p>This workspace is ready for school-managed records. No management records are available yet.</p>""",
)

FEE_CLASS_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fee Management - {{ class_name }} | Student Support Dashboard</title><style>:root{--ink:#f7f1e8;--muted:#9ca0ab;--line:#30333e;--amber:#f39a4b;--light:#ffc477;--coral:#e87551;--mint:#92c9ae}*{box-sizing:border-box}body{min-height:100vh;margin:0;background:radial-gradient(circle at 80% 0%,#382317 0,#15151d 30%,#0b0c12 70%);color:var(--ink);font:14px/1.5 "Trebuchet MS","Segoe UI",sans-serif}.top{display:flex;justify-content:space-between;align-items:center;padding:20px max(22px,5vw);border-bottom:1px solid #ffffff12;background:#0d0e14cc}.top a,.back{color:var(--light);text-decoration:none}.main{max-width:1240px;margin:auto;padding:45px max(22px,5vw) 70px}.heading{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px}.eyebrow{color:var(--light);font-size:11px;font-weight:bold;letter-spacing:.13em;text-transform:uppercase}.heading h1{margin:12px 0 0;font-size:clamp(30px,5vw,48px)}.panel{padding:24px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 20px 55px #0005}.table-wrap{overflow:auto}.fee-table{width:100%;border-collapse:collapse;min-width:1120px}.fee-table th,.fee-table td{text-align:left;padding:11px 9px;border-bottom:1px solid #ffffff12;vertical-align:top}.fee-table th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}.fee-cell{display:grid;gap:7px;min-width:112px}.fee-cell select,.fee-cell input[type=file]{width:112px;padding:6px;border:1px solid var(--line);border-radius:8px;background:#0e1017;color:var(--ink);font:11px inherit}.fee-cell input[type=file]{display:none}.fee-cell input[type=file].is-visible{display:block}.fee-status{font-size:11px;color:var(--muted)}.fee-status.locked{color:var(--mint)}.button{border:0;border-radius:8px;padding:7px 9px;background:linear-gradient(135deg,var(--amber),var(--coral));color:#1a120e;font:11px inherit;font-weight:bold;cursor:pointer}.button.secondary{background:#282b35;color:var(--light);text-decoration:none;font-size:13px;padding:10px 14px}.message{padding:11px 13px;border-radius:9px;margin-bottom:15px;background:#1b3b31;color:#b7f0ce}.review-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px 22px;margin:20px 0}.review-grid strong{display:block;color:var(--muted);font-size:11px;text-transform:uppercase}.review-photo{max-width:420px;max-height:280px;border:1px solid var(--line);border-radius:10px;margin:15px 0}.actions{display:flex;gap:10px;align-items:center}@media(max-width:650px){.top{display:block}.top span{display:block;margin-top:7px}.heading{align-items:flex-start;flex-direction:column}.main{padding-top:35px}.panel{padding:17px}.review-grid{grid-template-columns:1fr}}</style></head><body><header class="top"><strong>Student Support Dashboard</strong><span>{{ school_name }} &nbsp; <a href="/school-logout">School Logout</a></span></header><main class="main"><div class="heading"><div><span class="eyebrow">Fee management</span><h1>{{ class_name }}</h1></div><a class="back" href="/management/fees">&#8592; All Classes</a></div>{% if message %}<div class="message">{{ message }}</div>{% endif %}{% if review %}<section class="panel"><h2>Final Check</h2><div class="review-grid"><div><strong>Student Name</strong>{{ review.student_name }}</div><div><strong>Class</strong>{{ review.class_name }}</div><div><strong>UID Number</strong>{{ review.uid_number }}</div><div><strong>Roll Number</strong>{{ review.roll_number }}</div><div><strong>Month</strong>{{ review.month }}</div><div><strong>Fee Status</strong>{{ review.status }}</div></div>{% if review.receipt_data %}<img class="review-photo" src="data:{{ review.receipt_mime }};base64,{{ review.receipt_data }}" alt="Receipt photo preview">{% endif %}<form method="post" action="/management/fees?class={{ class_name|urlencode }}"><input type="hidden" name="fee_action" value="confirm"><input type="hidden" name="review_token" value="{{ review.token }}"><div class="actions"><button class="button" type="submit">Confirm and Lock</button><a class="button secondary" href="/management/fees?class={{ class_name|urlencode }}">Cancel</a></div></form></section>{% else %}<section class="panel"><div class="table-wrap"><table class="fee-table"><thead><tr><th>Roll Number</th><th>Student Name</th>{% for month in months %}<th>{{ month }}</th>{% endfor %}</tr></thead><tbody>{% for student in students %}<tr><td>{{ student.roll_number }}</td><td>{{ student.student_name }}</td>{% for month in months %}{% set fee = fee_statuses[student.id][month] %}<td><div class="fee-cell"><span class="fee-status{% if fee.locked %} locked{% endif %}">{{ fee.status }}{% if fee.locked %} · Locked{% endif %}</span>{% if not fee.locked %}<form method="post" enctype="multipart/form-data" action="/management/fees?class={{ class_name|urlencode }}"><input type="hidden" name="fee_action" value="review"><input type="hidden" name="student_id" value="{{ student.id }}"><input type="hidden" name="fee_month" value="{{ month }}"><select name="status" data-fee-status><option value="Pending" {% if fee.status == 'Pending' %}selected{% endif %}>Pending</option><option value="Paid">Paid</option><option value="Free">Free</option></select><input type="file" name="receipt_photo" accept="image/*" data-receipt-input><button class="button" type="submit">Final Check</button></form>{% endif %}</div></td>{% endfor %}</tr>{% else %}<tr><td colspan="14">No students are registered in {{ class_name }} yet.</td></tr>{% endfor %}</tbody></table></div><p class="actions"><a class="button secondary" href="/management/fees/download?class={{ class_name|urlencode }}">Download Fee Records</a></p></section>{% endif %}</main><script>document.querySelectorAll('[data-fee-status]').forEach(function(select){const form=select.form;const input=form.querySelector('[data-receipt-input]');const btn=form.querySelector('button[type="submit"]');function update(){const isPaid=select.value==='Paid';input.classList.toggle('is-visible',isPaid);input.required=isPaid;if(btn){btn.textContent=(select.value==='Pending')?'Save':'Final Check';}}select.addEventListener('change',update);update();});</script></body></html>
"""

STUDENT_MANAGEMENT_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Student Management - {{ class_name }}</title><style>:root{--ink:#f7f1e8;--muted:#9ca0ab;--line:#30333e;--amber:#f39a4b;--light:#ffc477;--coral:#e87551;--mint:#92c9ae}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 80% 0%,#382317 0,#15151d 30%,#0b0c12 70%);color:var(--ink);font:14px/1.5 "Trebuchet MS","Segoe UI",sans-serif}.top{display:flex;justify-content:space-between;align-items:center;padding:18px max(22px,5vw);border-bottom:1px solid #ffffff12;background:#0d0e14cc}.top a,.back{color:var(--light);text-decoration:none}.top span{color:var(--muted);font-size:12px}.main{max-width:1240px;margin:auto;padding:38px max(22px,5vw) 70px}.heading{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px}.eyebrow{color:var(--light);font-size:11px;font-weight:bold;letter-spacing:.13em;text-transform:uppercase}.heading h1{font-size:clamp(30px,4vw,46px);margin:8px 0 0}.count{color:var(--muted);font-size:13px}.panel{border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 14px 35px #0004;padding:22px;margin-bottom:18px}.panel h2{font-size:20px;margin:0 0 18px}.form-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.field{display:grid;gap:6px}.field.wide{grid-column:span 2}.field.full{grid-column:1/-1}.field label,.field>span{color:#d8d1cc;font-size:12px;font-weight:bold}.field input,.field textarea,.field select{width:100%;padding:10px 11px;border:1px solid var(--line);border-radius:9px;background:#0e1017;color:var(--ink);font:inherit;outline:0}.field textarea{min-height:70px;resize:vertical}.field input:focus,.field textarea:focus,.field select:focus{border-color:var(--amber);box-shadow:0 0 0 3px #f39a4b1c}.check-row{display:flex;align-items:center;gap:9px;color:#d8d1cc;font-size:13px}.check-row input{accent-color:var(--amber)}.siblings{display:grid;gap:9px}.sibling-row{display:grid;grid-template-columns:1fr 160px auto;gap:9px}.sibling-row button{border:1px solid #ffffff1c;border-radius:8px;background:#ffffff08;color:var(--light);cursor:pointer}.button{border:0;border-radius:10px;padding:11px 16px;background:linear-gradient(135deg,var(--amber),var(--coral));color:#1a120e;font:inherit;font-weight:bold;cursor:pointer}.button.secondary{background:#282b35;color:var(--light);text-decoration:none}.actions{display:flex;gap:10px;align-items:center;margin-top:3px}.message{padding:11px 13px;border-radius:9px;margin-bottom:15px}.success{background:#1b3b31;color:#b7f0ce}.error{background:#512c2859;color:#ffc5b3;border:1px solid #e8755159}.toolbar{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-bottom:15px}.search{display:flex;gap:8px;width:min(430px,100%)}.search input{flex:1;padding:10px 12px;border:1px solid var(--line);border-radius:9px;background:#0e1017;color:var(--ink);font:inherit}.table-wrap{overflow:auto}.student-table{width:100%;border-collapse:collapse;min-width:850px}.student-table th,.student-table td{text-align:left;padding:12px 10px;border-bottom:1px solid #ffffff0e;vertical-align:top}.student-table th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}.student-table td{color:#e5ddd5}.student-table small{display:block;color:var(--muted);margin-top:2px}.badge{display:inline-block;padding:4px 8px;border-radius:20px;background:#92c9ae1c;color:var(--mint);font-size:11px}.row-actions{display:flex;gap:7px}.row-actions form{display:inline}.link-button{border:0;background:none;color:var(--light);cursor:pointer;font:inherit;padding:0}.empty{text-align:center;color:var(--muted);padding:26px}.hidden{display:none!important}@media(max-width:950px){.form-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:560px){.top{display:block}.top span{display:block;margin-top:7px}.heading{align-items:flex-start;flex-direction:column}.form-grid{grid-template-columns:1fr}.field.wide,.field.full{grid-column:auto}.sibling-row{grid-template-columns:1fr 1fr}.sibling-row button{grid-column:1/-1;padding:8px}.toolbar{align-items:stretch;flex-direction:column}.search{width:100%}.panel{padding:17px}}</style></head><body><header class="top"><strong>Student Support Dashboard</strong><span>{{ school_name }} &nbsp; <a href="/school-logout">School Logout</a></span></header><main class="main"><div class="heading"><div><span class="eyebrow">Student management</span><h1>Student Management - {{ class_name }}</h1></div><div><span class="count">{{ students|length }} student{% if students|length != 1 %}s{% endif %}</span><br><a class="back" href="/?view=home">&#8592; Back to Dashboard</a></div></div>{% if message %}<div class="message success">{{ message }}</div>{% endif %}{% if error %}<div class="message error">{{ error }}</div>{% endif %}<section class="panel"><h2>{{ 'Edit Student' if edit_student else 'Add Student' }}</h2><form method="post" action="/students/{{ class_name|urlencode }}"><input type="hidden" name="student_id" value="{{ edit_student.id if edit_student else '' }}"><div class="form-grid"><div class="field"><label for="roll_number">Roll Number</label><input id="roll_number" name="roll_number" value="{{ form.roll_number }}" required inputmode="numeric" pattern="[0-9]{1,4}"></div><div class="field"><label for="uid_number">UID Number</label><input id="uid_number" name="uid_number" value="{{ form.uid_number }}" required maxlength="40"></div><div class="field wide"><label for="student_name">Student Name</label><input id="student_name" name="student_name" value="{{ form.student_name }}" required></div><div class="field"><label for="father_name">Father's Name</label><input id="father_name" name="father_name" value="{{ form.father_name }}" required></div><div class="field"><label for="mother_name">Mother's Name</label><input id="mother_name" name="mother_name" value="{{ form.mother_name }}" required></div><div class="field"><label for="parent_mobile">Parent's Mobile Number</label><input id="parent_mobile" name="parent_mobile" value="{{ form.parent_mobile }}" required inputmode="tel"></div><div class="field"><label for="second_mobile">Second Mobile Number</label><input id="second_mobile" name="second_mobile" value="{{ form.second_mobile }}" inputmode="tel"></div><div class="field full"><label for="address">Complete Address</label><textarea id="address" name="address" required>{{ form.address }}</textarea></div><div class="field full"><label class="check-row"><input type="checkbox" name="transport_student" {% if form.transport_student %}checked{% endif %}> Student uses school transport</label></div><div class="field full"><label class="check-row"><input id="has-sibling" type="checkbox" name="has_sibling" {% if siblings %}checked{% endif %}> Sibling studying in this school</label><div id="siblings" class="siblings {% if not siblings %}hidden{% endif %}">{% for sibling in siblings %}<div class="sibling-row"><input name="sibling_name" value="{{ sibling.name }}" placeholder="Sibling Name"><select name="sibling_class"><option value="">Sibling Class</option>{% for option in classes %}<option {% if sibling.class_name == option %}selected{% endif %}>{{ option }}</option>{% endfor %}</select><button type="button" class="remove-sibling">Remove</button></div>{% endfor %}{% if not siblings %}<div class="sibling-row"><input name="sibling_name" placeholder="Sibling Name"><select name="sibling_class"><option value="">Sibling Class</option>{% for option in classes %}<option>{{ option }}</option>{% endfor %}</select><button type="button" class="remove-sibling">Remove</button></div>{% endif %}<button type="button" id="add-sibling" class="button secondary">+ Add another sibling</button></div></div><div class="actions"><button class="button" type="submit">{{ 'Update Student' if edit_student else 'Save Student' }}</button>{% if edit_student %}<a class="button secondary" href="/students/{{ class_name|urlencode }}">Cancel</a>{% endif %}</div></div></form></section><section class="panel"><div class="toolbar"><h2>Students in {{ class_name }}</h2><form class="search" method="get"><input name="q" value="{{ query }}" placeholder="Search roll number, UID or name"><button class="button" type="submit">Search</button></form></div>{% if students %}<div class="table-wrap"><table class="student-table"><thead><tr><th>Roll Number</th><th>UID</th><th>Student Name</th><th>Father's Name</th><th>Mother's Name</th><th>Parent Mobile</th><th>Transport</th><th>Sibling</th><th>Actions</th></tr></thead><tbody>{% for student in students %}<tr><td>{{ student.roll_number }}</td><td>{{ student.uid_number }}</td><td>{{ student.student_name }}</td><td>{{ student.father_name }}</td><td>{{ student.mother_name }}</td><td>{{ student.parent_mobile }}{% if student.second_mobile %}<small>{{ student.second_mobile }}</small>{% endif %}</td><td>{% if student.transport_student %}<span class="badge">Yes</span>{% else %}No{% endif %}</td><td>{% if student.siblings %}{{ student.siblings|length }} sibling{% if student.siblings|length != 1 %}s{% endif %}{% else %}None{% endif %}</td><td><div class="row-actions"><a class="link-button" href="/students/{{ class_name|urlencode }}?edit={{ student.id }}">Edit</a><form method="post" action="/students/{{ class_name|urlencode }}/delete"><input type="hidden" name="student_id" value="{{ student.id }}"><button class="link-button" type="submit">Delete</button></form></div></td></tr>{% endfor %}</tbody></table></div>{% else %}<div class="empty">No students found in {{ class_name }}{% if query %} for “{{ query }}”{% endif %}.</div>{% endif %}</section></main><script>const siblingToggle=document.getElementById('has-sibling');const siblingBox=document.getElementById('siblings');const siblingTemplate=()=>{const row=document.createElement('div');row.className='sibling-row';row.innerHTML='<input name="sibling_name" placeholder="Sibling Name"><select name="sibling_class"><option value="">Sibling Class</option>{% for option in classes %}<option>{{ option }}</option>{% endfor %}</select><button type="button" class="remove-sibling">Remove</button>';return row;};siblingToggle?.addEventListener('change',()=>{siblingBox.classList.toggle('hidden',!siblingToggle.checked);});document.getElementById('add-sibling')?.addEventListener('click',()=>{siblingBox.insertBefore(siblingTemplate(),document.getElementById('add-sibling'));});siblingBox?.addEventListener('click',event=>{if(event.target.classList.contains('remove-sibling')){const rows=siblingBox.querySelectorAll('.sibling-row');if(rows.length>1)event.target.parentElement.remove();}});</script></body></html>
"""
TEMPLATE = TEMPLATE.replace("</style>", "<style>.management-subtitle{margin:7px 0 0;color:var(--muted);font-size:13px}.management-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.management-card{min-height:214px;display:flex;flex-direction:column;align-items:flex-start;padding:20px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 12px 28px #0003;color:var(--ink);text-decoration:none;transition:transform .2s,border-color .2s,box-shadow .2s}.management-card:hover,.management-card:focus-visible{border-color:#f39a4b99;box-shadow:0 16px 32px #0005,0 0 20px #f39a4b12;transform:translateY(-4px)}.management-card:focus-visible{outline:3px solid #ffc47766;outline-offset:3px}.management-icon{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:#f39a4b1c;color:var(--amber-light);font-size:21px;font-weight:bold}.calendar-icon{color:var(--sky);background:#88b8d01c}.exam-icon{color:var(--mint);background:#92c9ae1c}.report-icon{color:var(--amber-light)}.management-card-title{margin-top:22px;font-size:18px;font-weight:bold}.management-description{margin-top:8px;color:var(--muted);font-size:12px;line-height:1.55}.management-action{width:100%;margin-top:auto;padding-top:14px;border-top:1px solid #ffffff0e;color:var(--amber-light);font-size:12px;font-weight:bold}.management-action span{padding-left:6px;font-size:16px}@media(max-width:1000px){.management-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){.management-grid{grid-template-columns:1fr}.management-card{min-height:175px}}</style>")
TEMPLATE = TEMPLATE.replace('name="student_identifier" autocomplete="off" required', 'name="student_identifier" placeholder="Enter Student UID or Student ID" autocomplete="off" required')
TEMPLATE = TEMPLATE.replace('<div><span>Class</span><strong>{{ password_reset_student.class_name }}</strong></div>', '<div><span>Class</span><strong>{{ password_reset_student.class_name }}</strong></div><div><span>Student Login Username</span><strong>{{ password_reset_student.account_username }}</strong></div>')
TEMPLATE = TEMPLATE.replace('<div class="panel password-reset-card">', '<div class="panel password-reset-card reset-password-panel">')
TEMPLATE = TEMPLATE.replace("</style>", "<style>.account-support-section,.account-support-section .password-reset-card{background-color:#1a1c25!important;background-image:linear-gradient(145deg,#1a1c25,#15161d)!important}.account-support-section .password-reset-card{border:1px solid var(--line)!important;border-radius:17px!important;box-shadow:0 12px 28px #0003!important;padding:18px!important}</style>", 1)
TEMPLATE = TEMPLATE.replace("</style>", "<style>.account-support-section .password-reset-card{width:min(100%,920px);background:linear-gradient(145deg,#1a1c25,#15161d)!important;border:1px solid var(--line)!important;border-radius:18px!important;box-shadow:0 14px 35px #0004!important}.account-support-section .password-reset-form{gap:14px}.account-support-section .password-reset-form input{background:#0e1017!important;color:var(--ink)!important;border:1px solid var(--line)!important;border-radius:10px!important;padding:12px 13px!important;outline:0;transition:border-color .2s,box-shadow .2s}.account-support-section .password-reset-form input:focus{border-color:var(--amber)!important;box-shadow:0 0 0 3px #f39a4b1c!important}.account-support-section .password-reset-form .button,.account-support-section .reset-actions .button{border-radius:10px;background:linear-gradient(135deg,var(--amber),var(--coral))!important;box-shadow:0 10px 24px #e57c3b3d;color:#17120f;padding:12px 17px;font-weight:bold;transition:transform .2s,box-shadow .2s}.account-support-section .password-reset-form .button:hover,.account-support-section .reset-actions .button:hover{background:linear-gradient(135deg,#ffc477,#ef8961)!important;box-shadow:0 14px 28px #e57c3b55;transform:translateY(-2px)}.account-support-section .password-reset-form .button{height:45px}.account-support-section .verification-profile{max-width:850px}.account-support-section .reset-actions{max-width:850px;justify-content:flex-start}.account-support-section .reset-actions .secondary{background:#282b35!important;color:var(--light)!important;box-shadow:none}.account-support-section .reset-actions .secondary:hover{background:#343944!important;box-shadow:none}.account-support-section .reset-heading,.account-support-section .reset-summary,.account-support-section .password-reset-form{max-width:760px}</style>", 1)
TEMPLATE = TEMPLATE.replace("</style>", "<style>.memory-empty{display:grid;place-items:center;gap:10px;min-height:150px;padding:25px;border:1px dashed var(--line);border-radius:14px;background:#101821;color:var(--muted)}.memory-empty p{margin:0}</style>")
TEMPLATE = TEMPLATE.replace("</style>", "<style>.memories-subtitle{margin:7px 0 0;color:var(--muted);font-size:13px}.memories-section{position:relative}.memory-upload-button{white-space:nowrap}.memory-form{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px;padding:18px;border:1px solid var(--line);border-radius:14px;background:#101821}.memory-form label{display:grid;gap:6px;color:#d8d1cc;font-size:12px;font-weight:bold}.memory-form label:first-child{grid-column:1/-1}.memory-form input,.memory-form textarea{width:100%;padding:10px 11px;border:1px solid var(--line);border-radius:9px;background:#0e1017;color:var(--ink);font:inherit}.memory-form textarea{min-height:70px;resize:vertical}.memory-form-actions{display:flex;gap:10px;align-items:center;grid-column:1/-1}.memory-slider{display:grid;grid-template-columns:42px minmax(0,1fr) 42px;gap:12px;align-items:center}.memory-track{min-width:0}.memory-slide{display:none;position:relative;overflow:hidden;border:1px solid var(--line);border-radius:16px;background:#101821}.memory-slide.is-active{display:block}.memory-slide img{display:block;width:100%;height:260px;object-fit:cover;cursor:zoom-in}.memory-caption{padding:14px 16px}.memory-caption strong{font-size:16px}.memory-caption p{margin:5px 0;color:var(--muted);font-size:12px}.memory-caption small{color:var(--muted);font-size:11px}.memory-nav{width:40px;height:40px;border:1px solid var(--line);border-radius:50%;background:#151e28;color:var(--amber-light);font-size:20px;cursor:pointer}.memory-nav:hover{background:#f39a4b1c;border-color:#f39a4b99}.memory-delete{margin-top:9px;border:0;background:none;color:#ffb7a5;cursor:pointer;font:inherit;padding:0}.memory-lightbox{position:fixed;z-index:10;inset:0;display:grid;place-items:center;padding:30px;background:#05080dd9}.memory-lightbox[hidden]{display:none}.memory-lightbox img{max-width:min(100%,1000px);max-height:85vh;object-fit:contain;border:1px solid var(--line);border-radius:10px}.memory-lightbox-close{position:absolute;top:20px;right:25px;border:0;background:none;color:#fff;font-size:34px;cursor:pointer}@media(max-width:620px){.section-heading:has(.memory-upload-button){align-items:flex-start;flex-direction:column}.memory-upload-button{width:100%;text-align:center}.memory-form{grid-template-columns:1fr}.memory-form label:first-child,.memory-form-actions{grid-column:auto}.memory-slider{grid-template-columns:34px minmax(0,1fr) 34px;gap:7px}.memory-nav{width:34px;height:34px;font-size:16px}.memory-slide img{height:190px}}</style>")


STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '<form method="post" action="/students/{{ class_name|urlencode }}">',
    '<form method="post" action="/students/{{ class_name|urlencode }}" enctype="multipart/form-data">',
).replace(
    '<div class="field full"><label for="address">Complete Address</label>',
    '<div class="field"><label for="admission_date">Admission Date</label><input id="admission_date" name="admission_date" type="date" value="{{ form.admission_date }}" required></div><div class="field"><label for="student_photo">Student Photo</label><input id="student_photo" name="student_photo" type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png"></div><div class="field full"><label for="address">Complete Address</label>',
).replace(
    '<p><strong>{{ view_student.student_name }}</strong> · Roll {{ view_student.roll_number }} · UID {{ view_student.uid_number }}</p>',
    '<p>{% if view_student.photo_path %}<img class="student-photo" src="{{ url_for(\'student_photo\', student_id=view_student.id) }}" alt="Student photo">{% endif %}<strong>{{ view_student.student_name }}</strong> · Roll {{ view_student.roll_number }} · UID {{ view_student.uid_number }}{% if view_student.admission_date %} · Admission date {{ view_student.admission_date }}{% endif %}</p>',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace("</style>", "<style>.student-photo{width:58px;height:58px;object-fit:cover;border-radius:10px;border:1px solid var(--line);margin-right:10px;vertical-align:middle}</style>", 1)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace('<div class="row-actions">', '<div class="row-actions"><a class="link-button" href="/students/{{ class_name|urlencode }}/view/{{ student.id }}">View</a>')
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '<section class="panel"><h2>{{ \'Edit Student\' if edit_student else \'Add Student\' }}</h2>',
    '<section class="panel">{% if view_student %}<h2>Student Details</h2><p><strong>{{ view_student.student_name }}</strong> · Roll {{ view_student.roll_number }} · UID {{ view_student.uid_number }}</p><p>{{ view_student.father_name }} / {{ view_student.mother_name }} · {{ view_student.parent_mobile }}</p>{% endif %}<h2>{{ \'Edit Student\' if edit_student else \'Add Student\' }}</h2>',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '<section class="panel">{% if view_student %}',
    '<div class="student-section-links"><a class="student-section-card" href="#add-student"><span class="student-section-icon" aria-hidden="true">+</span><span><strong>Add Student</strong><small>Register a new student</small></span><span class="student-section-arrow" aria-hidden="true">&#8594;</span></a><a class="student-section-card" href="#student-list"><span class="student-section-icon group" aria-hidden="true">&#9673;</span><span><strong>Students in {{ class_name }}</strong><small>View and manage students</small></span><span class="student-section-arrow" aria-hidden="true">&#8594;</span></a></div><section id="add-student" class="panel">{% if view_student %}',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '</div></form></section><section class="panel"><div class="toolbar">',
    '</div></form></section><section id="student-list" class="panel"><div class="toolbar">',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '</style></head>',
    '</style><style>.student-section-links{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px;margin:0 0 18px}.student-section-links.is-hidden{display:none}.student-section-card{min-height:126px;display:flex;align-items:center;gap:14px;padding:20px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#1d202a,#15161d);box-shadow:0 14px 32px #0004;color:var(--ink);text-decoration:none;transition:transform .2s,border-color .2s,box-shadow .2s}.student-section-card:hover,.student-section-card:focus-visible{border-color:#f39a4b99;box-shadow:0 16px 34px #0005,0 0 22px #f39a4b12;transform:translateY(-3px)}.student-section-card:focus-visible{outline:3px solid #ffc47766;outline-offset:3px}.student-section-icon{display:grid;place-items:center;flex:0 0 43px;width:43px;height:43px;border-radius:13px;background:#f39a4b1c;color:var(--light);font-size:27px;font-weight:bold}.student-section-icon.group{color:var(--mint);background:#92c9ae1c;font-size:22px}.student-section-card strong{display:block;font-size:19px}.student-section-card small{display:block;margin-top:5px;color:var(--muted);font-size:12px}.student-section-arrow{margin-left:auto;color:var(--light);font-size:20px}.student-interface-panel{display:none}.student-interface-panel.is-open{display:block}.panel-close{float:right;border:1px solid #ffffff1c;border-radius:8px;background:#ffffff08;color:var(--light);padding:7px 10px;cursor:pointer;font:inherit}.panel-close:hover,.panel-close:focus-visible{background:#f39a4b1c;outline:3px solid #ffc47744;outline-offset:2px}@media(max-width:600px){.student-section-links{grid-template-columns:1fr}.student-section-card{min-height:108px}}</style></head>',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '<a class="student-section-card" href="#add-student">',
    '<a class="student-section-card" href="#add-student" data-panel-target="add-student">',
).replace(
    '<a class="student-section-card" href="#student-list">',
    '<a class="student-section-card" href="#student-list" data-panel-target="student-list">',
).replace(
    '<section id="add-student" class="panel">',
    '<section id="add-student" class="panel student-interface-panel"><button type="button" class="panel-close" data-panel-close>Back to options</button>',
).replace(
    '<section id="student-list" class="panel">',
    '<section id="student-list" class="panel student-interface-panel"><button type="button" class="panel-close" data-panel-close>Back to options</button>',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '</script></body></html>',
    '''</script><script>(function(){const links=document.querySelector('.student-section-links');const panels=document.querySelectorAll('.student-interface-panel');const openPanel=function(id){links.classList.add('is-hidden');panels.forEach(function(panel){panel.classList.toggle('is-open',panel.id===id);});const target=document.getElementById(id);if(target)target.scrollIntoView({behavior:'smooth',block:'start'});};const closePanels=function(){links.classList.remove('is-hidden');panels.forEach(function(panel){panel.classList.remove('is-open');});history.replaceState(null,'',window.location.pathname+window.location.search);};document.querySelectorAll('[data-panel-target]').forEach(function(link){link.addEventListener('click',function(event){event.preventDefault();openPanel(link.dataset.panelTarget);history.replaceState(null,'','#'+link.dataset.panelTarget);});});document.querySelectorAll('[data-panel-close]').forEach(function(button){button.addEventListener('click',closePanels);});if(window.location.hash==='#add-student'||window.location.hash==='#student-list')openPanel(window.location.hash.slice(1));else if(new URLSearchParams(window.location.search).has('edit')||new URLSearchParams(window.location.search).has('q'))openPanel('student-list');})();</script></body></html>''',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '</a></div><section id="add-student"',
    '</a><a class="student-section-card" href="#attendance-panel" data-panel-target="attendance-panel"><span class="student-section-icon attendance" aria-hidden="true">&#10003;</span><span><strong>Attendance Management</strong><small>Mark and manage daily attendance</small></span><span class="student-section-arrow" aria-hidden="true">&#8594;</span></a></div><section id="add-student"',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '</section></main><script>',
    '<section id="attendance-panel" class="panel student-interface-panel"><button type="button" class="panel-close" data-panel-close>Back to options</button><h2>Attendance - {{ class_name }}</h2><form method="post" action="/students/{{ class_name|urlencode }}/attendance"><div class="attendance-toolbar"><label for="attendance-date">Date<input id="attendance-date" type="date" name="attendance_date" value="{{ attendance_date }}" required onchange="window.location.href=window.location.pathname+\'?attendance=1&amp;attendance_date=\'+this.value+\'#attendance-panel\'"></label><label for="attendance-search">Search student<input id="attendance-search" type="search" placeholder="Roll number or name"></label><button type="button" class="button secondary" id="mark-all-present">Mark All Present</button><button type="button" class="button secondary" id="mark-all-absent">Mark All Absent</button></div><div class="attendance-list">{% if attendance_students %}<table class="attendance-table"><thead><tr><th>Roll Number</th><th>UID Number</th><th>Student Name</th><th>Attendance Status</th><th>Date</th></tr></thead><tbody>{% for student in attendance_students %}<tr class="attendance-row" data-search="{{ student.roll_number }} {{ student.student_name|lower }}"><td>{{ student.roll_number }}</td><td>{{ student.uid_number }}</td><td>{{ student.student_name }}</td><td><div class="attendance-choice"><label><input type="radio" name="status_{{ student.id }}" value="Present" {% if student.attendance_status == "Present" %}checked{% endif %}> Present</label><label><input type="radio" name="status_{{ student.id }}" value="Absent" {% if student.attendance_status == "Absent" %}checked{% endif %}> Absent</label></div></td><td>{{ attendance_date }}</td></tr>{% endfor %}</tbody></table>{% else %}<p class="empty">No students are registered in {{ class_name }} yet.</p>{% endif %}</div>{% if attendance_students %}<button class="button" type="submit">Save Attendance</button>{% endif %}</form></section></main><script>',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '.student-section-links{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));',
    '.student-section-links{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));',
).replace(
    '.student-section-icon.group{color:var(--mint);background:#92c9ae1c;font-size:22px}',
    '.student-section-icon.group{color:var(--mint);background:#92c9ae1c;font-size:22px}.student-section-icon.attendance{color:var(--sky);background:#88b8d01c;font-size:24px}.attendance-toolbar{display:flex;align-items:end;gap:10px;flex-wrap:wrap;margin-bottom:16px}.attendance-toolbar label{display:grid;gap:5px;color:var(--muted);font-size:12px}.attendance-toolbar input{padding:9px;border:1px solid var(--line);border-radius:9px;background:#0e1017;color:var(--ink);font:inherit}.attendance-list{overflow:auto;margin-bottom:18px}.attendance-table{width:100%;border-collapse:collapse;min-width:700px}.attendance-table th,.attendance-table td{text-align:left;padding:12px 10px;border-bottom:1px solid #ffffff0e}.attendance-table th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}.attendance-table td{color:#e5ddd5}.attendance-choice{display:flex;gap:14px;color:#d8d1cc;font-size:12px}.attendance-choice input{accent-color:var(--amber)}',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    'action="/students/{{ class_name|urlencode }}/attendance"',
    'action="{{ student_path }}/attendance"',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    'if(window.location.hash===\'#add-student\'||window.location.hash===\'#student-list\')openPanel(window.location.hash.slice(1));',
    'if(window.location.hash===\'#add-student\'||window.location.hash===\'#student-list\'||window.location.hash===\'#attendance-panel\')openPanel(window.location.hash.slice(1));else if(new URLSearchParams(window.location.search).has(\'attendance\'))openPanel(\'attendance-panel\');',
)
STUDENT_MANAGEMENT_TEMPLATE = STUDENT_MANAGEMENT_TEMPLATE.replace(
    '</body></html>',
    '<script>(function(){const rows=document.querySelectorAll(".attendance-row");const search=document.getElementById("attendance-search");const setStatus=function(status){rows.forEach(function(row){const input=row.querySelector("input[value=\\\""+status+"\\\"]");if(input)input.checked=true;});};search?.addEventListener("input",function(){const term=search.value.trim().toLowerCase();rows.forEach(function(row){row.hidden=term&&!row.dataset.search.toLowerCase().includes(term);});});document.getElementById("mark-all-present")?.addEventListener("click",()=>setStatus("Present"));document.getElementById("mark-all-absent")?.addEventListener("click",()=>setStatus("Absent"));})();</script></body></html>',
)


def batch_context(result, selected="all"):
    shown = result if selected == "all" else result[result["risk_status"] == ("At Risk" if selected == "risk" else "Low Risk")]
    rows = [{"index": index, "values": row.tolist()} for index, row in shown.astype(str).iterrows()]
    return {"total": len(result), "at_risk": int((result["risk_status"] == "At Risk").sum()), "low_risk": int((result["risk_status"] == "Low Risk").sum()), "rate": round((result["risk_status"] == "At Risk").mean() * 100, 1), "columns": shown.columns.tolist(), "rows": rows}


def student_detail(result, index):
    if index < 0 or index >= len(result):
        raise ValueError("That student could not be found in the current results.")
    row = result.iloc[index]
    values = row.to_dict()
    label = 1 if values["risk_status"] == "At Risk" else 0
    student_values = {column: values[column] for column in INPUT_COLUMNS}
    indicators = [
        {"label": "Attendance rate", "value": f"{float(values['attendance_rate']):.1f}%", "percent": min(100, max(0, float(values["attendance_rate"])))},
        {"label": "Study hours per week", "value": f"{float(values['study_hours_per_week']):.1f}", "percent": min(100, max(0, float(values["study_hours_per_week"]) / 40 * 100))},
        {"label": "Previous exam score", "value": f"{float(values['previous_exam_score']):.1f}", "percent": min(100, max(0, float(values["previous_exam_score"])))},
        {"label": "Assignment score", "value": f"{float(values['assignment_score']):.1f}", "percent": min(100, max(0, float(values["assignment_score"])))},
    ]
    profile_columns = [
        ("Age", "age"), ("Gender", "gender"), ("Program", "program"),
        ("Parental education", "parental_education"), ("Internet access", "internet_access"),
        ("Extracurricular activity", "extracurricular_activity"), ("Academic support", "academic_support"),
        ("Prediction probability", "risk_probability"),
    ]
    profile = [(label, values[column]) for label, column in profile_columns if column in values]
    return {"student_id": values.get("student_id", "Selected student"), "prediction": label, "explanation": single_explanation(student_values), "suggestions": suggestions(student_values, label), "profile": profile, "indicators": indicators, "later_outcome": values.get("final_exam_score")}


def student_history_context(school_id, student_query="", student_id=None):
    query = student_query.strip()
    with game_connection() as connection:
        if student_id is not None:
            student = connection.execute(
                "SELECT managed_students.*, game_users.id AS account_id, game_users.username, "
                "CASE WHEN game_users.id IS NULL THEN 'Not created' "
                "WHEN managed_students.active_student = 1 THEN 'Active' ELSE 'Inactive' END AS account_status, "
                "game_users.created_at AS account_created_at "
                "FROM managed_students LEFT JOIN game_users ON game_users.managed_student_id = managed_students.id "
                "AND game_users.school_id = managed_students.school_id "
                "WHERE managed_students.id = ? AND managed_students.school_id = ?",
                (student_id, school_id),
            ).fetchone()
            if student is None:
                return [], None, [], []
            fee_rows = connection.execute(
                "SELECT fee_month, status, receipt_data, updated_at FROM student_fee_status "
                "WHERE school_id = ? AND student_id = ? AND class_name = ?",
                (school_id, student_id, student["class_name"]),
            ).fetchall()
        elif query:
            pattern = f"%{query}%"
            matches = connection.execute(
                "SELECT id, student_name, uid_number, roll_number, class_name FROM managed_students "
                "WHERE school_id = ? AND (uid_number LIKE ? OR student_name LIKE ? COLLATE NOCASE OR class_name LIKE ? COLLATE NOCASE) "
                "ORDER BY class_name, student_name, id LIMIT 25",
                (school_id, pattern, pattern, pattern),
            ).fetchall()
            return [dict(row) for row in matches], None, [], []
        else:
            return [], None, [], []

    profile = dict(student)
    profile["photo_url"] = url_for("student_photo", student_id=student["id"]) if profile.get("photo_path") else None
    try:
        profile["siblings"] = json.loads(profile.get("siblings_json") or "[]")
    except (TypeError, ValueError):
        profile["siblings"] = []
    fee_by_month = {row["fee_month"]: dict(row) for row in fee_rows}
    fee_history = []
    for month in FEE_MONTHS:
        fee = fee_by_month.get(month, {"fee_month": month, "status": "Pending", "receipt_data": None, "updated_at": None})
        fee["receipt_preview"] = base64.b64encode(fee["receipt_data"]).decode("ascii") if fee.get("receipt_data") else None
        fee_history.append(fee)

    exam_history = student_examination_history(school_id, student["id"])
    return [], profile, fee_history, exam_history


def calculate_exam_grade(percentage):
    if percentage >= 90:
        return "A1"
    elif percentage >= 80:
        return "A2"
    elif percentage >= 70:
        return "B1"
    elif percentage >= 60:
        return "B2"
    elif percentage >= 50:
        return "C1"
    elif percentage >= 40:
        return "C2"
    elif percentage >= 33:
        return "D"
    else:
        return "E"


def seed_default_subjects(school_id, class_name):
    default_subjects = [
        "English",
        "Hindi",
        "Mathematics",
        "Science",
        "Social Studies",
        "Environmental Studies (EVS)",
        "Computer Science",
        "General Knowledge (GK)",
        "Art & Craft",
        "Physical Education / Sports",
        "Physics",
        "Chemistry",
        "Biology",
        "Accountancy",
        "Business Studies",
        "Economics",
    ]
    with game_connection() as connection:
        for idx, subj in enumerate(default_subjects, start=1):
            connection.execute(
                """
                INSERT INTO examination_subjects (school_id, class_name, subject_name, max_marks, pass_marks, is_active, subject_order)
                VALUES (?, ?, ?, 100, 33, 1, ?)
                ON CONFLICT(school_id, class_name, subject_name) DO UPDATE SET is_active = 1
                """,
                (school_id, class_name, subj, idx),
            )


def get_examination_context(school_id, class_name=None, status_filter=None, academic_session=None):
    with game_connection() as connection:
        classes_data = []
        for c_name in SCHOOL_CLASSES:
            subj_count = connection.execute(
                "SELECT COUNT(*) FROM examination_subjects WHERE school_id = ? AND class_name = ? AND is_active = 1",
                (school_id, c_name),
            ).fetchone()[0]
            student_count = connection.execute(
                "SELECT COUNT(*) FROM managed_students WHERE school_id = ? AND class_name = ?",
                (school_id, c_name),
            ).fetchone()[0]
            exam_count = connection.execute(
                "SELECT COUNT(*) FROM examinations WHERE school_id = ? AND class_name = ?",
                (school_id, c_name),
            ).fetchone()[0]
            classes_data.append({
                "class_name": c_name,
                "subject_count": subj_count,
                "student_count": student_count,
                "exam_count": exam_count,
            })

        query = "SELECT * FROM examinations WHERE school_id = ?"
        params = [school_id]
        if class_name:
            query += " AND class_name = ?"
            params.append(class_name)
        if status_filter:
            query += " AND status = ?"
            params.append(status_filter)
        if academic_session:
            query += " AND academic_session = ?"
            params.append(academic_session)

        query += " ORDER BY created_at DESC"
        exam_rows = connection.execute(query, params).fetchall()

        exams = []
        for row in exam_rows:
            e = dict(row)
            sub_c = connection.execute(
                "SELECT COUNT(*) FROM examination_subjects WHERE school_id = ? AND class_name = ? AND is_active = 1",
                (school_id, e["class_name"]),
            ).fetchone()[0]
            marks_entered = connection.execute(
                "SELECT COUNT(DISTINCT student_id) FROM student_exam_marks WHERE school_id = ? AND exam_id = ?",
                (school_id, e["id"]),
            ).fetchone()[0]
            stud_c = connection.execute(
                "SELECT COUNT(*) FROM managed_students WHERE school_id = ? AND class_name = ?",
                (school_id, e["class_name"]),
            ).fetchone()[0]
            e["subject_count"] = sub_c
            e["marks_entered_count"] = marks_entered
            e["total_students"] = stud_c
            exams.append(e)

        total_exams = connection.execute("SELECT COUNT(*) FROM examinations WHERE school_id = ?", (school_id,)).fetchone()[0]
        published_exams = connection.execute("SELECT COUNT(*) FROM examinations WHERE school_id = ? AND status = 'Published'", (school_id,)).fetchone()[0]
        draft_exams = connection.execute("SELECT COUNT(*) FROM examinations WHERE school_id = ? AND status = 'Draft'", (school_id,)).fetchone()[0]
        total_active_subjects = connection.execute("SELECT COUNT(*) FROM examination_subjects WHERE school_id = ? AND is_active = 1", (school_id,)).fetchone()[0]

        return {
            "classes_data": classes_data,
            "exams": exams,
            "total_exams": total_exams,
            "published_exams": published_exams,
            "draft_exams": draft_exams,
            "total_active_subjects": total_active_subjects,
        }


def get_student_exam_result_summary(school_id, exam_id, student_id):
    with game_connection() as connection:
        exam = connection.execute(
            "SELECT * FROM examinations WHERE school_id = ? AND id = ?",
            (school_id, exam_id),
        ).fetchone()
        if not exam:
            return None
        student = connection.execute(
            "SELECT * FROM managed_students WHERE school_id = ? AND id = ?",
            (school_id, student_id),
        ).fetchone()
        if not student:
            return None

        subjects = connection.execute(
            "SELECT * FROM examination_subjects WHERE school_id = ? AND class_name = ? AND is_active = 1 ORDER BY subject_order, subject_name",
            (school_id, exam["class_name"]),
        ).fetchall()

        marks_rows = connection.execute(
            "SELECT * FROM student_exam_marks WHERE school_id = ? AND exam_id = ? AND student_id = ?",
            (school_id, exam_id, student_id),
        ).fetchall()
        marks_map = {m["subject_id"]: dict(m) for m in marks_rows}

        subject_results = []
        total_max = 0
        total_obtained = 0
        is_pass = True
        has_any_marks = False

        for subj in subjects:
            m = marks_map.get(subj["id"])
            obtained = m["obtained_marks"] if m and m["obtained_marks"] is not None else None
            is_absent = bool(m["is_absent"]) if m else False
            if m and (obtained is not None or is_absent):
                has_any_marks = True

            max_m = subj["max_marks"]
            pass_m = subj["pass_marks"]
            total_max += max_m

            if is_absent:
                subj_status = "Absent"
                is_pass = False
            elif obtained is not None:
                total_obtained += obtained
                if obtained < pass_m:
                    subj_status = "Fail"
                    is_pass = False
                else:
                    subj_status = "Pass"
            else:
                subj_status = "Pending"

            subject_results.append({
                "subject_id": subj["id"],
                "subject_name": subj["subject_name"],
                "max_marks": max_m,
                "pass_marks": pass_m,
                "obtained_marks": obtained if not is_absent else "A",
                "is_absent": is_absent,
                "remarks": m["remarks"] if m else "",
                "status": subj_status,
            })

        percentage = round((total_obtained / total_max * 100), 2) if total_max > 0 else 0.0
        grade = calculate_exam_grade(percentage)

        return {
            "exam": dict(exam),
            "student": dict(student),
            "subject_results": subject_results,
            "total_max": total_max,
            "total_obtained": total_obtained,
            "percentage": percentage,
            "grade": grade,
            "is_pass": is_pass and has_any_marks,
            "has_any_marks": has_any_marks,
        }


def student_examination_history(school_id, student_id):
    with game_connection() as connection:
        student = connection.execute(
            "SELECT * FROM managed_students WHERE school_id = ? AND id = ?",
            (school_id, student_id),
        ).fetchone()
        if not student:
            return []

        published_exams = connection.execute(
            "SELECT * FROM examinations WHERE school_id = ? AND class_name = ? AND status = 'Published' ORDER BY created_at DESC",
            (school_id, student["class_name"]),
        ).fetchall()

        history = []
        for exam in published_exams:
            res = get_student_exam_result_summary(school_id, exam["id"], student_id)
            if res and res["has_any_marks"]:
                history.append({
                    "exam_id": exam["id"],
                    "exam_name": exam["exam_name"],
                    "academic_session": exam["academic_session"],
                    "total_obtained": res["total_obtained"],
                    "total_max": res["total_max"],
                    "percentage": res["percentage"],
                    "grade": res["grade"],
                    "is_pass": res["is_pass"],
                })
        return history


def class_sections(school_id):
    with game_connection() as connection:
        rows = connection.execute(
            "SELECT id, class_name, section_name FROM student_sections WHERE school_id = ? ORDER BY class_name, section_name, id",
            (school_id,),
        ).fetchall()
    grouped = {class_name: [] for class_name in SCHOOL_CLASSES}
    for row in rows:
        grouped.setdefault(row["class_name"], []).append(dict(row))
    return grouped


def normalize_section_name(value):
    name = re.sub(r"\s+", " ", (value or "").strip())
    return name if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,19}", name) else ""



TEMPLATE = TEMPLATE.replace("</head>", "<style>.account-support-section{background:transparent!important}.reset-password-panel{background-color:#1a1c25!important;background-image:linear-gradient(145deg,#1a1c25,#15161d)!important;border:1px solid var(--line)!important;border-radius:17px!important;box-shadow:0 12px 28px #0003!important;color:var(--ink)!important}.reset-password-panel .reset-summary{background:#1d202a!important;border:1px solid var(--line)!important;color:var(--ink)!important}.reset-password-panel .reset-summary span,.reset-password-panel .reset-heading p{color:var(--muted)!important}.reset-password-panel .password-reset-form input{background:#0e1017!important;color:var(--ink)!important;border-color:var(--line)!important}.reset-password-panel .password-reset-form input:focus{border-color:var(--amber)!important;box-shadow:0 0 0 3px #f39a4b1c!important}</style></head>", 1)


def render_page(view="home", **context):
    school = current_school()
    school_data = school_records(school["id"]) if school else pd.DataFrame()
    counts = school_data["risk_status"].value_counts() if "risk_status" in school_data else pd.Series(dtype=int)
    with game_connection() as connection:
        managed_counts = connection.execute("SELECT COUNT(*) AS total, COALESCE(SUM(active_student), 0) AS active FROM managed_students WHERE school_id = ?", (school["id"],)).fetchone() if school else {"total": 0, "active": 0}
    school_total = int(managed_counts["total"])
    total_students_ever = school_total
    active_students = int(managed_counts["active"])
    active_percentage = round(active_students / total_students_ever * 100, 1) if total_students_ever else 0
    user = current_user()
    memory_photos = []
    if school:
        with game_connection() as connection:
            memory_photos = [dict(row) for row in connection.execute("SELECT id, title, description, file_extension, uploaded_by, uploaded_at FROM school_memories WHERE school_id = ? ORDER BY uploaded_at DESC, id DESC", (school["id"],)).fetchall()]
    defaults = {"view": view, "school_name": school["school_name"] if school else "", "total_students": school_total, "total_students_ever": total_students_ever, "active_students": active_students, "active_percentage": active_percentage, "classes": SCHOOL_CLASSES, "sections_by_class": class_sections(school["id"]) if school else {}, "section_form_class": None, "section_error": None, "fee_months": FEE_MONTHS, "selected_class": "", "selected_month": "", "fee_students": [], "fee_summary": {"total": 0, "paid": 0, "pending": 0, "free": 0}, "at_risk_students": int(counts.get("At Risk", 0)), "low_risk_students": int(counts.get("Low Risk", 0)), "risk_rate": round(counts.get("At Risk", 0) / school_total * 100, 1) if school_total else 0, "avg_attendance": school_attendance_rate(school["id"]) if school else 0, "avg_study": round(school_data["study_hours_per_week"].mean(), 1) if school_total and "study_hours_per_week" in school_data else 0, "insights": dashboard_insights(), "model_metrics": MODEL_METRICS, "numeric_fields": [("age", "Age", 15, 22), ("study_hours_per_week", "Study Hours per Week", 0, 40), ("attendance_rate", "Attendance Rate (%)", 0, 100), ("previous_exam_score", "Previous Exam Score", 0, 100), ("assignment_score", "Assignment Score", 0, 100)], "select_fields": [(name, name.replace("_", " ").title(), options) for name, options in CATEGORICAL_COLUMNS.items()], "input_columns": INPUT_COLUMNS, "values": {}, "prediction": None, "suggestions": [], "explanation": "", "error": None, "batch": None, "filter": "all", "detail": None, "game_user": user, "game_summary": score_summary(user["id"]) if user else None, "game_error": session.pop("game_error", None)}
    defaults.update({"student_query": "", "student_matches": [], "student_profile": None, "student_fees": [], "memory_photos": memory_photos, "memory_form_open": False, "memory_message": None, "memory_error": None, "can_delete_memories": bool(school)})
    defaults.update(context)
    return render_template_string(TEMPLATE, **defaults)


def student_form_data(form):
    return {
        "roll_number": form.get("roll_number", "").strip(),
        "student_name": form.get("student_name", "").strip(),
        "father_name": form.get("father_name", "").strip(),
        "mother_name": form.get("mother_name", "").strip(),
        "parent_mobile": form.get("parent_mobile", "").strip(),
        "second_mobile": form.get("second_mobile", "").strip(),
        "address": form.get("address", "").strip(),
        "admission_date": form.get("admission_date", "").strip(),
        "transport_student": form.get("transport_student") == "on",
    }


def validate_student_form(form, require_roll=True):
    values = student_form_data(form)
    errors = []
    values["roll_number"] = ""
    name_pattern = r"[A-Za-z][A-Za-z .'-]{1,79}"
    for field, label in (("student_name", "Student name"), ("father_name", "Father's name"), ("mother_name", "Mother's name")):
        if not re.fullmatch(name_pattern, values[field]):
            errors.append(f"{label} contains an invalid value.")
    mobile_pattern = r"\+?[0-9]{10,15}"
    if not re.fullmatch(mobile_pattern, values["parent_mobile"]):
        errors.append("Parent mobile number must contain 10-15 digits.")
    if values["second_mobile"] and not re.fullmatch(mobile_pattern, values["second_mobile"]):
        errors.append("Second mobile number must contain 10-15 digits.")
    if not 5 <= len(values["address"]) <= 300:
        errors.append("Complete address must contain 5-300 characters.")
    if values["admission_date"]:
        try:
            date.fromisoformat(values["admission_date"])
        except ValueError:
            errors.append("Admission date must be a valid date.")
    elif not require_roll:
        values["admission_date"] = date.today().isoformat()
    siblings = []
    if form.get("has_sibling") == "on":
        for name, class_name in zip(form.getlist("sibling_name"), form.getlist("sibling_class")):
            name, class_name = name.strip(), class_name.strip()
            if not name and not class_name:
                continue
            if not re.fullmatch(name_pattern, name) or class_name not in SCHOOL_CLASSES:
                errors.append("Each sibling must have a valid name and class.")
                continue
            siblings.append({"name": name, "class_name": class_name})
    return values, siblings, errors


def save_student_photo(upload, school_id, student_id):
    if upload is None or not upload.filename:
        return None
    data = upload.read(MAX_STUDENT_PHOTO_BYTES + 1)
    if len(data) > MAX_STUDENT_PHOTO_BYTES:
        raise ValueError("Student photo must be 5 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data))
        if image.format not in ("JPEG", "PNG"):
            raise ValueError
        image.load()
        converted = io.BytesIO()
        image.convert("RGB").save(converted, format="JPEG", optimize=True, quality=88)
    except (OSError, ValueError):
        raise ValueError("Upload a valid JPG, JPEG, or PNG student photo.")
    filename = f"school_{int(school_id)}_student_{int(student_id)}_{uuid.uuid4().hex}.jpg"
    path = STUDENT_PHOTO_STORAGE / filename
    STUDENT_PHOTO_STORAGE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(converted.getvalue())
    return filename


def save_calendar_photo(upload, school_id, photo_num):
    if upload is None or not upload.filename:
        return None
    data = upload.read(MAX_CALENDAR_PHOTO_BYTES + 1)
    if len(data) > MAX_CALENDAR_PHOTO_BYTES:
        raise ValueError(f"Photo {photo_num} must be 5 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data))
        fmt = image.format
        if fmt not in ("JPEG", "PNG", "WEBP"):
            raise ValueError
        image.load()
        converted = io.BytesIO()
        ext_map = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
        ext = ext_map.get(fmt, "jpg")
        if fmt == "JPEG":
            image.convert("RGB").save(converted, format="JPEG", optimize=True, quality=90)
        elif fmt == "PNG":
            image.save(converted, format="PNG", optimize=True)
        elif fmt == "WEBP":
            image.save(converted, format="WEBP", optimize=True)
    except (OSError, ValueError):
        raise ValueError(f"Upload a valid JPG, JPEG, PNG, or WEBP image for Photo {photo_num}.")
    filename = f"school_{int(school_id)}_cal{int(photo_num)}_{uuid.uuid4().hex}.{ext}"
    CALENDAR_PHOTO_STORAGE.mkdir(parents=True, exist_ok=True)
    path = CALENDAR_PHOTO_STORAGE / filename
    path.write_bytes(converted.getvalue())
    return filename


def save_memory_photo(upload, school_id):
    if upload is None or not upload.filename:
        raise ValueError("Please select a photo.")
    original_filename = Path(upload.filename).name
    extension = Path(original_filename).suffix.lower().lstrip(".")
    if extension not in MEMORY_PHOTO_EXTENSIONS:
        raise ValueError("Upload a JPG, JPEG, PNG, or WEBP image.")
    data = upload.read(MAX_MEMORY_PHOTO_BYTES + 1)
    if len(data) > MAX_MEMORY_PHOTO_BYTES:
        raise ValueError("Photos must be 5 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data))
        if image.format not in ("JPEG", "PNG", "WEBP"):
            raise ValueError
        image.load()
        converted = io.BytesIO()
        if image.format == "JPEG":
            image.convert("RGB").save(converted, format="JPEG", optimize=True, quality=90)
            extension = "jpg"
        elif image.format == "PNG":
            image.save(converted, format="PNG", optimize=True)
            extension = "png"
        else:
            image.save(converted, format="WEBP", optimize=True)
            extension = "webp"
    except (OSError, ValueError):
        raise ValueError("Upload a valid JPG, JPEG, PNG, or WEBP image.")
    stored_filename = f"school_{int(school_id)}_memory_{uuid.uuid4().hex}.{extension}"
    MEMORY_PHOTO_STORAGE.mkdir(parents=True, exist_ok=True)
    (MEMORY_PHOTO_STORAGE / stored_filename).write_bytes(converted.getvalue())
    safe_original = re.sub(r"[^A-Za-z0-9._ -]", "", original_filename).strip(" .")[:120]
    return stored_filename, safe_original or f"memory.{extension}", extension


def save_report_file(upload, school_id):
    if upload is None or not upload.filename or not upload.filename.strip():
        raise ValueError("Please select a report file.")
    original_filename = Path(upload.filename).name
    extension = Path(original_filename).suffix.lower().lstrip(".")
    if extension not in REPORT_ALLOWED_EXTENSIONS:
        raise ValueError("Allowed report formats: PDF, DOCX, XLSX, JPG, JPEG, or PNG.")
    data = upload.read(MAX_REPORT_FILE_BYTES + 1)
    if len(data) > MAX_REPORT_FILE_BYTES:
        raise ValueError("Report files must be 10 MB or smaller.")
    safe_original = re.sub(r"[^A-Za-z0-9._ -]", "", original_filename).strip(" .")[:120]
    if not safe_original:
        raise ValueError("The uploaded filename is not valid.")
    stored_filename = f"school_{int(school_id)}_report_{uuid.uuid4().hex}.{extension}"
    REPORT_FILE_STORAGE.mkdir(parents=True, exist_ok=True)
    (REPORT_FILE_STORAGE / stored_filename).write_bytes(data)
    return stored_filename, safe_original, extension


def normalize_report_name(value):
    name = re.sub(r"[^A-Za-z0-9 ._()&'-]", "", (value or "")).strip()
    return re.sub(r"\s+", " ", name)[:120]


def student_login_credentials(uid_number, student_name):
    folded_name = unicodedata.normalize("NFKD", str(student_name or "")).encode("ascii", "ignore").decode("ascii")
    normalized_name = re.sub(r"[^a-z0-9]+", "_", folded_name.lower()).strip("_")
    normalized_uid = re.sub(r"[^a-z0-9]+", "", str(uid_number or "").lower())
    if not normalized_name or not re.fullmatch(r"[a-z][a-z0-9_]*", normalized_name) or not re.fullmatch(r"[a-z0-9]+", normalized_uid):
        raise ValueError("A valid UID and student name are required to generate student credentials.")
    username = f"{normalized_uid}@{normalized_name}"
    temporary_password = f"{normalized_uid}{normalized_name}"
    return username, temporary_password


def next_admission_uid(connection):
    sequence = connection.execute(
        "SELECT last_uid, uid_width FROM admission_uid_sequence WHERE id = 1"
    ).fetchone()
    if sequence is None:
        raise RuntimeError("Admission UID sequence is not initialized.")
    next_uid = sequence["last_uid"]
    while True:
        next_uid += 1
        candidate = f"UID{next_uid:0{sequence['uid_width']}d}"
        if connection.execute(
            "SELECT 1 FROM managed_students WHERE uid_number = ? LIMIT 1",
            (candidate,),
        ).fetchone() is None:
            connection.execute(
                "UPDATE admission_uid_sequence SET last_uid = ? WHERE id = 1",
                (next_uid,),
            )
            return candidate


def next_student_credentials(connection, student_name):
    while True:
        uid_number = next_admission_uid(connection)
        username, temporary_password = student_login_credentials(uid_number, student_name)
        if connection.execute(
            "SELECT 1 FROM game_users WHERE username = ? COLLATE NOCASE LIMIT 1",
            (username,),
        ).fetchone() is None:
            return uid_number, username, temporary_password


def next_roll_number(connection, school_id, class_name):
    next_number = connection.execute(
        "SELECT COALESCE(MAX(CAST(roll_number AS INTEGER)), 0) + 1 AS next_number "
        "FROM managed_students WHERE school_id = ? AND class_name = ?",
        (school_id, class_name),
    ).fetchone()["next_number"]
    return str(next_number)


def student_management_context(school_id, class_name, query="", edit_id=None, section_id=None):
    pattern = f"%{query}%"
    with game_connection() as connection:
        if section_id is None:
            rows = connection.execute(
                "SELECT * FROM managed_students WHERE school_id = ? AND class_name = ? AND (roll_number LIKE ? OR uid_number LIKE ? OR student_name LIKE ?) ORDER BY LOWER(student_name), student_name, id",
                (school_id, class_name, pattern, pattern, pattern),
            ).fetchall()
            edit_student = connection.execute(
                "SELECT * FROM managed_students WHERE id = ? AND school_id = ? AND class_name = ?",
                (edit_id, school_id, class_name),
            ).fetchone() if edit_id else None
        else:
            rows = connection.execute(
                "SELECT * FROM managed_students WHERE school_id = ? AND class_name = ? AND section_id = ? AND (roll_number LIKE ? OR uid_number LIKE ? OR student_name LIKE ?) ORDER BY LOWER(student_name), student_name, id",
                (school_id, class_name, section_id, pattern, pattern, pattern),
            ).fetchall()
            edit_student = connection.execute(
                "SELECT * FROM managed_students WHERE id = ? AND school_id = ? AND class_name = ? AND section_id = ?",
                (edit_id, school_id, class_name, section_id),
            ).fetchone() if edit_id else None
    students_for_view = []
    for row in rows:
        item = dict(row)
        item["siblings"] = json.loads(item["siblings_json"] or "[]")
        students_for_view.append(item)
    form = dict(edit_student) if edit_student else {"roll_number": "", "uid_number": "", "student_name": "", "father_name": "", "mother_name": "", "parent_mobile": "", "second_mobile": "", "address": "", "admission_date": date.today().isoformat(), "transport_student": 0}
    siblings = json.loads(edit_student["siblings_json"] or "[]") if edit_student else []
    return students_for_view, form, siblings, edit_student


def attendance_context(school_id, class_name, attendance_date, section_id=None):
    with game_connection() as connection:
        if section_id is None:
            rows = connection.execute(
                "SELECT s.id, s.roll_number, s.uid_number, s.student_name, a.status AS attendance_status FROM managed_students AS s LEFT JOIN student_attendance AS a ON a.student_id = s.id AND a.school_id = ? AND a.class_name = ? AND a.attendance_date = ? WHERE s.school_id = ? AND s.class_name = ? ORDER BY LOWER(s.student_name), s.student_name, s.id",
                (school_id, class_name, attendance_date, school_id, class_name),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT s.id, s.roll_number, s.uid_number, s.student_name, a.status AS attendance_status FROM managed_students AS s LEFT JOIN student_attendance AS a ON a.student_id = s.id AND a.school_id = ? AND a.class_name = ? AND a.attendance_date = ? WHERE s.school_id = ? AND s.class_name = ? AND s.section_id = ? ORDER BY LOWER(s.student_name), s.student_name, s.id",
                (school_id, class_name, attendance_date, school_id, class_name, section_id),
            ).fetchall()
    return [dict(row) for row in rows]


def class_notice_context(school_id, class_name, edit_id=None):
    with game_connection() as connection:
        notices = connection.execute(
            "SELECT * FROM class_notices WHERE school_id = ? AND class_name = ? AND status = 'Published' "
            "ORDER BY COALESCE(event_date, notice_date) DESC, created_at DESC, id DESC",
            (school_id, class_name),
        ).fetchall()
        edit_notice = None
        if edit_id is not None:
            edit_notice = connection.execute(
                "SELECT * FROM class_notices WHERE id = ? AND school_id = ? AND class_name = ?",
                (edit_id, school_id, class_name),
            ).fetchone()
    return [dict(row) for row in notices], dict(edit_notice) if edit_notice else None


def student_notice_context(user):
    with game_connection() as connection:
        rows = connection.execute(
            "SELECT n.* FROM class_notices AS n "
            "JOIN managed_students AS s ON s.school_id = n.school_id AND s.class_name = n.class_name "
            "WHERE s.id = ? AND s.school_id = ? AND s.active_student = 1 AND n.status = 'Published' "
            "ORDER BY COALESCE(n.event_date, n.notice_date) DESC, n.created_at DESC, n.id DESC",
            (user["managed_student_id"], user["school_id"]),
        ).fetchall()
    return [dict(row) for row in rows]


@app.route("/students/<path:class_name>/section/<path:section_name>", methods=["GET", "POST"])
@app.route("/students/<path:class_name>", methods=["GET", "POST"])
@school_required
def student_management(class_name, section_name=None):
    if class_name not in SCHOOL_CLASSES:
        return "Class not found.", 404
    school = current_school()
    section_id = None
    if section_name is not None:
        section_name = normalize_section_name(section_name)
        with game_connection() as connection:
            section = connection.execute("SELECT id FROM student_sections WHERE school_id = ? AND class_name = ? AND section_name = ?", (school["id"], class_name, section_name)).fetchone()
        if section is None:
            return "Section not found.", 404
        section_id = section["id"]
    query = request.args.get("q", "").strip()[:80]
    error = None
    message = request.args.get("message")
    credentials = None
    photo_path = None
    if request.method == "POST":
        if request.form.get("_action") == "delete":
            try:
                student_id = int(request.form.get("student_id", "-1"))
            except ValueError:
                student_id = -1
            with game_connection() as connection:
                if section_id is None:
                    connection.execute("DELETE FROM managed_students WHERE id = ? AND school_id = ? AND class_name = ?", (student_id, school["id"], class_name))
                else:
                    connection.execute("DELETE FROM managed_students WHERE id = ? AND school_id = ? AND class_name = ? AND section_id = ?", (student_id, school["id"], class_name, section_id))
                refresh_class_roll_numbers(connection, school["id"], class_name, section_id)
            return redirect(url_for("student_management", class_name=class_name, section_name=section_name, message="Student deleted."))
        student_id = request.form.get("student_id", "").strip()
        values, siblings, errors = validate_student_form(request.form, require_roll=bool(student_id))
        photo_upload = request.files.get("student_photo")
        if errors:
            error = " ".join(errors)
            edit_id = request.form.get("student_id")
            students_for_view, _, _, edit_student = student_management_context(school["id"], class_name, query, int(edit_id) if edit_id and edit_id.isdigit() else None, section_id)
            values["transport_student"] = request.form.get("transport_student") == "on"
        else:
            try:
                with game_connection() as connection:
                    if student_id:
                        connection.execute("BEGIN IMMEDIATE")
                        photo_path = save_student_photo(photo_upload, school["id"], int(student_id)) if photo_upload and photo_upload.filename else None
                        update_sql = "UPDATE managed_students SET student_name = ?, father_name = ?, mother_name = ?, parent_mobile = ?, second_mobile = ?, address = ?, transport_student = ?, siblings_json = ?, admission_date = ?, updated_at = datetime('now')"
                        update_values = [values["student_name"], values["father_name"], values["mother_name"], values["parent_mobile"], values["second_mobile"], values["address"], int(values["transport_student"]), json.dumps(siblings), values["admission_date"] or None]
                        if photo_path:
                            update_sql += ", photo_path = ?"
                            update_values.append(photo_path)
                        if section_id is None:
                            update_sql += " WHERE id = ? AND school_id = ? AND class_name = ?"
                            update_values.extend([int(student_id), school["id"], class_name])
                        else:
                            update_sql += " WHERE id = ? AND school_id = ? AND class_name = ? AND section_id = ?"
                            update_values.extend([int(student_id), school["id"], class_name, section_id])
                        connection.execute(update_sql, update_values)
                        refresh_class_roll_numbers(connection, school["id"], class_name, section_id)
                    else:
                        connection.execute("BEGIN IMMEDIATE")
                        duplicate_student = connection.execute(
                            "SELECT id FROM managed_students WHERE school_id = ? AND class_name = ? "
                            "AND student_name = ? COLLATE NOCASE AND parent_mobile = ? AND admission_date IS ?",
                            (school["id"], class_name, values["student_name"], values["parent_mobile"], values["admission_date"] or None),
                        ).fetchone()
                        if duplicate_student is not None:
                            raise ValueError("This student is already admitted in this class.")
                        values["uid_number"], username, temporary_password = next_student_credentials(connection, values["student_name"])
                        roll_number = next_roll_number(connection, school["id"], class_name)
                        student_cursor = connection.execute("INSERT INTO managed_students (school_id, class_name, roll_number, uid_number, student_name, father_name, mother_name, parent_mobile, second_mobile, address, transport_student, siblings_json, admission_date, section_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))", (school["id"], class_name, roll_number, values["uid_number"], values["student_name"], values["father_name"], values["mother_name"], values["parent_mobile"], values["second_mobile"], values["address"], int(values["transport_student"]), json.dumps(siblings), values["admission_date"], section_id))
                        photo_path = save_student_photo(photo_upload, school["id"], student_cursor.lastrowid) if photo_upload and photo_upload.filename else None
                        if photo_path:
                            connection.execute("UPDATE managed_students SET photo_path = ? WHERE id = ? AND school_id = ?", (photo_path, student_cursor.lastrowid, school["id"]))
                        connection.execute("INSERT INTO game_users (student_name, username, password_hash, must_change_password, managed_student_id, school_id, created_at) VALUES (?, ?, ?, 0, ?, ?, datetime('now'))", (values["student_name"], username, generate_password_hash(temporary_password), student_cursor.lastrowid, school["id"]))
                        refresh_class_roll_numbers(connection, school["id"], class_name, section_id)
                        credentials = {"username": username, "temporary_password": temporary_password}
                        message = f"Student added successfully with UID {values['uid_number']}. Provide these one-time credentials securely."
            except (sqlite3.IntegrityError, ValueError) as exc:
                if photo_path:
                    (STUDENT_PHOTO_STORAGE / Path(photo_path).name).unlink(missing_ok=True)
                logger.exception(
                    "Student admission failed: school_id=%s class=%s section_id=%s error_type=%s",
                    school["id"], class_name, section_id, type(exc).__name__,
                )
                error = str(exc) if isinstance(exc, ValueError) else "Student admission could not be completed because the record conflicted with existing data."
        students_for_view, current_form, current_siblings, edit_student = student_management_context(school["id"], class_name, query, int(request.form.get("student_id")) if request.form.get("student_id", "").isdigit() else None, section_id)
        current_form.update(values)
        siblings = siblings or current_siblings
    else:
        edit_id = request.args.get("edit", "")
        edit_id = int(edit_id) if edit_id.isdigit() else None
        students_for_view, current_form, siblings, edit_student = student_management_context(school["id"], class_name, query, edit_id, section_id)
    attendance_date = request.args.get("attendance_date", date.today().isoformat())
    attendance_students = attendance_context(school["id"], class_name, attendance_date, section_id)
    student_path = url_for("student_management", class_name=class_name, section_name=section_name) if section_name else url_for("student_management", class_name=class_name)
    notice_edit_id = request.args.get("notice_edit", "")
    notice_edit_id = int(notice_edit_id) if notice_edit_id.isdigit() else None
    notices, notice_edit = class_notice_context(school["id"], class_name, notice_edit_id)
    return render_template("student_management.html", class_name=class_name, section_name=section_name, student_path=student_path, students=students_for_view, school_name=school["school_name"], form=current_form, siblings=siblings, edit_student=edit_student, view_student=None, query=query, message=message, error=error, credentials=credentials, classes=SCHOOL_CLASSES, attendance_date=attendance_date, today_date=date.today().isoformat(), attendance_students=attendance_students, notices=notices, notice_edit=notice_edit)


@app.post("/students/<path:class_name>/notices")
@school_required
def manage_class_notice(class_name):
    if class_name not in SCHOOL_CLASSES:
        return "Class not found.", 404
    school = current_school()
    action = request.form.get("_action", "create")
    try:
        notice_id = int(request.form.get("notice_id", "-1"))
    except ValueError:
        notice_id = -1
    if action == "delete":
        with game_connection() as connection:
            connection.execute(
                "DELETE FROM class_notices WHERE id = ? AND school_id = ? AND class_name = ?",
                (notice_id, school["id"], class_name),
            )
        return redirect(url_for("student_management", class_name=class_name, open_notice=1, message="Notice deleted."))

    title = re.sub(r"\s+", " ", request.form.get("title", "").strip())[:120]
    description = request.form.get("description", "").strip()[:2000]
    notice_date = request.form.get("notice_date", "").strip() or date.today().isoformat()
    event_date = request.form.get("event_date", "").strip() or None
    priority = request.form.get("priority", "Normal").strip()
    if not title or not description or priority not in ("Normal", "Important"):
        return redirect(url_for("student_management", class_name=class_name, open_notice=1, error="Title, notice text, and a valid priority are required."))
    try:
        date.fromisoformat(notice_date)
        if event_date:
            date.fromisoformat(event_date)
    except ValueError:
        return redirect(url_for("student_management", class_name=class_name, open_notice=1, error="Notice dates must be valid dates."))

    with game_connection() as connection:
        if action == "update":
            connection.execute(
                "UPDATE class_notices SET title = ?, description = ?, notice_date = ?, event_date = ?, priority = ?, updated_at = datetime('now') "
                "WHERE id = ? AND school_id = ? AND class_name = ?",
                (title, description, notice_date, event_date, priority, notice_id, school["id"], class_name),
            )
            message = "Notice updated."
        else:
            connection.execute(
                "INSERT INTO class_notices (school_id, class_name, title, description, notice_date, event_date, priority, status, created_by, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'Published', ?, datetime('now'), datetime('now'))",
                (school["id"], class_name, title, description, notice_date, event_date, priority, school["username"]),
            )
            message = "Notice published."
    return redirect(url_for("student_management", class_name=class_name, open_notice=1, message=message))


@app.get("/student-photo/<int:student_id>")
def student_photo(student_id):
    school = current_school()
    user = current_user()
    if user is not None and user["managed_student_id"] is not None:
        school_id = user["school_id"]
        own_student_id = user["managed_student_id"]
    elif school is not None:
        school_id = school["id"]
        own_student_id = None
    else:
        return "Photo not found.", 404
    if own_student_id is not None and student_id != own_student_id:
        return "Photo not found.", 404
    with game_connection() as connection:
        student = connection.execute("SELECT photo_path FROM managed_students WHERE id = ? AND school_id = ? AND active_student = 1", (student_id, school_id)).fetchone()
    if student is None or not student["photo_path"]:
        return "Photo not found.", 404
    filename = Path(student["photo_path"]).name
    if filename != student["photo_path"] or not re.fullmatch(r"school_\d+_student_\d+_[a-f0-9]+\.jpg", filename):
        return "Photo not found.", 404
    path = STUDENT_PHOTO_STORAGE / filename
    if not path.is_file() or path.parent != STUDENT_PHOTO_STORAGE:
        return "Photo not found.", 404
    return send_file(path, mimetype="image/jpeg", max_age=3600)


@app.post("/students/<path:class_name>/section/<path:section_name>/attendance")
@app.post("/students/<path:class_name>/attendance")
@school_required
def save_attendance(class_name, section_name=None):
    if class_name not in SCHOOL_CLASSES:
        return "Class not found.", 404
    school = current_school()
    section_id = None
    if section_name:
        section_name = normalize_section_name(section_name)
        with game_connection() as connection:
            section = connection.execute("SELECT id FROM student_sections WHERE school_id = ? AND class_name = ? AND section_name = ?", (school["id"], class_name, section_name)).fetchone()
        if section is None:
            return "Section not found.", 404
        section_id = section["id"]
    attendance_date = request.form.get("attendance_date", "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", attendance_date):
        return "Invalid attendance date.", 400
    with game_connection() as connection:
        valid_students = {row[0] for row in connection.execute("SELECT id FROM managed_students WHERE school_id = ? AND class_name = ? AND section_id IS ?", (school["id"], class_name, section_id))}
        for key, status in request.form.items():
            if not key.startswith("status_") or status not in ("Present", "Absent"):
                continue
            try:
                student_id = int(key[7:])
            except ValueError:
                continue
            if student_id not in valid_students:
                continue
            connection.execute("INSERT INTO student_attendance (school_id, class_name, student_id, attendance_date, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now')) ON CONFLICT(school_id, class_name, student_id, attendance_date) DO UPDATE SET status = excluded.status, updated_at = datetime('now')", (school["id"], class_name, student_id, attendance_date, status))
    return redirect(url_for("student_management", class_name=class_name, section_name=section_name, attendance_date=attendance_date, attendance="1", message="Attendance saved."))


@app.post("/students/<path:class_name>/section/<path:section_name>/delete")
@app.post("/students/<path:class_name>/delete")
@school_required
def delete_student(class_name, section_name=None):
    if class_name not in SCHOOL_CLASSES:
        return "Class not found.", 404
    school = current_school()
    section_id = None
    if section_name:
        section_name = normalize_section_name(section_name)
        with game_connection() as connection:
            section = connection.execute("SELECT id FROM student_sections WHERE school_id = ? AND class_name = ? AND section_name = ?", (school["id"], class_name, section_name)).fetchone()
        if section is None:
            return "Section not found.", 404
        section_id = section["id"]
    try:
        student_id = int(request.form.get("student_id", "-1"))
    except ValueError:
        student_id = -1
    with game_connection() as connection:
        connection.execute("DELETE FROM managed_students WHERE id = ? AND school_id = ? AND class_name = ? AND section_id IS ?", (student_id, school["id"], class_name, section_id))
        refresh_class_roll_numbers(connection, school["id"], class_name, section_id)
    return redirect(url_for("student_management", class_name=class_name, section_name=section_name, message="Student deleted."))


@app.get("/students/<path:class_name>/section/<path:section_name>/view/<int:student_id>")
@app.get("/students/<path:class_name>/view/<int:student_id>")
@school_required
def view_student(class_name, student_id, section_name=None):
    if class_name not in SCHOOL_CLASSES:
        return "Class not found.", 404
    school = current_school()
    section_id = None
    if section_name:
        section_name = normalize_section_name(section_name)
        with game_connection() as connection:
            section = connection.execute("SELECT id FROM student_sections WHERE school_id = ? AND class_name = ? AND section_name = ?", (school["id"], class_name, section_name)).fetchone()
        if section is None:
            return "Section not found.", 404
        section_id = section["id"]
    with game_connection() as connection:
        row = connection.execute("SELECT * FROM managed_students WHERE id = ? AND school_id = ? AND class_name = ? AND section_id IS ?", (student_id, school["id"], class_name, section_id)).fetchone()
    if row is None:
        return "Student not found.", 404
    students_for_view, current_form, siblings, edit_student = student_management_context(school["id"], class_name, section_id=section_id)
    attendance_date = date.today().isoformat()
    attendance_students = attendance_context(school["id"], class_name, attendance_date, section_id)
    student_path = url_for("student_management", class_name=class_name, section_name=section_name) if section_name else url_for("student_management", class_name=class_name)
    notices, notice_edit = class_notice_context(school["id"], class_name)
    return render_template("student_management.html", class_name=class_name, section_name=section_name, student_path=student_path, school_name=school["school_name"], students=students_for_view, form=current_form, siblings=siblings, edit_student=None, view_student=dict(row), query="", message=None, error=None, classes=SCHOOL_CLASSES, attendance_date=attendance_date, today_date=date.today().isoformat(), attendance_students=attendance_students, notices=notices, notice_edit=notice_edit)


def password_reset_profile(school_id, identifier=None, student_id=None):
    with game_connection() as connection:
        params = [school_id, school_id]
        if student_id is not None:
            match_clause = "managed_students.id = ?"
            params.append(student_id)
        else:
            match_clause = "LOWER(managed_students.uid_number) = LOWER(?) OR CAST(managed_students.id AS TEXT) = ? OR LOWER(game_users.username) = LOWER(?)"
            params.extend([identifier, identifier, identifier])
        rows = connection.execute(
            "SELECT managed_students.*, game_users.id AS game_user_id, game_users.username AS account_username, "
            "CASE WHEN managed_students.active_student = 1 THEN 'Active' ELSE 'Inactive' END AS account_status "
            "FROM managed_students LEFT JOIN game_users "
            "ON game_users.managed_student_id = managed_students.id "
            "AND game_users.school_id = ? "
            "WHERE managed_students.school_id = ? AND (" + match_clause + ")",
            params,
        ).fetchall()
    if len(rows) != 1:
        return None, "ambiguous" if len(rows) > 1 else "not_found"
    profile = rows[0]
    if profile["game_user_id"] is None:
        return None, "account_not_created"
    return profile, "found"


@app.post("/management/student-password-reset/find")
@school_required
def find_student_for_password_reset():
    school_id = session["school_id"]
    identifier = request.form.get("student_identifier", "").strip()
    profile, lookup_status = password_reset_profile(school_id, identifier=identifier) if identifier else (None, "not_found")
    if profile is None:
        session.pop("password_reset_pending_student_id", None)
        session.pop("password_reset_confirmed_student_id", None)
        if lookup_status == "account_not_created":
            return "Student account not created.", 404
        return "Student not found in this school.", 404
    session["password_reset_pending_student_id"] = profile["id"]
    session.pop("password_reset_confirmed_student_id", None)
    return render_page("home", password_reset_step="verify", password_reset_student=dict(profile))


@app.post("/management/student-password-reset/confirm")
@school_required
def confirm_student_password_reset():
    school_id = session["school_id"]
    student_id = session.get("password_reset_pending_student_id")
    profile, _ = password_reset_profile(school_id, student_id=student_id) if student_id else (None, "not_found")
    if profile is None:
        session.pop("password_reset_pending_student_id", None)
        session.pop("password_reset_confirmed_student_id", None)
        return "Student not found in this school.", 404
    session["password_reset_confirmed_student_id"] = profile["id"]
    return render_page("home", password_reset_step="reset", password_reset_student=dict(profile))


@app.post("/management/student-password-reset/search-another")
@school_required
def search_another_student():
    session.pop("password_reset_pending_student_id", None)
    session.pop("password_reset_confirmed_student_id", None)
    return redirect(url_for("home"))


@app.post("/management/student-password-reset")
@school_required
def reset_student_password():
    school_id = session["school_id"]
    student_id = session.get("password_reset_confirmed_student_id")
    profile, _ = password_reset_profile(school_id, student_id=student_id) if student_id else (None, "not_found")
    if profile is None:
        return "Student not found in this school.", 404

    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")
    if len(new_password) < 6 or new_password != confirm_password:
        return "Invalid password reset request.", 400

    with game_connection() as connection:
        updated = connection.execute(
            "UPDATE game_users SET password_hash = ? "
            "WHERE id = ? AND managed_student_id = ? AND school_id = ?",
            (generate_password_hash(new_password), profile["game_user_id"], profile["id"], school_id),
        )
        if updated.rowcount != 1:
            return "Student not found in this school.", 404

    session.pop("password_reset_pending_student_id", None)
    session.pop("password_reset_confirmed_student_id", None)
    return "Student password has been reset successfully."


def render_management_page(title, description, classes=None, **context):
    school = current_school()
    template_context = {"reports": [], "report_form_open": False, "message": None, "error": None}
    template_context.update(context)
    return render_template_string(MANAGEMENT_TEMPLATE, title=title, description=description, school_name=school["school_name"], classes=classes or [], **template_context)


def fee_management_context(school_id, class_name):
    with game_connection() as connection:
        students = connection.execute(
            "SELECT id, roll_number, student_name FROM managed_students WHERE school_id = ? AND class_name = ? ORDER BY CASE WHEN roll_number GLOB '[0-9]*' THEN CAST(roll_number AS INTEGER) ELSE 2147483647 END, id",
            (school_id, class_name),
        ).fetchall()
        fee_rows = connection.execute(
            "SELECT student_id, fee_month, status, locked FROM student_fee_status WHERE school_id = ? AND class_name = ?",
            (school_id, class_name),
        ).fetchall()
    fee_statuses = {student["id"]: {month: {"status": "Pending", "locked": False} for month in FEE_MONTHS} for student in students}
    for row in fee_rows:
        if row["student_id"] in fee_statuses and row["fee_month"] in FEE_MONTHS:
            fee_statuses[row["student_id"]][row["fee_month"]] = {"status": row["status"], "locked": bool(row["locked"])}
    return [dict(student) for student in students], fee_statuses


def fee_analysis_context(school_id, class_name, fee_month):
    with game_connection() as connection:
        rows = connection.execute(
            "SELECT s.roll_number, s.student_name, s.uid_number, "
            "COALESCE(f.status, 'Pending') AS status, f.receipt_data "
            "FROM managed_students AS s "
            "LEFT JOIN student_fee_status AS f ON f.student_id = s.id AND f.school_id = s.school_id "
            "AND f.class_name = s.class_name AND f.fee_month = ? "
            "WHERE s.school_id = ? AND s.class_name = ? "
            "ORDER BY CASE WHEN s.roll_number GLOB '[0-9]*' THEN CAST(s.roll_number AS INTEGER) ELSE 2147483647 END, s.id",
            (fee_month, school_id, class_name),
        ).fetchall()
    students = [dict(row) for row in rows]
    return students, {
        "total": len(students),
        "paid": sum(student["status"] == "Paid" for student in students),
        "pending": sum(student["status"] == "Pending" for student in students),
        "free": sum(student["status"] == "Free" for student in students),
    }


def render_fee_class(school, class_name, message=None, review=None):
    students, fee_statuses = fee_management_context(school["id"], class_name)
    return render_template_string(FEE_CLASS_TEMPLATE, school_name=school["school_name"], class_name=class_name, students=students, months=FEE_MONTHS, fee_statuses=fee_statuses, message=message, review=review)


def staged_receipt(token):
    path = FEE_RECEIPT_STAGING / f"{token}.png"
    return path if path.is_file() else None


def build_fee_review(school, class_name, student_id, fee_month, status, receipt_path=None, token=None):
    with game_connection() as connection:
        student = connection.execute("SELECT id, uid_number, student_name, roll_number FROM managed_students WHERE id = ? AND school_id = ? AND class_name = ?", (student_id, school["id"], class_name)).fetchone()
    if student is None:
        return None
    review = {"school_id": school["id"], "class_name": class_name, "student_id": student_id, "fee_month": fee_month, "student_name": student["student_name"], "uid_number": student["uid_number"], "roll_number": student["roll_number"], "month": fee_month, "status": status, "token": token, "receipt_data": None, "receipt_mime": "image/png"}
    if receipt_path:
        review["receipt_data"] = base64.b64encode(receipt_path.read_bytes()).decode("ascii")
    return review


@app.route("/management/fees", methods=["GET", "POST"])
@school_required
def management_fees():
    class_name = request.args.get("class", "").strip()
    if not class_name:
        return render_management_page("Fee Management", "Manage student fees, payments, pending dues and fee history.", SCHOOL_CLASSES)
    if class_name not in SCHOOL_CLASSES:
        return "Class not found.", 404
    school = current_school()
    if request.method == "POST":
        action = request.form.get("fee_action", "")
        if action == "confirm":
            review = session.get("fee_review")
            review_token = request.form.get("review_token", "")
            if not review or review.get("school_id") != school["id"] or review.get("class_name") != class_name or review.get("token") != review_token:
                return render_fee_class(school, class_name, "The review expired. Please start again.")
            student_id, fee_month, status = review["student_id"], review["fee_month"], review["status"]
            receipt_path = staged_receipt(review["token"]) if status == "Paid" else None
            if status == "Paid" and receipt_path is None:
                session.pop("fee_review", None)
                return render_fee_class(school, class_name, "The receipt photo is missing. Please upload it again.")
            with game_connection() as connection:
                student = connection.execute("SELECT id FROM managed_students WHERE id = ? AND school_id = ? AND class_name = ?", (student_id, school["id"], class_name)).fetchone()
                current = connection.execute("SELECT locked FROM student_fee_status WHERE school_id = ? AND class_name = ? AND student_id = ? AND fee_month = ?", (school["id"], class_name, student_id, fee_month)).fetchone()
                if student is None or current and current["locked"]:
                    session.pop("fee_review", None)
                    return render_fee_class(school, class_name, "This fee record is already locked.")
                receipt_data = receipt_path.read_bytes() if receipt_path else None
                connection.execute(
                    "INSERT INTO student_fee_status (school_id, class_name, student_id, fee_month, status, locked, receipt_filename, receipt_data, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?, datetime('now'), datetime('now')) ON CONFLICT(school_id, class_name, student_id, fee_month) DO UPDATE SET status = excluded.status, locked = 1, receipt_filename = excluded.receipt_filename, receipt_data = excluded.receipt_data, updated_at = datetime('now')",
                    (school["id"], class_name, student_id, fee_month, status, f"receipt_{review['token']}.png" if receipt_data else None, receipt_data),
                )
            if receipt_path:
                receipt_path.unlink(missing_ok=True)
            session.pop("fee_review", None)
            return redirect(url_for("management_fees", **{"class": class_name, "message": "Fee record confirmed and locked."}))

        try:
            student_id = int(request.form.get("student_id", "-1"))
        except ValueError:
            student_id = -1
        fee_month = request.form.get("fee_month", "")
        status = request.form.get("status", "")
        if fee_month not in FEE_MONTHS or status not in FEE_STATUSES:
            return render_fee_class(school, class_name, "Invalid fee selection.")
        with game_connection() as connection:
            student = connection.execute("SELECT id FROM managed_students WHERE id = ? AND school_id = ? AND class_name = ?", (student_id, school["id"], class_name)).fetchone()
            current = connection.execute("SELECT locked FROM student_fee_status WHERE school_id = ? AND class_name = ? AND student_id = ? AND fee_month = ?", (school["id"], class_name, student_id, fee_month)).fetchone()
        if student is None or current and current["locked"]:
            return render_fee_class(school, class_name, "This fee record is already locked or unavailable.")
        if status == "Pending":
            with game_connection() as connection:
                connection.execute("INSERT INTO student_fee_status (school_id, class_name, student_id, fee_month, status, locked, created_at, updated_at) VALUES (?, ?, ?, ?, 'Pending', 0, datetime('now'), datetime('now')) ON CONFLICT(school_id, class_name, student_id, fee_month) DO UPDATE SET status = 'Pending', updated_at = datetime('now') WHERE locked = 0", (school["id"], class_name, student_id, fee_month))
            return redirect(url_for("management_fees", **{"class": class_name, "message": "Pending status saved."}))
        receipt_path = None
        token = uuid.uuid4().hex
        if status == "Paid":
            upload = request.files.get("receipt_photo")
            if upload is None or not upload.filename:
                return render_fee_class(school, class_name, "A receipt photo is required for Paid status.")
            data = upload.read(MAX_RECEIPT_BYTES + 1)
            if len(data) > MAX_RECEIPT_BYTES:
                return render_fee_class(school, class_name, "Receipt photo must be 5 MB or smaller.")
            try:
                image = Image.open(io.BytesIO(data))
                image.verify()
                image = Image.open(io.BytesIO(data))
                if image.format not in ("JPEG", "PNG", "WEBP"):
                    raise ValueError
                image.load()
                converted = io.BytesIO()
                image.convert("RGB").save(converted, format="PNG", optimize=True)
                FEE_RECEIPT_STAGING.mkdir(parents=True, exist_ok=True)
                receipt_path = FEE_RECEIPT_STAGING / f"{token}.png"
                receipt_path.write_bytes(converted.getvalue())
            except (OSError, ValueError):
                return render_fee_class(school, class_name, "Upload a valid JPEG, PNG, or WEBP receipt image.")
        session["fee_review"] = {"school_id": school["id"], "class_name": class_name, "student_id": student_id, "fee_month": fee_month, "status": status, "token": token}
        review = build_fee_review(school, class_name, student_id, fee_month, status, receipt_path, token)
        return render_fee_class(school, class_name, review=review)
    return render_fee_class(school, class_name, request.args.get("message"))


@app.get("/management/fees/download")
@school_required
def download_fee_records():
    class_name = request.args.get("class", "").strip()
    fee_month = request.args.get("month", "").strip()
    school = current_school()
    if fee_month:
        if class_name not in SCHOOL_CLASSES or fee_month not in FEE_MONTHS:
            return "Class or month not found.", 404
        rows, _ = fee_analysis_context(school["id"], class_name, fee_month)
        download_name = f"fee_records_{class_name.replace(' ', '_')}_{fee_month}.xlsx"
    else:
        with game_connection() as connection:
            query = (
                "SELECT f.status, f.receipt_data, s.student_name, s.uid_number, s.roll_number, f.class_name, f.fee_month "
                "FROM student_fee_status AS f "
                "JOIN managed_students AS s ON s.id = f.student_id AND s.school_id = f.school_id "
                "WHERE f.school_id = ? AND f.locked = 1"
            )
            params = [school["id"]]
            if class_name:
                if class_name not in SCHOOL_CLASSES:
                    return "Class not found.", 404
                query += " AND f.class_name = ?"
                params.append(class_name)
                download_name = f"fee_records_{class_name.replace(' ', '_')}.xlsx"
            else:
                download_name = f"fee_records_all_classes.xlsx"
            rows = [dict(row) for row in connection.execute(query, params).fetchall()]

    def month_sort_key(month):
        try:
            return FEE_MONTHS.index(month)
        except ValueError:
            return 999

    def roll_sort_key(roll):
        try:
            return int(roll)
        except (ValueError, TypeError):
            return 999999

    sorted_rows = sorted(rows, key=lambda r: (r.get("class_name", class_name), roll_sort_key(r["roll_number"]), month_sort_key(r.get("fee_month", fee_month))))

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Fee Records"
    sheet.append(["Student Name", "Class", "UID Number", "Roll Number", "Month", "Fee Status", "Receipt Photo"])
    sheet.row_dimensions[1].height = 25

    for row in sorted_rows:
        roll = int(row["roll_number"]) if str(row["roll_number"]).isdigit() else row["roll_number"]
        has_receipt = bool(row["receipt_data"])
        status_val = row["status"]
        sheet.append([
            row["student_name"],
            row.get("class_name", class_name),
            row["uid_number"],
            roll,
            row.get("fee_month", fee_month),
            status_val,
            ""
        ])
        current_row = sheet.max_row
        if has_receipt:
            try:
                img_io = io.BytesIO(row["receipt_data"])
                image = ExcelImage(img_io)
                orig_w, orig_h = image.width, image.height
                if orig_w > 0 and orig_h > 0:
                    scale = min(160 / orig_w, 75 / orig_h, 1.0)
                    image.width = int(orig_w * scale)
                    image.height = int(orig_h * scale)
                sheet.add_image(image, f"G{current_row}")
                sheet.row_dimensions[current_row].height = 65
            except Exception:
                sheet.cell(row=current_row, column=7, value="[Image Attached]")
        else:
            sheet.cell(row=current_row, column=7, value="No Receipt" if status_val == "Paid" else "-")

    sheet.column_dimensions["A"].width = 24
    sheet.column_dimensions["B"].width = 16
    sheet.column_dimensions["C"].width = 20
    sheet.column_dimensions["D"].width = 14
    sheet.column_dimensions["E"].width = 16
    sheet.column_dimensions["F"].width = 14
    sheet.column_dimensions["G"].width = 26

    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name=download_name, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


CALENDAR_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Academic Calendar | Student Support Dashboard</title><style>:root{--ink:#f7f1e8;--muted:#9ca0ab;--line:#30333e;--amber:#f39a4b;--light:#ffc477;--coral:#e87551;--mint:#92c9ae}*{box-sizing:border-box}body{min-height:100vh;margin:0;background:radial-gradient(circle at 80% 0%,#382317 0,#15151d 30%,#0b0c12 70%);color:var(--ink);font:14px/1.5 "Trebuchet MS","Segoe UI",sans-serif}.top{display:flex;justify-content:space-between;align-items:center;padding:20px max(22px,5vw);border-bottom:1px solid #ffffff12;background:#0d0e14cc}.top a,.back{color:var(--light);text-decoration:none}.main{max-width:960px;margin:auto;padding:50px 22px 70px}.heading{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:24px}.eyebrow{color:var(--light);font-size:11px;font-weight:bold;letter-spacing:.13em;text-transform:uppercase}.heading h1{margin:8px 0 0;font-size:clamp(30px,5vw,46px)}.panel{padding:28px;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 20px 55px #0005;margin-bottom:24px}.panel h2{margin:0 0 16px;font-size:20px;color:var(--ink)}.panel p{color:var(--muted);font-size:14px;margin-top:0}.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.field{display:grid;gap:8px}.field label{color:#d8d1cc;font-size:13px;font-weight:bold}.field input[type=file]{width:100%;padding:10px;border:1px solid var(--line);border-radius:10px;background:#0e1017;color:var(--ink);font:inherit}.field input[type=file]::file-selector-button{background:#282b35;color:var(--light);border:1px solid var(--line);border-radius:6px;padding:6px 12px;margin-right:10px;cursor:pointer;font-weight:bold}.button{display:inline-block;border:0;border-radius:10px;padding:11px 18px;background:linear-gradient(135deg,var(--amber),var(--coral));color:#1a120e;font:inherit;font-weight:bold;cursor:pointer;text-decoration:none}.button.secondary{background:#282b35;color:var(--light);font-size:12px;padding:7px 12px}.button.danger{background:#512c28;color:#ffc5b3;border:1px solid #e8755159}.actions{margin-top:20px;display:flex;gap:12px;align-items:center}.message{padding:12px 16px;border-radius:10px;margin-bottom:20px;background:#1b3b31;color:#b7f0ce;border:1px solid #92c9ae33}.error{padding:12px 16px;border-radius:10px;margin-bottom:20px;background:#512c2859;color:#ffc5b3;border:1px solid #e8755159}.calendar-photos-grid{display:grid;gap:20px;grid-template-columns:{% if photo1 and photo2 %}repeat(2,minmax(0,1fr)){% else %}1fr{% endif %};margin-top:16px}.photo-card{border:1px solid var(--line);border-radius:16px;background:#12141c;padding:16px;display:flex;flex-direction:column;gap:12px}.photo-card-header{display:flex;justify-content:space-between;align-items:center}.photo-card-title{font-weight:bold;color:var(--light);font-size:14px}.photo-preview{width:100%;max-height:480px;object-fit:contain;border-radius:10px;background:#08090d;border:1px solid #ffffff12}.empty-state{color:var(--muted);font-size:14px;margin:12px 0}@media(max-width:700px){.form-grid,.calendar-photos-grid{grid-template-columns:1fr}.main{padding-top:30px}.panel{padding:20px}}</style></head><body><header class="top"><strong>Student Support Dashboard</strong><span>{{ school_name }} &nbsp; <a href="/school-logout">School Logout</a></span></header><main class="main"><div class="heading"><div><span class="eyebrow">School management</span><h1>Academic Calendar</h1></div><a class="back" href="/?view=home">&#8592; Back to Dashboard</a></div>{% if message %}<div class="message">{{ message }}</div>{% endif %}{% if error %}<div class="error">{{ error }}</div>{% endif %}<section class="panel"><h2>Upload Academic Calendar Photos</h2><p>Attach up to two academic calendar photos (JPG, JPEG, PNG, WEBP — max 5 MB each).</p><form method="post" enctype="multipart/form-data" action="/management/calendar"><div class="form-grid"><div class="field"><label for="calendar_photo_1">Academic Calendar Photo 1</label><input id="calendar_photo_1" name="calendar_photo_1" type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"></div><div class="field"><label for="calendar_photo_2">Academic Calendar Photo 2</label><input id="calendar_photo_2" name="calendar_photo_2" type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"></div></div><div class="actions"><button class="button" type="submit">Save Academic Calendar</button></div></form></section><section class="panel"><h2>Academic Calendar Display</h2>{% if photo1 or photo2 %}<div class="calendar-photos-grid">{% if photo1 %}<div class="photo-card"><div class="photo-card-header"><span class="photo-card-title">Academic Calendar Photo 1</span><form method="post" action="/management/calendar" style="margin:0"><input type="hidden" name="delete_photo" value="1"><button class="button secondary danger" type="submit">Remove</button></form></div><img class="photo-preview" src="/calendar-photo/1" alt="Academic Calendar Photo 1"></div>{% endif %}{% if photo2 %}<div class="photo-card"><div class="photo-card-header"><span class="photo-card-title">Academic Calendar Photo 2</span><form method="post" action="/management/calendar" style="margin:0"><input type="hidden" name="delete_photo" value="2"><button class="button secondary danger" type="submit">Remove</button></form></div><img class="photo-preview" src="/calendar-photo/2" alt="Academic Calendar Photo 2"></div>{% endif %}</div>{% else %}<p class="empty-state">This workspace is ready for school-managed records. No management records are available yet.</p>{% endif %}</section></main></body></html>
"""


@app.route("/management/calendar", methods=["GET", "POST"])
@school_required
def management_calendar():
    school = current_school()
    school_id = school["id"]
    error = None
    message = request.args.get("message")

    with game_connection() as connection:
        cal = connection.execute(
            "SELECT photo1_filename, photo2_filename FROM school_academic_calendar WHERE school_id = ?",
            (school_id,),
        ).fetchone()

    photo1_filename = cal["photo1_filename"] if cal else None
    photo2_filename = cal["photo2_filename"] if cal else None

    if request.method == "POST":
        delete_photo = request.form.get("delete_photo")
        if delete_photo in ("1", "2"):
            target_file = photo1_filename if delete_photo == "1" else photo2_filename
            if target_file:
                safe_name = Path(target_file).name
                if safe_name == target_file:
                    (CALENDAR_PHOTO_STORAGE / safe_name).unlink(missing_ok=True)
            with game_connection() as connection:
                if delete_photo == "1":
                    connection.execute(
                        "UPDATE school_academic_calendar SET photo1_filename = NULL, updated_at = datetime('now') WHERE school_id = ?",
                        (school_id,),
                    )
                else:
                    connection.execute(
                        "UPDATE school_academic_calendar SET photo2_filename = NULL, updated_at = datetime('now') WHERE school_id = ?",
                        (school_id,),
                    )
            return redirect(url_for("management_calendar", message=f"Academic Calendar Photo {delete_photo} removed."))

        upload1 = request.files.get("calendar_photo_1")
        upload2 = request.files.get("calendar_photo_2")

        has_file1 = upload1 is not None and bool(upload1.filename and upload1.filename.strip())
        has_file2 = upload2 is not None and bool(upload2.filename and upload2.filename.strip())

        if not has_file1 and not has_file2:
            return render_template_string(
                CALENDAR_TEMPLATE,
                school_name=school["school_name"],
                photo1=photo1_filename,
                photo2=photo2_filename,
                message=None,
                error="Please select at least one photo to upload.",
            )

        new_photo1 = photo1_filename
        new_photo2 = photo2_filename

        try:
            if has_file1:
                saved_1 = save_calendar_photo(upload1, school_id, 1)
                if photo1_filename:
                    (CALENDAR_PHOTO_STORAGE / Path(photo1_filename).name).unlink(missing_ok=True)
                new_photo1 = saved_1

            if has_file2:
                saved_2 = save_calendar_photo(upload2, school_id, 2)
                if photo2_filename:
                    (CALENDAR_PHOTO_STORAGE / Path(photo2_filename).name).unlink(missing_ok=True)
                new_photo2 = saved_2
        except ValueError as exc:
            return render_template_string(
                CALENDAR_TEMPLATE,
                school_name=school["school_name"],
                photo1=photo1_filename,
                photo2=photo2_filename,
                message=None,
                error=str(exc),
            )

        with game_connection() as connection:
            connection.execute(
                "INSERT INTO school_academic_calendar (school_id, photo1_filename, photo2_filename, created_at, updated_at) "
                "VALUES (?, ?, ?, datetime('now'), datetime('now')) "
                "ON CONFLICT(school_id) DO UPDATE SET photo1_filename = excluded.photo1_filename, photo2_filename = excluded.photo2_filename, updated_at = datetime('now')",
                (school_id, new_photo1, new_photo2),
            )

        return redirect(url_for("management_calendar", message="Academic Calendar updated successfully."))

    return render_template_string(
        CALENDAR_TEMPLATE,
        school_name=school["school_name"],
        photo1=photo1_filename,
        photo2=photo2_filename,
        message=message,
        error=error,
    )


@app.get("/calendar-photo/<int:photo_num>")
@school_required
def calendar_photo(photo_num):
    if photo_num not in (1, 2):
        return "Photo not found.", 404
    school = current_school()
    if school is None:
        return "Unauthorized.", 401
    school_id = school["id"]
    with game_connection() as connection:
        cal = connection.execute(
            "SELECT photo1_filename, photo2_filename FROM school_academic_calendar WHERE school_id = ?",
            (school_id,),
        ).fetchone()
    if cal is None:
        return "Photo not found.", 404
    filename = cal["photo1_filename"] if photo_num == 1 else cal["photo2_filename"]
    if not filename:
        return "Photo not found.", 404
    safe_name = Path(filename).name
    if safe_name != filename or not re.fullmatch(r"school_\d+_cal[12]_[a-f0-9]+\.(jpg|jpeg|png|webp)", filename):
        return "Photo not found.", 404
    path = CALENDAR_PHOTO_STORAGE / safe_name
    if not path.is_file() or path.parent.resolve() != CALENDAR_PHOTO_STORAGE.resolve():
        return "Photo not found.", 404
    ext = path.suffix.lower()
    mimetype = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
    return send_file(path, mimetype=mimetype, max_age=3600)


EXAM_COMMON_STYLE = """
:root{--ink:#f7f1e8;--muted:#9ca0ab;--line:#30333e;--amber:#f39a4b;--light:#ffc477;--coral:#e87551;--mint:#92c9ae;--card-bg:linear-gradient(145deg,#1a1c25,#15161d)}
*{box-sizing:border-box}
body{min-height:100vh;margin:0;background:radial-gradient(circle at 80% 0%,#382317 0,#15151d 30%,#0b0c12 70%);color:var(--ink);font:14px/1.5 "Trebuchet MS","Segoe UI",sans-serif}
.top{display:flex;justify-content:space-between;align-items:center;padding:20px max(22px,5vw);border-bottom:1px solid #ffffff12;background:#0d0e14cc}
.top a,.back{color:var(--light);text-decoration:none}
.main{max-width:1120px;margin:auto;padding:40px 22px 70px}
.heading{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:24px}
.eyebrow{color:var(--light);font-size:11px;font-weight:bold;letter-spacing:.13em;text-transform:uppercase}
.heading h1{margin:8px 0 0;font-size:clamp(24px,4vw,38px)}
.panel{padding:24px;border:1px solid var(--line);border-radius:18px;background:var(--card-bg);box-shadow:0 20px 55px #0005;margin-bottom:24px}
.panel h2,.panel h3{margin:0 0 14px;color:var(--ink)}
.panel p{color:var(--muted);font-size:13px;margin-top:0}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px}
.stat-card{padding:18px;border:1px solid var(--line);border-radius:14px;background:#12141c}
.stat-card span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.stat-card b{display:block;font-size:24px;color:var(--light);margin-top:4px}
.form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}
.field{display:grid;gap:6px}
.field label{color:#d8d1cc;font-size:12px;font-weight:bold}
.field input,.field select{width:100%;padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:#0e1017;color:var(--ink);font:inherit}
.button{display:inline-block;border:0;border-radius:8px;padding:9px 16px;background:linear-gradient(135deg,var(--amber),var(--coral));color:#1a120e;font:inherit;font-weight:bold;cursor:pointer;text-decoration:none}
.button.secondary{background:#282b35;color:var(--light);font-size:12px;padding:6px 12px;border:1px solid var(--line)}
.button.secondary:hover{background:#353845}
.button.danger{background:#512c28;color:#ffc5b3;border:1px solid #e8755159}
.actions{margin-top:18px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.message{padding:12px 16px;border-radius:10px;margin-bottom:20px;background:#1b3b31;color:#b7f0ce;border:1px solid #92c9ae33}
.error{padding:12px 16px;border-radius:10px;margin-bottom:20px;background:#512c2859;color:#ffc5b3;border:1px solid #e8755159}
.table-wrap{overflow-x:auto;margin-top:16px}
table.results{width:100%;border-collapse:collapse;font-size:13px}
table.results th,table.results td{padding:10px 12px;text-align:left;border-bottom:1px solid #ffffff12}
table.results th{background:#0e1017;color:var(--light);font-weight:600;font-size:11px;text-transform:uppercase}
table.results tr:hover{background:#ffffff05}
.badge{display:inline-block;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:bold;text-transform:uppercase}
.badge.published,.badge.pass,.badge.locked{background:#1c3d2f;color:#85e3b5;border:1px solid #2e604a}
.badge.draft,.badge.pending{background:#3d361c;color:#e3cb85;border:1px solid #60552e}
.badge.fail,.badge.archived,.badge.absent{background:#3d1c1c;color:#e38585;border:1px solid #602e2e}
.flex-between{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
.mark-input{width:75px !important;padding:6px 8px !important;text-align:center}
.absent-check{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:var(--muted);cursor:pointer;margin-left:4px}
.nav-tabs{display:flex;gap:10px;margin-bottom:24px;border-bottom:1px solid var(--line);padding-bottom:12px;flex-wrap:wrap}
.nav-tab{padding:8px 16px;border-radius:8px;background:#12141c;color:var(--muted);text-decoration:none;font-weight:bold;font-size:13px;border:1px solid var(--line)}
.nav-tab.active,.nav-tab:hover{background:var(--amber);color:#1a120e}
.report-card{border:2px solid var(--amber);padding:30px;border-radius:18px;background:#12141c;margin-top:24px}
.report-header{text-align:center;border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:20px}
.report-header h2{margin:4px 0;color:var(--light);font-size:22px}
@media print{
  body{background:#fff;color:#000}
  .top,.heading .back,.nav-tabs,.actions,.form-grid,.filter-panel,.no-print{display:none !important}
  .panel,.report-card{border:1px solid #000;box-shadow:none;background:#fff;color:#000;padding:15px;margin:0}
  .report-header h2{color:#000}
  table.results th{background:#eee;color:#000}
  table.results th,table.results td{border:1px solid #ccc;color:#000}
  .badge{border:1px solid #000;color:#000;background:none}
}
@media(max-width:700px){.main{padding-top:20px}.panel{padding:16px}.form-grid{grid-template-columns:1fr}}
"""

EXAM_DASHBOARD_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Examination Management | Student Support</title><style>""" + EXAM_COMMON_STYLE + """</style></head><body>
<header class="top"><strong>Student Support Dashboard</strong><span>{{ school_name }} &nbsp; <a href="/school-logout">School Logout</a></span></header>
<main class="main">
<div class="heading"><div><span class="eyebrow">School Management</span><h1>Examination & Result Management</h1></div><a class="back" href="/?view=home">&#8592; Back to Dashboard</a></div>

<div class="nav-tabs">
  <a class="nav-tab active" href="/management/examinations">Exams Dashboard</a>
  <a class="nav-tab" href="/management/examinations/subjects">Subject Management</a>
  <a class="nav-tab" href="/management/examinations/results">Result & Reports</a>
</div>

{% if message %}<div class="message">{{ message }}</div>{% endif %}
{% if error %}<div class="error">{{ error }}</div>{% endif %}

<div class="stats-grid">
  <div class="stat-card"><span>Total Examinations</span><b>{{ total_exams }}</b></div>
  <div class="stat-card"><span>Published Exams</span><b>{{ published_exams }}</b></div>
  <div class="stat-card"><span>Draft Exams</span><b>{{ draft_exams }}</b></div>
  <div class="stat-card"><span>Active Subjects</span><b>{{ total_active_subjects }}</b></div>
</div>

<section class="panel">
  <h2>Create New Examination</h2>
  <form method="post" action="/management/examinations/create">
    <div class="form-grid">
      <div class="field">
        <label for="class_name">Select Class *</label>
        <select id="class_name" name="class_name" required>
          <option value="">-- Select Class --</option>
          {% for c in classes %}
          <option value="{{ c }}">{{ c }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="field">
        <label for="exam_name">Examination Name *</label>
        <input id="exam_name" name="exam_name" placeholder="e.g. Mid Term Exam 2026" required>
      </div>
      <div class="field">
        <label for="academic_session">Academic Session *</label>
        <input id="academic_session" name="academic_session" value="2025-2026" required>
      </div>
      <div class="field">
        <label for="passing_percentage">Passing %</label>
        <input id="passing_percentage" name="passing_percentage" type="number" step="0.1" value="33" required>
      </div>
      <div class="field">
        <label for="start_date">Start Date</label>
        <input id="start_date" name="start_date" type="date">
      </div>
      <div class="field">
        <label for="end_date">End Date</label>
        <input id="end_date" name="end_date" type="date">
      </div>
    </div>
    <div class="actions">
      <button class="button" type="submit">Create Examination</button>
    </div>
  </form>
</section>

<section class="panel">
  <div class="flex-between">
    <h2>All Examinations</h2>
    <form method="get" action="/management/examinations" class="no-print" style="display:flex;gap:10px;align-items:center;">
      <select name="class" onchange="this.form.submit()">
        <option value="">All Classes</option>
        {% for c in classes %}
        <option value="{{ c }}" {% if selected_class == c %}selected{% endif %}>{{ c }}</option>
        {% endfor %}
      </select>
      <select name="status" onchange="this.form.submit()">
        <option value="">All Statuses</option>
        <option value="Draft" {% if selected_status == 'Draft' %}selected{% endif %}>Draft</option>
        <option value="Published" {% if selected_status == 'Published' %}selected{% endif %}>Published</option>
      </select>
    </form>
  </div>

  {% if exams %}
  <div class="table-wrap">
    <table class="results">
      <thead>
        <tr>
          <th>Class</th>
          <th>Exam Name</th>
          <th>Session</th>
          <th>Configured Subjects</th>
          <th>Marks Entered</th>
          <th>Pass %</th>
          <th>Status</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {% for exam in exams %}
        <tr>
          <td><b>{{ exam.class_name }}</b></td>
          <td>{{ exam.exam_name }}</td>
          <td>{{ exam.academic_session }}</td>
          <td>{{ exam.subject_count }} subjects</td>
          <td>{{ exam.marks_entered_count }} / {{ exam.total_students }} students</td>
          <td>{{ exam.passing_percentage }}%</td>
          <td><span class="badge {{ exam.status|lower }}">{{ exam.status }}</span></td>
          <td>
            <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
              <a class="button secondary" href="/management/examinations/{{ exam.id }}/marks">Enter Marks</a>
              <a class="button secondary" href="/management/examinations/results?class={{ exam.class_name|urlencode }}&exam_id={{ exam.id }}">Results</a>
              <form method="post" action="/management/examinations/{{ exam.id }}/status" style="margin:0">
                <input type="hidden" name="status" value="{{ 'Published' if exam.status == 'Draft' else 'Draft' }}">
                <button class="button secondary" type="submit">{{ 'Publish' if exam.status == 'Draft' else 'Unpublish' }}</button>
              </form>
              <form method="post" action="/management/examinations/{{ exam.id }}/delete" style="margin:0" onsubmit="return confirm('Are you sure you want to delete this examination and all its marks?');">
                <button class="button secondary danger" type="submit">Delete</button>
              </form>
            </div>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <p style="color:var(--muted);margin-top:16px;">No examinations created yet. Use the form above to create an examination for any class from PG to Class 12.</p>
  {% endif %}
</section>
</main></body></html>"""

EXAM_SUBJECTS_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Subject Management | Student Support</title><style>""" + EXAM_COMMON_STYLE + """</style></head><body>
<header class="top"><strong>Student Support Dashboard</strong><span>{{ school_name }} &nbsp; <a href="/school-logout">School Logout</a></span></header>
<main class="main">
<div class="heading"><div><span class="eyebrow">School Management</span><h1>Subject Management</h1></div><a class="back" href="/management/examinations">&#8592; Back to Examinations</a></div>

<div class="nav-tabs">
  <a class="nav-tab" href="/management/examinations">Exams Dashboard</a>
  <a class="nav-tab active" href="/management/examinations/subjects">Subject Management</a>
  <a class="nav-tab" href="/management/examinations/results">Result & Reports</a>
</div>

{% if message %}<div class="message">{{ message }}</div>{% endif %}
{% if error %}<div class="error">{{ error }}</div>{% endif %}

<section class="panel">
  <div class="flex-between">
    <h2>Select Class to Manage Subjects</h2>
    <form method="get" action="/management/examinations/subjects" style="display:flex;gap:10px;align-items:center;">
      <select name="class" onchange="this.form.submit()" style="padding:9px 14px;border-radius:8px;background:#0e1017;color:var(--ink);border:1px solid var(--line);">
        {% for c in classes %}
        <option value="{{ c }}" {% if selected_class == c %}selected{% endif %}>{{ c }}</option>
        {% endfor %}
      </select>
    </form>
  </div>
</section>

<section class="panel">
  <div class="flex-between">
    <h2>Add New Subject for {{ selected_class }}</h2>
    <a class="button secondary" href="/management/examinations/subjects/seed?class={{ selected_class|urlencode }}">Seed Default 16 Subjects</a>
  </div>
  <form method="post" action="/management/examinations/subjects" style="margin-top:16px;">
    <input type="hidden" name="class_name" value="{{ selected_class }}">
    <div class="form-grid">
      <div class="field">
        <label for="subject_name">Subject Name *</label>
        <input id="subject_name" name="subject_name" placeholder="e.g. Mathematics" required>
      </div>
      <div class="field">
        <label for="max_marks">Maximum Marks *</label>
        <input id="max_marks" name="max_marks" type="number" value="100" required>
      </div>
      <div class="field">
        <label for="pass_marks">Passing Marks *</label>
        <input id="pass_marks" name="pass_marks" type="number" value="33" required>
      </div>
      <div class="field">
        <label for="display_order">Display Order</label>
        <input id="display_order" name="display_order" type="number" value="{{ (subjects|length) + 1 }}">
      </div>
    </div>
    <div class="actions">
      <button class="button" type="submit">Add Subject</button>
    </div>
  </form>
</section>

<section class="panel">
  <h2>Configured Subjects for {{ selected_class }} ({{ subjects|length }})</h2>
  {% if subjects %}
  <div class="table-wrap">
    <table class="results">
      <thead>
        <tr>
          <th>Order</th>
          <th>Subject Name</th>
          <th>Max Marks</th>
          <th>Pass Marks</th>
          <th>Status</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
        {% for s in subjects %}
        <tr>
          <td>{{ s.subject_order }}</td>
          <td><b>{{ s.subject_name }}</b></td>
          <td>{{ s.max_marks }}</td>
          <td>{{ s.pass_marks }}</td>
          <td><span class="badge {% if s.is_active %}pass{% else %}fail{% endif %}">{{ 'Active' if s.is_active else 'Inactive' }}</span></td>
          <td>
            <form method="post" action="/management/examinations/subjects/toggle" style="margin:0">
              <input type="hidden" name="subject_id" value="{{ s.id }}">
              <input type="hidden" name="class_name" value="{{ selected_class }}">
              <button class="button secondary" type="submit">{{ 'Deactivate' if s.is_active else 'Activate' }}</button>
            </form>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <p style="color:var(--muted);margin-top:16px;">No subjects added for {{ selected_class }} yet. Click "Seed Default 16 Subjects" above or add subjects manually.</p>
  {% endif %}
</section>
</main></body></html>"""

EXAM_MARKS_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Marks Entry | Student Support</title><style>""" + EXAM_COMMON_STYLE + """</style></head><body>
<header class="top"><strong>Student Support Dashboard</strong><span>{{ school_name }} &nbsp; <a href="/school-logout">School Logout</a></span></header>
<main class="main">
<div class="heading"><div><span class="eyebrow">Marks Entry</span><h1>{{ exam.exam_name }} ({{ exam.class_name }})</h1></div><a class="back" href="/management/examinations">&#8592; Back to Examinations</a></div>

<div class="nav-tabs">
  <a class="nav-tab" href="/management/examinations">Exams Dashboard</a>
  <a class="nav-tab" href="/management/examinations/subjects?class={{ exam.class_name|urlencode }}">Subject Management</a>
  <a class="nav-tab" href="/management/examinations/results?class={{ exam.class_name|urlencode }}&exam_id={{ exam.id }}">Result & Reports</a>
</div>

{% if message %}<div class="message">{{ message }}</div>{% endif %}
{% if error %}<div class="error">{{ error }}</div>{% endif %}

<div class="stats-grid">
  <div class="stat-card"><span>Exam</span><b>{{ exam.exam_name }}</b></div>
  <div class="stat-card"><span>Class</span><b>{{ exam.class_name }}</b></div>
  <div class="stat-card"><span>Session</span><b>{{ exam.academic_session }}</b></div>
  <div class="stat-card"><span>Students Count</span><b>{{ students|length }}</b></div>
</div>

{% if not subjects %}
<div class="error">No active subjects found for {{ exam.class_name }}. Please <a href="/management/examinations/subjects?class={{ exam.class_name|urlencode }}" style="color:var(--light)">configure subjects</a> first.</div>
{% elif not students %}
<div class="error">No enrolled students found in {{ exam.class_name }}. Please add students to {{ exam.class_name }} first.</div>
{% else %}
<section class="panel">
  <div class="flex-between">
    <h2>Student Marks Entry Matrix</h2>
    <span style="color:var(--muted);font-size:12px;">Enter obtained marks or check <b>A</b> for Absent</span>
  </div>

  <form method="post" action="/management/examinations/{{ exam.id }}/marks">
    <div class="table-wrap">
      <table class="results">
        <thead>
          <tr>
            <th>Roll No</th>
            <th>Student Name</th>
            <th>UID</th>
            {% for subj in subjects %}
            <th>{{ subj.subject_name }}<br><small style="color:var(--muted)">Max: {{ subj.max_marks }} | Pass: {{ subj.pass_marks }}</small></th>
            {% endfor %}
            <th>Remarks</th>
          </tr>
        </thead>
        <tbody>
          {% for stud in students %}
          {% set stud_id = stud.id %}
          <tr>
            <td>{{ stud.roll_number }}</td>
            <td><b>{{ stud.student_name }}</b></td>
            <td><small style="color:var(--muted)">{{ stud.uid_number }}</small></td>
            {% for subj in subjects %}
            {% set subj_id = subj.id %}
            {% set mark_key = stud_id ~ '_' ~ subj_id %}
            {% set m = marks_map.get(mark_key) %}
            <td>
              <div style="display:flex;align-items:center;">
                <input class="mark-input" name="marks_{{ stud_id }}_{{ subj_id }}" type="number" step="0.5" min="0" max="{{ subj.max_marks }}" value="{{ m.obtained_marks if m and m.obtained_marks is not none else '' }}" placeholder="Marks">
                <label class="absent-check">
                  <input type="checkbox" name="absent_{{ stud_id }}_{{ subj_id }}" value="1" {% if m and m.is_absent %}checked{% endif %}> A
                </label>
              </div>
            </td>
            {% endfor %}
            <td>
              {% set first_subj_key = stud_id ~ '_' ~ subjects[0].id %}
              {% set first_m = marks_map.get(first_subj_key) %}
              <input name="remarks_{{ stud_id }}" value="{{ first_m.remarks if first_m else '' }}" placeholder="Remarks" style="padding:6px;border-radius:6px;background:#0e1017;color:var(--ink);border:1px solid var(--line);font-size:12px;">
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    <div class="actions" style="margin-top:20px;">
      <button class="button" type="submit">Save All Marks</button>
      <a class="button secondary" href="/management/examinations/results?class={{ exam.class_name|urlencode }}&exam_id={{ exam.id }}">View Result Sheet</a>
    </div>
  </form>
</section>
{% endif %}
</main></body></html>"""

EXAM_RESULTS_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Examination Results & Reports | Student Support</title><style>""" + EXAM_COMMON_STYLE + """</style></head><body>
<header class="top"><strong>Student Support Dashboard</strong><span>{{ school_name }} &nbsp; <a href="/school-logout">School Logout</a></span></header>
<main class="main">
<div class="heading"><div><span class="eyebrow">Reports & Analytics</span><h1>Examination Results</h1></div><a class="back" href="/management/examinations">&#8592; Back to Examinations</a></div>

<div class="nav-tabs no-print">
  <a class="nav-tab" href="/management/examinations">Exams Dashboard</a>
  <a class="nav-tab" href="/management/examinations/subjects">Subject Management</a>
  <a class="nav-tab active" href="/management/examinations/results">Result & Reports</a>
</div>

{% if message %}<div class="message no-print">{{ message }}</div>{% endif %}
{% if error %}<div class="error no-print">{{ error }}</div>{% endif %}

<section class="panel filter-panel no-print">
  <form method="get" action="/management/examinations/results">
    <div class="form-grid">
      <div class="field">
        <label for="class">Select Class</label>
        <select id="class" name="class" onchange="this.form.submit()">
          {% for c in classes %}
          <option value="{{ c }}" {% if selected_class == c %}selected{% endif %}>{{ c }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="field">
        <label for="exam_id">Select Examination</label>
        <select id="exam_id" name="exam_id" onchange="this.form.submit()">
          {% if exams %}
            {% for e in exams %}
            <option value="{{ e.id }}" {% if selected_exam and selected_exam.id == e.id %}selected{% endif %}>{{ e.exam_name }} ({{ e.academic_session }})</option>
            {% endfor %}
          {% else %}
            <option value="">No exams created for {{ selected_class }}</option>
          {% endif %}
        </select>
      </div>
      <div class="field" style="display:flex;align-items:end;">
        <button class="button" type="submit">View Results</button>
      </div>
    </div>
  </form>
</section>

{% if selected_exam and results %}
<div class="stats-grid">
  <div class="stat-card"><span>Total Students</span><b>{{ results|length }}</b></div>
  <div class="stat-card"><span>Pass Rate</span><b>{{ pass_rate }}%</b></div>
  <div class="stat-card"><span>Class Highest</span><b>{{ highest_percentage }}%</b></div>
  <div class="stat-card"><span>Class Average</span><b>{{ average_percentage }}%</b></div>
</div>

<section class="panel">
  <div class="flex-between">
    <div>
      <h2>Class {{ selected_class }} — {{ selected_exam.exam_name }} Result Sheet</h2>
      <p style="color:var(--muted);margin:0;">Academic Session: {{ selected_exam.academic_session }} | Passing Percentage: {{ selected_exam.passing_percentage }}%</p>
    </div>
    <button class="button secondary no-print" onclick="window.print()">Print Result Sheet</button>
  </div>

  <div class="table-wrap">
    <table class="results">
      <thead>
        <tr>
          <th>Roll No</th>
          <th>Student Name</th>
          <th>UID</th>
          {% for subj in subjects %}
          <th>{{ subj.subject_name }}</th>
          {% endfor %}
          <th>Total Marks</th>
          <th>%</th>
          <th>Grade</th>
          <th>Result</th>
          <th class="no-print">Report Card</th>
        </tr>
      </thead>
      <tbody>
        {% for r in results %}
        <tr>
          <td>{{ r.student.roll_number }}</td>
          <td><b>{{ r.student.student_name }}</b></td>
          <td><small style="color:var(--muted)">{{ r.student.uid_number }}</small></td>
          {% for sr in r.subject_results %}
          <td>
            {% if sr.is_absent %}
              <span class="badge absent">A</span>
            {% elif sr.obtained_marks is not none %}
              {{ sr.obtained_marks }} <small style="color:var(--muted)">/{{ sr.max_marks }}</small>
            {% else %}
              <span style="color:var(--muted)">—</span>
            {% endif %}
          </td>
          {% endfor %}
          <td><b>{{ r.total_obtained }}</b> / {{ r.total_max }}</td>
          <td><b>{{ r.percentage }}%</b></td>
          <td><span class="badge locked">{{ r.grade }}</span></td>
          <td><span class="badge {% if r.is_pass %}pass{% else %}fail{% endif %}">{{ 'PASSED' if r.is_pass else 'FAILED' }}</span></td>
          <td class="no-print">
            <a class="button secondary" href="/management/examinations/results?class={{ selected_class|urlencode }}&exam_id={{ selected_exam.id }}&student_id={{ r.student.id }}">Report Card</a>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</section>

{% if single_report %}
<section class="report-card" id="student-report-card">
  <div class="report-header">
    <span class="eyebrow" style="color:var(--amber)">Official Academic Performance Report</span>
    <h2>{{ school_name }}</h2>
    <p style="margin:4px 0 0;color:var(--muted);">{{ selected_exam.exam_name }} (Session {{ selected_exam.academic_session }})</p>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:20px;padding:16px;background:#0e1017;border-radius:12px;border:1px solid var(--line);">
    <div><span style="color:var(--muted);font-size:11px;display:block">STUDENT NAME</span><b>{{ single_report.student.student_name }}</b></div>
    <div><span style="color:var(--muted);font-size:11px;display:block">CLASS</span><b>{{ single_report.student.class_name }}</b></div>
    <div><span style="color:var(--muted);font-size:11px;display:block">ROLL NUMBER</span><b>{{ single_report.student.roll_number }}</b></div>
    <div><span style="color:var(--muted);font-size:11px;display:block">UID NUMBER</span><b>{{ single_report.student.uid_number }}</b></div>
    <div><span style="color:var(--muted);font-size:11px;display:block">FATHER NAME</span><b>{{ single_report.student.father_name }}</b></div>
  </div>

  <div class="table-wrap">
    <table class="results">
      <thead>
        <tr>
          <th>Subject</th>
          <th>Max Marks</th>
          <th>Pass Marks</th>
          <th>Obtained Marks</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {% for sr in single_report.subject_results %}
        <tr>
          <td><b>{{ sr.subject_name }}</b></td>
          <td>{{ sr.max_marks }}</td>
          <td>{{ sr.pass_marks }}</td>
          <td><b>{{ sr.obtained_marks }}</b></td>
          <td><span class="badge {% if sr.status == 'Pass' %}pass{% elif sr.status == 'Fail' %}fail{% else %}pending{% endif %}">{{ sr.status }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div style="display:flex;justify-content:space-between;align-items:center;margin-top:24px;padding-top:16px;border-top:1px solid var(--line);flex-wrap:wrap;gap:16px;">
    <div>
      <span style="display:block;color:var(--muted);font-size:12px;">TOTAL SCORE: <b>{{ single_report.total_obtained }} / {{ single_report.total_max }}</b></span>
      <span style="display:block;color:var(--muted);font-size:12px;">PERCENTAGE: <b>{{ single_report.percentage }}%</b> &nbsp;|&nbsp; GRADE: <b>{{ single_report.grade }}</b></span>
    </div>
    <div>
      <span class="badge {% if single_report.is_pass %}pass{% else %}fail{% endif %}" style="font-size:14px;padding:6px 14px;">FINAL RESULT: {{ 'PASSED' if single_report.is_pass else 'FAILED' }}</span>
    </div>
  </div>

  <div class="actions no-print" style="margin-top:24px;">
    <button class="button" onclick="window.print()">Print Report Card</button>
    <a class="button secondary" href="/management/examinations/results?class={{ selected_class|urlencode }}&exam_id={{ selected_exam.id }}">Close Report Card</a>
  </div>
</section>
{% endif %}

{% elif selected_exam %}
<section class="panel">
  <p style="color:var(--muted)">No student marks entered for this exam yet. Go to <a href="/management/examinations/{{ selected_exam.id }}/marks" style="color:var(--light)">Marks Entry</a> to enter student marks.</p>
</section>
{% endif %}

</main></body></html>"""


@app.get("/management/examinations")
@school_required
def management_examinations():
    school = current_school()
    selected_class = request.args.get("class", "").strip()
    selected_status = request.args.get("status", "").strip()
    message = request.args.get("message")
    error = request.args.get("error")

    ctx = get_examination_context(
        school["id"],
        class_name=selected_class if selected_class else None,
        status_filter=selected_status if selected_status else None,
    )

    return render_template_string(
        EXAM_DASHBOARD_TEMPLATE,
        school_name=school["school_name"],
        classes=SCHOOL_CLASSES,
        selected_class=selected_class,
        selected_status=selected_status,
        message=message,
        error=error,
        **ctx,
    )


@app.post("/management/examinations/create")
@school_required
def management_examinations_create():
    school = current_school()
    class_name = request.form.get("class_name", "").strip()
    exam_name = request.form.get("exam_name", "").strip()
    academic_session = request.form.get("academic_session", "").strip()
    passing_percentage_str = request.form.get("passing_percentage", "33").strip()
    start_date = request.form.get("start_date", "").strip() or None
    end_date = request.form.get("end_date", "").strip() or None

    if class_name not in SCHOOL_CLASSES:
        return redirect(url_for("management_examinations", error="Invalid class selected."))
    if not exam_name or not academic_session:
        return redirect(url_for("management_examinations", error="Exam name and academic session are required."))

    try:
        passing_percentage = float(passing_percentage_str)
    except ValueError:
        passing_percentage = 33.0

    with game_connection() as connection:
        try:
            connection.execute(
                """
                INSERT INTO examinations (school_id, class_name, exam_name, academic_session, passing_percentage, start_date, end_date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'Draft')
                """,
                (school["id"], class_name, exam_name, academic_session, passing_percentage, start_date, end_date),
            )
            active_subj = connection.execute(
                "SELECT COUNT(*) FROM examination_subjects WHERE school_id = ? AND class_name = ? AND is_active = 1",
                (school["id"], class_name),
            ).fetchone()[0]
            if active_subj == 0:
                seed_default_subjects(school["id"], class_name)
        except sqlite3.IntegrityError:
            return redirect(url_for("management_examinations", error=f"An examination with name '{exam_name}' already exists for {class_name} ({academic_session})."))

    return redirect(url_for("management_examinations", message=f"Examination '{exam_name}' created successfully for {class_name}."))


@app.post("/management/examinations/<int:exam_id>/status")
@school_required
def management_examinations_status(exam_id):
    school = current_school()
    new_status = request.form.get("status", "").strip()
    if new_status not in ["Draft", "Published", "Archived"]:
        new_status = "Draft"
    with game_connection() as connection:
        connection.execute(
            "UPDATE examinations SET status = ?, updated_at = datetime('now', 'localtime') WHERE school_id = ? AND id = ?",
            (new_status, school["id"], exam_id),
        )
    return redirect(url_for("management_examinations", message=f"Examination status updated to {new_status}."))


@app.post("/management/examinations/<int:exam_id>/delete")
@school_required
def management_examinations_delete(exam_id):
    school = current_school()
    with game_connection() as connection:
        connection.execute(
            "DELETE FROM student_exam_marks WHERE school_id = ? AND exam_id = ?",
            (school["id"], exam_id),
        )
        connection.execute(
            "DELETE FROM examinations WHERE school_id = ? AND id = ?",
            (school["id"], exam_id),
        )
    return redirect(url_for("management_examinations", message="Examination and all associated marks deleted successfully."))


@app.get("/management/examinations/subjects")
@school_required
def management_examinations_subjects():
    school = current_school()
    selected_class = request.args.get("class", SCHOOL_CLASSES[0]).strip()
    if selected_class not in SCHOOL_CLASSES:
        selected_class = SCHOOL_CLASSES[0]

    message = request.args.get("message")
    error = request.args.get("error")

    with game_connection() as connection:
        subjects = connection.execute(
            "SELECT * FROM examination_subjects WHERE school_id = ? AND class_name = ? ORDER BY subject_order, subject_name",
            (school["id"], selected_class),
        ).fetchall()

    return render_template_string(
        EXAM_SUBJECTS_TEMPLATE,
        school_name=school["school_name"],
        classes=SCHOOL_CLASSES,
        selected_class=selected_class,
        subjects=[dict(s) for s in subjects],
        message=message,
        error=error,
    )


@app.post("/management/examinations/subjects")
@school_required
def management_examinations_subjects_post():
    school = current_school()
    class_name = request.form.get("class_name", "").strip()
    subject_name = request.form.get("subject_name", "").strip()
    max_marks_str = request.form.get("max_marks", "100").strip()
    pass_marks_str = request.form.get("pass_marks", "33").strip()
    display_order_str = request.form.get("display_order", "1").strip()

    if class_name not in SCHOOL_CLASSES:
        return redirect(url_for("management_examinations_subjects", error="Invalid class selected."))
    if not subject_name:
        return redirect(url_for("management_examinations_subjects", class_name=class_name, error="Subject name is required."))

    try:
        max_marks = float(max_marks_str)
        pass_marks = float(pass_marks_str)
        display_order = int(display_order_str)
    except ValueError:
        return redirect(url_for("management_examinations_subjects", class_name=class_name, error="Invalid max/pass marks or order."))

    with game_connection() as connection:
        try:
            connection.execute(
                """
                INSERT INTO examination_subjects (school_id, class_name, subject_name, max_marks, pass_marks, is_active, subject_order)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (school["id"], class_name, subject_name, max_marks, pass_marks, display_order),
            )
        except sqlite3.IntegrityError:
            connection.execute(
                """
                UPDATE examination_subjects SET max_marks = ?, pass_marks = ?, is_active = 1, subject_order = ?
                WHERE school_id = ? AND class_name = ? AND subject_name = ?
                """,
                (max_marks, pass_marks, display_order, school["id"], class_name, subject_name),
            )

    return redirect(url_for("management_examinations_subjects", **{"class": class_name, "message": f"Subject '{subject_name}' added for {class_name}."}))


@app.get("/management/examinations/subjects/seed")
@school_required
def management_examinations_subjects_seed():
    school = current_school()
    class_name = request.args.get("class", "").strip()
    if class_name in SCHOOL_CLASSES:
        seed_default_subjects(school["id"], class_name)
        return redirect(url_for("management_examinations_subjects", **{"class": class_name, "message": f"Default 16 subjects seeded for {class_name}."}))
    return redirect(url_for("management_examinations_subjects", error="Invalid class selected."))


@app.post("/management/examinations/subjects/toggle")
@school_required
def management_examinations_subjects_toggle():
    school = current_school()
    subject_id = request.form.get("subject_id")
    class_name = request.form.get("class_name", "").strip()
    with game_connection() as connection:
        connection.execute(
            "UPDATE examination_subjects SET is_active = CASE WHEN is_active = 1 THEN 0 ELSE 1 END WHERE school_id = ? AND id = ?",
            (school["id"], subject_id),
        )
    return redirect(url_for("management_examinations_subjects", **{"class": class_name, "message": "Subject status updated."}))


@app.get("/management/examinations/<int:exam_id>/marks")
@school_required
def management_examinations_marks(exam_id):
    school = current_school()
    message = request.args.get("message")
    error = request.args.get("error")

    with game_connection() as connection:
        exam = connection.execute(
            "SELECT * FROM examinations WHERE school_id = ? AND id = ?",
            (school["id"], exam_id),
        ).fetchone()
        if not exam:
            return redirect(url_for("management_examinations", error="Examination not found."))

        subjects = connection.execute(
            "SELECT * FROM examination_subjects WHERE school_id = ? AND class_name = ? AND is_active = 1 ORDER BY subject_order, subject_name",
            (school["id"], exam["class_name"]),
        ).fetchall()

        students = connection.execute(
            "SELECT * FROM managed_students WHERE school_id = ? AND class_name = ? ORDER BY roll_number, student_name",
            (school["id"], exam["class_name"]),
        ).fetchall()

        marks_rows = connection.execute(
            "SELECT * FROM student_exam_marks WHERE school_id = ? AND exam_id = ?",
            (school["id"], exam_id),
        ).fetchall()

        marks_map = {}
        for m in marks_rows:
            key = f"{m['student_id']}_{m['subject_id']}"
            marks_map[key] = dict(m)

    return render_template_string(
        EXAM_MARKS_TEMPLATE,
        school_name=school["school_name"],
        exam=dict(exam),
        subjects=[dict(s) for s in subjects],
        students=[dict(s) for s in students],
        marks_map=marks_map,
        message=message,
        error=error,
    )


@app.post("/management/examinations/<int:exam_id>/marks")
@school_required
def management_examinations_marks_post(exam_id):
    school = current_school()
    with game_connection() as connection:
        exam = connection.execute(
            "SELECT * FROM examinations WHERE school_id = ? AND id = ?",
            (school["id"], exam_id),
        ).fetchone()
        if not exam:
            return redirect(url_for("management_examinations", error="Examination not found."))

        subjects = connection.execute(
            "SELECT * FROM examination_subjects WHERE school_id = ? AND class_name = ? AND is_active = 1",
            (school["id"], exam["class_name"]),
        ).fetchall()

        students = connection.execute(
            "SELECT * FROM managed_students WHERE school_id = ? AND class_name = ?",
            (school["id"], exam["class_name"]),
        ).fetchall()

        for stud in students:
            stud_id = stud["id"]
            remarks = request.form.get(f"remarks_{stud_id}", "").strip()
            for subj in subjects:
                subj_id = subj["id"]
                mark_val = request.form.get(f"marks_{stud_id}_{subj_id}", "").strip()
                absent_val = request.form.get(f"absent_{stud_id}_{subj_id}", "").strip()

                is_absent = 1 if absent_val == "1" else 0
                obtained_marks = None
                if not is_absent and mark_val != "":
                    try:
                        obtained_marks = float(mark_val)
                    except ValueError:
                        obtained_marks = None

                if obtained_marks is not None or is_absent:
                    connection.execute(
                        """
                        INSERT INTO student_exam_marks (school_id, exam_id, student_id, subject_id, obtained_marks, is_absent, remarks, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
                        ON CONFLICT(school_id, exam_id, student_id, subject_id) DO UPDATE SET
                          obtained_marks = excluded.obtained_marks,
                          is_absent = excluded.is_absent,
                          remarks = excluded.remarks,
                          updated_at = excluded.updated_at
                        """,
                        (school["id"], exam_id, stud_id, subj_id, obtained_marks, is_absent, remarks),
                    )

    return redirect(url_for("management_examinations_marks", exam_id=exam_id, message="Student marks saved successfully."))


@app.get("/management/examinations/results")
@school_required
def management_examinations_results():
    school = current_school()
    selected_class = request.args.get("class", SCHOOL_CLASSES[0]).strip()
    if selected_class not in SCHOOL_CLASSES:
        selected_class = SCHOOL_CLASSES[0]

    exam_id_str = request.args.get("exam_id", "").strip()
    student_id_str = request.args.get("student_id", "").strip()
    message = request.args.get("message")
    error = request.args.get("error")

    with game_connection() as connection:
        exams = connection.execute(
            "SELECT * FROM examinations WHERE school_id = ? AND class_name = ? ORDER BY created_at DESC",
            (school["id"], selected_class),
        ).fetchall()

        selected_exam = None
        if exam_id_str:
            try:
                e_id = int(exam_id_str)
                selected_exam = connection.execute(
                    "SELECT * FROM examinations WHERE school_id = ? AND id = ?",
                    (school["id"], e_id),
                ).fetchone()
            except ValueError:
                selected_exam = None

        if not selected_exam and exams:
            selected_exam = exams[0]

        results = []
        subjects = []
        pass_rate = 0.0
        highest_percentage = 0.0
        average_percentage = 0.0
        single_report = None

        if selected_exam:
            subjects = connection.execute(
                "SELECT * FROM examination_subjects WHERE school_id = ? AND class_name = ? AND is_active = 1 ORDER BY subject_order, subject_name",
                (school["id"], selected_exam["class_name"]),
            ).fetchall()

            students = connection.execute(
                "SELECT * FROM managed_students WHERE school_id = ? AND class_name = ? ORDER BY roll_number, student_name",
                (school["id"], selected_exam["class_name"]),
            ).fetchall()

            passed_count = 0
            total_pct_sum = 0.0

            for stud in students:
                res = get_student_exam_result_summary(school["id"], selected_exam["id"], stud["id"])
                if res:
                    results.append(res)
                    if res["is_pass"]:
                        passed_count += 1
                    if res["percentage"] > highest_percentage:
                        highest_percentage = res["percentage"]
                    total_pct_sum += res["percentage"]

            if results:
                pass_rate = round((passed_count / len(results)) * 100, 1)
                average_percentage = round(total_pct_sum / len(results), 1)

            if student_id_str:
                try:
                    s_id = int(student_id_str)
                    single_report = get_student_exam_result_summary(school["id"], selected_exam["id"], s_id)
                except ValueError:
                    single_report = None

    return render_template_string(
        EXAM_RESULTS_TEMPLATE,
        school_name=school["school_name"],
        classes=SCHOOL_CLASSES,
        selected_class=selected_class,
        exams=[dict(e) for e in exams],
        selected_exam=dict(selected_exam) if selected_exam else None,
        subjects=[dict(s) for s in subjects],
        results=results,
        pass_rate=pass_rate,
        highest_percentage=highest_percentage,
        average_percentage=average_percentage,
        single_report=single_report,
        message=message,
        error=error,
    )


@app.get("/management/reports")
@school_required
def management_reports():
    school = current_school()
    with game_connection() as connection:
        reports = connection.execute(
            "SELECT id, report_name, description, original_filename, file_extension, uploaded_by, uploaded_at FROM school_reports WHERE school_id = ? ORDER BY uploaded_at DESC, id DESC",
            (school["id"],),
        ).fetchall()
    return render_management_page(
        "Reports",
        "View attendance, academic, examination, fee and student performance reports.",
        reports=[dict(report) for report in reports],
        report_form_open=request.args.get("add") == "1",
        message=request.args.get("message"),
        error=request.args.get("error"),
    )


@app.post("/management/reports")
@school_required
def management_reports_upload():
    school = current_school()
    report_name = normalize_report_name(request.form.get("report_name"))
    description = re.sub(r"\s+", " ", request.form.get("description", "").strip())[:500]
    upload = request.files.get("report_file")
    if not report_name:
        return redirect(url_for("management_reports", add="1", error="Report name is required."))
    try:
        stored_filename, original_filename, extension = save_report_file(upload, school["id"])
    except ValueError as exc:
        return redirect(url_for("management_reports", add="1", error=str(exc)))
    try:
        with game_connection() as connection:
            connection.execute(
                "INSERT INTO school_reports (school_id, report_name, description, original_filename, stored_filename, file_extension, uploaded_by, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))",
                (school["id"], report_name, description or None, original_filename, stored_filename, extension, school["username"]),
            )
    except sqlite3.Error:
        (REPORT_FILE_STORAGE / stored_filename).unlink(missing_ok=True)
        return redirect(url_for("management_reports", add="1", error="The report could not be saved."))
    return redirect(url_for("management_reports", message="Report added successfully."))


@app.get("/management/reports/<int:report_id>/file")
@school_required
def download_report_file(report_id):
    school = current_school()
    with game_connection() as connection:
        report = connection.execute(
            "SELECT original_filename, stored_filename, file_extension FROM school_reports WHERE id = ? AND school_id = ?",
            (report_id, school["id"]),
        ).fetchone()
    if report is None:
        return "Report not found.", 404
    stored_filename = report["stored_filename"]
    safe_name = Path(stored_filename).name
    if safe_name != stored_filename or not re.fullmatch(r"school_\d+_report_[a-f0-9]+\.(pdf|docx|xlsx|jpg|jpeg|png)", safe_name):
        return "Report not found.", 404
    path = REPORT_FILE_STORAGE / safe_name
    if not path.is_file() or path.parent.resolve() != REPORT_FILE_STORAGE.resolve():
        return "Report file not found.", 404
    mimetypes = {
        "pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "jpg": "image/jpeg",
        "jpeg": "image/jpeg", "png": "image/png",
    }
    return send_file(path, as_attachment=True, download_name=report["original_filename"], mimetype=mimetypes[report["file_extension"]])


@app.post("/management/reports/<int:report_id>/delete")
@school_required
def delete_report_file(report_id):
    school = current_school()
    with game_connection() as connection:
        report = connection.execute(
            "SELECT stored_filename FROM school_reports WHERE id = ? AND school_id = ?",
            (report_id, school["id"]),
        ).fetchone()
        if report is None:
            return redirect(url_for("management_reports", error="Report not found."))
        stored_filename = report["stored_filename"]
        connection.execute("DELETE FROM school_reports WHERE id = ? AND school_id = ?", (report_id, school["id"]))
    safe_name = Path(stored_filename).name
    if safe_name == stored_filename:
        (REPORT_FILE_STORAGE / safe_name).unlink(missing_ok=True)
    return redirect(url_for("management_reports", message="Report deleted successfully."))


GAME_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cognitive Game</title><style>
body{margin:0;background:#eef3f4;color:#203040;font:14px/1.5 Arial,sans-serif}.top{background:#173442;color:#fff;padding:22px max(22px,5vw);display:flex;justify-content:space-between}.top a{color:#d6e8ea}.main{max-width:900px;margin:auto;padding:34px 22px}.panel{background:#fff;border:1px solid #d8e2e6;border-radius:7px;padding:24px;box-shadow:0 12px 28px #20304012;margin-bottom:18px}.muted{color:#657783}.button{display:inline-block;background:#176b87;color:#fff;border:0;border-radius:4px;padding:11px 16px;font-weight:bold;text-decoration:none;cursor:pointer}.game-question{font-size:25px;margin:20px 0}.game-options{display:grid;gap:10px}.game-options button{background:#f4f6f7;border:1px solid #d8e2e6;padding:14px;text-align:left;border-radius:5px;cursor:pointer;font:inherit}.game-options button:hover{border-color:#176b87;background:#eaf6f8}.score-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.score-card{background:#f4f6f7;padding:14px;border-radius:5px}.score-card span{display:block;color:#657783;font-size:12px}.score-card b{font-size:22px}.history{width:100%;border-collapse:collapse}.history th,.history td{text-align:left;padding:10px;border-bottom:1px solid #d8e2e6}.score-bars{display:flex;align-items:end;gap:8px;height:130px;padding:15px 0}.score-bar{flex:1;background:#176b87;min-height:4px;border-radius:4px 4px 0 0;position:relative}.score-bar span{position:absolute;bottom:-22px;width:100%;text-align:center;font-size:11px;color:#657783}@media(max-width:650px){.score-cards{grid-template-columns:repeat(2,1fr)}.top{display:block}.top a{display:block;margin-top:8px}}
</style></head><body><header class="top"><strong>Student Cognitive Game</strong><a href="/?view=home">Back to Dashboard</a></header><main class="main"><div class="panel"><span class="muted">Welcome, {{ user['student_name'] }}</span><h1>Game-based cognitive performance</h1><p class="muted">Answer five short pattern, memory, and logic questions.</p>{% if question %}<p class="muted">Question {{ position }} of {{ total }}</p><div class="game-question">{{ question.prompt }}</div><form class="game-options" method="post">{% for option in question.options %}<button name="answer" value="{{ option }}">{{ option }}</button>{% endfor %}</form>{% else %}<p>This is an entertainment/educational game score, not a standardized IQ test.</p><a class="button" href="/game?start=1">Play Game</a>{% endif %}</div>{% if result %}<div class="panel"><h1>Game Complete!</h1><div class="score-cards"><div class="score-card"><span>Score</span><b>{{ result.score }}/{{ result.total }}</b></div><div class="score-card"><span>Accuracy</span><b>{{ result.accuracy }}%</b></div><div class="score-card"><span>Cognitive Level</span><b>{{ result.level }}</b></div><div class="score-card"><span>Best Score</span><b>{{ summary.best }}</b></div></div><p class="muted">These are game-performance levels, not IQ classifications.</p><a class="button" href="/game?start=1">Play Again</a></div>{% endif %}<div class="panel"><h2>My Progress</h2><div class="score-cards"><div class="score-card"><span>Games Played</span><b>{{ summary.games }}</b></div><div class="score-card"><span>Average Score</span><b>{{ summary.average }}</b></div><div class="score-card"><span>Best Score</span><b>{{ summary.best }}</b></div><div class="score-card"><span>Latest Score</span><b>{{ summary.latest }}</b></div></div>{% if summary.history %}<div class="score-bars">{% for score in summary.history|reverse %}<div class="score-bar" style="height:{{ (score['score'] / score['total_questions'] * 100)|int }}%"><span>{{ score['score'] }}/{{ score['total_questions'] }}</span></div>{% endfor %}</div><h3>My Game History</h3><table class="history"><tr><th>Date</th><th>Score</th><th>Accuracy</th><th>Level</th></tr>{% for score in summary.history %}<tr><td>{{ score['played_at'] }}</td><td>{{ score['score'] }}/{{ score['total_questions'] }}</td><td>{{ score['accuracy']|int }}%</td><td>{{ score['level'] }}</td></tr>{% endfor %}</table>{% else %}<p class="muted">Your completed games will appear here.</p>{% endif %}</div></main></body></html>
"""
GAME_TEMPLATE = GAME_TEMPLATE.replace('<strong>Student Cognitive Game</strong><a href="/?view=home">Back to Dashboard</a>', '<strong>Student Cognitive Game</strong><span><a href="/profile">Profile</a> &nbsp; <a href="/?view=home">Back to Dashboard</a></span>')
GAME_TEMPLATE = GAME_TEMPLATE.replace('</div></div><p class="muted">These are game-performance levels, not IQ classifications.</p>', '</div></div><p class="muted">These are game-performance levels, not IQ classifications.</p>{% if result.review %}<h2>Question Review</h2><div class="review-list">{% for item in result.review %}<div class="review-item"><b>{{ loop.index }}. {{ item.question }}</b><span>Your answer: {{ item.selected }}</span><span>Correct answer: {{ item.correct }}</span><strong class="{% if item.is_correct %}review-correct{% else %}review-wrong{% endif %}">{{ "Correct" if item.is_correct else "Wrong" }}</strong></div>{% endfor %}</div>{% endif %}')
GAME_TEMPLATE = GAME_TEMPLATE.replace('</style></head><body>', '.review-list{display:grid;gap:10px;margin:18px 0}.review-item{display:grid;gap:4px;background:#f4f6f7;border:1px solid #d8e2e6;border-radius:5px;padding:13px}.review-item span{color:#657783}.review-correct{color:#176b3a}.review-wrong{color:#a4472b}</style></head><body>')


STUDENT_DASHBOARD_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Student Dashboard</title><style>
:root{--ink:#f7f1e8;--muted:#aaa8b1;--line:#343640;--amber:#f39a4b;--light:#ffc477;--coral:#e87551}*{box-sizing:border-box}body{min-height:100vh;margin:0;background:radial-gradient(circle at 80% 0%,#382317 0,#15151d 30%,#0b0c12 70%);color:var(--ink);font:14px/1.5 "Trebuchet MS","Segoe UI",sans-serif}.top{display:flex;justify-content:space-between;align-items:center;padding:18px max(22px,5vw);border-bottom:1px solid #ffffff12;background:#0d0e14cc}.top a{color:var(--light);text-decoration:none}.main{max-width:1000px;margin:auto;padding:42px max(22px,5vw) 70px}.heading{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px}.eyebrow{color:var(--light);font-size:11px;font-weight:bold;letter-spacing:.13em;text-transform:uppercase}.heading h1{margin:8px 0 0;font-size:clamp(30px,5vw,46px)}.panel{border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 14px 35px #0004;padding:24px;margin-bottom:18px}.profile-layout{display:grid;grid-template-columns:160px 1fr;gap:24px}.student-photo{width:150px;height:180px;object-fit:cover;border-radius:12px;border:1px solid var(--line)}.photo-placeholder{display:grid;place-items:center;width:150px;height:180px;border:1px dashed var(--line);border-radius:12px;color:var(--muted);text-align:center}.details{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.details div{min-width:0}.details span{display:block;color:var(--muted);font-size:11px;font-weight:bold;text-transform:uppercase}.details strong{display:block;margin-top:4px;overflow-wrap:anywhere}.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.field{display:grid;gap:6px}.field.full{grid-column:1/-1}.field label{color:#d8d1cc;font-size:12px;font-weight:bold}.field input,.field textarea{width:100%;padding:10px 11px;border:1px solid var(--line);border-radius:9px;background:#0e1017;color:var(--ink);font:inherit}.field textarea{min-height:90px;resize:vertical}.actions{display:flex;gap:10px;align-items:center;margin-top:18px}.button{display:inline-block;border:0;border-radius:10px;padding:11px 16px;background:linear-gradient(135deg,var(--amber),var(--coral));color:#1a120e;font:inherit;font-weight:bold;cursor:pointer;text-decoration:none}.button.secondary{background:#282b35;color:var(--light)}.message{padding:11px 13px;border-radius:9px;margin-bottom:15px;background:#1b3b31;color:#b7f0ce}.error{padding:11px 13px;border-radius:9px;margin-bottom:15px;background:#512c2859;color:#ffc5b3;border:1px solid #e8755159}@media(max-width:650px){.top{display:block}.top a{display:inline-block;margin:7px 12px 0 0}.heading{align-items:flex-start;flex-direction:column}.profile-layout{grid-template-columns:1fr}.details,.form-grid{grid-template-columns:1fr}.field.full{grid-column:auto}}
</style></head><body><header class="top"><strong>Student Dashboard</strong><span><a href="/change-password">Change Password</a> <a href="/logout">Logout</a></span></header><main class="main"><div class="heading"><div><span class="eyebrow">Private student account</span><h1>Welcome, {{ student.student_name }}</h1></div><a class="top-link" href="/game">Cognitive Game</a></div>{% if message %}<div class="message">{{ message }}</div>{% endif %}{% if error %}<div class="error">{{ error }}</div>{% endif %}<section class="panel"><div class="profile-layout"><div>{% if student.photo_path %}<img class="student-photo" src="{{ url_for('student_photo', student_id=student.id) }}" alt="Profile photo">{% else %}<div class="photo-placeholder">No profile photo</div>{% endif %}</div><div><h2>Student Profile</h2><div class="details"><div><span>Student Name</span><strong>{{ student.student_name }}</strong></div><div><span>UID Number</span><strong>{{ student.uid_number }}</strong></div><div><span>Class</span><strong>{{ student.class_name }}</strong></div><div><span>Roll Number</span><strong>{{ student.roll_number }}</strong></div><div><span>Father's Name</span><strong>{{ student.father_name }}</strong></div><div><span>Mother's Name</span><strong>{{ student.mother_name }}</strong></div><div><span>Parent Contact</span><strong>{{ student.parent_mobile }}</strong></div><div><span>Second Contact</span><strong>{{ student.second_mobile or 'Not provided' }}</strong></div><div class="field full"><span>Address</span><strong>{{ student.address }}</strong></div></div><div class="actions"><a class="button" href="/student-dashboard?edit=1">Edit Profile</a></div></div></div></section>{% if edit %}<section class="panel"><h2>Edit Profile</h2><form method="post" action="/student-dashboard"><div class="form-grid"><div class="field"><label for="student_name">Student Name</label><input id="student_name" name="student_name" value="{{ student.student_name }}" required></div><div class="field"><label for="parent_mobile">Parent Contact</label><input id="parent_mobile" name="parent_mobile" value="{{ student.parent_mobile }}" required inputmode="tel"></div><div class="field"><label for="second_mobile">Second Contact</label><input id="second_mobile" name="second_mobile" value="{{ student.second_mobile or '' }}" inputmode="tel"></div><div class="field full"><label for="address">Address</label><textarea id="address" name="address" required>{{ student.address }}</textarea></div></div><div class="actions"><button class="button" type="submit">Save Changes</button><a class="button secondary" href="/student-dashboard">Cancel</a></div></form></section>{% endif %}</main></body></html>
"""


FEE_STATUS_BLOCK = """<section class="panel fee-status-panel"><div class="fee-section-heading"><div><span class="eyebrow">School finance</span><h2>Fee Status</h2><p class="muted">Read-only records from your school's Fee Management.</p></div></div><div class="fee-summary"><div><span>Total Fee</span><strong>{{ fee_status.total_fee }}</strong></div><div><span>Paid Amount</span><strong>{{ fee_status.paid_amount }}</strong></div><div><span>Due Amount</span><strong>{{ fee_status.due_amount }}</strong></div></div><div class="fee-current-status"><span>Current Status</span><strong>{{ fee_status.current_status }}</strong></div>{% if fee_status.monthly %}<div class="fee-history"><h3>Monthly Fee Status</h3><div class="fee-table-wrap"><table><thead><tr><th>Month</th><th>Fee Amount</th><th>Paid Amount</th><th>Due Amount</th><th>Status</th><th>Receipt</th></tr></thead><tbody>{% for fee in fee_status.monthly %}<tr><td>{{ fee.fee_month }}</td><td>Not recorded</td><td>Not recorded</td><td>Not recorded</td><td><span class="fee-status-pill {{ fee.status|lower }}">{{ fee.status }}</span></td><td>{% if fee.receipt_available %}<a class="fee-receipt-link" href="{{ url_for('student_fee_receipt', fee_id=fee.id) }}">View Receipt</a>{% else %}Receipt not available{% endif %}</td></tr>{% endfor %}</tbody></table></div></div>{% else %}<p class="muted fee-empty">No fee records are available yet.</p>{% endif %}</section>"""
FEE_HISTORY_BLOCK = """<section class="panel fee-status-panel"><div class="fee-section-heading"><div><span class="eyebrow">FEE HISTORY</span><h2>Fee History</h2><p class="muted">Read-only monthly records from your school's Fee Management.</p></div></div><div class="fee-history"><div class="fee-table-wrap"><table><thead><tr><th>Month</th><th>Status</th><th>Receipt / Photo</th><th>Updated</th></tr></thead><tbody>{% for fee in fee_status.monthly %}<tr><td>{{ fee.fee_month }}</td><td><span class="fee-status-pill {{ fee.status|lower }}">{{ fee.status }}</span></td><td>{% if fee.receipt_available %}<a class="fee-receipt-link" href="{{ url_for('student_fee_receipt', fee_id=fee.id) }}">View Receipt</a>{% else %}No receipt on file{% endif %}</td><td>{{ fee.updated_at or '—' }}</td></tr>{% endfor %}</tbody></table></div></div></section>"""
STUDENT_NOTICE_BLOCK = """<section class="panel class-notices-panel"><div class="class-notices-heading"><div><span class="eyebrow">CLASS NOTICES</span><h2>Class Notices</h2><p class="muted">Notices for {{ student.class_name }}</p></div></div>{% if class_notices %}<div class="class-notice-list">{% for notice in class_notices %}<article class="class-notice-item"><div class="class-notice-top"><h3>{{ notice.title }}</h3><span class="notice-priority {{ notice.priority|lower }}">{{ notice.priority }}</span></div><p>{{ notice.description }}</p><div class="class-notice-meta">{% if notice.event_date %}<span>Test/Event date: {{ notice.event_date }}</span>{% endif %}<span>Posted: {{ notice.notice_date }}</span><span>Published: {{ notice.created_at }}</span></div></article>{% endfor %}</div>{% else %}<p class="muted">No notices have been published for your class.</p>{% endif %}</section>"""
STUDENT_DASHBOARD_TEMPLATE = STUDENT_DASHBOARD_TEMPLATE.replace('{% if edit %}', STUDENT_NOTICE_BLOCK + '{% if edit %}', 1)
STUDENT_DASHBOARD_TEMPLATE = STUDENT_DASHBOARD_TEMPLATE.replace('</main>', FEE_HISTORY_BLOCK + '</main>', 1)
STUDENT_DASHBOARD_TEMPLATE = STUDENT_DASHBOARD_TEMPLATE.replace('</style>', '<style>.class-notices-panel{padding:24px}.class-notices-heading{margin-bottom:18px}.class-notices-heading h2{margin:0}.class-notices-heading p{margin:5px 0 0}.class-notice-list{display:grid;gap:12px}.class-notice-item{padding:17px;border:1px solid var(--line);border-radius:13px;background:#ffffff05}.class-notice-top{display:flex;align-items:start;justify-content:space-between;gap:14px}.class-notice-top h3{margin:0;font-size:18px}.class-notice-item p{margin:9px 0;color:#e8e2dd;white-space:pre-wrap}.class-notice-meta{display:flex;flex-wrap:wrap;gap:8px 16px;color:var(--muted);font-size:12px}.notice-priority{padding:4px 8px;border-radius:14px;background:#ffffff12;color:var(--muted);font-size:11px;font-weight:bold}.notice-priority.important{background:#f39a4b20;color:var(--light)}@media(max-width:650px){.class-notice-top{display:block}.class-notice-top .notice-priority{display:inline-block;margin-top:8px}}</style></head>', 1)
STUDENT_DASHBOARD_TEMPLATE = STUDENT_DASHBOARD_TEMPLATE.replace('</style>', '<style>.fee-status-panel{padding:24px}.fee-section-heading{display:flex;justify-content:space-between;align-items:start;margin-bottom:18px}.fee-section-heading h2{margin:0}.fee-section-heading p{margin:5px 0 0}.fee-summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.fee-summary>div{padding:15px;border:1px solid var(--line);border-radius:12px;background:#ffffff05}.fee-summary span,.fee-current-status span{display:block;color:var(--muted);font-size:11px;font-weight:bold;text-transform:uppercase}.fee-summary strong{display:block;margin-top:5px;color:var(--light);font-size:19px}.fee-current-status{display:flex;align-items:center;gap:10px;margin-top:14px;padding:12px 14px;border:1px solid var(--line);border-radius:10px;background:#ffffff05}.fee-current-status strong{color:var(--light)}.fee-history{margin-top:22px}.fee-history h3{margin:0 0 12px}.fee-table-wrap{overflow:auto}.fee-history table{width:100%;min-width:700px;border-collapse:collapse}.fee-history th,.fee-history td{text-align:left;padding:11px 10px;border-bottom:1px solid #ffffff12}.fee-history th{color:var(--muted);font-size:11px;text-transform:uppercase}.fee-history td{color:#e8e2dd}.fee-status-pill{display:inline-block;padding:4px 8px;border-radius:14px;background:#ffffff12;font-size:11px;font-weight:bold}.fee-status-pill.paid{background:#92c9ae20;color:var(--mint)}.fee-status-pill.pending{background:#ffc47720;color:var(--light)}.fee-status-pill.free{background:#88b8d020;color:var(--sky)}.fee-receipt-link{color:var(--light);font-weight:bold;text-decoration:none}.fee-receipt-link:hover{text-decoration:underline}.fee-empty{margin:0}@media(max-width:650px){.fee-summary{grid-template-columns:1fr}.fee-section-heading{display:block}}</style></head>', 1)


PROFILE_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Profile</title><style>body{font:14px Arial;background:#eef3f4;color:#203040;margin:0}.main{max-width:520px;margin:50px auto;padding:22px}.panel{background:#fff;border:1px solid #d8e2e6;border-radius:7px;padding:24px}label{display:grid;gap:5px;margin:12px 0;font-weight:bold}input{padding:10px;font:inherit;border:1px solid #bdcbd1;border-radius:4px}.button{background:#176b87;color:#fff;border:0;border-radius:4px;padding:11px 16px;font-weight:bold;cursor:pointer}a{color:#176b87}</style></head><body><main class="main"><div class="panel"><h1>Profile</h1><p>Signed in as {{ user['username'] }}</p><form method="post"><label>Student name<input name="student_name" value="{{ user['student_name'] }}" required></label><label>New password <span>(leave blank to keep current)</span><input name="password" type="password" minlength="6"></label><button class="button" type="submit">Save Profile</button></form>{% if message %}<p>{{ message }}</p>{% endif %}<p><a href="/game">Back to Game</a></p></div></main></body></html>
"""

CHANGE_PASSWORD_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Change Temporary Password</title><style>:root{--ink:#f7f1e8;--muted:#aaa8b1;--line:#343640;--amber:#f39a4b;--coral:#e87551}*{box-sizing:border-box}body{min-height:100vh;margin:0;display:grid;place-items:center;background:radial-gradient(circle at 80% 0%,#382317 0,#15151d 30%,#0b0c12 70%);color:var(--ink);font:14px/1.5 "Trebuchet MS","Segoe UI",sans-serif}.panel{width:min(440px,calc(100% - 40px));padding:30px;border:1px solid #ffffff17;border-radius:20px;background:linear-gradient(145deg,#1a1c25,#15161d);box-shadow:0 20px 55px #0008}.eyebrow{color:#ffc477;font-size:11px;font-weight:bold;letter-spacing:.13em;text-transform:uppercase}h1{margin:9px 0;font-size:29px}p{color:var(--muted)}label{display:grid;gap:6px;margin-top:16px;color:#d8d1cc;font-size:12px;font-weight:bold}input{padding:12px;border:1px solid var(--line);border-radius:10px;background:#0e1017;color:var(--ink);font:inherit}input:focus{outline:0;border-color:var(--amber);box-shadow:0 0 0 3px #f39a4b1c}.button{width:100%;margin-top:22px;padding:13px;border:0;border-radius:10px;background:linear-gradient(135deg,var(--amber),var(--coral));color:#1a120e;font:inherit;font-weight:bold;cursor:pointer}.error{margin-top:16px;padding:11px;border-radius:9px;background:#512c2859;color:#ffc5b3}</style></head><body><main class="panel"><span class="eyebrow">Student account</span><h1>Set a new password</h1><p>Your temporary password can only be used once. Set a new password to continue.</p><form method="post"><label>New password<input name="password" type="password" minlength="6" autocomplete="new-password" required></label><label>Confirm new password<input name="confirm_password" type="password" minlength="6" autocomplete="new-password" required></label><button class="button" type="submit">Save new password</button></form>{% if error %}<div class="error">{{ error }}</div>{% endif %}</main></body></html>
"""


def student_fee_context(user):
    with game_connection() as connection:
        rows = connection.execute(
            "SELECT id, fee_month, status, receipt_data, updated_at "
            "FROM student_fee_status "
            "WHERE school_id = ? AND student_id = ? "
            "ORDER BY CASE fee_month "
            "WHEN 'April' THEN 1 WHEN 'May' THEN 2 WHEN 'June' THEN 3 "
            "WHEN 'July' THEN 4 WHEN 'August' THEN 5 WHEN 'September' THEN 6 "
            "WHEN 'October' THEN 7 WHEN 'November' THEN 8 WHEN 'December' THEN 9 "
            "WHEN 'January' THEN 10 WHEN 'February' THEN 11 WHEN 'March' THEN 12 ELSE 99 END",
            (user["school_id"], user["managed_student_id"]),
        ).fetchall()
    fee_by_month = {row["fee_month"]: row for row in rows}
    monthly = []
    for month in FEE_MONTHS:
        row = fee_by_month.get(month)
        monthly.append(
            {
                "id": row["id"] if row else None,
                "fee_month": month,
                "status": row["status"] if row else "Pending",
                "receipt_available": bool(row["receipt_data"]) if row else False,
                "updated_at": row["updated_at"] if row else None,
            }
        )
    return {
        "has_records": bool(rows),
        "monthly": monthly,
    }


@app.route("/profile", methods=["GET", "POST"])
@student_required
def profile():
    user = current_user()
    if user is None:
        return redirect(url_for("home"))
    if user["must_change_password"]:
        return redirect(url_for("change_password"))
    message = None
    if request.method == "POST":
        name = request.form.get("student_name", "").strip()
        password = request.form.get("password", "")
        if not name or password and len(password) < 6:
            message = "Enter a name and, if changing it, a password with at least 6 characters."
        else:
            with game_connection() as connection:
                if password:
                    connection.execute("UPDATE game_users SET student_name = ?, password_hash = ? WHERE id = ?", (name, generate_password_hash(password), user["id"]))
                else:
                    connection.execute("UPDATE game_users SET student_name = ? WHERE id = ?", (name, user["id"]))
                connection.execute(
                    "UPDATE managed_students SET student_name = ?, updated_at = datetime('now') WHERE id = ? AND school_id = ? AND active_student = 1",
                    (name, user["managed_student_id"], user["school_id"]),
                )
            user = current_user()
            message = "Profile updated."
    return render_template_string(PROFILE_TEMPLATE, user=user, message=message)


@app.route("/change-password", methods=["GET", "POST"])
def change_password():
    user = current_user()
    if user is None:
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        if len(password) < 6:
            error = "Your new password must contain at least 6 characters."
        elif password != confirm_password:
            error = "The passwords do not match."
        else:
            with game_connection() as connection:
                connection.execute("UPDATE game_users SET password_hash = ?, must_change_password = 0 WHERE id = ?", (generate_password_hash(password), user["id"]))
            return redirect(url_for("student_dashboard" if user["managed_student_id"] else "game"))
    return render_template_string(CHANGE_PASSWORD_TEMPLATE, error=error)


@app.post("/register")
def register():
    name = request.form.get("student_name", "").strip()
    username = request.form.get("username", "").strip().lower()
    password = request.form.get("password", "")
    if not name or len(username) < 3 or len(password) < 6:
        session["game_error"] = "Enter a name, a username with at least 3 characters, and a password with at least 6 characters."
        return redirect(url_for("game"))
    try:
        with game_connection() as connection:
            cursor = connection.execute("INSERT INTO game_users (student_name, username, password_hash, created_at, school_id) VALUES (?, ?, ?, datetime('now'), ?)", (name, username, generate_password_hash(password), session.get("school_id")))
            session["game_user_id"] = cursor.lastrowid
    except sqlite3.IntegrityError:
        session["game_error"] = "That username is already in use. Choose another one."
    return redirect(url_for("game"))


@app.get("/student-fee-receipt/<int:fee_id>")
@student_required
def student_fee_receipt(fee_id):
    user = current_user()
    with game_connection() as connection:
        receipt = connection.execute(
            "SELECT receipt_data FROM student_fee_status "
            "WHERE id = ? AND student_id = ? AND school_id = ?",
            (fee_id, user["managed_student_id"], user["school_id"]),
        ).fetchone()
    if receipt is None or not receipt["receipt_data"]:
        return "Receipt not found.", 404
    return Response(receipt["receipt_data"], mimetype="image/png", headers={"Content-Disposition": "inline; filename=fee_receipt.png"})


@app.route("/student-dashboard", methods=["GET", "POST"])
@student_required
def student_dashboard():
    user = current_user()
    error = None
    message = None
    with game_connection() as connection:
        student = connection.execute(
            "SELECT managed_students.*, game_users.username AS account_username "
            "FROM managed_students JOIN game_users ON game_users.managed_student_id = managed_students.id "
            "AND game_users.school_id = managed_students.school_id "
            "WHERE managed_students.id = ? AND managed_students.school_id = ? AND managed_students.active_student = 1",
            (user["managed_student_id"], user["school_id"]),
        ).fetchone()
    if student is None:
        session["game_error"] = "Student account not found. Please contact the school administration."
        session.pop("game_user_id", None)
        return redirect(url_for("login"))
    if request.method == "POST":
        name = request.form.get("student_name", "").strip()
        parent_mobile = request.form.get("parent_mobile", "").strip()
        second_mobile = request.form.get("second_mobile", "").strip()
        address = request.form.get("address", "").strip()
        errors = []
        if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,79}", name):
            errors.append("Student name contains an invalid value.")
        if not re.fullmatch(r"\+?[0-9]{10,15}", parent_mobile):
            errors.append("Parent mobile number must contain 10-15 digits.")
        if second_mobile and not re.fullmatch(r"\+?[0-9]{10,15}", second_mobile):
            errors.append("Second mobile number must contain 10-15 digits.")
        if not 5 <= len(address) <= 300:
            errors.append("Complete address must contain 5-300 characters.")
        if errors:
            error = " ".join(errors)
            student = dict(student)
            student.update({"student_name": name, "parent_mobile": parent_mobile, "second_mobile": second_mobile, "address": address})
        else:
            try:
                with game_connection() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "UPDATE managed_students SET student_name = ?, parent_mobile = ?, second_mobile = ?, address = ?, updated_at = datetime('now') "
                        "WHERE id = ? AND school_id = ? AND active_student = 1",
                        (name, parent_mobile, second_mobile, address, user["managed_student_id"], user["school_id"]),
                    )
                    connection.execute(
                        "UPDATE game_users SET student_name = ? WHERE id = ? AND managed_student_id = ? AND school_id = ?",
                        (name, user["id"], user["managed_student_id"], user["school_id"]),
                    )
                message = "Profile updated."
                return redirect(url_for("student_dashboard"))
            except (sqlite3.IntegrityError, ValueError) as exc:
                logger.exception(
                    "Student profile update failed: school_id=%s student_id=%s error_type=%s",
                    user["school_id"], user["managed_student_id"], type(exc).__name__,
                )
                error = "Your profile could not be updated."
    return render_template_string(STUDENT_DASHBOARD_TEMPLATE, student=student, fee_status=student_fee_context(user), class_notices=student_notice_context(user), edit=request.args.get("edit") == "1" or request.method == "POST", error=error, message=message)


SCHOOL_AUTH_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ title }}</title><style>
@keyframes auth-in{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}@keyframes float-in{from{opacity:0;transform:translateY(12px) scale(.96)}to{opacity:1;transform:translateY(0) scale(1)}}:root{--ink:#f7f1e8;--muted:#a8a6af;--line:#343640;--card:#181a23;--amber:#f39a4b;--amber-light:#ffc477;--coral:#e87551;--mint:#92c9ae}*{box-sizing:border-box}body{min-height:100vh;margin:0;background:radial-gradient(circle at 12% 0%,#382317 0,transparent 34%),radial-gradient(circle at 92% 100%,#192a2b 0,transparent 31%),#0b0c12;color:var(--ink);font:14px/1.5 "Trebuchet MS","Segoe UI",sans-serif}.auth-shell{width:min(1080px,calc(100% - 44px));min-height:650px;margin:32px auto;display:grid;grid-template-columns:1fr 1fr;overflow:hidden;border:1px solid #ffffff17;border-radius:28px;background:#12141ccc;box-shadow:0 25px 70px #0009;animation:auth-in .5s ease-out both}.auth-visual{position:relative;display:flex;flex-direction:column;justify-content:space-between;min-height:650px;padding:44px;background:linear-gradient(145deg,#241914,#171820 58%,#102024);overflow:hidden}.auth-visual:after{content:"";position:absolute;width:340px;height:340px;right:-100px;bottom:-90px;border-radius:50%;background:#f08d4824;filter:blur(30px)}.auth-brand{position:relative;z-index:1;color:#e7d9cd;font-weight:bold;font-size:16px}.auth-brand span{color:var(--amber-light)}.auth-visual h1{position:relative;z-index:1;max-width:420px;margin:0 0 16px;font-size:clamp(34px,4vw,52px);line-height:1.04;letter-spacing:-.03em}.auth-visual p{position:relative;z-index:1;max-width:360px;color:#c6bfc0;font-size:15px}.visual-art{position:absolute;inset:0;pointer-events:none}.visual-glow{position:absolute;width:240px;height:240px;right:23%;top:32%;border-radius:50%;background:radial-gradient(circle,#f39a4b50,transparent 68%);filter:blur(3px)}.visual-ring{position:absolute;right:12%;top:34%;width:290px;height:120px;border:1px solid #ffbe7759;border-radius:50%;transform:rotate(-22deg)}.visual-ring.second{right:18%;top:38%;width:220px;height:90px;transform:rotate(55deg)}.visual-star{position:absolute;right:39%;top:39%;display:grid;place-items:center;width:68px;height:68px;border-radius:22px;background:linear-gradient(145deg,#ffc477,#d76242);color:#251711;font-size:30px;box-shadow:0 14px 32px #df784a73;transform:rotate(8deg);animation:float-in .7s .2s ease-out both}.visual-tag{position:absolute;z-index:1;right:9%;padding:9px 12px;border:1px solid #ffffff1c;border-radius:11px;background:#20232bd9;color:#e8ddd2;font-size:12px;box-shadow:0 8px 22px #0005}.visual-tag span{color:var(--amber-light);font-size:17px;margin-right:5px}.tag-top{top:29%}.tag-bottom{top:57%;right:7%}.auth-form-wrap{display:flex;align-items:center;padding:44px clamp(28px,5vw,74px);background:#151720e8}.auth-card{width:100%;max-width:390px;margin:auto}.auth-eyebrow{color:var(--amber-light);font-size:11px;font-weight:bold;letter-spacing:.13em;text-transform:uppercase}.auth-card h2{margin:10px 0 8px;font-size:32px;letter-spacing:-.02em}.auth-card>p{margin:0 0 28px;color:var(--muted)}.error{margin:0 0 18px;padding:12px 14px;border:1px solid #e8755159;border-radius:11px;background:#512c2859;color:#ffc5b3}.auth-form label{display:block;margin:16px 0 7px;color:#d8d1cc;font-size:12px;font-weight:bold}.input-wrap{position:relative}.auth-form input{width:100%;padding:13px 14px;border:1px solid var(--line);border-radius:11px;background:#0e1017;color:var(--ink);font:inherit;outline:0;transition:border-color .2s,box-shadow .2s,background .2s}.auth-form input:focus{border-color:var(--amber);background:#12141c;box-shadow:0 0 0 4px #f39a4b1c}.password-input{padding-right:48px!important}.password-toggle{position:absolute;right:7px;top:7px;width:36px;height:36px;border:0;border-radius:8px;background:transparent;color:#aaa5a2;cursor:pointer;font-size:17px}.password-toggle:hover,.password-toggle:focus{background:#ffffff0d;color:var(--amber-light);outline:0}.auth-submit{width:100%;margin-top:24px;padding:14px;border:0;border-radius:11px;background:linear-gradient(135deg,var(--amber),var(--coral));color:#1a120e;font:inherit;font-weight:bold;cursor:pointer;box-shadow:0 12px 25px #e57c3b3b;transition:transform .2s,box-shadow .2s,filter .2s}.auth-submit:hover{filter:brightness(1.08);transform:translateY(-2px);box-shadow:0 16px 30px #e57c3b59}.auth-submit:focus{outline:3px solid #ffc47766;outline-offset:3px}.auth-link{margin:24px 0 0!important;color:var(--muted)!important;font-size:13px}.auth-link a{color:var(--amber-light);font-weight:bold;text-decoration:none}.auth-link a:hover{text-decoration:underline}@media(max-width:760px){.auth-shell{grid-template-columns:1fr;min-height:0;margin:18px auto}.auth-visual{min-height:300px;padding:30px}.auth-visual h1{max-width:480px;font-size:38px}.visual-art{opacity:.62}.auth-form-wrap{padding:38px 28px 44px}}@media(max-width:480px){.auth-shell{width:min(100% - 24px,420px);border-radius:21px}.auth-visual{min-height:265px;padding:25px}.auth-visual h1{font-size:32px}.auth-visual p{font-size:13px}.auth-form-wrap{padding:32px 21px 38px}.auth-card h2{font-size:28px}}@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
</style></head><body><main class="auth-shell"><section class="auth-visual"><div class="auth-brand"><span>✦</span> Student Support</div><div><h1>Empowering schools with smarter student support.</h1><p>Bring clearer signals and more human conversations into every learner's journey.</p></div><div class="visual-art" aria-hidden="true"><div class="visual-glow"></div><div class="visual-ring"></div><div class="visual-ring second"></div><div class="visual-star">✦</div><div class="visual-tag tag-top"><span>✓</span> Progress</div><div class="visual-tag tag-bottom"><span>↗</span> Support</div></div></section><section class="auth-form-wrap"><div class="auth-card"><span class="auth-eyebrow">School workspace</span><h2>{{ title }}</h2><p>Sign in to access your private Student Risk Prediction dashboard.</p>{% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}<form class="auth-form" method="post"><label>Username<input name="username" autocomplete="username" required></label><label>Password<div class="input-wrap"><input id="school-password" class="password-input" name="password" type="password" autocomplete="current-password" required><button class="password-toggle" type="button" aria-label="Show password" aria-controls="school-password">&#128065;</button></div></label><button class="auth-submit" type="submit">Log In <span aria-hidden="true">&#8594;</span></button></form><p class="auth-link">New school? <a href="/school-register">Create a school account</a></p></div></section></main><script>document.querySelector('.password-toggle')?.addEventListener('click',function(){const input=document.getElementById('school-password');const visible=input.type==='text';input.type=visible?'password':'text';this.setAttribute('aria-label',visible?'Show password':'Hide password');});</script></body></html>
"""
SCHOOL_REGISTER_TEMPLATE = SCHOOL_AUTH_TEMPLATE.replace(
    '<form class="auth-form" method="post"><label>Username',
    '<form class="auth-form" method="post"><label>School name<input name="school_name" required></label><label>Username',
).replace('>Log In</button>', '>Create School Account</button>')

SCHOOL_AUTH_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ title }}</title><style>
:root{--ink:#f7f1e8;--muted:#aaa8b1;--line:#343640;--amber:#f39a4b;--light:#ffc477;--coral:#e87551;--mint:#92c9ae}*{box-sizing:border-box}body{min-height:100vh;margin:0;background:radial-gradient(circle at 12% 0%,#382317 0,transparent 34%),radial-gradient(circle at 92% 100%,#192a2b 0,transparent 31%),#0b0c12;color:var(--ink);font:14px/1.5 "Trebuchet MS","Segoe UI",sans-serif}.auth-shell{width:min(1080px,calc(100% - 44px));min-height:650px;margin:32px auto;display:grid;grid-template-columns:1fr 1fr;overflow:hidden;border:1px solid #ffffff17;border-radius:28px;background:#12141ccc;box-shadow:0 25px 70px #0009}.auth-visual{position:relative;display:flex;flex-direction:column;justify-content:space-between;min-height:650px;padding:44px;background:linear-gradient(145deg,#241914,#171820 58%,#102024);overflow:hidden}.auth-brand{position:relative;z-index:1;color:#e7d9cd;font-weight:bold;font-size:16px}.auth-brand span{color:var(--light)}.auth-visual h1{position:relative;z-index:1;max-width:420px;margin:0 0 16px;font-size:clamp(34px,4vw,52px);line-height:1.04;letter-spacing:-.03em}.auth-visual p{position:relative;z-index:1;max-width:360px;color:#c6bfc0;font-size:15px}.visual-art{position:absolute;inset:0;pointer-events:none}.visual-glow{position:absolute;width:240px;height:240px;right:23%;top:32%;border-radius:50%;background:radial-gradient(circle,#f39a4b50,transparent 68%)}.visual-ring{position:absolute;right:12%;top:34%;width:290px;height:120px;border:1px solid #ffbe7759;border-radius:50%;transform:rotate(-22deg)}.visual-ring.second{right:18%;top:38%;width:220px;height:90px;transform:rotate(55deg)}.visual-star{position:absolute;right:39%;top:39%;display:grid;place-items:center;width:68px;height:68px;border-radius:22px;background:linear-gradient(145deg,#ffc477,#d76242);color:#251711;font-size:30px;box-shadow:0 14px 32px #df784a73;transform:rotate(8deg)}.visual-tag{position:absolute;right:9%;padding:9px 12px;border:1px solid #ffffff1c;border-radius:11px;background:#20232bd9;color:#e8ddd2;font-size:12px}.visual-tag span{color:var(--light);font-size:17px;margin-right:5px}.tag-top{top:29%}.tag-bottom{top:57%;right:7%}.auth-form-wrap{display:flex;align-items:center;padding:44px clamp(28px,5vw,74px);background:#151720e8}.auth-card{width:100%;max-width:390px;margin:auto}.auth-eyebrow{color:var(--light);font-size:11px;font-weight:bold;letter-spacing:.13em;text-transform:uppercase}.auth-card h2{margin:10px 0 8px;font-size:32px;letter-spacing:-.02em}.auth-card>p{margin:0 0 22px;color:var(--muted)}.mode-tabs{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:4px;margin-bottom:24px;border:1px solid var(--line);border-radius:12px;background:#0e1017}.mode-tab{position:relative;border:0;border-radius:9px;padding:10px 8px;background:transparent;color:var(--muted);font:inherit;font-weight:bold;cursor:pointer}.mode-tab:after{content:"";position:absolute;inset:0;border-radius:9px;background:linear-gradient(135deg,var(--amber),var(--coral));opacity:0;transform:scale(.92);transition:opacity .2s,transform .2s;z-index:0}.mode-tab span{position:relative;z-index:1}.mode-tab[aria-selected="true"]{color:#1b130e}.mode-tab[aria-selected="true"]:after{opacity:1;transform:scale(1)}.mode-tab:focus-visible{outline:3px solid #ffc47766;outline-offset:2px}.error{margin:0 0 18px;padding:12px 14px;border:1px solid #e8755159;border-radius:11px;background:#512c2859;color:#ffc5b3}.auth-form label{display:block;margin:16px 0 7px;color:#d8d1cc;font-size:12px;font-weight:bold}.input-wrap{position:relative}.auth-form input{width:100%;padding:13px 14px;border:1px solid var(--line);border-radius:11px;background:#0e1017;color:var(--ink);font:inherit;outline:0;transition:border-color .2s,box-shadow .2s}.auth-form input:focus{border-color:var(--amber);box-shadow:0 0 0 4px #f39a4b1c}.password-input{padding-right:48px!important}.password-toggle{position:absolute;right:7px;top:7px;width:36px;height:36px;border:0;border-radius:8px;background:transparent;color:#aaa5a2;cursor:pointer;font-size:17px}.password-toggle:hover,.password-toggle:focus{background:#ffffff0d;color:var(--light);outline:0}.auth-submit{width:100%;margin-top:24px;padding:14px;border:0;border-radius:11px;background:linear-gradient(135deg,var(--amber),var(--coral));color:#1a120e;font:inherit;font-weight:bold;cursor:pointer;box-shadow:0 12px 25px #e57c3b3b;transition:transform .2s,box-shadow .2s}.auth-submit:hover{transform:translateY(-2px);box-shadow:0 16px 30px #e57c3b59}.auth-submit:focus-visible{outline:3px solid #ffc47766;outline-offset:3px}.auth-link{margin:24px 0 0!important;color:var(--muted)!important;font-size:13px}.auth-link a{color:var(--light);font-weight:bold;text-decoration:none}.auth-link a:hover{text-decoration:underline}.school-only[hidden]{display:none}@media(max-width:760px){.auth-shell{grid-template-columns:1fr;min-height:0;margin:18px auto}.auth-visual{min-height:300px;padding:30px}.auth-visual h1{max-width:480px;font-size:38px}.visual-art{opacity:.62}.auth-form-wrap{padding:38px 28px 44px}}@media(max-width:480px){.auth-shell{width:min(100% - 24px,420px);border-radius:21px}.auth-visual{min-height:265px;padding:25px}.auth-visual h1{font-size:32px}.auth-visual p{font-size:13px}.auth-form-wrap{padding:32px 21px 38px}.auth-card h2{font-size:28px}}
</style></head><body><main class="auth-shell"><section class="auth-visual"><div class="auth-brand"><span>✦</span> Student Support</div><div><h1>Empowering schools with smarter student support.</h1><p>Bring clearer signals and more human conversations into every learner's journey.</p></div><div class="visual-art" aria-hidden="true"><div class="visual-glow"></div><div class="visual-ring"></div><div class="visual-ring second"></div><div class="visual-star">✦</div><div class="visual-tag tag-top"><span>✓</span> Progress</div><div class="visual-tag tag-bottom"><span>↗</span> Support</div></div></section><section class="auth-form-wrap"><div class="auth-card"><span class="auth-eyebrow">Student support workspace</span><h2>{{ title }}</h2><p>Choose your account type to continue securely.</p><div class="mode-tabs" role="tablist" aria-label="Login type"><button class="mode-tab" type="button" role="tab" aria-selected="{{ 'true' if mode == 'student' else 'false' }}" aria-controls="login-form" data-mode="student"><span>Student Login</span></button><button class="mode-tab" type="button" role="tab" aria-selected="{{ 'true' if mode == 'school' else 'false' }}" aria-controls="login-form" data-mode="school"><span>School Login</span></button></div>{% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}<form id="login-form" class="auth-form" method="post" action="{{ '/school-login' if mode == 'school' else '/login' }}"><div class="school-only" {% if mode != 'school' %}hidden{% endif %}><label for="school-name">School name<input id="school-name" name="school_name" autocomplete="organization"></label></div><label for="login-username"><span class="student-only" {% if mode != 'student' %}hidden{% endif %}>Student Username/Email</span><span class="school-only" {% if mode != 'school' %}hidden{% endif %}>School Username</span><input id="login-username" name="username" autocomplete="username" required></label><label for="login-password">Password<div class="input-wrap"><input id="login-password" class="password-input" name="password" type="password" autocomplete="current-password" required><button class="password-toggle" type="button" aria-label="Show password" aria-controls="login-password">&#128065;</button></div></label><button class="auth-submit" type="submit"><span class="student-only" {% if mode != 'student' %}hidden{% endif %}>Login as Student</span><span class="school-only" {% if mode != 'school' %}hidden{% endif %}>Login as School</span><span aria-hidden="true">&#8594;</span></button></form><p class="auth-link school-only" {% if mode != 'school' %}hidden{% endif %}>New school? <a href="/school-register">Create a school account</a></p></div></section></main><script>(function(){const form=document.getElementById('login-form');const tabs=document.querySelectorAll('.mode-tab');const schoolOnly=document.querySelectorAll('.school-only');const studentOnly=document.querySelectorAll('.student-only');const username=document.getElementById('login-username');const schoolName=document.getElementById('school-name');const setMode=function(mode){const school=mode==='school';form.action=school?'/school-login':'/login';tabs.forEach(function(tab){const active=tab.dataset.mode===mode;tab.setAttribute('aria-selected',active?'true':'false');});schoolOnly.forEach(function(element){element.hidden=!school;});studentOnly.forEach(function(element){element.hidden=school;});username.setAttribute('aria-label',school?'School Username':'Student Username or Email');if(schoolName)schoolName.required=school;};tabs.forEach(function(tab){tab.addEventListener('click',function(){setMode(tab.dataset.mode);});tab.addEventListener('keydown',function(event){if(event.key==='ArrowRight'||event.key==='ArrowLeft'){event.preventDefault();tabs[tab.dataset.mode==='student'?1:0].focus();tabs[tab.dataset.mode==='student'?1:0].click();}});});const toggle=document.querySelector('.password-toggle');toggle.addEventListener('click',function(){const input=document.getElementById('login-password');const visible=input.type==='text';input.type=visible?'password':'text';toggle.setAttribute('aria-label',visible?'Show password':'Hide password');});})();</script></body></html>
"""
SCHOOL_AUTH_TEMPLATE = SCHOOL_AUTH_TEMPLATE.replace(
    "action=\"{{ '/school-login' if mode == 'school' else '/login' }}\"",
    "action=\"{{ '/school-register' if title == 'Create School Account' else ('/school-login' if mode == 'school' else '/login') }}\"",
)
SCHOOL_AUTH_TEMPLATE = SCHOOL_AUTH_TEMPLATE.replace(
    '<div class="visual-ring"></div><div class="visual-ring second"></div>',
    '<div class="visual-ring"><i></i></div><div class="visual-ring second"><i></i></div><div class="visual-ring third"><i></i></div>',
)
SCHOOL_AUTH_TEMPLATE = SCHOOL_AUTH_TEMPLATE.replace(
    '</style>',
    '''<style>
@keyframes orbit-one{to{transform:rotate(360deg)}}@keyframes orbit-two{to{transform:rotate(-360deg)}}@keyframes orbit-three{to{transform:rotate(360deg)}}@keyframes star-pulse{0%,100%{box-shadow:0 14px 32px #df784a73;transform:rotate(8deg) scale(1)}50%{box-shadow:0 14px 38px #f39a4b99;transform:rotate(8deg) scale(1.04)}}@keyframes depth-drift{0%,100%{transform:translate3d(0,0,0);opacity:.22}50%{transform:translate3d(18px,-12px,0);opacity:.55}}@keyframes visual-in{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:translateX(0)}}@keyframes panel-in{from{opacity:0;transform:translateX(18px)}to{opacity:1;transform:translateX(0)}}@keyframes field-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}.auth-visual{animation:visual-in .65s ease-out both}.auth-form-wrap{animation:panel-in .65s .08s ease-out both}.visual-art{z-index:0}.visual-ring{transform-origin:50% 50%;animation:orbit-one 30s linear infinite}.visual-ring.second{animation:orbit-two 38s linear infinite}.visual-ring.third{right:22%;top:30%;width:345px;height:145px;transform:rotate(12deg);border-color:#f39a4b2e;animation:orbit-three 46s linear infinite}.visual-ring i{position:absolute;left:50%;top:-4px;width:8px;height:8px;border-radius:50%;background:var(--light);box-shadow:0 0 12px var(--amber);transform:translateX(-50%)}.visual-ring.second i{width:6px;height:6px;top:auto;bottom:-3px;background:var(--mint);box-shadow:0 0 10px var(--mint)}.visual-ring.third i{width:5px;height:5px;top:auto;bottom:10px;background:#e9d6b8;box-shadow:0 0 9px #e9d6b8}.visual-star{animation:star-pulse 5s ease-in-out infinite}.visual-art:before,.visual-art:after{content:"";position:absolute;left:22%;top:22%;width:3px;height:3px;border-radius:50%;background:#f8dfb7;box-shadow:48px 92px 0 #ffc477,185px 22px 0 #92c9ae,272px 174px 0 #f39a4b,320px 58px 0 #e9d6b8;animation:depth-drift 12s ease-in-out infinite}.visual-art:after{left:73%;top:68%;width:2px;height:2px;box-shadow:-90px -80px 0 #ffc477,35px -110px 0 #92c9ae,-170px 40px 0 #e9d6b8;animation-delay:-5s;animation-duration:15s}.auth-card>*{animation:field-in .45s ease-out both}.auth-card>.auth-eyebrow{animation-delay:.22s}.auth-card>h2{animation-delay:.28s}.auth-card>p{animation-delay:.34s}.auth-card>.mode-tabs{animation-delay:.4s}.auth-card>.error{animation-delay:.44s}.auth-form label:nth-child(1){animation-delay:.46s}.auth-form label:nth-child(2){animation-delay:.52s}.auth-submit{animation:field-in .45s .58s ease-out both}.auth-link{animation:field-in .45s .64s ease-out both}.auth-submit:hover{transform:translateY(-2px);box-shadow:0 15px 28px #e57c3b4d}@media(prefers-reduced-motion:reduce){.auth-visual,.auth-form-wrap,.auth-card>*, .auth-submit{animation:none!important}.visual-ring{animation:none!important}.visual-star{animation:none!important}.visual-art:before,.visual-art:after{animation:none!important;opacity:.35}.visual-star{transform:rotate(8deg)} }
</style>''',
)
SCHOOL_REGISTER_TEMPLATE = SCHOOL_AUTH_TEMPLATE


@app.route("/school-login", methods=["GET", "POST"])
def school_login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        with game_connection() as connection:
            school = connection.execute("SELECT * FROM schools WHERE username = ?", (username,)).fetchone()
        if school is None or not check_password_hash(school["password_hash"], password):
            error = "Invalid school username or password."
        else:
            session["school_id"] = school["id"]
            return redirect(url_for("home"))
    return render_template_string(SCHOOL_AUTH_TEMPLATE, title="School Login", error=error, mode="school" if request.method == "POST" else "student")


@app.route("/school-register", methods=["GET", "POST"])
def school_register():
    error = None
    if request.method == "POST":
        school_name = request.form.get("school_name", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        if not school_name or len(username) < 3 or len(password) < 6:
            error = "Enter a school name, username of at least 3 characters, and password of at least 6 characters."
        else:
            try:
                with game_connection() as connection:
                    cursor = connection.execute("INSERT INTO schools (school_name, username, password_hash, created_at) VALUES (?, ?, ?, datetime('now'))", (school_name, username, generate_password_hash(password)))
                    session["school_id"] = cursor.lastrowid
                return redirect(url_for("home"))
            except sqlite3.IntegrityError:
                error = "That school username is already in use."
    login_template = SCHOOL_REGISTER_TEMPLATE.replace('href="/school-register">Create a school account', 'href="/school-login">Back to school login')
    return render_template_string(login_template, title="Create School Account", error=error, mode="school")


@app.get("/school-logout")
def school_logout():
    session.pop("school_id", None)
    return redirect(url_for("school_login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        error = session.pop("game_error", None)
        return render_template_string(SCHOOL_AUTH_TEMPLATE, title="Student Login", error=error, mode="student")
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    with game_connection() as connection:
        user = connection.execute(
            "SELECT game_users.* FROM game_users "
            "LEFT JOIN managed_students ON managed_students.id = game_users.managed_student_id "
            "WHERE game_users.username = ? COLLATE NOCASE AND (game_users.managed_student_id IS NULL OR "
            "(game_users.school_id = managed_students.school_id AND managed_students.active_student = 1))",
            (username,),
        ).fetchone()
    if user is None:
        session["game_error"] = "Student account not found. Please contact the school administration."
    elif not check_password_hash(user["password_hash"], password):
        session["game_error"] = "Invalid username or password."
    else:
        session["game_user_id"] = user["id"]
        if user["must_change_password"]:
            return redirect(url_for("change_password"))
        return redirect(url_for("student_dashboard" if user["managed_student_id"] else "game"))
    return redirect(url_for("login"))


@app.get("/logout")
def logout():
    session.pop("game_user_id", None)
    session.pop("game_questions", None)
    session.pop("game_position", None)
    session.pop("game_correct", None)
    session.pop("game_answers", None)
    session.pop("game_review", None)
    return redirect(url_for("login"))


@app.route("/game", methods=["GET", "POST"])
def game():
    user = current_user()
    if user is None:
        session["game_error"] = "Log in to play and save your game history."
        return redirect(url_for("home"))
    if user["must_change_password"]:
        return redirect(url_for("change_password"))
    if request.args.get("start") == "1":
        session["game_questions"] = random.sample(range(len(GAME_QUESTIONS)), 5)
        session["game_position"] = 0
        session["game_correct"] = 0
        session["game_answers"] = []
        session.pop("game_review", None)
        return redirect(url_for("game"))

    result = None
    question = None
    question_ids = session.get("game_questions", [])
    position = session.get("game_position", 0)
    if question_ids and position < len(question_ids):
        question = GAME_QUESTIONS[question_ids[position]]
        if request.method == "POST":
            selected_answer = request.form.get("answer", "")
            is_correct = selected_answer == question["answer"]
            if is_correct:
                session["game_correct"] = session.get("game_correct", 0) + 1
            answers = session.get("game_answers", [])
            answers.append({"question": question["prompt"], "selected": selected_answer, "correct": question["answer"], "is_correct": is_correct})
            session["game_answers"] = answers
            position += 1
            session["game_position"] = position
            if position < len(question_ids):
                question = GAME_QUESTIONS[question_ids[position]]
            else:
                correct = session.get("game_correct", 0)
                accuracy = correct / len(question_ids) * 100
                level = game_level(accuracy)
                with game_connection() as connection:
                    connection.execute("INSERT INTO game_scores (user_id, score, total_questions, accuracy, level, played_at) VALUES (?, ?, ?, ?, ?, datetime('now'))", (user["id"], correct, len(question_ids), accuracy, level))
                session.pop("game_questions", None)
                session.pop("game_position", None)
                session.pop("game_correct", None)
                result = {"score": correct, "total": len(question_ids), "accuracy": round(accuracy), "level": level}
                result["review"] = session.get("game_answers", [])
                session["game_review"] = result["review"]
                question = None

    if result is None and session.get("game_review"):
        summary = score_summary(user["id"])
        latest = summary["history"][0]
        result = {"score": latest["score"], "total": latest["total_questions"], "accuracy": round(latest["accuracy"]), "level": latest["level"], "review": session["game_review"]}
    return render_template_string(GAME_TEMPLATE, user=user, question=question, position=position + 1, total=len(question_ids) or 5, result=result, summary=score_summary(user["id"]))


@app.get("/student-history")
@school_required
def student_history():
    school = current_school()
    student_query = request.args.get("q", "").strip()
    student_id = request.args.get("student_id", "").strip()
    try:
        selected_student_id = int(student_id) if student_id else None
    except ValueError:
        selected_student_id = None
    student_matches, student_profile, student_fees, student_exam_history = student_history_context(
        school["id"], student_query, selected_student_id
    )
    return render_template_string(
        STUDENT_HISTORY_TEMPLATE,
        school_name=school["school_name"],
        student_query=student_query,
        student_matches=student_matches,
        student_profile=student_profile,
        student_fees=student_fees,
        student_exam_history=student_exam_history,
    )


@app.route("/", methods=["GET", "POST"])
@app.route("/dashboard", methods=["GET", "POST"])
@school_required
def home():
    school = current_school()
    view = request.args.get("view", "home")
    if view == "single":
        student_query = request.args.get("student_query", "").strip()
        student_id = request.args.get("student_id", "").strip()
        try:
            selected_student_id = int(student_id) if student_id else None
        except ValueError:
            selected_student_id = None
        student_matches, student_profile, student_fees, student_exam_history = student_history_context(
            school["id"], student_query, selected_student_id
        )
        return render_page(
            view,
            student_query=student_query,
            student_matches=student_matches,
            student_profile=student_profile,
            student_fees=student_fees,
            student_exam_history=student_exam_history,
        )
    if view == "batch":
        class_name = request.args.get("class", "").strip()
        fee_month = request.args.get("month", "").strip()
        if class_name and class_name not in SCHOOL_CLASSES or fee_month and fee_month not in FEE_MONTHS:
            return "Class or month not found.", 404
        fee_students, fee_summary = (fee_analysis_context(school["id"], class_name, fee_month)
                                     if class_name and fee_month else ([], {"total": 0, "paid": 0, "pending": 0, "free": 0}))
        return render_page(view, selected_class=class_name, selected_month=fee_month, fee_students=fee_students, fee_summary=fee_summary)
    if view == "detail" and not school_records(school["id"]).empty:
        try:
            result = school_records(school["id"])
            selected_index = int(request.args.get("index", "-1"))
            return render_page(view, detail=student_detail(result, selected_index))
        except (TypeError, ValueError, KeyError) as problem:
            return render_page("batch", error=str(problem))
    section_form_class = request.args.get("section_class", "").strip()
    if section_form_class not in SCHOOL_CLASSES:
        section_form_class = None
    return render_page(view, section_form_class=section_form_class, section_error=request.args.get("section_error"), message=request.args.get("message"), memory_form_open=request.args.get("memory_add") == "1", memory_message=request.args.get("memory_message"), memory_error=request.args.get("memory_error"))


@app.post("/school-memories")
@school_required
def upload_school_memory():
    school = current_school()
    title = re.sub(r"\s+", " ", request.form.get("title", "").strip())[:120]
    description = re.sub(r"\s+", " ", request.form.get("description", "").strip())[:500]
    try:
        stored_filename, original_filename, extension = save_memory_photo(request.files.get("memory_photo"), school["id"])
    except ValueError as exc:
        return redirect(url_for("home", memory_add="1", memory_error=str(exc)))
    try:
        with game_connection() as connection:
            connection.execute("INSERT INTO school_memories (school_id, title, description, original_filename, stored_filename, file_extension, uploaded_by, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))", (school["id"], title or None, description or None, original_filename, stored_filename, extension, school["username"]))
    except sqlite3.Error:
        (MEMORY_PHOTO_STORAGE / stored_filename).unlink(missing_ok=True)
        return redirect(url_for("home", memory_add="1", memory_error="The school memory could not be saved."))
    return redirect(url_for("home", memory_message="School memory uploaded successfully."))


@app.get("/school-memories/<int:memory_id>/photo")
@school_required
def school_memory_photo(memory_id):
    school = current_school()
    with game_connection() as connection:
        memory = connection.execute("SELECT original_filename, stored_filename, file_extension FROM school_memories WHERE id = ? AND school_id = ?", (memory_id, school["id"])).fetchone()
    if memory is None:
        return "Photo not found.", 404
    filename = memory["stored_filename"]
    if Path(filename).name != filename or not re.fullmatch(r"school_\d+_memory_[a-f0-9]+\.(jpg|jpeg|png|webp)", filename):
        return "Photo not found.", 404
    path = MEMORY_PHOTO_STORAGE / filename
    if not path.is_file() or path.parent.resolve() != MEMORY_PHOTO_STORAGE.resolve():
        return "Photo not found.", 404
    mimetype = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[memory["file_extension"]]
    return send_file(path, mimetype=mimetype, max_age=3600)


@app.post("/school-memories/<int:memory_id>/delete")
@school_required
def delete_school_memory(memory_id):
    school = current_school()
    with game_connection() as connection:
        memory = connection.execute("SELECT stored_filename FROM school_memories WHERE id = ? AND school_id = ?", (memory_id, school["id"])).fetchone()
        if memory is None:
            return redirect(url_for("home", memory_error="Photo not found."))
        connection.execute("DELETE FROM school_memories WHERE id = ? AND school_id = ?", (memory_id, school["id"]))
    filename = Path(memory["stored_filename"]).name
    if filename == memory["stored_filename"]:
        (MEMORY_PHOTO_STORAGE / filename).unlink(missing_ok=True)
    return redirect(url_for("home", memory_message="School memory deleted."))


@app.post("/sections")
@school_required
def create_student_section():
    school = current_school()
    class_name = request.form.get("class_name", "").strip()
    section_name = normalize_section_name(request.form.get("section_name"))
    if class_name not in SCHOOL_CLASSES:
        return redirect(url_for("home", section_class=class_name, section_error="Invalid class selected."))
    if not section_name:
        return redirect(url_for("home", section_class=class_name, section_error="Use a section name with letters or numbers, up to 20 characters."))
    with game_connection() as connection:
        try:
            connection.execute(
                "INSERT INTO student_sections (school_id, class_name, section_name, created_at) VALUES (?, ?, ?, datetime('now'))",
                (school["id"], class_name, section_name),
            )
        except sqlite3.IntegrityError:
            return redirect(url_for("home", section_class=class_name, section_error="That section already exists for this class."))
    return redirect(url_for("home", message=f"Section {class_name}({section_name}) created successfully."))


@app.get("/download-results")
@school_required
def download_results():
    school = current_school()
    result = school_records(school["id"])
    if result.empty:
        return "No batch results are available.", 404
    return Response(result.to_csv(index=False).encode("utf-8"), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=student_risk_results.csv"})


if __name__ == "__main__":
    app.run(debug=True)
