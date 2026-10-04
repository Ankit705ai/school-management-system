import gc
import tempfile
import unittest
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

import app as app_module


class PasswordResetAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.database_directory = tempfile.TemporaryDirectory()
        self.original_database_path = app_module.GAME_DB_PATH
        app_module.GAME_DB_PATH = Path(self.database_directory.name) / "student_game.db"
        app_module.initialize_game_database()
        app_module.app.config.update(TESTING=True)
        self.client = app_module.app.test_client()

        with app_module.game_connection() as connection:
            self.school_a = connection.execute(
                "INSERT INTO schools (school_name, username, password_hash, created_at) VALUES (?, ?, ?, datetime('now'))",
                ("School A", "school-a", generate_password_hash("school-a-password")),
            ).lastrowid
            self.school_b = connection.execute(
                "INSERT INTO schools (school_name, username, password_hash, created_at) VALUES (?, ?, ?, datetime('now'))",
                ("School B", "school-b", generate_password_hash("school-b-password")),
            ).lastrowid
            self.student_a = self._insert_student(connection, self.school_a, "UID-A")
            self.student_b = self._insert_student(connection, self.school_b, "UID-B")
            self.student_without_account = self._insert_student(connection, self.school_a, "UID-NO-ACCOUNT", create_account=False)

    def tearDown(self):
        app_module.GAME_DB_PATH = self.original_database_path
        gc.collect()
        self.database_directory.cleanup()

    @staticmethod
    def _insert_student(connection, school_id, uid_number, create_account=True):
        student_id = connection.execute(
            "INSERT INTO managed_students (school_id, class_name, roll_number, uid_number, student_name, "
            "father_name, mother_name, parent_mobile, address, created_at, updated_at) "
            "VALUES (?, 'Class 1', ?, ?, 'Student Name', 'Father', 'Mother', '1234567890', 'Address', datetime('now'), datetime('now'))",
            (school_id, uid_number, uid_number),
        ).lastrowid
        if create_account:
            connection.execute(
                "INSERT INTO game_users (student_name, username, password_hash, managed_student_id, school_id, created_at) "
                "VALUES ('Student Name', ?, ?, ?, ?, datetime('now'))",
                (f"{uid_number.lower()}@school", generate_password_hash("original-password"), student_id, school_id),
            )
        return student_id

    def _login_as_school(self, school_id):
        with self.client.session_transaction() as session:
            session["school_id"] = school_id

    def _password_hash(self, student_id):
        with app_module.game_connection() as connection:
            return connection.execute(
                "SELECT password_hash FROM game_users WHERE managed_student_id = ?", (student_id,)
            ).fetchone()["password_hash"]

    def _find_and_confirm(self, identifier):
        find_response = self.client.post(
            "/management/student-password-reset/find",
            data={"student_identifier": identifier, "school_id": self.school_b},
        )
        if find_response.status_code != 200:
            return find_response, None
        confirm_response = self.client.post("/management/student-password-reset/confirm")
        return find_response, confirm_response

    def test_school_a_can_reset_school_a_student(self):
        self._login_as_school(self.school_a)

        find_response, confirm_response = self._find_and_confirm("UID-A")
        self.assertEqual(find_response.status_code, 200)
        self.assertIn("Student Name", find_response.get_data(as_text=True))
        self.assertEqual(confirm_response.status_code, 200)
        self.assertIn('name="new_password"', confirm_response.get_data(as_text=True))
        self.assertTrue(check_password_hash(self._password_hash(self.student_a), "original-password"))

        response = self.client.post(
            "/management/student-password-reset",
            data={"new_password": "new-a-password", "confirm_password": "new-a-password"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "Student password has been reset successfully.")
        self.assertTrue(check_password_hash(self._password_hash(self.student_a), "new-a-password"))

    def test_lookup_accepts_login_username_and_student_id(self):
        self._login_as_school(self.school_a)

        username_response, _ = self._find_and_confirm("UID-A@SCHOOL")
        self.assertEqual(username_response.status_code, 200)
        username_html = username_response.get_data(as_text=True)
        self.assertIn("uid-a@school", username_html)
        self.assertIn("UID-A", username_html)

        student_id_response, _ = self._find_and_confirm(str(self.student_a))
        self.assertEqual(student_id_response.status_code, 200)
        self.assertIn("Student Name", student_id_response.get_data(as_text=True))

    def test_student_without_game_account_is_not_resettable(self):
        self._login_as_school(self.school_a)

        response = self.client.post(
            "/management/student-password-reset/find",
            data={"student_identifier": "UID-NO-ACCOUNT"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_data(as_text=True), "Student account not created.")

    def test_reset_requires_confirmed_student_session(self):
        self._login_as_school(self.school_a)
        original_hash = self._password_hash(self.student_a)

        response = self.client.post(
            "/management/student-password-reset",
            data={"new_password": "attacker-password", "confirm_password": "attacker-password"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_data(as_text=True), "Student not found in this school.")
        self.assertEqual(self._password_hash(self.student_a), original_hash)

    def test_school_a_cannot_reset_school_b_student(self):
        self._login_as_school(self.school_a)
        original_hash = self._password_hash(self.student_b)

        response = self.client.post(
            "/management/student-password-reset/find",
            data={"student_identifier": "UID-B"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_data(as_text=True), "Student not found in this school.")
        self.assertEqual(self._password_hash(self.student_b), original_hash)

    def test_school_b_cannot_reset_school_a_student(self):
        self._login_as_school(self.school_b)
        original_hash = self._password_hash(self.student_a)

        response = self.client.post(
            "/management/student-password-reset/find",
            data={"student_identifier": "UID-A"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_data(as_text=True), "Student not found in this school.")
        self.assertEqual(self._password_hash(self.student_a), original_hash)

    def test_forged_student_id_and_url_parameters_do_not_bypass_school_scope(self):
        self._login_as_school(self.school_a)
        original_hash = self._password_hash(self.student_b)

        response = self.client.post(
            f"/management/student-password-reset/find?student_id={self.student_b}&school_id={self.school_b}",
            data={"student_identifier": str(self.student_b), "student_id": str(self.student_b), "school_id": self.school_b},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_data(as_text=True), "Student not found in this school.")
        self.assertEqual(self._password_hash(self.student_b), original_hash)

    def test_dashboard_replaces_prediction_section_with_reset_form(self):
        self._login_as_school(self.school_a)

        response = self.client.get("/?view=home")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for label in ("ACCOUNT SUPPORT", "Reset Student Password", "Student UID / Student ID", "Find Student"):
            self.assertIn(label, html)
        self.assertNotIn('name="new_password"', html)
        self.assertNotIn('name="confirm_password"', html)
        self.assertNotIn('>Reset Password</button>', html)
        for removed_label in ("Confidence in the system", "Random Forest", "Accuracy", "Precision", "Recall", "Make a prediction"):
            self.assertNotIn(removed_label, html)
        self.assertEqual(html.count("/management/student-password-reset/find"), 1)


if __name__ == "__main__":
    unittest.main()
