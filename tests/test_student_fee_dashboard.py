import gc
import tempfile
import unittest
from pathlib import Path

import app as app_module


class StudentFeeDashboardTests(unittest.TestCase):
    def setUp(self):
        self.database_directory = tempfile.TemporaryDirectory()
        self.original_database_path = app_module.GAME_DB_PATH
        app_module.GAME_DB_PATH = Path(self.database_directory.name) / "student_game.db"
        app_module.initialize_game_database()
        app_module.app.config.update(TESTING=True)
        self.client = app_module.app.test_client()

        with app_module.game_connection() as connection:
            self.school_a = self._insert_school(connection, "School A", "school-a")
            self.school_b = self._insert_school(connection, "School B", "school-b")
            self.student_a, self.user_a = self._insert_student(connection, self.school_a, "UID-A", "student-a")
            self.student_b, self.user_b = self._insert_student(connection, self.school_b, "UID-B", "student-b")
            self.fee_a = self._insert_fee(connection, self.school_a, self.student_a, "April", "Paid", b"receipt-a")
            self.fee_b = self._insert_fee(connection, self.school_b, self.student_b, "April", "Pending", None)

    def tearDown(self):
        app_module.GAME_DB_PATH = self.original_database_path
        gc.collect()
        self.database_directory.cleanup()

    @staticmethod
    def _insert_school(connection, name, username):
        return connection.execute(
            "INSERT INTO schools (school_name, username, password_hash, created_at) VALUES (?, ?, ?, datetime('now'))",
            (name, username, "hash"),
        ).lastrowid

    @staticmethod
    def _insert_student(connection, school_id, uid, username):
        student_id = connection.execute(
            "INSERT INTO managed_students (school_id, class_name, roll_number, uid_number, student_name, father_name, mother_name, parent_mobile, address, created_at, updated_at) "
            "VALUES (?, 'Class 1', '1', ?, 'Student Name', 'Father', 'Mother', '1234567890', 'Address', datetime('now'), datetime('now'))",
            (school_id, uid),
        ).lastrowid
        user_id = connection.execute(
            "INSERT INTO game_users (student_name, username, password_hash, managed_student_id, school_id, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("Student Name", username, "hash", student_id, school_id),
        ).lastrowid
        return student_id, user_id

    @staticmethod
    def _insert_fee(connection, school_id, student_id, month, status, receipt_data):
        return connection.execute(
            "INSERT INTO student_fee_status (school_id, class_name, student_id, fee_month, status, locked, receipt_data, created_at, updated_at) VALUES (?, 'Class 1', ?, ?, ?, 1, ?, datetime('now'), datetime('now'))",
            (school_id, student_id, month, status, receipt_data),
        ).lastrowid

    def _login_as_student(self, user_id):
        with self.client.session_transaction() as session:
            session["game_user_id"] = user_id

    def test_student_sees_only_own_fee_status(self):
        self._login_as_student(self.user_a)

        response = self.client.get("/student-dashboard")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("FEE HISTORY", html)
        self.assertIn("Fee History", html)
        self.assertIn("April", html)
        self.assertIn("Paid", html)
        self.assertIn(f"/student-fee-receipt/{self.fee_a}", html)
        self.assertNotIn(f"/student-fee-receipt/{self.fee_b}", html)
        self.assertNotIn("Total Fee", html)
        self.assertNotIn("Paid Amount", html)
        self.assertNotIn("Due Amount", html)

    def test_fee_history_is_the_last_dashboard_section(self):
        self._login_as_student(self.user_a)

        html = self.client.get("/student-dashboard").get_data(as_text=True)
        notices_position = html.find('class="panel class-notices-panel"')
        fee_history_position = html.find('<section class="panel fee-status-panel"')

        self.assertGreaterEqual(notices_position, 0)
        self.assertGreaterEqual(fee_history_position, 0)
        self.assertLess(notices_position, fee_history_position)
        self.assertEqual(html.rfind("<section"), fee_history_position)

    def test_student_cannot_access_another_students_receipt(self):
        self._login_as_student(self.user_a)

        response = self.client.get(f"/student-fee-receipt/{self.fee_b}")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_data(as_text=True), "Receipt not found.")

    def test_existing_receipt_is_served_without_creating_records(self):
        self._login_as_student(self.user_a)
        with app_module.game_connection() as connection:
            before_count = connection.execute("SELECT COUNT(*) FROM student_fee_status").fetchone()[0]

        response = self.client.get(f"/student-fee-receipt/{self.fee_a}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"receipt-a")
        with app_module.game_connection() as connection:
            after_count = connection.execute("SELECT COUNT(*) FROM student_fee_status").fetchone()[0]
        self.assertEqual(before_count, after_count)

    def test_second_student_sees_second_students_status(self):
        self._login_as_student(self.user_b)

        response = self.client.get("/student-dashboard")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Pending", html)
        self.assertNotIn(f"/student-fee-receipt/{self.fee_a}", html)


if __name__ == "__main__":
    unittest.main()
