import gc
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

import app as app_module


class ClassNoticeTests(unittest.TestCase):
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
            self.student_a9, self.user_a9 = self._insert_student(connection, self.school_a, "Class 9", "a9")
            self.student_a8, self.user_a8 = self._insert_student(connection, self.school_a, "Class 8", "a8")
            self.student_b9, self.user_b9 = self._insert_student(connection, self.school_b, "Class 9", "b9")

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
    def _insert_student(connection, school_id, class_name, username):
        student_id = connection.execute(
            "INSERT INTO managed_students (school_id, class_name, roll_number, uid_number, student_name, father_name, mother_name, parent_mobile, address, created_at, updated_at) "
            "VALUES (?, ?, '1', ?, 'Student Name', 'Father', 'Mother', '1234567890', 'Address', datetime('now'), datetime('now'))",
            (school_id, class_name, f"UID-{username}"),
        ).lastrowid
        user_id = connection.execute(
            "INSERT INTO game_users (student_name, username, password_hash, managed_student_id, school_id, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("Student Name", username, "hash", student_id, school_id),
        ).lastrowid
        return student_id, user_id

    def _login_as_school(self, school_id):
        with self.client.session_transaction() as session:
            session["school_id"] = school_id

    def _login_as_student(self, user_id):
        with self.client.session_transaction() as session:
            session["game_user_id"] = user_id

    @staticmethod
    def _class_url(class_name):
        return f"/students/{quote(class_name, safe='')}"

    def _publish_notice(self, school_id, class_name, title="Unit Test"):
        self._login_as_school(school_id)
        return self.client.post(
            f"{self._class_url(class_name)}/notices",
            data={
                "title": title,
                "description": "Mathematics test notice.",
                "notice_date": "2026-10-04",
                "event_date": "2026-10-10",
                "priority": "Important",
            },
        )

    def test_staff_can_create_and_view_class_notice(self):
        response = self._publish_notice(self.school_a, "Class 9")

        self.assertEqual(response.status_code, 302)
        page = self.client.get(f"{self._class_url('Class 9')}?open_notice=1")
        html = page.get_data(as_text=True)
        self.assertIn("Class Notice - Class 9", html)
        self.assertIn("Unit Test", html)
        self.assertIn("Existing Notices", html)

    def test_notice_is_visible_only_to_matching_class_and_school(self):
        self._publish_notice(self.school_a, "Class 9")

        self._login_as_student(self.user_a9)
        class_nine_html = self.client.get("/student-dashboard").get_data(as_text=True)
        self.assertIn("Unit Test", class_nine_html)

        self._login_as_student(self.user_a8)
        class_eight_html = self.client.get("/student-dashboard").get_data(as_text=True)
        self.assertNotIn("Unit Test", class_eight_html)

        self._publish_notice(self.school_b, "Class 9", title="School B Notice")
        self._login_as_student(self.user_a9)
        school_a_html = self.client.get("/student-dashboard").get_data(as_text=True)
        self.assertNotIn("School B Notice", school_a_html)

        self._login_as_student(self.user_b9)
        school_b_html = self.client.get("/student-dashboard").get_data(as_text=True)
        self.assertIn("School B Notice", school_b_html)

    def test_staff_can_edit_and_delete_notice(self):
        self._publish_notice(self.school_a, "Class 9")
        with app_module.game_connection() as connection:
            notice_id = connection.execute("SELECT id FROM class_notices WHERE school_id = ?", (self.school_a,)).fetchone()["id"]

        edit_response = self.client.post(
            f"{self._class_url('Class 9')}/notices",
            data={
                "_action": "update",
                "notice_id": str(notice_id),
                "title": "Updated Test",
                "description": "Updated notice.",
                "notice_date": "2026-10-04",
                "event_date": "2026-10-11",
                "priority": "Normal",
            },
        )
        self.assertEqual(edit_response.status_code, 302)
        self._login_as_student(self.user_a9)
        self.assertIn("Updated Test", self.client.get("/student-dashboard").get_data(as_text=True))

        self._login_as_school(self.school_a)
        delete_response = self.client.post(
            f"{self._class_url('Class 9')}/notices",
            data={"_action": "delete", "notice_id": str(notice_id)},
        )
        self.assertEqual(delete_response.status_code, 302)
        self._login_as_student(self.user_a9)
        self.assertNotIn("Updated Test", self.client.get("/student-dashboard").get_data(as_text=True))

    def test_students_and_logged_out_users_cannot_manage_notices(self):
        response = self.client.post(
            f"{self._class_url('Class 9')}/notices",
            data={"title": "Unauthorized", "description": "No", "priority": "Normal"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/school-login", response.location)

        self._login_as_student(self.user_a9)
        response = self.client.post(
            f"{self._class_url('Class 9')}/notices",
            data={"title": "Unauthorized", "description": "No", "priority": "Normal"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/school-login", response.location)


if __name__ == "__main__":
    unittest.main()
