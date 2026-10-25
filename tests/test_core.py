import csv
from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from attendance_core import (
    attendance_for_date, create_person_folder, descriptor, euclidean_distance,
    init_database, load_descriptors, match_face, parse_date, record_attendance, validate_name,
)
from app import create_app

ARTIFACTS = Path(__file__).resolve().parents[1] / ".test-artifacts"
ARTIFACTS.mkdir(exist_ok=True)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "attendance.db"
        self.now = datetime(2026, 1, 2, 9, 30, 0)

    def test_initialize_and_query_missing_database(self):
        self.assertEqual(attendance_for_date("2026-01-02", self.db), [])
        init_database(self.db)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual([row[1] for row in conn.execute("PRAGMA table_info(attendance)")],
                             ["name", "time", "date"])

    def test_duplicate_preserves_first_time(self):
        self.assertTrue(record_attendance("Demo Person", self.db, self.now))
        self.assertFalse(record_attendance("Demo Person", self.db, self.now.replace(hour=10)))
        self.assertEqual(attendance_for_date("2026-01-02", self.db), [("Demo Person", "09:30:00")])

    def test_different_dates_and_names(self):
        record_attendance("Demo Person", self.db, self.now)
        self.assertTrue(record_attendance("Demo Person", self.db, self.now.replace(day=3)))
        self.assertTrue(record_attendance("Other Person", self.db, self.now))
        self.assertEqual(len(attendance_for_date("2026-01-02", self.db)), 2)
        self.assertEqual(len(attendance_for_date("2026-01-03", self.db)), 1)

    def test_database_rejects_direct_duplicate_insert(self):
        record_attendance("Demo Person", self.db, self.now)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO attendance VALUES (?, ?, ?)", ("Demo Person", "10:00:00", "2026-01-02"))

    def test_legacy_duplicate_data_is_not_deleted(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("CREATE TABLE attendance(name TEXT, time TEXT, date DATE)")
            conn.executemany("INSERT INTO attendance VALUES (?, ?, ?)",
                             [("Demo Person", "09:00:00", "2026-01-02")] * 2)
        with self.assertRaises(sqlite3.IntegrityError):
            init_database(self.db)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0], 2)

    def test_name_validation_and_folder_safety(self):
        for name in ["", " ", "../escape", "a/b", "a\\b", "<script>", "unknown", "--", "x" * 81]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                create_person_folder(name, self.root / "faces")
        folder, number = create_person_folder(" Demo Person ", self.root / "faces")
        self.assertEqual(number, 1)
        self.assertEqual(folder.name, "person_1_Demo Person")
        with self.assertRaises(ValueError):
            create_person_folder("Demo Person", self.root / "faces")
        (self.root / "faces" / "unrelated").mkdir()
        self.assertEqual(create_person_folder("Other Person", self.root / "faces")[1], 2)
        self.assertEqual(validate_name("O'Neil"), "O'Neil")

    def test_invalid_date(self):
        for value in [None, "", "bad", "2026-02-30", "2026-01-02' OR 1=1 --"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_date(value)
        self.assertEqual(parse_date("2026-01-02"), "2026-01-02")


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.csv = Path(self.temp.name) / "features.csv"
        self.feature = np.zeros(128)
        self.feature[-1] = 1

    def write_rows(self, rows):
        with self.csv.open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerows(rows)

    def test_missing_empty_and_malformed_csv(self):
        with self.assertRaises(ValueError):
            load_descriptors(self.csv)
        for rows in [[], [["Demo Person", 1]], [["Demo Person", *(["bad"] * 128)]],
                     [["Demo Person", *(["nan"] * 128)]], [["Demo Person", *([0] * 128)]],
                     [["Demo Person", *self.feature]] * 2]:
            self.write_rows(rows)
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                load_descriptors(self.csv)

    def test_valid_csv_including_zero_first_coordinate(self):
        self.write_rows([["Demo Person", *self.feature]])
        names, features = load_descriptors(self.csv)
        self.assertEqual(names, ["Demo Person"])
        np.testing.assert_array_equal(features[0], self.feature)

    def test_invalid_descriptor(self):
        for values in [[], [1] * 127, [1] * 129, [0] * 128, [float("inf")] * 128]:
            with self.subTest(size=len(values)), self.assertRaises(ValueError):
                descriptor(values)

    def test_euclidean_distance(self):
        self.assertEqual(euclidean_distance([0, 0], [3, 4]), 5)

    def test_threshold_nearest_and_unknown(self):
        base = self.feature.copy()
        base[0] = 0.39
        self.assertEqual(match_face(base, ["Demo Person"], [self.feature]), "Demo Person")
        base[0] = 0.4
        self.assertEqual(match_face(base, ["Demo Person"], [self.feature]), "unknown")
        base[0] = 0.41
        self.assertEqual(match_face(base, ["Demo Person"], [self.feature]), "unknown")
        self.assertEqual(match_face(base, [], []), "unknown")
        self.assertEqual(match_face(base, ["Other", "Nearest"], [self.feature, base]), "Nearest")
        with self.assertRaises(ValueError):
            match_face(base, ["Mismatch"], [])


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "attendance.db"
        self.app = create_app(self.db)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Select Date", response.data)

    def test_missing_database_shows_no_data(self):
        response = self.client.post("/attendance", data={"selected_date": "2026-01-02"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No attendance data", response.data)

    def test_bad_dates_return_validation_message(self):
        for data in [{}, {"selected_date": "invalid"}, {"selected_date": "2026-02-30"}]:
            response = self.client.post("/attendance", data=data)
            self.assertEqual(response.status_code, 400)
            self.assertIn(b"Choose a valid date", response.data)

    def test_filtering(self):
        record_attendance("Demo Person", self.db, datetime(2026, 1, 2, 9))
        record_attendance("Other Person", self.db, datetime(2026, 1, 3, 9))
        response = self.client.post("/attendance", data={"selected_date": "2026-01-02"})
        self.assertIn(b"Demo Person", response.data)
        self.assertNotIn(b"Other Person", response.data)

    def test_unavailable_database_is_not_internal_error(self):
        self.app.config["DATABASE"] = Path(self.temp.name) / "missing" / "attendance.db"
        with self.assertLogs(self.app.logger, level="ERROR"):
            response = self.client.post("/attendance", data={"selected_date": "2026-01-02"})
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"Attendance storage is unavailable", response.data)


if __name__ == "__main__":
    unittest.main()
