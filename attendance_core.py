"""Storage and descriptor validation shared by the local attendance pages."""

from contextlib import closing
from datetime import date, datetime
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FACE_DIR = ROOT / "data" / "data_faces_from_camera"
FEATURES_CSV = ROOT / "data" / "features_all.csv"
MODEL_DIR = ROOT / "data" / "data_dlib"
DATABASE = ROOT / "attendance.db"
MATCH_THRESHOLD = 0.4


def validate_name(name):
    name = name.strip()
    if not name or len(name) > 80 or not all(
        char.isalnum() or char in " _-'" for char in name
    ) or not any(char.isalnum() for char in name) or name.casefold() == "unknown":
        raise ValueError("Use a name of 1-80 letters/numbers, spaces, underscores, hyphens or apostrophes; 'unknown' is reserved.")
    return name


def create_person_folder(name, root=FACE_DIR):
    name = validate_name(name)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for item in root.iterdir():
        if item.is_dir() and item.name.count("_") >= 2 and item.name.split("_", 2)[-1] == name:
            raise ValueError("This name is already enrolled. Use a distinct name for each person.")
    ids = [int(match.group(1)) for item in root.iterdir()
           if item.is_dir() and (match := re.fullmatch(r"person_(\d+)(?:_.*)?", item.name))]
    number = max(ids, default=0) + 1
    # Refuse to overwrite an existing enrollment directory.
    folder = root / f"person_{number}_{name}"
    folder.mkdir()
    return folder, number


def parse_date(value):
    try:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        raise ValueError("Choose a valid date in YYYY-MM-DD format.") from None


def init_database(path=DATABASE):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS attendance (
            name TEXT, time TEXT, date DATE, UNIQUE(name, date))""")
        # Older schemas must also enforce same-day uniqueness.
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS attendance_name_date ON attendance(name, date)")


def record_attendance(name, path=DATABASE, now=None):
    name = validate_name(name)
    now = now or datetime.now()
    init_database(path)
    with closing(sqlite3.connect(path)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO attendance(name, time, date) VALUES (?, ?, ?) "
            "ON CONFLICT(name, date) DO NOTHING",
            (name, now.strftime("%H:%M:%S"), now.date().isoformat()),
        )
        return cursor.rowcount == 1


def attendance_for_date(value, path=DATABASE):
    value = parse_date(value)
    init_database(path)
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(
            "SELECT name, time FROM attendance WHERE date = ? ORDER BY time, name", (value,)
        ).fetchall()


def descriptor(values):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Descriptors must contain 128 finite numeric values.") from None
    if result.shape != (128,) or not np.isfinite(result).all() or not np.any(result):
        raise ValueError("Descriptors must contain 128 finite numeric values and cannot be all zero.")
    return result


def load_descriptors(path=FEATURES_CSV):
    names, features = [], []
    try:
        rows = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        raise ValueError("No face descriptors found. Register faces and run Extract CSV first.") from None
    except pd.errors.EmptyDataError:
        raise ValueError("The descriptor CSV is empty. Register faces and run Extract CSV first.") from None
    except pd.errors.ParserError:
        raise ValueError("Malformed descriptor CSV. Run Extract CSV again.") from None
    for line, row in enumerate(rows.itertuples(index=False, name=None), 1):
        try:
            if len(row) != 129:
                raise ValueError("Expected a name followed by 128 values.")
            name, feature = validate_name(row[0]), descriptor(row[1:])
            if name in names:
                raise ValueError("Duplicate name; each person must have a distinct name.")
        except ValueError as error:
            raise ValueError(f"Invalid descriptor row {line}: {error}") from error
        names.append(name)
        features.append(feature)
    if not names:
        raise ValueError("The descriptor CSV is empty. Register faces and run Extract CSV first.")
    return names, features


def euclidean_distance(first, second):
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


def match_face(values, names, features):
    values = descriptor(values)
    if len(names) != len(features):
        raise ValueError("Names and descriptors must have the same length.")
    if not features:
        return "unknown"
    distances = [euclidean_distance(values, descriptor(feature)) for feature in features]
    best = int(np.argmin(distances))
    return names[best] if distances[best] < MATCH_THRESHOLD else "unknown"
