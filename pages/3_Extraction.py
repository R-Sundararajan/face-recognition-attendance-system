"""Generate one mean pretrained dlib descriptor per enrolled person."""

import csv
import os
from pathlib import Path
import re
import tempfile

import cv2
import dlib
import numpy as np
import streamlit as st

from attendance_core import FACE_DIR, FEATURES_CSV, MODEL_DIR, descriptor, validate_name


def extract_features(root=FACE_DIR, destination=FEATURES_CSV):
    root, destination = Path(root), Path(destination)
    if not root.is_dir():
        raise ValueError("No enrollment directory found. Register and save face images first.")
    people = sorted(item for item in root.iterdir()
                    if item.is_dir() and re.fullmatch(r"person_\d+(?:_.*)?", item.name))
    if not people:
        raise ValueError("No registered people found. Register and save face images first.")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(MODEL_DIR / "shape_predictor_68_face_landmarks.dat"))
    model = dlib.face_recognition_model_v1(str(MODEL_DIR / "dlib_face_recognition_resnet_model_v1.dat"))
    rows, skipped, names = [], 0, set()
    for person in people:
        name = validate_name(person.name.split("_", 2)[-1] if person.name.count("_") >= 2 else person.name)
        if name in names:
            raise ValueError("Duplicate enrollment names found. Use a distinct name for each person.")
        names.add(name)
        features = []
        for photo in sorted(person.iterdir()):
            if not photo.is_file():
                continue
            image = cv2.imread(str(photo))
            if image is None:
                skipped += 1
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detector(image, 1)
            if len(faces) != 1:
                skipped += 1
                continue
            features.append(descriptor(model.compute_face_descriptor(image, predictor(image, faces[0]))))
        if not features:
            raise ValueError(f"Enrollment folder {person.name} has no usable single-face images. Existing CSV was not changed.")
        rows.append([name, *descriptor(np.mean(features, axis=0))])
    # Replace only after every person has a valid descriptor; failures preserve the old CSV.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=destination.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            csv.writer(output).writerows(rows)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(rows), skipped


def main():
    st.set_page_config(page_title="Face recognition attendance system")
    st.title("Facial Feature Extraction")
    if st.button("Extract CSV"):
        try:
            with st.spinner("Extracting face descriptors..."):
                people, skipped = extract_features()
        except (ValueError, OSError, RuntimeError) as error:
            st.error(str(error))
        else:
            st.success(f"Saved descriptors for {people} people to data/features_all.csv.")
            if skipped:
                st.warning(f"Skipped {skipped} unreadable images or images without exactly one face.")


if __name__ == "__main__":
    main()
