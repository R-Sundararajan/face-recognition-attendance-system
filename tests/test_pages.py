import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from streamlit.testing.v1 import AppTest

import attendance_core as core

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".test-artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def load_page(filename):
    spec = importlib.util.spec_from_file_location(filename, ROOT / "pages" / filename)
    module = importlib.util.module_from_spec(spec)
    with patch("streamlit.markdown"), patch("streamlit.set_page_config"):
        spec.loader.exec_module(module)
    return module


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.page = load_page("3_Extraction.py")
        self.temp = tempfile.TemporaryDirectory(dir=ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.faces = self.root / "faces"
        self.destination = self.root / "features.csv"

    def test_missing_enrollment_does_not_touch_csv(self):
        self.destination.write_text("unchanged", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.page.extract_features(self.faces, self.destination)
        self.assertEqual(self.destination.read_text(), "unchanged")

    def test_mean_extraction_and_atomic_failure(self):
        folder, _ = core.create_person_folder("Demo Person", self.faces)
        for name in ["a.jpg", "b.jpg"]:
            (folder / name).touch()
        detector = Mock(return_value=[object()])
        model = Mock()
        model.compute_face_descriptor.side_effect = [np.ones(128), np.ones(128) * 3]
        with patch.object(self.page.dlib, "get_frontal_face_detector", return_value=detector), \
             patch.object(self.page.dlib, "shape_predictor"), \
             patch.object(self.page.dlib, "face_recognition_model_v1", return_value=model), \
             patch.object(self.page.cv2, "imread", return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
            self.assertEqual(self.page.extract_features(self.faces, self.destination), (1, 0))
            names, features = core.load_descriptors(self.destination)
            self.assertEqual(names, ["Demo Person"])
            np.testing.assert_array_equal(features[0], np.ones(128) * 2)
            original = self.destination.read_bytes()
            detector.return_value = []
            with self.assertRaises(ValueError):
                self.page.extract_features(self.faces, self.destination)
            self.assertEqual(self.destination.read_bytes(), original)

    def test_streamlit_button_controls_extraction(self):
        folder, _ = core.create_person_folder("Demo Person", self.faces)
        (folder / "synthetic.jpg").touch()
        model = Mock()
        model.compute_face_descriptor.return_value = np.ones(128)
        with patch.object(core, "FACE_DIR", self.faces), patch.object(core, "FEATURES_CSV", self.destination), \
             patch("dlib.get_frontal_face_detector", return_value=Mock(return_value=[object()])), \
             patch("dlib.shape_predictor"), patch("dlib.face_recognition_model_v1", return_value=model), \
             patch("cv2.imread", return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
            app = AppTest.from_file(str(ROOT / "pages" / "3_Extraction.py")).run()
            self.assertFalse(app.exception)
            self.assertFalse(self.destination.exists())
            app.run()
            self.assertFalse(self.destination.exists())
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(self.destination.exists())
            self.assertEqual(len(app.success), 1)
            original = self.destination.read_bytes()
            app.run()
            self.assertEqual(self.destination.read_bytes(), original)

    def test_streamlit_failure_has_no_success(self):
        with patch.object(core, "FACE_DIR", self.faces), patch.object(core, "FEATURES_CSV", self.destination):
            app = AppTest.from_file(str(ROOT / "pages" / "3_Extraction.py")).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 1)
            self.assertEqual(len(app.success), 0)
            self.assertFalse(self.destination.exists())


class CameraSafetyTests(unittest.TestCase):
    def setUp(self):
        self.register = load_page("2_New_Register.py")
        self.attendance = load_page("1_Attendance_taker.py")

    def test_clear_cancel_preserves_enrollment(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temp:
            folder, _ = core.create_person_folder("Demo Person", temp)
            register = self.register.Face_Register.__new__(self.register.Face_Register)
            register.path_photos_from_camera = temp
            register.win = Mock()
            with patch.object(self.register.messagebox, "askyesno", return_value=False):
                register.GUI_clear_data()
            self.assertTrue(folder.exists())

    def test_clear_confirmed_only_removes_enrollment_and_csv(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temp:
            root = Path(temp)
            folder, _ = core.create_person_folder("Demo Person", root)
            unrelated = root / "unrelated"
            unrelated.mkdir()
            features = root / "features.csv"
            features.touch()
            register = self.register.Face_Register.__new__(self.register.Face_Register)
            register.path_photos_from_camera = temp
            register.win = Mock()
            register.log_all = {}
            register.label_cnt_face_in_database = {}
            with patch.object(self.register.messagebox, "askyesno", return_value=True), \
                 patch.object(self.register, "FEATURES_CSV", features):
                register.GUI_clear_data()
            self.assertFalse(folder.exists())
            self.assertFalse(features.exists())
            self.assertTrue(unrelated.exists())
            self.assertFalse(register.face_folder_created_flag)

    def test_registration_failed_read_is_safe(self):
        register = self.register.Face_Register.__new__(self.register.Face_Register)
        register.cap = Mock()
        register.cap.read.return_value = (False, None)
        self.assertEqual(register.get_frame(), (False, None))

    def test_invalid_registration_cannot_save_to_previous_person(self):
        register = self.register.Face_Register.__new__(self.register.Face_Register)
        register.input_name = Mock()
        register.input_name.get.return_value = "../escape"
        register.log_all = {}
        register.face_folder_created_flag = True
        register.current_face_dir = "previous-person"
        register.GUI_get_input_name()
        self.assertFalse(register.face_folder_created_flag)
        self.assertEqual(register.current_face_dir, "")

    def test_registration_close_releases_camera(self):
        register = self.register.Face_Register.__new__(self.register.Face_Register)
        register.cap = Mock()
        register.win = Mock()
        register.close()
        register.cap.release.assert_called_once()
        register.win.destroy.assert_called_once()

    def test_pages_do_not_open_camera_until_button_is_clicked(self):
        with patch("cv2.VideoCapture") as camera:
            for filename in ["1_Attendance_taker.py", "2_New_Register.py"]:
                app = AppTest.from_file(str(ROOT / "pages" / filename)).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.button), 1)
            camera.assert_not_called()

    def test_attendance_releases_camera_after_failure(self):
        recognizer = self.attendance.Face_Recognizer.__new__(self.attendance.Face_Recognizer)
        camera = Mock()
        camera.isOpened.return_value = True
        recognizer.process = Mock(side_effect=ValueError("Invalid CSV"))
        with patch.object(self.attendance.cv2, "VideoCapture", return_value=camera), \
             patch.object(self.attendance.cv2, "destroyAllWindows") as destroy, \
             patch.object(self.attendance, "init_database"):
            with self.assertRaises(ValueError):
                recognizer.run()
            camera.release.assert_called_once()
            destroy.assert_called_once()

    def test_unavailable_camera_is_released(self):
        recognizer = self.attendance.Face_Recognizer.__new__(self.attendance.Face_Recognizer)
        camera = Mock()
        camera.isOpened.return_value = False
        with patch.object(self.attendance.cv2, "VideoCapture", return_value=camera), \
             patch.object(self.attendance.cv2, "destroyAllWindows"):
            with self.assertRaises(RuntimeError):
                recognizer.run()
        camera.release.assert_called_once()

    def test_failed_image_write_does_not_claim_success(self):
        register = self.register.Face_Register.__new__(self.register.Face_Register)
        register.face_folder_created_flag = True
        register.current_frame_faces_cnt = 1
        register.out_of_range_flag = False
        register.ss_cnt = 0
        register.face_ROI_height = register.face_ROI_width = 1
        register.face_ROI_height_start = register.face_ROI_width_start = 0
        register.hh = register.ww = 0
        register.raw_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        register.current_face_dir = "unused"
        register.log_all = {}
        with patch.object(self.register.cv2, "imwrite", return_value=False):
            register.save_current_face()
        self.assertEqual(register.ss_cnt, 0)
        self.assertIn("could not be saved", register.log_all["text"])

    def test_each_detected_face_is_matched_and_unknown_is_not_recorded(self):
        with patch.object(self.attendance.dlib, "shape_predictor"), \
             patch.object(self.attendance.dlib, "face_recognition_model_v1"):
            recognizer = self.attendance.Face_Recognizer()
        recognizer.get_face_database = Mock(return_value=True)
        recognizer.face_name_known_list = ["Demo Person"]
        recognizer.face_features_known_list = [np.ones(128)]
        recognizer.face_reco_model.compute_face_descriptor.side_effect = [np.ones(128), np.ones(128) * 3]
        recognizer.attendance = Mock(return_value=True)
        camera = Mock()
        camera.isOpened.side_effect = [True, False]
        camera.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        face = self.attendance.dlib.rectangle(10, 10, 40, 40)
        with patch.object(self.attendance, "detector", return_value=[face, face]), \
             patch.object(self.attendance.cv2, "waitKey", return_value=ord("q")), \
             patch.object(self.attendance.st, "success"):
            recognizer.process(camera)
        self.assertEqual(recognizer.current_frame_face_name_list, ["Demo Person", "unknown"])
        recognizer.attendance.assert_called_once_with("Demo Person")


if __name__ == "__main__":
    unittest.main()
