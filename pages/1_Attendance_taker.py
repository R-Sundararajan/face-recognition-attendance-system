import dlib
import time
import logging
import sqlite3
import cv2
import streamlit as st

from attendance_core import (
    MODEL_DIR, init_database, load_descriptors, match_face,
    euclidean_distance, record_attendance,
)

# Hide the Deploy button, header, and footer using CSS
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}  /* Hide the menu */
        footer {visibility: hidden;}      /* Hide the footer */
        header {visibility: hidden;}      /* Hide the header */
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(""" 
 <style>
  div {    
 }  
#left {   
 text-align: left;  
 }  
 #center {    
 text-align: center;
 backgroundColor = "#000000"  
 }  
 #right{    
 text-align: right; }
  </style>
  <div id ="center">  
  <h1> Opening Attendance Taker </h1>
  </div>
  """,
              unsafe_allow_html= True )
# Use dlib's frontal face detector.
detector = dlib.get_frontal_face_detector()

class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        self.predictor = dlib.shape_predictor(str(MODEL_DIR / "shape_predictor_68_face_landmarks.dat"))
        self.face_reco_model = dlib.face_recognition_model_v1(str(MODEL_DIR / "dlib_face_recognition_resnet_model_v1.dat"))

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        #  Save the features of faces in the database
        self.face_features_known_list = []

        # / Save the name of faces in the database
        self.face_name_known_list = []

        #  List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        #  cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        #  Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        #  Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    #  "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        self.face_name_known_list, self.face_features_known_list = load_descriptors()
        logging.info("Loaded registered face descriptors")
        return True

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        return euclidean_distance(feature_1, feature_2)

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # cv2 window / putText on cv2 window window
    def draw_note(self, img_rd):
        #  / Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with Deep Learning", (20, 40), self.font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)
    # insert data in database5
    def attendance(self, name):
        return record_attendance(name)

    #  Face detection and recognition wit OT from input video stream
    def process(self, stream):
        # 1.  Get faces known from "features.all.csv"
        global count
        if self.get_face_database():

            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                if not flag or img_rd is None:
                    st.error("No camera frame received. Check webcam permissions and other camera apps.")
                    break
                rgb_frame = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
                kk = cv2.waitKey(1)

                # 2.  Detect faces for frame X
                faces = detector(rgb_frame, 0)
                self.reclassify_interval_cnt += 1

                # 3.  Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.  Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5.  update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1  if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt < self.reclassify_interval):
                    logging.debug("scene 1:   No face cnt changes in this frame!!!")

                    self.current_frame_face_position_list = []


                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                    #  Multi-faces in current frame, use centroid-tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # 6.2 Write names under ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    self.draw_note(img_rd)

                # 6.2  If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1  Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                    # 6.2.2 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug("  scene 2.2  Get faces in this frame and do f  ace recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = self.predictor(rgb_frame, faces[i])
                            self.current_frame_face_feature_list.append(
                                self.face_reco_model.compute_face_descriptor(rgb_frame, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("  For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])
                            self.current_frame_face_X_e_distance_list = []

                            # 6.2.2.2  Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            nam = match_face(
                                self.current_frame_face_feature_list[k],
                                self.face_name_known_list, self.face_features_known_list,
                            )
                            self.current_frame_face_name_list[k] = nam
                            if nam != "unknown" and self.attendance(nam):
                                st.success(f"{nam} is marked as present.")
                        # 7.  / Add note on cv2 window
                        self.draw_note(img_rd)

                # 8.  'q'  / Press 'q' to exit
                if kk == ord('q'):
                    break
                    
                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                logging.debug("Frame ends\n\n")

    def run(self):
        # cap = cv2.VideoCapture("video.mp4")
        # Get video stream from video file
        cap = cv2.VideoCapture(0)  # Get video stream from camera
        try:
            if not cap.isOpened():
                raise RuntimeError("Cannot open the default webcam. Check permissions and other camera apps.")
            init_database()
            self.process(cap)
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()

if __name__ == '__main__':
    if st.button("Start attendance camera"):
        try:
            main()
        except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
            st.error(str(error))
