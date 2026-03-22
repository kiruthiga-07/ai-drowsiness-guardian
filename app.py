import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time


# -----------------------------
# UI
# -----------------------------

st.set_page_config(
    page_title="Driver Drowsiness Alert",
    page_icon="🚗"
)

st.title("🚗 AI Drowsiness Detector")
st.write("Alert if eyes closed for 2 seconds")


# -----------------------------
# Video Processor
# -----------------------------

class DrowsinessProcessor(VideoProcessorBase):

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        self.closed_start = None
        self.drowsy = False


    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)


        if results.multi_face_landmarks:

            for face in results.multi_face_landmarks:

                # eye points
                top = face.landmark[159].y
                bottom = face.landmark[145].y

                distance = abs(top - bottom)


                if distance < 0.012:

                    if self.closed_start is None:
                        self.closed_start = time.time()

                    elif time.time() - self.closed_start > 2:
                        self.drowsy = True

                else:

                    self.closed_start = None
                    self.drowsy = False


                if self.drowsy:

                    cv2.putText(
                        img,
                        "WAKE UP !!!",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3,
                    )

                else:

                    cv2.putText(
                        img,
                        "AWAKE",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )


        return frame.from_ndarray(
            img,
            format="bgr24"
        )


# -----------------------------
# Camera
# -----------------------------

webrtc_streamer(
    key="drowsy",
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)
