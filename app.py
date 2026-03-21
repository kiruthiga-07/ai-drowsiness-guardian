import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import time
import threading

# Page Configuration
st.set_page_config(page_title="Deep Focus Monitor", page_icon="🧘")
st.title("🧘 Deep Focus Monitor")
st.write("Real-time AI monitoring. Stay in the frame to keep your timer running!")

# Shared variable to communicate between the video thread and the UI
lock = threading.Lock()
container = {"is_focused": False}

# The Video Processor Class
class FaceDetector(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        with lock:
            container["is_focused"] = len(faces) > 0

        # Draw a rectangle around the face for visual feedback
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "FOCUSED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

# Start the WebRTC Streamer
webrtc_streamer(key="focus-monitor", video_processor_factory=FaceDetector)

# Focus Timer Logic in the UI
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "focus_seconds" not in st.session_state:
    st.session_state.focus_seconds = 0

placeholder = st.empty()

# UI Update Loop
while True:
    with lock:
        focused = container["is_focused"]

    if focused:
        st.session_state.focus_seconds += 1
        status = "✅ You are focused!"
        color = "green"
    else:
        status = "⚠️ LOOK AT THE SCREEN!"
        color = "red"

    with placeholder.container():
        st.markdown(f"<h2 style='color: {color}; text-align: center;'>{status}</h2>", unsafe_allow_html=True)
        st.metric("Total Focus Time", f"{st.session_state.focus_seconds} seconds")
    
    time.sleep(1) # Update the timer every second
