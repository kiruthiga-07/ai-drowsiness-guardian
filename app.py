import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import threading

# Page Setup
st.set_page_config(page_title="Deep Focus AI", page_icon="🧘")
st.title("🧘 Deep Focus AI")

# We use a simple class to store our "Focus Points"
class SessionState:
    def __init__(self):
        self.focus_count = 0
        self.is_focused = False

# This persists the data even when the video is running
if "state" not in st.session_state:
    st.session_state.state = SessionState()

class FocusProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # Update the focus count directly
        if len(faces) > 0:
            st.session_state.state.focus_count += 1
            st.session_state.state.is_focused = True
            color = (0, 255, 0)
            label = "FOCUS MODE ON"
        else:
            st.session_state.state.is_focused = False
            color = (0, 0, 255)
            label = "NOT DETECTED"

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame.from_ndarray(img, format="bgr24")

# --- UI DISPLAY ---
col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="focus-stream",
        video_processor_factory=FocusProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Stats")
    # This automatically refreshes the numbers
    total_sec = st.session_state.state.focus_count // 20 # Approximation of seconds
    st.metric("Total Focus Time", f"{total_sec} sec")
    
    if st.session_state.state.is_focused:
        st.success("✨ YOU ARE HERE!")
    else:
        st.error("😴 WHERE ARE YOU??")

# Add an auto-refresh so the timer numbers move
if st.button("Refresh Stats"):
    st.rerun()
