import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import time
import threading

# Page Setup
st.set_page_config(page_title="Deep Focus AI", page_icon="🧘")
st.title("🧘 Deep Focus AI")
st.write("The timer starts automatically when you look at the screen!")

# --- SHARED DATA ---
# This dictionary shares data between the video thread and the web page
lock = threading.Lock()
data_container = {"is_focused": False}

# --- VIDEO PROCESSING ---
class FocusProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # Update the focus status globally
        with lock:
            data_container["is_focused"] = len(faces) > 0

        # Add visual fun - Draw green/red boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, "FOCUS MODE ON", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame.from_ndarray(img, format="bgr24")

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    # Start the live camera
    ctx = webrtc_streamer(
        key="focus-stream",
        video_processor_factory=FocusProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Standard for web streaming
        media_stream_constraints={"video": True, "audio": False},
    )

# Timer Session State
if "total_time" not in st.session_state:
    st.session_state.total_time = 0

with col2:
    st.subheader("Stats")
    timer_placeholder = st.empty()
    status_placeholder = st.empty()

# --- THE LOOP ---
# This updates the UI without needing to refresh the whole page
if ctx.state.playing:
    while True:
        with lock:
            focused = data_container["is_focused"]

        if focused:
            st.session_state.total_time += 1
            status_text = "✨ YOU ARE KILLING IT! ✨"
            status_color = "#28a745" # Green
        else:
            status_text = "😴 WHERE ARE YOU??"
            status_color = "#dc3545" # Red

        # Update the UI instantly
        timer_placeholder.metric("Focus Duration", f"{st.session_state.total_time} sec")
        status_placeholder.markdown(
            f"<div style='padding:20px; border-radius:10px; background-color:{status_color}; color:white; text-align:center; font-weight:bold;'>"
            f"{status_text}</div>", 
            unsafe_allow_html=True
        )

        time.sleep(1) # Wait 1 second before checking again
