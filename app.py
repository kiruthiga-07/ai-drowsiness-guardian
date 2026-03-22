import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

# Title and UI
st.set_page_config(page_title="Auto-Focus Monitor", page_icon="👁️")
st.title("👁️ Real-Time Focus Monitor")
st.write("The timer below will count up as long as the AI sees your face!")

# This class handles the video frames automatically
class FaceFocusTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for faster detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # If a face is found, increment the internal counter
        if len(faces) > 0:
            self.count += 1
            # Draw green box
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "FOCUSED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Draw red warning
            cv2.putText(img, "NOT DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img

# Create the WebRTC streamer
# This is what turns on your camera automatically
ctx = webrtc_streamer(key="example", video_transformer_factory=FaceFocusTransformer)

if ctx.video_transformer:
    # Display the "Focus Score" (Total frames where a face was seen)
    st.metric("Focus Points", ctx.video_transformer.count)
    
    if ctx.video_transformer.count > 0:
        st.success("Great job! You are staying in the frame.")
    else:
        st.warning("Position yourself in front of the camera to start the timer.")

st.info("Note: Click 'START' in the video window above to turn on your webcam.")
