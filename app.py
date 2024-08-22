import streamlit as st
import numpy as np
import json
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Attempt to import OpenCV with the headless option
try:
   import cv2
except ImportError as e:
   st.error(f"Failed to import cv2: {e}")
   st.stop()

# Attempt to import DeepFace and handle potential import errors
try:
   from deepface import DeepFace
except ImportError as e:
   st.error(f"Failed to import DeepFace: {e}")
   st.stop()

# Function to analyze facial attributes using DeepFace
def analyze_frame(frame):
   try:
      result = DeepFace.analyze(img_path=frame, actions=['age', 'gender', 'race', 'emotion'],
                                     enforce_detection=False,
                                     detector_backend="opencv",
                                     align=True,
                                     silent=False)
      return result
   except Exception as e:
        st.error(f"Failed to analyze frame: {e}")
        return None

def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.9  # Adjust the transparency of the overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  # White rectangle
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_position = 15  # Where the first text is put into the overlay
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        text_position += 20

    return frame

def save_result_to_json(data, filename='attendance.json'):
       if os.path.exists(filename):
           with open(filename, 'r') as file:
               attendance_data = json.load(file)
       else:
           attendance_data = []

       attendance_data.append(data)

       with open(filename, 'w') as file:
           json.dump(attendance_data, file, indent=4)

class VideoTransformer(VideoTransformerBase):
       def __init__(self, user_type, action):
           self.user_type = user_type
           self.action = action

       def transform(self, frame):
           img = frame.to_ndarray(format="bgr24")

           # Analyze the frame using DeepFace
           result = analyze_frame(img)
           if result is None:
               return img

           # Extract the face coordinates
           face_coordinates = result[0]["region"]
           x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']

           # Draw bounding box around the face
           cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
           text = f"{result[0]['dominant_emotion']}"
           cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

           # Overlay white rectangle with text on the frame
           texts = [
               f"Age: {result[0]['age']}",
               f"Face Confidence: {round(result[0]['face_confidence'], 3)}",
               f"Gender: {result[0]['dominant_gender']} {round(result[0]['gender'][result[0]['dominant_gender']], 3)}",
               f"Race: {result[0]['dominant_race']}",
               f"Dominant Emotion: {result[0]['dominant_emotion']} {round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}",
           ]

           img_with_overlay = overlay_text_on_frame(img, texts)

           # Save result to JSON
           data = {
               "user_type": self.user_type,
               "action": self.action,
               "age": result[0]['age'],
               "gender": result[0]['dominant_gender'],
               "race": result[0]['dominant_race'],
               "emotion": result[0]['dominant_emotion'],
               "face_confidence": result[0]['face_confidence'],
               "timestamp": datetime.now().isoformat()
           }
           save_result_to_json(data)

           return img_with_overlay

def main():
       activities = ["Webcam Face Detection", "About"]
       choice = st.sidebar.selectbox("Select Activity", activities)
       st.sidebar.markdown(
           """     
           """)
       if choice == "Webcam Face Detection":
           html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                               <h4 style="color:white;text-align:center;">
                                               Smart Attendance</h4>
                                               </div>
                                               </br>"""
           st.markdown(html_temp_home1, unsafe_allow_html=True)
           user_type = st.selectbox("Select User Type", ["Employee", "Customer"])
           action = None
           if user_type == "Employee":
               action = st.selectbox("Select Action", ["In", "Out"])
           
           if st.button("Start Detection", key="start_button"):  # Add a unique key for the start button
               webrtc_streamer(
                   key="example",
                   mode=WebRtcMode.SENDRECV,
                   video_transformer_factory=lambda: VideoTransformer(user_type, action),
                   media_stream_constraints={"video": True, "audio": False},
               )

       elif choice == "About":
           st.subheader("About this app")

           html_temp4 = """
                                          		<div style="background-color:#98AFC7;padding:10px">
                                          		<h4 style="color:white;text-align:center;">Aplikasi ini dapat digunakan untuk melakukan pengenalan wajah pada gambar atau video. </h4>
                                          		<h4 style="color:white;text-align:center;">Terimakasih sudah berkunjung</h4>
                                          		</div>
                                          		<br></br>
                                          		<br></br>"""

           st.markdown(html_temp4, unsafe_allow_html=True)

       else:
           pass

if __name__ == "__main__":
    main()
