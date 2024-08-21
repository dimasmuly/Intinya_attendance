from kivy.uix.gridlayout import accumulate
import streamlit as st
import numpy as np
from deepface import DeepFace
import json
import os
from datetime import datetime

# Use opencv-python-headless to avoid libGL.so.1 issues
import cv2

# Function to analyze facial attributes using DeepFace
def analyze_frame(frame):
    result = DeepFace.analyze(img_path=frame, actions=['age', 'gender', 'race', 'emotion'],
                              enforce_detection=False,
                              detector_backend="opencv",
                              align=True,
                              silent=False)
    return result

def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.9  # Adjust the transparency of the overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  # White rectangle
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_position = 15 # Where the first text is put into the overlay
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

def facesentiment(user_type, action):
    # st.title("Real-Time Facial Analysis with Streamlit")
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)  # Use default local camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error("Failed to open webcam. Please check your camera.")
        return

    stframe = st.image([])  # Placeholder for the webcam feed

    stop_button = st.button("Stop", key="stop_button")  # Add a unique key for the stop button

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is captured successfully
        if not ret:
            st.error("Failed to capture frame from camera")
            st.info("Please turn off the other app that is using the camera and restart app")
            st.stop()

        # Analyze the frame using DeepFace
        result = analyze_frame(frame)

        # Extract the face coordinates
        face_coordinates = result[0]["region"]
        x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{result[0]['dominant_emotion']}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Convert the BGR frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Overlay white rectangle with text on the frame
        texts = [
            f"Age: {result[0]['age']}",
            f"Face Confidence: {round(result[0]['face_confidence'],3)}",
            f"Gender: {result[0]['dominant_gender']} {round(result[0]['gender'][result[0]['dominant_gender']], 3)}",
            f"Race: {result[0]['dominant_race']}",
            f"Dominant Emotion: {result[0]['dominant_emotion']} {round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}",
        ]

        frame_with_overlay = overlay_text_on_frame(frame_rgb, texts)

        # Display the frame in Streamlit
        stframe.image(frame_with_overlay, channels="RGB")

        # Save result to JSON
        data = {
            "user_type": user_type,
            "action": action,
            "age": result[0]['age'],
            "gender": result[0]['dominant_gender'],
            "race": result[0]['dominant_race'],
            "emotion": result[0]['dominant_emotion'],
            "face_confidence": result[0]['face_confidence'],
            "timestamp": datetime.now().isoformat()
        }
        save_result_to_json(data)

        # Add a delay to avoid overloading the system
        if stop_button:
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Face Analysis Application #
    # st.title("Real Time Face Emotion Detection Application")
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
            facesentiment(user_type, action)

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
