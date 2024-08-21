import streamlit as st
import numpy as np
from deepface import DeepFace
import json
import os
from datetime import datetime
import cv2
from PIL import Image

# Function to analyze facial attributes using DeepFace
def analyze_frame(frame):
    # Convert image array to file path for DeepFace
    temp_filename = 'temp_frame.jpg'
    cv2.imwrite(temp_filename, frame)
    result = DeepFace.analyze(img_path=temp_filename, actions=['age', 'gender', 'race', 'emotion'],
                              enforce_detection=False,
                              detector_backend="opencv",
                              align=True,
                              silent=False)
    os.remove(temp_filename)
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
    stframe = st.empty()  # Placeholder for the webcam feed
    stop_button = st.button("Stop", key="stop_button")  # Add a unique key for the stop button

    while True:
        frame = st.camera_input("Webcam feed")

        if frame is not None:
            # Convert frame to an image array
            frame = np.array(frame)
            
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
            
            # Check if the stop button is pressed
            if stop_button:
                break
        else:
            st.warning("No camera input detected.")

def main():
    st.title("Real-Time Face Emotion Detection Application")
    activities = ["Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    if choice == "Webcam Face Detection":
        st.markdown("<div style='background-color:#6D7B8D;padding:10px'><h4 style='color:white;text-align:center;'>Smart Attendance</h4></div>", unsafe_allow_html=True)
        user_type = st.selectbox("Select User Type", ["Employee", "Customer"])
        action = None
        if user_type == "Employee":
            action = st.selectbox("Select Action", ["In", "Out"])
        
        if st.button("Start Detection", key="start_button"):
            facesentiment(user_type, action)
    
    elif choice == "About":
        st.subheader("About this app")
        st.markdown("<div style='background-color:#98AFC7;padding:10px'><h4 style='color:white;text-align:center;'>Aplikasi ini dapat digunakan untuk melakukan pengenalan wajah pada gambar atau video. Terimakasih sudah berkunjung</h4></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
