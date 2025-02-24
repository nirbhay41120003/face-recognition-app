import cv2
import streamlit as st
import face_recognition as fr
import pickle
import os
import numpy as np

# Initialize trained model flag
trainedModel = False

# Load trained embeddings if file exists
trainmap = {"embeddings": [], "label": []}
if os.path.isfile("trainmodel.pkl"):
    trainmap = pickle.load(open("trainmodel.pkl", "rb"))
    trainedModel = True

# Real-time Face Recognition function
def realTimefeed(_bool, isBoundingBox, isCap, name):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while _bool:
        _, frame = cap.read()
        if isBoundingBox:
            facelocation = fr.face_locations(frame, model='hog')
            faceEmbeddings = fr.face_encodings(frame, facelocation)

            for (top, right, bottom, left), faceEmbedding in zip(facelocation, faceEmbeddings):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                if trainedModel and trainmap["embeddings"]:
                    faceDis = fr.face_distance(trainmap["embeddings"], faceEmbedding)
                    minFaceDisIdx = np.argmin(faceDis)
                    minFaceDis = faceDis[minFaceDisIdx]

                    if minFaceDis < 0.5:
                        label = trainmap["label"][minFaceDisIdx]
                    else:
                        label = "Unknown"
                else:
                    label = "Unknown"

                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

        if isCap:
            cv2.imwrite(f"{name}.jpg", frame)
            cap.release()
            break

    cap.release()

# Training function
def train(name):
    if os.path.exists(f"{name}.jpg"):
        face_img = fr.load_image_file(f"{name}.jpg")
        face_bbs = fr.face_locations(face_img)
        face_emds = fr.face_encodings(face_img, face_bbs)

        if face_emds:  # Ensure there is at least one face detected
            for embd in face_emds:
                trainmap["embeddings"].append(embd)
                trainmap["label"].append(name)

            pickle.dump(trainmap, open("trainmodel.pkl", "wb"))
            st.write("Training complete:", trainmap)
        else:
            st.write("No face detected. Please try capturing a clearer image.")
    else:
        st.write("Error: Image not found. Capture an image first.")

# Streamlit UI
st.title("Real-Time Face Recognition")

tabs = ['Real-Time Feed', 'Training']
choice = st.sidebar.selectbox("Select Mode", tabs)

isCap = False

if choice == 'Real-Time Feed':
    if st.button('Start'):
        realTimefeed(True, True, False, "")
    if st.button('Exit'):
        realTimefeed(False, False, False, "")

else:  # Training Mode
    name = st.text_input("Enter your name:")
    
    if name:
        if st.button("Capture Image"):
            isCap = True
            realTimefeed(True, False, isCap, name)

        if st.button("Train Model"):
            train(name)