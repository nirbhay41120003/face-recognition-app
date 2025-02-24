import cv2
import streamlit as st
import face_recognition as fr
import pickle
import os
import numpy as np

# Real-time Face Recognition Function
def realTimefeed(_bool, isBoundingBox, isCap, name):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)  # Change to 1 if using an external webcam

    if not cap.isOpened():
        st.write("Error: Could not access webcam.")
        return

    while _bool:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        if isBoundingBox:
            facelocation = fr.face_locations(frame, model='hog')
            faceEmbeddings = fr.face_encodings(frame, facelocation)
            for (top, right, bottom, left), faceEmbedding in zip(facelocation, faceEmbeddings):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                if trainedModel:
                    match = fr.compare_faces(trainmap["embeddings"], faceEmbedding)
                    faceDis = fr.face_distance(trainmap["embeddings"], faceEmbedding)
                    minFaceDis = min(faceDis)
                    minFaceDisIdx = np.argmin(faceDis)

                    if match[minFaceDisIdx] and minFaceDis < 0.5:
                        label = trainmap["label"][minFaceDisIdx]
                        cv2.putText(frame, f"{label}", (left, top - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown", (left, top - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (left, top - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

        if isCap:
            for _ in range(5):  # Capture a few frames first
                ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"{name}.jpg", frame)
                st.success(f"Image captured as {name}.jpg")
            else:
                st.error("Failed to capture image.")
            isCap = False

    cap.release()

# Training Function
def train(name):
    if not os.path.exists(f"{name}.jpg"):
        st.write("Error: Capture an image first.")
        return

    faceimg = fr.load_image_file(f"{name}.jpg")
    facebb = fr.face_locations(faceimg)
    faceemd = fr.face_encodings(faceimg, facebb)

    if len(faceemd) == 0:
        st.write("No face detected. Try capturing a clearer image.")
        return

    for embd in faceemd:
        trainmap["embeddings"].append(embd)
        trainmap["label"].append(name)

    pickle.dump(trainmap, open("trainmodel.pkl", "wb"))
    st.write("Training complete! Face added to the model.")

# Load or Initialize Training Data
trainmap = {"embeddings": [], "label": []}
st.title("Real-Time Face Recognition")

tabs = ['Real-time Feed', 'Training']
choice = st.sidebar.selectbox("Mode", tabs)
isCap = False
trainedModel = False
if os.path.isfile("trainmodel.pkl"):
    trainmap = pickle.load(open("trainmodel.pkl", "rb"))
    trainedModel = True

if choice == tabs[0]:  # Real-time feed mode
    if st.button('Start'):
        realTimefeed(True, True, isCap, "")
    if st.button('Exit'):
        realTimefeed(False, True, isCap, "")

else:  # Training mode
    name = st.text_input("Enter your name:")
    if name!="":
        if st.button("Capture"):
            isCap = True  # Capture the image
        if st.button("Train"):
            train(name)  # Train with the captured image
            
        realTimefeed(True,False,isCap,name)
