import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import time
import numpy as np

# Path to trained model
MODEL_PATH = "runs/detect/transbot_office_train/weights/best.pt"

# Load YOLO model
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="Transbot Object Detection", layout="wide")
st.title("🤖 Transbot SE Object Detection Interface")

# Sidebar controls
st.sidebar.header("⚙️ Control Panel")
mode = st.sidebar.radio("Select Mode:", ("Laptop Camera", "Upload Image", "Upload Video"))
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
st.sidebar.write(f"Current Confidence: **{confidence:.2f}**")
# Laptop Camera Mode
# ==========================
if mode == "Laptop Camera":
    st.write("Camera mode: Press Start Detection")
    start_btn = st.button("Start Detection")
    stop_btn = st.button("Stop Detection")
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while start_btn and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to read from camera")
            break

        results = model.predict(frame, conf=confidence)
        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR")

        # Display detections
        boxes = results[0].boxes
        if len(boxes) > 0:
            st.subheader("🧾 Detected Objects")
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf_score = float(boxes.conf[i])
                st.write(f"- {model.names[cls_id]} ({conf_score:.2f})")

        if stop_btn:
            break

    cap.release()
    st.write("Camera stopped.")

# Upload Image Mode

elif mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model.predict(img, conf=confidence)
        annotated_frame = results[0].plot()
        st.image(annotated_frame, channels="BGR", use_container_width=True)
        st.success("Detection Complete!")

        # Show detected objects
        boxes = results[0].boxes
        if len(boxes) > 0:
            st.subheader("🧾 Detected Objects")
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf_score = float(boxes.conf[i])
                st.write(f"- {model.names[cls_id]} ({conf_score:.2f})")

# Upload Video Mode
elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.video(tfile.name)
        run_detection = st.button("Run Detection")
        if run_detection:
            results = model.predict(source=tfile.name, conf=confidence)
            st.success("Video Detection Completed!")

            boxes = results[0].boxes
            if len(boxes) > 0:
                st.subheader("🧾 Detected Objects in Video")
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf_score = float(boxes.conf[i])
                    st.write(f"- {model.names[cls_id]} ({conf_score:.2f})")

# Capture Options (Extras)
st.sidebar.header("📸 Capture Options")

if st.sidebar.button("Capture Image from Camera"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
        st.image(frame, channels="BGR")
        st.success("Image Captured!")
    cap.release()

if st.sidebar.button("Capture Video from Camera"):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("captured_video.mp4", fourcc, 20.0, (640, 480))
    stframe = st.empty()
    st.write("Recording for 5 seconds...")
    start_time = time.time()
    while int(time.time() - start_time) < 5:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            stframe.image(frame, channels="BGR")
    cap.release()
    out.release()
    st.success("Video Captured!")
