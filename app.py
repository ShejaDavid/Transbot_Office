import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import time
import numpy as np
import shutil

# --- Configuration & Initialization ---
MODEL_PATH = "runs/detect/transbot_office_train/weights/best.pt"
OUTPUT_VIDEO_PATH = "processed_video_output.mp4" 

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

st.set_page_config(page_title="Transbot Object Detection", layout="wide")
st.title("🤖 Transbot SE Object Detection Interface")

# Initialize session state for camera stability
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

def start_camera_cb():
    st.session_state.camera_running = True

def stop_camera_cb():
    st.session_state.camera_running = False

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Control Panel")
mode = st.sidebar.radio("Select Mode:", ("Laptop Camera", "Upload Image", "Upload Video"))
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
st.sidebar.write(f"Current Confidence: **{confidence:.2f}**")

# --- Laptop Camera Mode (Live Detection) ---
if mode == "Laptop Camera":
    st.subheader("Live Camera Detection")
    
    start_btn = st.button("Start Detection", on_click=start_camera_cb, disabled=st.session_state.camera_running)
    stop_btn = st.button("Stop Detection", on_click=stop_camera_cb, disabled=not st.session_state.camera_running)
    
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    
    if st.session_state.camera_running:
        st.info("Live detection active.")
        
        while st.session_state.camera_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame.")
                stop_camera_cb()
                break
                
            results = model.predict(frame, conf=confidence, verbose=False)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            boxes = results[0].boxes
            if len(boxes) > 0:
                with st.expander("🧾 Detected Objects (Current Frame)", expanded=True):
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i])
                        conf_score = float(boxes.conf[i])
                        st.write(f"- **{model.names[cls_id]}** ({conf_score:.2f})")
                        
        cap.release()
        if not st.session_state.camera_running:
             stframe.empty()
             st.success("Camera stopped.")
    
    else:
        cap.release()
        st.warning("Detection is paused.")

# --- Upload Image Mode ---
elif mode == "Upload Image":
    st.subheader("Image Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Decode uploaded file into an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(img, channels="BGR", caption="Original Image", use_container_width=True)
        
        # YOLO inference
        results = model.predict(img, conf=confidence, verbose=False)
        annotated_frame = results[0].plot()
        
        st.image(annotated_frame, channels="BGR", caption="Detected Objects", use_container_width=True)
        st.success("Detection Complete!")

        # Show detected objects list
        boxes = results[0].boxes
        if len(boxes) > 0:
            st.subheader("🧾 Detected Objects")
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf_score = float(boxes.conf[i])
                st.write(f"- **{model.names[cls_id]}** ({conf_score:.2f})")
        else:
            st.info("No objects detected.")

# --- Upload Video Mode ---
elif mode == "Upload Video":
    st.subheader("Video Detection")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.type.split('/')[-1]}")
        tfile.write(uploaded_video.read())
        tfile.close()

        st.video(tfile.name, caption="Uploaded Video")
        
        run_detection = st.button("Run Detection and Display Video")
        
        if run_detection:
            st.info("Processing video... This may take a moment.")
            
            # Clean up previous output file/folder
            if os.path.exists(OUTPUT_VIDEO_PATH):
                os.remove(OUTPUT_VIDEO_PATH)
            
            # YOLO predicts and saves the annotated video automatically
            results = model.predict(
                source=tfile.name, 
                conf=confidence, 
                save=True, 
                project='st_output', 
                name='video_run', 
                exist_ok=True,
                verbose=False
            )
            
            try:
                # Find the path where YOLO saved the video
                save_dir = results[0].save_dir
                saved_video_path = os.path.join(save_dir, os.path.basename(tfile.name))
                
                # Copy the saved video to the root for display
                shutil.copy(saved_video_path, OUTPUT_VIDEO_PATH)

                st.success("Video Detection Completed! Displaying result below.")
                st.video(OUTPUT_VIDEO_PATH, caption="Annotated Video")
                
                # Clean up temporary files
                os.unlink(tfile.name)
                shutil.rmtree(save_dir)
                
            except Exception as e:
                st.error(f"Could not find or display processed video. Error: {e}")
                os.unlink(tfile.name)


# --- Capture Options (Extras) ---
st.sidebar.header("📸 Capture Options")

# FIX: Run detection on the captured image
if st.sidebar.button("Capture Image from Camera"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Run YOLO detection on the captured frame
        results = model.predict(frame, conf=confidence, verbose=False)
        annotated_frame = results[0].plot()
        
        # Display the annotated frame in the sidebar
        st.sidebar.image(annotated_frame, channels="BGR", caption="Detected Image")
        st.sidebar.success("Image Captured and Detected!")
        
        # Display detection list in the main area
        st.subheader("🧾 Captured Image Detections")
        boxes = results[0].boxes
        if len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf_score = float(boxes.conf[i])
                st.write(f"- **{model.names[cls_id]}** ({conf_score:.2f})")
        else:
            st.info("No objects detected in the captured image.")
    else:
        st.sidebar.error("Failed to access camera.")


if st.sidebar.button("Capture Video from Camera"):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    capture_output_path = "captured_video.mp4"
    out = cv2.VideoWriter(capture_output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    stframe_rec = st.empty()
    st.info("Recording for 5 seconds...")
    
    start_time = time.time()
    while int(time.time() - start_time) < 5:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            stframe_rec.image(frame, channels="BGR", width=200)
        
    cap.release()
    out.release()
    stframe_rec.empty()
    
    st.sidebar.video(capture_output_path, caption="5s Video Captured")
    st.success(f"Video Captured!")