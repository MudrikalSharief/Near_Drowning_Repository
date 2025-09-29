"""
Streamlit app: Upload a video -> run YOLOv8 (detection or tracking) -> view & download annotated video.

How to run:
  1. Install dependencies (preferably in a virtualenv):
     pip install ultralytics streamlit lap

  2. Save this file as `streamlit_yolov8_video_app.py` and from the terminal run:
     streamlit run streamlit_yolov8_video_app.py

Notes:
  - If you want to use GPU, make sure you have a CUDA-enabled PyTorch installed. Ultralytics will use the GPU if available.
  - If your default Streamlit upload limit is too small for your test videos, create a file `.streamlit/config.toml` with:
      [server]
      maxUploadSize = 1024  # size in MB

This script writes temporary files to a temp folder and cleans them up after the run (unless you check the "Keep output folder after run" option).
"""

# pip install ultralytics streamlit lap
# how tot run?
# streamlit run c:/Users/Lenovo Slim i3/Documents/GitHub/Near_Drowning_Repository/Model/yolov8/streamlit_yolov_8_video_app.py
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import shutil
import glob
import cv2

# --- UI: page title and short help ---
st.set_page_config(page_title="YOLOv8 Video Analyzer", layout="wide")
st.title("YOLOv8 Video Analyzer")
st.write(
    "Upload a video, choose a model (or upload your own .pt), pick a mode, and run.\n"
    "- **Real-time Preview**: See detections as the video is processed.\n"
    "- **Full Video Output**: Get an annotated video to play and download."
)

# --- Sidebar: settings ---
st.sidebar.header("Settings")

# Default model path
default_model_path = "C:/Users/Lenovo Slim i3/Documents/GitHub/Near_Drowning_Repository/Model/yolov8/PD_default.pt"

use_custom_model = st.sidebar.checkbox("Upload custom .pt model (optional)")
model_file = None
if use_custom_model:
    model_file = st.sidebar.file_uploader("Upload model (.pt)", type=["pt"])
else:
    st.sidebar.markdown(f"**Default model**: `{default_model_path}`")

mode = st.sidebar.radio("Mode", ["Real-time Preview", "Full Video Output"])

tracker_option = st.sidebar.selectbox(
    "Tracker (only for Full Video Output)",
    ["None (detection only)", "bytetrack.yaml", "strongsort.yaml"],
    index=1,
)

save_everything = st.sidebar.checkbox("Keep output folder after run", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("Dependencies: `ultralytics`, `streamlit`, `lap`, `opencv-python`")

# --- Main UI: video upload ---
video_file = st.file_uploader("Upload a video (mp4, mov, avi, mkv)", type=["mp4", "mov", "avi", "mkv"])

if video_file is None:
    st.info("Upload a video to get started — try a short clip first to test the flow.")
else:
    # show uploaded video preview
    st.video(video_file)

    # Run button
    run = st.button("Run model on this video")

    if run:
        # 1) Create a temporary working directory
        tmpdir = tempfile.mkdtemp(prefix="yolov8_streamlit_")
        try:
            st.info(f"Created temporary working directory: `{tmpdir}`")

            # 2) Save uploaded video
            ext = os.path.splitext(video_file.name)[1]
            input_path = os.path.join(tmpdir, f"input_video{ext}")
            with open(input_path, "wb") as f:
                f.write(video_file.getbuffer())
            st.success(f"Saved uploaded video to `{input_path}`")

            # 3) Prepare model
            if use_custom_model and model_file is not None:
                model_path = os.path.join(tmpdir, "custom_model.pt")
                with open(model_path, "wb") as f:
                    f.write(model_file.getbuffer())
                st.success("Saved uploaded model to temporary file")
            else:
                model_path = default_model_path
                st.write(f"Using model: `{model_path}`")

            with st.spinner("Loading YOLO model..."):
                model = YOLO(model_path)
            st.success("Model loaded ✅")

            # -----------------------------
            # MODE 1: Real-time Preview
            # -----------------------------
            if mode == "Real-time Preview":
                st.info("Running in Real-time Preview mode")
                cap = cv2.VideoCapture(input_path)
                stframe = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
                    annotated_frame = results[0].plot()
                    stframe.image(annotated_frame, channels="BGR", use_column_width=True)

                cap.release()
                st.success("Preview finished ✅")

            # -----------------------------
            # MODE 2: Full Video Output
            # -----------------------------
            else:
                st.info("Running in Full Video Output mode")
                out_dir = os.path.join(tmpdir, "output")
                os.makedirs(out_dir, exist_ok=True)
                result_name = "result"

                try:
                    if tracker_option == "None (detection only)":
                        with st.spinner("Running detection and saving annotated video..."):
                            results = model.predict(source=input_path, save=True, project=out_dir, name=result_name)
                    else:
                        with st.spinner(f"Running tracker `{tracker_option}` and saving annotated video..."):
                            results = model.track(source=input_path, tracker=tracker_option, save=True, project=out_dir, name=result_name)

                    st.success("Processing finished ✅")

                    # Locate annotated video
                    search_dir = os.path.join(out_dir, result_name)
                    candidate_files = glob.glob(os.path.join(search_dir, "**", "*.*"), recursive=True)
                    annotated_video = None
                    for f in candidate_files:
                        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                            annotated_video = f
                            break

                    if annotated_video:
                        st.video(annotated_video)
                        with open(annotated_video, "rb") as vf:
                            video_bytes = vf.read()
                            st.download_button(
                                label="Download annotated video",
                                data=video_bytes,
                                file_name=os.path.basename(annotated_video),
                                mime="video/mp4",
                            )

                        if save_everything:
                            final_dest = os.path.join(os.getcwd(), f"yolov8_output_{os.path.basename(tmpdir)}")
                            shutil.copytree(search_dir, final_dest)
                            st.info(f"Saved output directory to `{final_dest}`")
                    else:
                        st.error("Annotated video not found.")

                except Exception as run_err:
                    st.error(f"Error while running the model: {run_err}")

        finally:
            if not save_everything:
                try:
                    shutil.rmtree(tmpdir)
                    st.write("Temporary files cleaned up")
                except Exception:
                    pass
            else:
                st.write(f"Temporary working directory kept at `{tmpdir}`")

# End of app


# ----------------------------
# Short developer notes (inside the file):
# - If you want to run batch processing for many videos, replace the single file uploader
#   with `st.file_uploader(..., accept_multiple_files=True)` and loop through the list.
# - If annotated video is not saved, try inspecting the `out_dir` folder on disk; ultralytics may save
#   in slightly different nested paths depending on version. The glob search above is conservative.
# - For production or multi-user hosting, consider:
#     * Running the model behind an API (Flask/FastAPI) and queueing jobs
#     * Using a GPU-enabled machine to speed up inference
#     * Storing outputs in persistent storage instead of temp dirs
# ----------------------------
