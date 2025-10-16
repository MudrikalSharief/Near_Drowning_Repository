import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO
import time
import shutil
from collections import defaultdict
import base64

def play_alert_sound(audio_path):
    with open(audio_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    md = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# --- Audio Utility Function (Base64 Embedded - The actual alert) ---
def create_audio_html(audio_file_path):
    """
    Reads the MP3 file, encodes it, and embeds an invisible audio tag with autoplay.
    It relies on the audio context being 'unlocked' by an earlier user action (the primer).
    """
    try:
        with open(audio_file_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        data_uri = f"data:audio/mp3;base64,{audio_b64}"
    except FileNotFoundError:
        st.error(f"❌ Audio file not found at: {audio_file_path}")
        return ""
    except Exception as e:
        st.error(f"❌ Error encoding audio file: {e}")
        return ""

    # The 'autoplay' attribute should work now that the context is unlocked.
    html_code = f"""
    <audio id="alert_audio" src="{data_uri}" preload="auto" autoplay></audio>
    <style>
        #alert_audio {{ display: none; }}
    </style>
    """
    return html_code
# --- End Audio Utility Function (The actual alert) ---

# -------------------------------------------------------------

# --- Audio Primer Utility (To unlock browser autoplay) ---
def create_audio_primer_html(audio_file_path):
    """
    Generates an audio element and attempts a play() operation once.
    This is called immediately after a user click to satisfy browser autoplay policy.
    """
    try:
        with open(audio_file_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        data_uri = f"data:audio/mp3;base64,{audio_b64}"
    except Exception:
        return ""

    # Use JavaScript to attempt playback immediately on rendering
    html_code = f"""
    <audio id="primer_audio" src="{data_uri}" preload="auto" controls></audio>
    <script>
        var audio = document.getElementById('primer_audio');
        if (audio) {{
            // Attempt an immediate play on load (within the user-interaction context)
            audio.volume = 0.01; // Keep it quiet
            audio.play().catch(function(error) {{
                // This is expected if playback is blocked, but the context may still be unlocked
                console.warn("Primer play failed, but the browser audio context might be unlocked:", error);
            }});
        }}
    </script>
    <style>
        #primer_audio {{ display: none; }}
    </style>
    """
    return html_code
# --- End Audio Primer Utility ---

# -------------------------------------------------------------
# --- Configuration & Utility Functions ---

DEFAULT_MODEL_NAME = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/Model/yolov8/best.pt" 
AUDIO_FILE_PATH = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/Model/yolov8/alert.mp3"
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dictionary to store object start times: {track_id: start_time_seconds}
object_start_times = {} 
# Dictionary to store the class ID of the object when it was first seen: {track_id: class_id}
object_class_map = {}


@st.cache_resource
def load_model(model_path, device):
    """Load YOLOv8 model once and cache it."""
    try:
        model = YOLO(model_path)
        model.to(device)
        return model
    except Exception as e:
        if model_path == DEFAULT_MODEL_NAME and not os.path.exists(model_path):
            st.error(f"❌ Error: Default model `{model_path}` not found. Please place it in the application directory or upload a custom model.")
        else:
            st.error(f"❌ Error loading model from `{model_path}`: {e}")
        st.stop()
        
# --- Streamlit Page Setup ---

st.set_page_config(
    page_title="YOLOv8 + ByteTrack Object Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👁️ Real-Time Object Tracking with YOLOv8 & ByteTrack")
st.caption(
    "Alert triggers when a tracked object maintains its ID and class for a specified duration."
)
st.markdown("---")

# --- Session State Initialization ---
if "selected_video_path" not in st.session_state:
    st.session_state.selected_video_path = None
if "video_name" not in st.session_state:
    st.session_state.video_name = "None Selected"
if "selected_classes" not in st.session_state:
    st.session_state.selected_classes = []


# --- Sidebar: Model & Technical Settings ---
model = None
model_path = None
use_custom_model = False
tmpdir = None 
alert_min_duration = 3.0 # Default alert duration

with st.sidebar:
    st.header("⚙️ Model & Technical Settings")
    
    # 1. Model Configuration (Upload/Default)
    use_custom_model = st.checkbox("Upload custom YOLO model (.pt)", value=False)
    model_display = DEFAULT_MODEL_NAME
    
    if use_custom_model:
        model_file = st.file_uploader("Upload custom model file (.pt)", type=["pt"])
        if model_file:
            tmpdir = tempfile.mkdtemp()
            model_path = os.path.join(tmpdir, model_file.name)
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())
            model_display = model_file.name
        else:
            st.warning("Please upload a custom model file.")
    else:
        model_path = DEFAULT_MODEL_NAME
        st.info(f"Using default model: `{model_path}`")
    
    # 2. Device Selection
    device = st.selectbox("Processing Device", ["cuda", "cpu"], index=0, help="Choose 'cuda' for GPU acceleration if available.")
    
    st.markdown("---")
    
    # --- 🌟 ALERT SETTINGS (Duration Based) 🌟 ---
    st.header("🔔 Duration Alert Settings")
    
    # Input for the required duration
    alert_min_duration = st.number_input(
        "Alert if same object ID/Class is tracked for (Seconds)",
        min_value=0.1,
        value=3.0,
        step=0.5,
        format="%.1f",
        help="The minimum continuous duration for a specific object to trigger an alarm. ByteTrack assigns the unique ID."
    )
    
    # Placeholder for the target class selection (filled after model load)
    alert_class_selector = st.empty()
    
    st.markdown("---")
    st.markdown("Powered by: `ultralytics` (YOLOv8) & `ByteTrack`")
    
# --- Model Loading Status & Class Selection ---
target_alert_class_name = None

if model_path and (os.path.exists(model_path) or (use_custom_model and model_path)):
    with st.spinner(f"⏳ Loading model: **{model_display}** on **{device}**..."):
        model = load_model(model_path, device)
        
        if model and hasattr(model.names, '__len__') and len(model.names) > 0:
            all_classes = list(model.names.values())
            
            st.sidebar.header("🎯 Object Filter")
            
            st.session_state.selected_classes = st.sidebar.multiselect(
                "Select classes to display:",
                options=all_classes,
                default=all_classes, 
                help="Only detections belonging to the selected classes will be shown."
            )
            
            # Populate the Alert Class Selector
            target_alert_class_name = alert_class_selector.selectbox(
                "Target Class for Duration Alert:",
                options=all_classes,
                index=all_classes.index("person") if "person" in all_classes else 0,
                help="The specific class whose duration will be monitored for the alert."
            )
            
    st.success(f"✅ Model **{model_display}** loaded successfully!")
else:
    if not use_custom_model:
        st.error("🛑 Please place the default model file or check the 'Upload custom model' box.")
    elif use_custom_model and not model_path:
        st.error("🛑 Waiting for custom model upload...")
    st.stop()


# --- Main Area: Video Upload & Selection ---

col_upload, col_current = st.columns([1, 2])

with col_upload:
    st.subheader("1. Video Source")
    uploaded_video = st.file_uploader(
        "Upload a video file (.mp4, .mov, etc.)",
        type=["mp4", "mov", "avi", "mkv"],
        key="main_upload" 
    )
    
    # Logic to save the uploaded file and update session state
    if uploaded_video:
        save_path = os.path.join(UPLOAD_DIR, uploaded_video.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
            
        st.session_state.selected_video_path = save_path
        st.session_state.video_name = uploaded_video.name
        st.success(f"✅ Video **'{uploaded_video.name}'** uploaded and selected!")
    elif uploaded_video is None and st.session_state.selected_video_path:
        st.session_state.selected_video_path = None
        st.session_state.video_name = "None Selected"

with col_current:
    st.subheader("2. Selected Video Preview")
    
    if st.session_state.selected_video_path:
        st.info(f"Video selected: **{st.session_state.video_name}**")
        st.video(st.session_state.selected_video_path, format="video/mp4", start_time=0)
    else:
        st.warning("⚠️ No video selected. Please upload one on the left.")

# --- Main Area: Run Detection (MODIFIED for Audio Primer and Alert) --- 
st.markdown("---") 

st.header("3. Run Detection & Tracking")

if not st.session_state.selected_video_path:
    st.warning("Please upload or select a video above before running detection.")
elif not st.session_state.selected_classes:
    st.warning("⚠️ Please select at least one class in the **Object Filter** sidebar to run detection.")
else:
    # Detection Settings
    conf_threshold = st.slider(
        "Confidence Threshold (min probability for detection)", 
        0.1, 1.0, 0.4, 0.05,
        key="conf_slider",
        help="Lowering this value may increase detections but also false positives."
    )
    
    selected_class_ids = [k for k, v in model.names.items() if v in st.session_state.selected_classes]
    
    # Get the ID of the target alert class
    target_alert_class_id = next(
        (k for k, v in model.names.items() if v == target_alert_class_name), 
        None
    )

    if target_alert_class_id is None:
        st.error(f"Internal error: Could not find class ID for '{target_alert_class_name}'.")
        st.stop()


    # Alert placeholder setup
    alert_placeholder = st.empty()
    alert_triggered_for_id = None # Store the ID that triggered the current alert

    # Run button
    if st.button("▶️ START DETECTION & TRACKING", type="primary", use_container_width=True):
        
        # --- 1. 🔊 AUDIO PRIMER EXECUTION (THE CRITICAL STEP) 🔊
        audio_file_path = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/Model/yolov8/alert.mp3" 
        
        # Use a temporary placeholder to insert and immediately clear the primer audio
        primer_placeholder = st.empty()
        primer_placeholder.markdown(
            create_audio_primer_html(audio_file_path), 
            unsafe_allow_html=True
        )
        time.sleep(0.1) # A short pause to ensure the HTML is rendered before clearing
        primer_placeholder.empty() # Remove the primer element after a fraction of a second
        st.info("Audio context primed. The alert sound should now play when triggered.")
        # --------------------------------------------------------

        # Reset tracking history for a new run
        object_start_times.clear()
        object_class_map.clear()
        
        # 🌟 FIX: Initialize the variable used for tracking the maximum count
        total_objects_tracked = 0
        
        # --- Real-Time Output Section ---
        st.subheader("⚡️ Real-Time Tracking Output")
        
        # 1. Setup Metrics Container
        col_metrics_1, col_metrics_2, col_metrics_3 = st.columns(3)
        metric_fps = col_metrics_1.empty()
        metric_frame = col_metrics_2.empty()
        metric_objects = col_metrics_3.empty()
        
        # 2. Setup Video Container
        st.markdown("### Processed Video Stream")
        stframe = st.empty() 
         # --- Detection Loop (Frame-based Duration Alert) ---
        cap = cv2.VideoCapture(st.session_state.selected_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None:
            fps = 30  # fallback if metadata is missing
        frame_index = 0

        # Trackers
        first_seen_frame = {}     # track_id -> first frame number
        last_seen_frame = {}      # track_id -> last frame number
        alert_triggered_ids = set()

        with st.spinner("Processing video frames..."):
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_index += 1

                # Run YOLOv8 + ByteTrack
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=conf_threshold,
                    classes=selected_class_ids,
                    verbose=False
                )

                annotated_frame = results[0].plot()
                boxes = results[0].boxes
                current_detected_ids = set()

                # --- Process detected objects ---
                if boxes.id is not None:
                    for box in boxes.data:
                        track_id = int(box[4].item())
                        class_id = int(box[5].item())
                        current_detected_ids.add(track_id)

                        # Only watch the target class
                        if class_id != target_alert_class_id:
                            continue

                        # Record when object first appeared
                        if track_id not in first_seen_frame:
                            first_seen_frame[track_id] = frame_index

                        last_seen_frame[track_id] = frame_index

                        # Compute how long (in seconds) it’s been seen
                        elapsed_frames = frame_index - first_seen_frame[track_id]
                        elapsed_seconds = elapsed_frames / fps

                        # Trigger alert once when duration reached
                        if elapsed_seconds >= alert_min_duration and track_id not in alert_triggered_ids:
                            alert_triggered_ids.add(track_id)
                            alert_placeholder.error(
                                f"## 🚨 ALERT! Object ID **{track_id}** "
                                f"({target_alert_class_name}) detected for {elapsed_seconds:.1f}s 🚨",
                                icon="⚠️"
                            )
                            play_alert_sound(AUDIO_FILE_PATH)
                            st.toast(f"⚠️ Alert triggered for ID {track_id} ({target_alert_class_name})")

                # --- Reset timers for disappeared objects ---
                inactive_ids = [
                    tid for tid in list(first_seen_frame.keys())
                    if tid not in current_detected_ids and
                    (frame_index - last_seen_frame.get(tid, frame_index)) > (fps * 1.0)  # 1 second inactivity
                ]
                for tid in inactive_ids:
                    first_seen_frame.pop(tid, None)
                    last_seen_frame.pop(tid, None)
                    if tid in alert_triggered_ids:
                        alert_triggered_ids.remove(tid)
                    alert_placeholder.empty()

                # --- Display Stats ---
                metric_frame.metric("Frame Count", frame_index)
                metric_objects.metric("Tracked Objects", len(current_detected_ids))

                stframe.image(
                    annotated_frame,
                    channels="BGR",
                    use_container_width=True,
                    caption=f"Processing: {st.session_state.video_name}"
                )

        cap.release()
        stframe.empty()
        alert_placeholder.empty()
        st.balloons()
        st.success(f"✅ Detection complete — Processed {frame_index} frames.")


        # Clean up temporary model directory
        if use_custom_model and tmpdir:
            try:
                shutil.rmtree(tmpdir)
            except OSError as e:
                st.warning(f"Could not clean up temp directory: {e}")