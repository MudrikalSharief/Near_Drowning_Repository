import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO
import time
import shutil
import base64
import numpy as np 

# ==============================
# 🔊 Utility Functions
# ==============================

def play_alert_sound(audio_path, alert_id):
    """Play an audio alert using base64 and assign a unique ID so it can be stopped."""
    try:
        with open(audio_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()

        # Give a unique audio tag ID per alert
        audio_tag_id = f"alert_audio_{alert_id}"

        md = f"""
            <audio id="{audio_tag_id}" autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            <script>
                if (!window.activeAudios) window.activeAudios = {{}};
                window.activeAudios['{audio_tag_id}'] = document.getElementById('{audio_tag_id}');
            </script>
        """
        st.markdown(md, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Alert sound file not found at: {audio_path}")


def stop_alert_sound(alert_id=None):
    """Stop specific or all active alert sounds."""
    if alert_id:
        stop_script = f"""
            <script>
                if (window.activeAudios && window.activeAudios['alert_audio_{alert_id}']) {{
                    let a = window.activeAudios['alert_audio_{alert_id}'];
                    a.pause();
                    a.currentTime = 0;
                    delete window.activeAudios['alert_audio_{alert_id}'];
                }}
            </script>
        """
    else:
        stop_script = """
            <script>
                if (window.activeAudios) {
                    for (const id in window.activeAudios) {
                        let a = window.activeAudios[id];
                        a.pause();
                        a.currentTime = 0;
                    }
                    window.activeAudios = {};
                }
            </script>
        """
    st.markdown(stop_script, unsafe_allow_html=True)


def create_audio_primer_html(audio_file_path):
    """Prime browser audio context for autoplay permission."""
    try:
        with open(audio_file_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        data_uri = f"data:audio/mp3;base64,{audio_b64}"
    except Exception:
        return ""

    html_code = f"""
    <audio id="primer_audio" src="{data_uri}" preload="auto"></audio>
    <script>
        var audio = document.getElementById('primer_audio');
        if (audio) {{
            audio.volume = 0.01; 
            audio.play().catch(function(error) {{
                console.warn("Primer play failed:", error);
            }});
        }}
    </script>
    <style>
        #primer_audio {{ display: none; }}
    </style>
    """
    return html_code

def scale_bbox_to_original(x1, y1, x2, y2, scale_x, scale_y):
    """Scales bounding box coordinates from a resized frame back to the original frame size."""
    # NOTE: The original code had a bug here, scaling x2 and y2 by scale_y.
    # It should be x2 by scale_x and y2 by scale_y. Correcting the logic:
    x1_orig = int(x1 * scale_x)
    y1_orig = int(y1 * scale_y)
    x2_orig = int(x2 * scale_x) # Corrected from scale_y
    y2_orig = int(y2 * scale_y)
    return x1_orig, y1_orig, x2_orig, y2_orig

# ==============================
# ⚙️ Configuration
# ==============================

# NOTE: Update these paths to match your system configuration
DEFAULT_MODEL_NAME = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/Model/yolov8/best.pt"
AUDIO_FILE_PATH = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/Model/yolov8/alert.mp3"
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ==============================
# 🧠 Cached Model Loader
# ==============================

@st.cache_resource
def load_model(model_path, device):
    """Load YOLOv8 model once and cache it."""
    try:
        model = YOLO(model_path)
        model.to(device)
        return model
    except Exception as e:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: `{model_path}`")
        else:
            st.error(f"❌ Error loading model: {e}")
        st.stop()


# ==============================
# 🎨 Streamlit Setup
# ==============================

st.set_page_config(page_title="Near-Drowning Detection Tracker", layout="wide")
st.title("🏊 Duration-Based Object Tracker & Alert System")
st.caption("Monitors near-drowning incidents based on tracked duration.")
st.markdown("---")


# ==============================
# 🔄 Session State Setup
# ==============================

if "source_type" not in st.session_state:
    st.session_state.source_type = 'Upload Video File'
if "video_source" not in st.session_state:
    st.session_state.video_source = None # Stores path for video file OR camera index for webcam
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "selected_classes" not in st.session_state:
    st.session_state.selected_classes = []
if "first_seen_frame" not in st.session_state:
    st.session_state.first_seen_frame = {}
if "last_seen_frame" not in st.session_state:
    st.session_state.last_seen_frame = {}
if "alert_triggered_ids_current" not in st.session_state:
    st.session_state.alert_triggered_ids_current = set()
if "dismissed_alerts" not in st.session_state:
    st.session_state.dismissed_alerts = set()
if "stop_sound_pending" not in st.session_state:
    st.session_state.stop_sound_pending = None 
if "camera_index" not in st.session_state:
    st.session_state.camera_index = 0
if "camera_cap" not in st.session_state:
    st.session_state.camera_cap = None


# 🧩 Handle any pending sound stop (after button rerun)
if st.session_state.stop_sound_pending is not None:
    stop_alert_sound(alert_id=st.session_state.stop_sound_pending)
    st.session_state.stop_sound_pending = None

# Callback for stopping the webcam feed
def stop_detection():
    st.session_state.is_running = False
    stop_alert_sound()
    st.session_state.first_seen_frame.clear()
    st.session_state.last_seen_frame.clear()
    st.session_state.alert_triggered_ids_current.clear()
    st.session_state.dismissed_alerts.clear()
    # Release camera object if it exists
    if st.session_state.camera_cap:
        st.session_state.camera_cap.release()
        st.session_state.camera_cap = None
    # Rerun to clear the video output and show the start button
    st.rerun() 

# ==============================
# 🧩 Sidebar Settings
# ==============================

model_path = DEFAULT_MODEL_NAME
use_custom_model = False
tmpdir = None

with st.sidebar:
    st.header("⚙️ Model & Performance")

    with st.expander("Model Configuration"):
        use_custom_model = st.checkbox("Upload custom YOLO model (.pt)", value=False)
        model_display = os.path.basename(DEFAULT_MODEL_NAME)

        if use_custom_model:
            model_file = st.file_uploader("Upload YOLO model", type=["pt"])
            if model_file:
                tmpdir = tempfile.mkdtemp()
                model_path = os.path.join(tmpdir, model_file.name)
                with open(model_path, "wb") as f:
                    f.write(model_file.getbuffer())
                st.success(f"✅ Custom model uploaded: {model_file.name}")
            else:
                st.warning("Please upload a .pt file.")
        else:
            st.info(f"Using default model: `{model_display}`")

        device = st.selectbox("Device", ["cuda", "cpu"], index=0)

    st.markdown("---")

    with st.spinner("Loading YOLOv8 model..."):
        model = load_model(model_path, device)
        all_classes = list(model.names.values())

    st.header("🎯 Detection Settings")
    
    st.session_state.selected_classes = st.multiselect(
        "Classes to Detect:",
        options=all_classes,
        default=all_classes,
        help="Filter which objects are passed to the tracker."
    )
    
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05, help="Minimum confidence score for a detection to be considered valid.")
    
    st.markdown("---")

    st.header("🔔 Alert Rules")
    
    alert_min_duration = st.number_input(
        "Target Duration (seconds)",
        min_value=0.1,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="Time an object must be tracked before an alert is triggered."
    )

    default_index = all_classes.index("near-drowning") if "near-drowning" in all_classes else 0
    target_alert_class_name = st.selectbox(
        "Alert Target Class:",
        options=all_classes,
        index=default_index,
        help="The specific class that triggers the duration alert (e.g., 'near-drowning')."
    )
    
    st.markdown("---")
    
    # ⭐ Frame Rate Limiter Setting
    max_fps_limit = st.slider(
        "Max Processing FPS (for speed)", 
        min_value=5, 
        max_value=60, 
        value=30, 
        step=5,
        help="Limits the number of frames processed per second to save CPU/GPU resources."
    )


# ==============================
# 📤 Step 1: Select Source (with Live Preview)
# ==============================

st.header("Step 1: Select Input Source")

# Use st.radio for clear selection between video file and webcam
st.session_state.source_type = st.radio(
    "Choose your input source:",
    ['Upload Video File', 'Use Webcam'],
    horizontal=True,
    key='source_radio'
)

# Placeholders for source display and info
source_col, info_col = st.columns([3, 1])
source_display = source_col.empty()

# Reset video_source when switching modes
st.session_state.video_source = None

if st.session_state.source_type == 'Upload Video File':
    # --- Video Upload Logic ---
    uploaded_video = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        save_path = os.path.join(UPLOAD_DIR, uploaded_video.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.session_state.video_source = save_path
        info_col.success(f"✅ Video uploaded: {uploaded_video.name}")
        # Show video preview
        source_display.video(st.session_state.video_source)
    else:
        info_col.info("Please upload a video file to proceed.")


elif st.session_state.source_type == 'Use Webcam':
    # --- Webcam Live Preview Logic ---
    st.session_state.camera_index = st.number_input("Enter Camera Index (0 for default)", min_value=0, value=0, step=1, key='camera_index_input')
    st.session_state.video_source = st.session_state.camera_index
    
    info_col.info(f"Webcam at index: {st.session_state.camera_index}")

    if not st.session_state.is_running:
        # We need a separate capture object for the PREVIEW to keep it running
        # This cap object will be closed and re-opened when starting the main detection loop
        cap_preview = cv2.VideoCapture(st.session_state.camera_index)
        
        if cap_preview.isOpened():
            st.session_state.camera_cap = cap_preview # Store it temporarily to manage its state
            success_prev, frame_prev = cap_preview.read()
            if success_prev:
                source_display.image(frame_prev, channels="BGR", caption="Webcam Live Preview (No Detection Running)", use_container_width=True)
            else:
                source_display.error("Cannot read camera feed. Check index or permissions.")
            
            # Release the preview cap immediately so the main loop can re-open it later
            cap_preview.release()
            st.session_state.camera_cap = None
        else:
            source_display.error("Webcam not found or access denied. Check camera index.")


st.markdown("---")


# ==============================
# 🚀 Step 2: Run Detection
# ==============================

st.header("Step 2: Run Detection & Tracking")

source_is_ready = (st.session_state.video_source is not None)

selected_class_ids = [k for k, v in model.names.items() if v in st.session_state.selected_classes]
target_alert_class_id = next((k for k, v in model.names.items() if v == target_alert_class_name), None)


# Button logic
if st.session_state.is_running:
    # Dedicated Stop button when detection is running
    st.button("🛑 STOP DETECTION", type="secondary", use_container_width=True, on_click=stop_detection)
elif source_is_ready and model:
    # Dedicated Start button when source is ready
    st.button("▶️ START DETECTION & TRACKING", type="primary", use_container_width=True, key="start_main_button")
    if st.session_state.start_main_button:
        # Set running state and rerun to start the loop below
        st.session_state.is_running = True
        st.rerun() # <<< CORRECTED: Replaced st.experimental_rerun() with st.rerun()
else:
    st.warning("Please ensure a source is selected and the model is loaded.")


# --- Main Detection Loop ---
if st.session_state.is_running:
    
    # Reset tracking/alert state for a new run
    st.session_state.first_seen_frame.clear()
    st.session_state.last_seen_frame.clear()
    st.session_state.alert_triggered_ids_current.clear()
    st.session_state.dismissed_alerts.clear()
    
    # Prepare layout
    col_video, col_alert = st.columns([3, 1])
    # Use the source_display placeholder from Step 1 for the video output in detection mode
    # If not possible (due to scope), use a new one:
    stframe = col_video.empty() 
    alert_placeholder = col_alert.empty()
    
    # Prime browser audio permission
    primer_placeholder = st.empty()
    primer_placeholder.markdown(create_audio_primer_html(AUDIO_FILE_PATH), unsafe_allow_html=True)
    time.sleep(0.1)
    primer_placeholder.empty()

    # Initialize video capture (path for file, index for webcam)
    cap = cv2.VideoCapture(st.session_state.video_source)
    
    if not cap.isOpened():
        st.error("Failed to open capture source. Stopping detection.")
        stop_detection()
        st.rerun() # <<< CORRECTED: Replaced st.experimental_rerun() with st.rerun()

    # Get FPS for video files, assume 30 for live camera if unavailable
    if st.session_state.source_type == 'Upload Video File':
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
    else:
        # For live camera, use the max_fps_limit as the effective processing rate.
        fps = max_fps_limit 
        
    frame_index = 0
    
    # ⭐ FRAME RATE LIMITING SETUP
    if st.session_state.source_type == 'Upload Video File' and fps > max_fps_limit:
        SKIP_INTERVAL = max(1, round(fps / max_fps_limit))
        st.info(f"Video FPS: {fps:.2f}. **Target FPS is {max_fps_limit}**. Skipping {SKIP_INTERVAL - 1} frame(s) for every {SKIP_INTERVAL} processed.")
    else:
        SKIP_INTERVAL = 1
        st.info(f"Processing every frame (Target: {max_fps_limit} FPS).")

    # ⭐ INPUT RESOLUTION SETUP (For maximum speed)
    NEW_WIDTH = 640
    NEW_HEIGHT = 480
    st.info(f"Processing frames internally resized to {NEW_WIDTH}x{NEW_HEIGHT} for maximum speed.")


    with st.spinner("Processing feed..."):
        while st.session_state.is_running and cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                if st.session_state.source_type == 'Upload Video File':
                    st.session_state.is_running = False # Stop on end of file
                    break
                else:
                    st.error("Error reading from camera. Trying again...")
                    time.sleep(0.5) 
                    continue
            
            frame_index += 1

            # ⭐ IMPLEMENT FRAME SKIPPING
            if (frame_index - 1) % SKIP_INTERVAL != 0:
                continue 


            # ---------------------------------------------
            # ⭐ OPTIMIZED DETECTION BLOCK 
            # ---------------------------------------------
            
            # 1. Calculate scaling factors
            h_orig, w_orig, _ = frame.shape
            scale_x = w_orig / NEW_WIDTH
            scale_y = h_orig / NEW_HEIGHT
            
            # 2. Resize the frame for FAST processing
            resized_frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_LINEAR)

            # 3. Run YOLO detection and tracking
            results = model.track(
                resized_frame, 
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_threshold,
                classes=selected_class_ids,
                verbose=False
            )
            
            annotated_frame = frame.copy() 
            boxes = results[0].boxes
            current_target_class_ids = set()

            if boxes.id is not None:
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    track_id = int(boxes.id[i].item())
                    conf = float(boxes.conf[i].item())
                    
                    # Get scaled coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    
                    # 4. Scale the coordinates back to the original size
                    x1_orig, y1_orig, x2_orig, y2_orig = scale_bbox_to_original(
                        x1, y1, x2, y2, scale_x, scale_y
                    )
                    
                    # Plot the box and label
                    class_name = model.names[class_id]
                    label = f"id:{track_id} {class_name} {conf:.2f}"
                    
                    color = (0, 0, 255) if class_name == target_alert_class_name else (255, 0, 0)
                    
                    cv2.rectangle(annotated_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
                    cv2.putText(annotated_frame, label, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Tracking and Alert Logic
                    if class_id == target_alert_class_id:
                        current_target_class_ids.add(track_id)
                        
                        if track_id not in st.session_state.first_seen_frame:
                            st.session_state.first_seen_frame[track_id] = frame_index
                        st.session_state.last_seen_frame[track_id] = frame_index

                        elapsed_frames = frame_index - st.session_state.first_seen_frame[track_id]
                        elapsed_seconds = elapsed_frames / fps

                        if elapsed_seconds >= alert_min_duration and track_id not in st.session_state.alert_triggered_ids_current:
                            st.session_state.alert_triggered_ids_current.add(track_id)
                            if track_id not in st.session_state.dismissed_alerts:
                                play_alert_sound(AUDIO_FILE_PATH, alert_id=track_id)
                                st.toast(f"⚠️ Alert triggered for ID {track_id} ({target_alert_class_name})")
                            
            # Cleanup inactive/vanished targets
            inactive_ids = [
                tid for tid in list(st.session_state.first_seen_frame.keys())
                if tid not in current_target_class_ids and
                (frame_index - st.session_state.last_seen_frame.get(tid, frame_index)) > fps
            ]
            for tid in inactive_ids:
                st.session_state.first_seen_frame.pop(tid, None)
                st.session_state.last_seen_frame.pop(tid, None)
                if tid in st.session_state.alert_triggered_ids_current:
                    stop_alert_sound(alert_id=tid)
                st.session_state.alert_triggered_ids_current.discard(tid)
                st.session_state.dismissed_alerts.discard(tid)

            # Render active alerts
            with col_alert:
                with alert_placeholder.container():
                    st.markdown("##### Current Alerts:")
                    active_alerts = [
                        tid for tid in st.session_state.alert_triggered_ids_current
                        if tid not in st.session_state.dismissed_alerts
                    ]
                    if not active_alerts:
                        st.info("No active alerts.")
                    for alert_id in active_alerts:
                        alert_box = st.container(border=True)
                        alert_col_text, alert_col_close = alert_box.columns([3, 1])
                        alert_col_text.markdown(f"**🚨 ID {alert_id} is {target_alert_class_name}**")
                        if alert_col_close.button("❌", key=f"dismiss_{alert_id}_{frame_index}"):
                            st.session_state.dismissed_alerts.add(alert_id)
                            st.session_state.stop_sound_pending = alert_id
                            st.rerun() # <<< CORRECTED: Replaced st.experimental_rerun() with st.rerun()

            # Update the detection output frame
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

    # --- End of while loop ---
    cap.release()
    stop_alert_sound()  
    st.session_state.is_running = False 

    if st.session_state.source_type == 'Upload Video File':
        st.balloons()
        st.success(f"✅ Detection complete — processed {frame_index} total frames.")
        if use_custom_model and tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    elif st.session_state.source_type == 'Use Webcam':
        st.success("Webcam stream stopped.")
        
    st.rerun() # <<< CORRECTED: Replaced st.experimental_rerun() with st.rerun()