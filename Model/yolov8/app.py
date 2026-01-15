import streamlit as st
import streamlit.components.v1 as components
import base64
import time
import cv2
import os
import tempfile
from ultralytics import YOLO
import shutil
import numpy as np


# ==============================
# 🔊 Utility Functions
# ==============================


# Global flag to track what sounds should be playing
if "playing_audio_ids" not in st.session_state:
    st.session_state.playing_audio_ids = {}


def get_audio_web_component():
    """
    Create a single Web Audio API component that we control via JavaScript.
    Uses window.parent to expose functions to parent Streamlit context.
    """
    html_code = """
    <html>
    <body>
    <script>
        // Create audio context in parent window so it persists
        if (!window.parent.audioContext) {
            window.parent.audioContext = new (window.parent.AudioContext || window.parent.webkitAudioContext)();
            window.parent.playingAudios = {};
            console.log('✓ Audio context initialized');
        }
        
        function loadAudioBuffer(audioData) {
            return new Promise((resolve, reject) => {
                try {
                    const binaryString = atob(audioData);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }
                    window.parent.audioContext.decodeAudioData(bytes.buffer, resolve, reject);
                } catch (e) {
                    console.error('Decode error:', e);
                    reject(e);
                }
            });
        }
        
        window.parent.playAlert = function(alertId, audioBase64) {
            console.log('playAlert called for ID:', alertId);
            
            // Stop existing audio
            if (window.parent.playingAudios[alertId]) {
                try {
                    window.parent.playingAudios[alertId].stop(0);
                } catch (e) {}
            }
            
            // Load and play
            loadAudioBuffer(audioBase64).then(buffer => {
                console.log('Buffer loaded, playing...');
                const source = window.parent.audioContext.createBufferSource();
                source.buffer = buffer;
                source.loop = true;
                source.connect(window.parent.audioContext.destination);
                source.start(0);
                window.parent.playingAudios[alertId] = source;
                console.log('✓ Audio playing for ID:', alertId);
            }).catch(err => console.error('Audio load error:', err));
        };
        
        window.parent.stopAlert = function(alertId) {
            console.log('stopAlert called for ID:', alertId);
            if (window.parent.playingAudios[alertId]) {
                try {
                    window.parent.playingAudios[alertId].stop(0);
                    console.log('✓ Audio stopped for ID:', alertId);
                } catch (e) {
                    console.error('Stop error:', e);
                }
                delete window.parent.playingAudios[alertId];
            }
        };
        
        window.parent.stopAllAlerts = function() {
            console.log('stopAllAlerts called');
            for (let alertId in window.parent.playingAudios) {
                window.parent.stopAlert(alertId);
            }
        };
    </script>
    </body>
    </html>
    """
    return html_code


def render_web_audio_manager():
    """Render the Web Audio component once."""
    components.html(get_audio_web_component(), height=10, width=10)


# Cache audio data globally to avoid re-encoding
_cached_audio_data = None

def load_cached_audio(audio_path):
    """Load and cache audio data."""
    global _cached_audio_data
    if _cached_audio_data is None:
        try:
            with open(audio_path, "rb") as f:
                _cached_audio_data = base64.b64encode(f.read()).decode()
            print(f"✓ Audio cached ({len(_cached_audio_data)} chars)")
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
    return _cached_audio_data


def play_alert_sound(audio_path, alert_id):
    """Play alert sound using Web Audio API."""
    if alert_id not in st.session_state.playing_audio_ids:
        try:
            # Get cached audio data
            audio_data = load_cached_audio(audio_path)
            if not audio_data:
                return
            
            st.session_state.playing_audio_ids[alert_id] = True
            print(f"Playing alert sound for ID: {alert_id}")
            
            # Call parent window function
            js_code = f"""
            <script>
                if (window.parent.playAlert) {{
                    window.parent.playAlert({alert_id}, '{audio_data}');
                }}
            </script>
            """
            components.html(js_code, height=1, width=1)
        except Exception as e:
            print(f"Error in play_alert_sound: {e}")


def stop_alert_sound(alert_id=None):
    """Stop alert sound using Web Audio API."""
    try:
        if alert_id is not None:
            st.session_state.playing_audio_ids.pop(alert_id, None)
            print(f"Stopping alert sound for ID: {alert_id}")
            
            js_code = f"""
            <script>
                if (window.parent.stopAlert) {{
                    window.parent.stopAlert({alert_id});
                }}
            </script>
            """
            components.html(js_code, height=1, width=1)
        else:
            st.session_state.playing_audio_ids.clear()
            print("Stopping all alert sounds")
            
            js_code = """
            <script>
                if (window.parent.stopAllAlerts) {
                    window.parent.stopAllAlerts();
                }
            </script>
            """
            components.html(js_code, height=1, width=1)
    except Exception as e:
        print(f"Error in stop_alert_sound: {e}")


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
    x1_orig = int(x1 * scale_x)
    y1_orig = int(y1 * scale_y)
    x2_orig = int(x2 * scale_x) 
    y2_orig = int(y2 * scale_y)
    return x1_orig, y1_orig, x2_orig, y2_orig

# ==============================
# ⚙️ Configuration
# ==============================

# NOTE: Update these paths to match your system configuration
DEFAULT_MODEL_NAME = "C:/Users/Lenovo/Near_Drowning_Repository/Model/yolov8/best.pt"
AUDIO_FILE_PATH = "C:/Users/Lenovo/Near_Drowning_Repository/Model/yolov8/alert.mp3"
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
# ⭐ Camera Scanning Function
# ==============================

@st.cache_data(show_spinner="Scanning for available cameras...")
def get_available_cameras():
    """Scans for available camera indices (0 to 9) and returns them as a list of descriptive strings."""
    available_cameras = []
    # Test up to 10 indices (0 to 9)
    for i in range(10): 
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # Use CAP_DSHOW for faster initialization on Windows
            if cap.isOpened():
                # Try to read a frame to confirm functionality
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(f"Camera {i} (Index: {i})")
                else:
                    available_cameras.append(f"Camera {i} (Index: {i}) - Inaccessible")
                cap.release()
            else:
                # If cap doesn't open, we skip/ignore it silently unless it's index 0
                if i == 0 and not available_cameras:
                    available_cameras.append(f"Camera 0 (Default) - Not Found")
            cap.release()
        except Exception:
             # Ignore exceptions during scan
             pass
            
    if not available_cameras:
        available_cameras.append("No Cameras Found (Try Index 0)")
        return available_cameras, 0
    
    # Return list of names and the index of the default camera (index 0)
    default_selection_index = next((i for i, name in enumerate(available_cameras) if "Index: 0" in name), 0)
    return available_cameras, default_selection_index

# ==============================
# 🎨 Streamlit Setup
# ==============================

st.set_page_config(page_title="Near-Drowning Detection Tracker", layout="wide")
st.title("🏊 Duration-Based Object Tracker & Alert System")
st.caption("Monitors near-drowning incidents based on tracked duration.")
render_web_audio_manager()

# ==============================
# ⭐ NEW: Dynamic Height CSS + Fullscreen
# ==============================

VIDEO_CONTAINER_CSS = """
<style>
/* Normal mode - constrained height */
.stImage {
    overflow: hidden !important;
}

.stImage > div {
    max-height: 100vh;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}

.stImage img {
    max-width: 100%;
    max-height: none !important;
    width: 100%;
    height: auto;
    object-fit: fill;
    display: block; 
}

/* Make fullscreen button always visible without hovering */
.stElementToolbar {
    opacity: 1 !important;
    visibility: visible !important;
    position: absolute !important;
    top: 0 !important;
    right: 0 !important;
    pointer-events: auto !important;
}

/* Fullscreen mode - fill entire screen while keeping aspect ratio */
:fullscreen .stImage {
    width: 100vw !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    max-height: none !important;
    overflow: hidden !important;
    overflow-y: hidden !important;
    overflow-x: hidden !important;
}

:fullscreen .stImage > div {
    width: 100vw !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    max-height: none !important;
    display: flex;
}

:fullscreen .stImage img {
    width: 100vw !important;
    height: 100vh !important;
    max-width: none !important;
    max-height: none !important;
    object-fit: cover;
    object-position: center;
}

/* Fullscreen only: overflow hidden for data-testid elements */
:fullscreen [data-testid="stImage"] {
    overflow: hidden !important;
}

:fullscreen [data-testid="stImageContainer"] {
    overflow: hidden !important;
}

/* Remove height constraint from Streamlit emotion cache element in fullscreen */
:fullscreen .st-emotion-cache-sa0fr5 {
    height: 100% !important;
    max-height: none !important;
}



</style>
"""
st.markdown(VIDEO_CONTAINER_CSS, unsafe_allow_html=True)

st.markdown("---")


# ==============================
# 🔄 Session State Setup
# ==============================

if "source_type" not in st.session_state:
    st.session_state.source_type = 'Upload Video File'
if "video_source" not in st.session_state:
    st.session_state.video_source = None 
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "selected_classes" not in st.session_state:
    st.session_state.selected_classes = []

if "accumulated_frames" not in st.session_state:
    st.session_state.accumulated_frames = {} 
if "last_seen_frame" not in st.session_state:
    st.session_state.last_seen_frame = {}
if "alert_triggered_ids_current" not in st.session_state:
    st.session_state.alert_triggered_ids_current = set()
if "dismissed_alerts" not in st.session_state:
    st.session_state.dismissed_alerts = set()
if "stop_sound_pending" not in st.session_state:
    st.session_state.stop_sound_pending = None 
if "alert_rows" not in st.session_state:
    st.session_state.alert_rows = {}
if "camera_index" not in st.session_state:
    st.session_state.camera_index = 0
if "camera_cap" not in st.session_state:
    st.session_state.camera_cap = None
if "last_rendered_alerts" not in st.session_state:
    st.session_state.last_rendered_alerts = set()
if "stop_pending_ids" not in st.session_state:
    st.session_state.stop_pending_ids = set()


# 🧩 Handle any pending sound stops (top of every run)
if st.session_state.stop_sound_pending is not None:
    stop_alert_sound(alert_id=st.session_state.stop_sound_pending)
    st.session_state.stop_sound_pending = None

# Also process any pending IDs in the set (from button callbacks)
for pending_id in list(st.session_state.stop_pending_ids):
    stop_alert_sound(alert_id=pending_id)
st.session_state.stop_pending_ids.clear()

# Callback for stopping the webcam feed
def stop_detection():
    st.session_state.is_running = False
    stop_alert_sound()
    st.session_state.accumulated_frames.clear() 
    st.session_state.last_seen_frame.clear()
    st.session_state.alert_triggered_ids_current.clear()
    st.session_state.dismissed_alerts.clear()
    # Release camera object if it exists
    if st.session_state.camera_cap:
        st.session_state.camera_cap.release()
        st.session_state.camera_cap = None
    # Rerun to clear the video output and show the start button
    st.rerun()

# Callback for dismissing an alert and stopping its sound
def dismiss_alert_callback(alert_id):
    """Called when ❌ button is clicked."""
    print(f"Alert ID {alert_id} dismissed by user.")
    # Mark the audio to be stopped (will execute at top of next run)
    st.session_state.stop_pending_ids.add(alert_id)
    # Mark as dismissed
    st.session_state.dismissed_alerts.add(alert_id)
    st.session_state.alert_triggered_ids_current.discard(alert_id)

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
        value=2.0, 
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
    
    grace_period_seconds = st.number_input(
        "Detection Grace Period (seconds)",
        min_value=0.1,
        value=1.0, 
        step=0.1,
        format="%.1f",
        help="How long an object can be 'lost' (e.g., underwater) before its accumulated timer resets."
    )

    st.markdown("---")
    
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

st.session_state.source_type = st.radio(
    "Choose your input source:",
    ['Upload Video File', 'Use Webcam'],
    horizontal=True,
    key='source_radio'
)

# Placeholders for source display and info
source_col, info_col = st.columns([3, 1])
source_display = source_col.empty()

# Reset video_source when switching modes (only when not actively running)
if not st.session_state.is_running:
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
    # --- Webcam Dropdown Logic ---
    
    # Get the list of available cameras and the default index
    camera_options, default_index = get_available_cameras()
    
    selected_camera_name = st.selectbox(
        "Select Camera:",
        options=camera_options,
        index=default_index,
        key='camera_select'
    )
    
    # Extract the camera index from the selected string (e.g., "Camera 0 (Index: 0)" -> 0)
    try:
        # Assumes format is "Camera X (Index: Y)"
        index_str = selected_camera_name.split('(Index: ')[-1].replace(')', '').strip()
        st.session_state.camera_index = int(index_str)
    except:
        st.session_state.camera_index = 0 # Fallback to 0 if parsing fails
        
    st.session_state.video_source = st.session_state.camera_index
    
    info_col.info(f"Using camera index: **{st.session_state.camera_index}**")

    if not st.session_state.is_running and "No Cameras Found" not in selected_camera_name:
        # We need a separate capture object for the PREVIEW 
        cap_preview = cv2.VideoCapture(st.session_state.camera_index, cv2.CAP_DSHOW)
        
        if cap_preview.isOpened():
            st.session_state.camera_cap = cap_preview
            success_prev, frame_prev = cap_preview.read()
            if success_prev:
                source_display.image(frame_prev, channels="BGR", width='stretch')
            else:
                source_display.error(f"Cannot read camera feed from index {st.session_state.camera_index}. Check index or permissions.")
            
            # Release the preview cap
            cap_preview.release()
            st.session_state.camera_cap = None
        else:
            source_display.error(f"Webcam at index {st.session_state.camera_index} not found or access denied. Try a different index.")


st.markdown("---")


# ==============================
# 🚀 Step 2: Run Detection
# ==============================

st.header("Step 2: Run Detection & Tracking")

source_is_ready = (st.session_state.video_source is not None)

selected_class_ids = [k for k, v in model.names.items() if v in st.session_state.selected_classes]
target_alert_class_id = next((k for k, v in model.names.items() if v == target_alert_class_name), None)


# --- Main Detection Loop ---
if st.session_state.is_running:
    # Dedicated Stop button when detection is running
    st.button("🛑 STOP DETECTION", type="secondary", width='stretch', on_click=stop_detection)

    # Reset tracking/alert state for a new run
  
    

    
    # Prepare layout
    col_video, col_alert = st.columns([3, 1])
    stframe = col_video.empty() 
    alert_placeholder = col_alert.empty()
    
    # Prime browser audio permission
    primer_placeholder = st.empty()
    primer_placeholder.markdown(create_audio_primer_html(AUDIO_FILE_PATH), unsafe_allow_html=True)
    time.sleep(0.1)
    primer_placeholder.empty()
    # Initialize Web Audio component


    # Initialize video capture (path for file, index for webcam)
    # Note: Use CAP_DSHOW for webcam if OS is Windows
    if st.session_state.source_type == 'Use Webcam':
        cap = cv2.VideoCapture(st.session_state.video_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(st.session_state.video_source)
    
    if not cap.isOpened():
        st.error("Failed to open capture source. Stopping detection.")
        stop_detection()
        st.rerun() 

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
        print(f"[INFO] Video FPS: {fps:.2f}. Target FPS is {max_fps_limit}. Skipping {SKIP_INTERVAL - 1} frame(s).")
        st.info(f"Video FPS: {fps:.2f}. **Target FPS is {max_fps_limit}**. Skipping {SKIP_INTERVAL - 1} frame(s) for every {SKIP_INTERVAL} processed.")
    else:
        SKIP_INTERVAL = 1
        print(f"[INFO] Processing every frame (Target: {max_fps_limit} FPS).")
        st.info(f"Processing every frame (Target: {max_fps_limit} FPS).")

    # ⭐ INPUT RESOLUTION SETUP (For maximum speed)
    NEW_WIDTH = 640
    NEW_HEIGHT = 480
    print(f"[INFO] Processing frames internally resized to {NEW_WIDTH}x{NEW_HEIGHT}.")
    st.info(f"Processing frames internally resized to {NEW_WIDTH}x{NEW_HEIGHT} for maximum speed.")

    # ⭐ Calculate grace period in frames
    GRACE_PERIOD_FRAMES = grace_period_seconds * fps
    print(f"[INFO] Grace Period: {grace_period_seconds:.1f}s = {GRACE_PERIOD_FRAMES:.0f} frames.")
    st.info(f"Timer grace period: {grace_period_seconds}s. (Timer resets if object is lost for longer)")
    

    with st.spinner("Processing feed..."):
        while st.session_state.is_running and cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                if st.session_state.source_type == 'Upload Video File':
                    st.session_state.is_running = False 
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
            
            h_orig, w_orig, _ = frame.shape
            scale_x = w_orig / NEW_WIDTH
            scale_y = h_orig / NEW_HEIGHT
            
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
            
            # Dictionary to hold the final labels to be drawn on the frame
            drawing_labels = {}
            # Temporary list for current frame's detections to print later
            frame_detections = [] 

            if boxes.id is not None:
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    track_id = int(boxes.id[i].item())
                    conf = float(boxes.conf[i].item())
                    
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    
                    x1_orig, y1_orig, x2_orig, y2_orig = scale_bbox_to_original(
                        x1, y1, x2, y2, scale_x, scale_y
                    )
                    
                    class_name = model.names[class_id]
                    
                    # --- TRACKING AND LAST SEEN FRAME LOGIC ---
                    if class_id == target_alert_class_id:
                        current_target_class_ids.add(track_id)
                        
                        # Initialize the Event Timer if this is a new track
                        if track_id not in st.session_state.accumulated_frames:
                            print(f"[FRAME {frame_index:05d}] NEW TRACK: ID {track_id} detected.")
                            st.session_state.accumulated_frames[track_id] = 0 # Start event time
                        
                        # Crucial: Update the last seen frame *only* when it's the target class
                        st.session_state.last_seen_frame[track_id] = frame_index
                    
                    # Store drawing info (time will be added after accumulation below)
                    drawing_labels[track_id] = {
                        'class': class_name,
                        'conf': conf,
                        'coords': (x1_orig, y1_orig, x2_orig, y2_orig),
                        'color': (0, 0, 255) if class_name == target_alert_class_name else (255, 0, 0)
                    }

            # ---------------------------------------------
            # ⭐ CUMULATIVE TIME ACCUMULATION & ALERT CHECK 
            # ---------------------------------------------

            all_tracked_ids = list(st.session_state.accumulated_frames.keys())

            for track_id in all_tracked_ids:
                
                # Increment the accumulated time for this tracked ID by the time step.
                st.session_state.accumulated_frames[track_id] += SKIP_INTERVAL
                
                elapsed_frames = st.session_state.accumulated_frames[track_id]
                elapsed_seconds = elapsed_frames / fps
                
               
                # Check for Alert Trigger
                if elapsed_seconds >= alert_min_duration and track_id not in st.session_state.alert_triggered_ids_current:
                    # Trigger only if the ID was detected in the last few frames (safety check)
                    if (frame_index - st.session_state.last_seen_frame.get(track_id, 0)) <= GRACE_PERIOD_FRAMES:
                        st.session_state.alert_triggered_ids_current.add(track_id)
                        if track_id not in st.session_state.dismissed_alerts:
                            play_alert_sound(AUDIO_FILE_PATH, alert_id=track_id)
                            print(f"[FRAME {frame_index:05d}] ALERT TRIGGERED: ID {track_id} at {elapsed_seconds:.1f}s (Cumulative)!")

                # Update terminal debug message
                if track_id in current_target_class_ids:
                    frame_detections.append(f"ID {track_id} ({elapsed_seconds:.1f}s)")
                
                # Update the drawing label with the calculated time
                if track_id in drawing_labels:
                    drawing_labels[track_id]['label'] = f"id:{track_id} {drawing_labels[track_id]['class']} {drawing_labels[track_id]['conf']:.2f} [{elapsed_seconds:.1f}s]"


            # ⭐ TERMINAL PRINT - Current Detections
            # if frame_detections:
            #     print(f"[FRAME {frame_index:05d}] Active Targets: {', '.join(frame_detections)}")

            # --- DRAWING BLOCK ---
            for track_id, data in drawing_labels.items():
                x1, y1, x2, y2 = data['coords']
                label = data.get('label', f"id:{track_id} {data['class']} {data['conf']:.2f}") 
                color = data['color']
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


            # ---------------------------------------------
            # ⭐ EVENT END/CLEANUP LOGIC 
            # ---------------------------------------------
            
            # Inactive IDs are those whose last detection as the target class is older than the GRACE PERIOD.
            inactive_ids = [
                tid for tid in list(st.session_state.accumulated_frames.keys())
                if (frame_index - st.session_state.last_seen_frame.get(tid, 0)) > GRACE_PERIOD_FRAMES
            ]
            
            for tid in inactive_ids:
                current_time_lost = (frame_index - st.session_state.last_seen_frame.get(tid, 0)) / fps
                print(f"[FRAME {frame_index:05d}] RESET TIMER (EVENT ENDED): ID {tid} history ({st.session_state.accumulated_frames.get(tid, 0)/fps:.1f}s) reset. Last seen {current_time_lost:.1f}s ago (Grace Period: {grace_period_seconds:.1f}s).")

                # NEW: debug prints
                print("Calling stop_alert_sound for tid:", tid)
                print("alert_triggered_ids_current before discard:", st.session_state.alert_triggered_ids_current)

                st.session_state.accumulated_frames.pop(tid, None)
                st.session_state.last_seen_frame.pop(tid, None)
                if tid in st.session_state.alert_triggered_ids_current:
                    stop_alert_sound(alert_id=tid)
                st.session_state.alert_triggered_ids_current.discard(tid)
                st.session_state.dismissed_alerts.discard(tid)

                # NEW: after discard
                print("alert_triggered_ids_current after discard:", st.session_state.alert_triggered_ids_current)


            # Sync audio with alert state
            # Remove audio for any IDs that are no longer in active alerts
            for audio_id in list(st.session_state.playing_audio_ids.keys()):
                if audio_id not in st.session_state.alert_triggered_ids_current:
                    print("the alert is done now stopping the:", audio_id)
                    stop_alert_sound(alert_id=audio_id)


            # Render active alerts (UI element remains). Only render when alert set changes to avoid duplicate key errors.
            active_alerts = [
                tid for tid in st.session_state.alert_triggered_ids_current
                if tid not in st.session_state.dismissed_alerts
            ]
            
            # Only re-render if the set of active alerts has changed
            if set(active_alerts) != st.session_state.last_rendered_alerts:
                alert_placeholder.empty()
                with alert_placeholder:
                    st.markdown("##### Current Alerts:")
                    
                    if not active_alerts:
                        st.info("No active alerts.")
                    else:
                        for alert_id in active_alerts:
                            alert_box = st.container(border=True)
                            alert_col_text, alert_col_close = alert_box.columns([3, 1])
                            
                            current_secs = st.session_state.accumulated_frames.get(alert_id, 0) / fps
                            alert_col_text.markdown(f"**🚨 ID {alert_id} ({target_alert_class_name})** [{current_secs:.1f}s]")

                            # Dismiss button with callback
                            alert_col_close.button(
                                "❌",
                                key=f"dismiss_{alert_id}",
                                on_click=dismiss_alert_callback,
                                args=(alert_id,)
                            )
                
                st.session_state.last_rendered_alerts = set(active_alerts)
       
            # Update the detection output frame
            # The custom CSS applied globally handles the height constraint here.
            stframe.image(annotated_frame, channels="BGR", width='stretch')

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
        
    st.rerun()


elif source_is_ready and model:
    # Dedicated Start button when source is ready
    st.button("▶️ START DETECTION & TRACKING", type="primary", width='stretch', key="start_main_button")
    if st.session_state.start_main_button:

        st.session_state.accumulated_frames.clear() 
        st.session_state.last_seen_frame.clear()
        st.session_state.alert_triggered_ids_current.clear()
        st.session_state.dismissed_alerts.clear()

        # Set running state and rerun to start the loop below
        st.session_state.is_running = True
        st.rerun() 
else:
    st.warning("Please ensure a source is selected and the model is loaded.")
    