#pip install PyQt5 opencv-python ultralytics numpy simpleaudio

import sys
import os
import time
import cv2
import numpy as np
import simpleaudio as sa
from ultralytics import YOLO

# PyQt Imports
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QSlider, QComboBox, 
    QSpinBox, QDoubleSpinBox, QMessageBox, QGroupBox
)
from PyQt5.QtCore import (
    QThread, pyqtSignal, QTimer, Qt, QObject
)
from PyQt5.QtGui import QImage, QPixmap

# --- CONFIGURATION (UPDATE THESE PATHS) ---
DEFAULT_MODEL_NAME = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/Model/yolov8/best.pt"
AUDIO_FILE_PATH = "C:/Users/Lenovo/Marcelino-Portfolio/Near_Drowning_Repository/Model/yolov8/alert.mp3"
# -------------------------------------------

# --- Utility: Scale Bounding Box ---
def scale_bbox_to_original(x1, y1, x2, y2, scale_x, scale_y):
    x1_orig = int(x1 * scale_x)
    y1_orig = int(y1 * scale_y)
    x2_orig = int(x2 * scale_x) 
    y2_orig = int(y2 * scale_y)
    return x1_orig, y1_orig, x2_orig, y2_orig

# =======================================================
# 🔊 AUDIO WORKER (QThread) - Handles Controllable Playback
# =======================================================

class AudioWorker(QObject):
    """Worker object to run audio playback in a separate thread."""
    playback_finished = pyqtSignal()

    def __init__(self, audio_file_path):
        super().__init__()
        self.file_path = audio_file_path
        self._is_running = True
        self.playback_handle = None
        self.wave_obj = None

        if os.path.exists(self.file_path):
            try:
                # Pre-load the wave object
                self.wave_obj = sa.WaveObject.from_wave_file(self.file_path)
            except Exception as e:
                print(f"Error loading audio file: {e}")
                self.wave_obj = None

    def start_playback(self):
        """Starts the audio loop."""
        if self.wave_obj is None:
            print("Audio file not loaded.")
            self.playback_finished.emit()
            return

        self._is_running = True
        try:
            # Start playing and store the handle. Loop it indefinitely (-1)
            self.playback_handle = self.wave_obj.play()

            # Wait until the shared flag or internal stop is called
            while self._is_running and self.playback_handle.is_playing():
                time.sleep(0.1) 
            
            # If the loop exited because self._is_running was set to False, stop the sound
            if self.playback_handle.is_playing():
                self.playback_handle.stop()
                
        except Exception as e:
            print(f"Error during audio playback: {e}")

        self._is_running = False
        self.playback_finished.emit() # Signal the main thread when truly done

    def stop_playback(self):
        """Immediate stop command from the main thread."""
        self._is_running = False
        if self.playback_handle and self.playback_handle.is_playing():
            self.playback_handle.stop()

# =======================================================
# 🧠 VIDEO PROCESSING WORKER (QThread)
# =======================================================

class VideoWorker(QObject):
    """Worker object to handle all OpenCV/YOLO processing."""
    
    # Signals to communicate results back to the main UI thread
    ImageUpdate = pyqtSignal(QImage)
    AlertTriggered = pyqtSignal(int, float) # ID, Duration
    DebugMessage = pyqtSignal(str)
    
    # Configuration passed from the main UI
    config = {
        'model_path': DEFAULT_MODEL_NAME,
        'device': 'cpu', # Use CPU by default for safety in cross-platform code
        'conf_threshold': 0.4,
        'alert_min_duration': 2.0,
        'grace_period_seconds': 5.0,
        'max_fps_limit': 30,
        'target_alert_class_name': 'near-drowning',
        'video_source': 0, # Default to webcam index 0
        'selected_classes': []
    }
    
    def __init__(self):
        super().__init__()
        self._is_running = False
        
        # Tracking State (Equivalent to Streamlit session state)
        self.model = None
        self.accumulated_frames = {} 
        self.last_seen_frame = {}
        self.alert_triggered_ids_current = set()
        self.dismissed_alerts = set()
        self.frame_index = 0
        self.fps = 30 # Default

    def load_model(self):
        """Loads the YOLOv8 model."""
        if not os.path.exists(self.config['model_path']):
            self.DebugMessage.emit(f"❌ Model not found: {self.config['model_path']}")
            return False
            
        try:
            self.model = YOLO(self.config['model_path'])
            self.model.to(self.config['device'])
            self.DebugMessage.emit("✅ Model loaded successfully.")
            
            # Update classes based on loaded model
            self.all_classes = list(self.model.names.values())
            self.target_alert_class_id = next((k for k, v in self.model.names.items() 
                                               if v == self.config['target_alert_class_name']), None)
            
            return True
        except Exception as e:
            self.DebugMessage.emit(f"❌ Error loading model: {e}")
            return False

    def run_detection(self):
        """The main detection and tracking loop."""
        
        if not self.load_model():
            self.stop_detection()
            return
            
        self._is_running = True
        
        # --- CAPTURE SETUP ---
        cap = cv2.VideoCapture(self.config['video_source'])
        
        if not cap.isOpened():
            self.DebugMessage.emit("❌ Failed to open capture source.")
            self.stop_detection()
            return
            
        # Get FPS or use limit
        cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.fps = min(cap_fps, self.config['max_fps_limit'])
        
        if self.config['video_source'] == 0: # Webcam
             self.fps = self.config['max_fps_limit']
             
        SKIP_INTERVAL = max(1, round(cap_fps / self.fps)) if cap_fps > self.fps else 1
        self.GRACE_PERIOD_FRAMES = self.config['grace_period_seconds'] * self.fps
        
        self.DebugMessage.emit(f"Processing at {self.fps:.1f} FPS (Skip: {SKIP_INTERVAL})")

        # --- LOOP START ---
        NEW_WIDTH, NEW_HEIGHT = 640, 480
        
        while self._is_running and cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                if isinstance(self.config['video_source'], str): # Video file
                    break
                # Webcam retry
                time.sleep(0.5) 
                continue
            
            self.frame_index += 1

            if (self.frame_index - 1) % SKIP_INTERVAL != 0:
                continue 

            # ---------------------------------------------
            # ⭐ DETECTION AND TRACKING
            # ---------------------------------------------
            h_orig, w_orig, _ = frame.shape
            scale_x = w_orig / NEW_WIDTH
            scale_y = h_orig / NEW_HEIGHT
            
            resized_frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_LINEAR)

            results = self.model.track(
                resized_frame, 
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.config['conf_threshold'],
                classes=[k for k, v in self.model.names.items() if v in self.config['selected_classes']],
                verbose=False
            )
            
            annotated_frame = frame.copy() 
            boxes = results[0].boxes
            current_target_class_ids = set()
            drawing_labels = {}

            if boxes.id is not None:
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    track_id = int(boxes.id[i].item())
                    conf = float(boxes.conf[i].item())
                    
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1_orig, y1_orig, x2_orig, y2_orig = scale_bbox_to_original(x1, y1, x2, y2, scale_x, scale_y)
                    class_name = self.model.names[class_id]
                    
                    # --- TRACKING LOGIC ---
                    if class_id == self.target_alert_class_id:
                        current_target_class_ids.add(track_id)
                        
                        if track_id not in self.accumulated_frames:
                            self.accumulated_frames[track_id] = 0
                            self.DebugMessage.emit(f"[NEW TRACK] ID {track_id}")
                            
                        self.last_seen_frame[track_id] = self.frame_index
                    
                    drawing_labels[track_id] = {
                        'class': class_name,
                        'conf': conf,
                        'coords': (x1_orig, y1_orig, x2_orig, y2_orig),
                        'color': (0, 0, 255) if class_name == self.config['target_alert_class_name'] else (255, 0, 0)
                    }

            # ---------------------------------------------
            # ⭐ ALERT CHECK & TIME ACCUMULATION
            # ---------------------------------------------
            all_tracked_ids = list(self.accumulated_frames.keys())

            for track_id in all_tracked_ids:
                self.accumulated_frames[track_id] += SKIP_INTERVAL
                elapsed_frames = self.accumulated_frames[track_id]
                elapsed_seconds = elapsed_frames / cap_fps
                
                # Check for Alert Trigger
                if elapsed_seconds >= self.config['alert_min_duration'] and \
                   track_id not in self.alert_triggered_ids_current:
                    
                    # Safety check: is it still considered "active"?
                    if (self.frame_index - self.last_seen_frame.get(track_id, 0)) <= self.GRACE_PERIOD_FRAMES:
                        self.alert_triggered_ids_current.add(track_id)
                        if track_id not in self.dismissed_alerts:
                            self.AlertTriggered.emit(track_id, elapsed_seconds)
                            self.DebugMessage.emit(f"⚠️ ALERT: ID {track_id} triggered at {elapsed_seconds:.1f}s.")
                
                # Update drawing label
                if track_id in drawing_labels:
                    is_alerting = "🚨" if track_id in self.alert_triggered_ids_current and track_id not in self.dismissed_alerts else ""
                    drawing_labels[track_id]['label'] = f"{is_alerting}id:{track_id} {drawing_labels[track_id]['class']} [{elapsed_seconds:.1f}s]"


            # --- DRAWING BLOCK ---
            for track_id, data in drawing_labels.items():
                x1, y1, x2, y2 = data['coords']
                label = data.get('label', f"id:{track_id} {data['class']} {data['conf']:.2f}") 
                color = data['color']
                
                # Use red color for active alerts
                if track_id in self.alert_triggered_ids_current and track_id not in self.dismissed_alerts:
                    color = (0, 0, 255) # BGR: Red
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3) # Thicker box
                else:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


            # ---------------------------------------------
            # ⭐ EVENT END/CLEANUP LOGIC
            # ---------------------------------------------
            inactive_ids = [
                tid for tid in list(self.accumulated_frames.keys())
                if (self.frame_index - self.last_seen_frame.get(tid, 0)) > self.GRACE_PERIOD_FRAMES
            ]
            
            for tid in inactive_ids:
                if tid in self.alert_triggered_ids_current:
                    # Signal main thread to stop the sound for this ID
                    self.AlertTriggered.emit(tid, -1) # Use -1 duration to signal reset/stop
                    
                self.DebugMessage.emit(f"[RESET TIMER] ID {tid} timer reset.")
                
                self.accumulated_frames.pop(tid, None)
                self.last_seen_frame.pop(tid, None)
                self.alert_triggered_ids_current.discard(tid)
                self.dismissed_alerts.discard(tid)

            # Send the frame back to the UI thread
            self.convert_to_qt(annotated_frame)
            
        # --- LOOP END ---
        cap.release()
        self.stop_detection()
        self.DebugMessage.emit("Detection stopped or video ended.")


    def convert_to_qt(self, cv_img):
        """Converts an OpenCV image (BGR) to a QImage."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        # QImage(data, width, height, bytesPerLine, format)
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.ImageUpdate.emit(qt_image)
        
    def stop_detection(self):
        """Called to safely stop the worker thread."""
        self._is_running = False
        
        # Cleanup state for next run
        self.accumulated_frames.clear()
        self.last_seen_frame.clear()
        self.alert_triggered_ids_current.clear()
        self.dismissed_alerts.clear()
        
        # Clear model resources if necessary
        self.model = None

# =======================================================
# 🖼️ MAIN APPLICATION WINDOW
# =======================================================

class MainWindow(QMainWindow):
    
    # Custom signal to send a stop command to a specific AudioWorker instance
    stopAudioSignal = pyqtSignal(int) 

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏊 PyQt Near-Drowning Detection Tracker")
        self.setGeometry(100, 100, 1200, 800)

        self.video_thread = None
        self.video_worker = None
        
        # Dictionary to manage multiple audio threads (one per active alert ID)
        self.active_audio_workers = {} 

        # --- UI Initialization ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 1. Video and Alert Display Column
        video_col = QVBoxLayout()
        self.image_label = QLabel("Waiting for Video Source...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        video_col.addWidget(self.image_label, 5) # Takes more space
        
        self.console_output = QLabel("Console Output:")
        video_col.addWidget(self.console_output, 1)

        # 2. Controls and Settings Column
        controls_col = QVBoxLayout()
        
        # Add Widgets
        self.setup_source_group(controls_col)
        self.setup_alert_group(controls_col)
        self.setup_model_group(controls_col)
        self.setup_run_group(controls_col)
        self.setup_alert_list(controls_col)

        controls_col.addStretch(1) # Push everything to the top

        main_layout.addLayout(video_col, 5) # 5 parts wide
        main_layout.addLayout(controls_col, 3) # 3 parts wide

        # Initial state setup
        self.update_alert_list()
        self.init_workers()

    # --- Setup Groups ---

    def setup_source_group(self, layout):
        group = QGroupBox("Input Source")
        vbox = QVBoxLayout()
        
        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 10)
        self.camera_index_spin.setValue(0)
        vbox.addWidget(QLabel("Webcam Index (0 for default):"))
        vbox.addWidget(self.camera_index_spin)
        
        self.video_path_label = QLabel("Or use a video file path:")
        self.video_path_input = QLineEdit("path/to/video.mp4")
        # In a real app, you'd use a QFileDialog button here
        vbox.addWidget(self.video_path_label)
        vbox.addWidget(self.video_path_input)
        
        self.use_webcam_radio = QRadioButton("Use Webcam (Index 0)")
        self.use_file_radio = QRadioButton("Use Video File")
        self.use_webcam_radio.setChecked(True)
        vbox.addWidget(self.use_webcam_radio)
        vbox.addWidget(self.use_file_radio)
        
        group.setLayout(vbox)
        layout.addWidget(group)


    def setup_model_group(self, layout):
        group = QGroupBox("Model & Performance")
        vbox = QVBoxLayout()
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(40)
        self.conf_slider.setSingleStep(5)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        
        self.conf_label = QLabel(f"Confidence Threshold: {self.conf_slider.value() / 100:.2f}")
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(30)
        
        vbox.addWidget(self.conf_label)
        vbox.addWidget(self.conf_slider)
        vbox.addWidget(QLabel("Max Processing FPS:"))
        vbox.addWidget(self.fps_spin)

        group.setLayout(vbox)
        layout.addWidget(group)

    def setup_alert_group(self, layout):
        group = QGroupBox("Alert Rules")
        vbox = QVBoxLayout()

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 10.0)
        self.duration_spin.setSingleStep(0.1)
        self.duration_spin.setValue(2.0)
        
        self.grace_spin = QDoubleSpinBox()
        self.grace_spin.setRange(0.1, 10.0)
        self.grace_spin.setSingleStep(0.5)
        self.grace_spin.setValue(5.0)

        # In a real app, you'd load the model first to populate this combo box
        self.class_combo = QComboBox()
        self.class_combo.addItem("near-drowning") 
        self.class_combo.addItem("person")

        vbox.addWidget(QLabel("Target Duration (seconds):"))
        vbox.addWidget(self.duration_spin)
        vbox.addWidget(QLabel("Detection Grace Period (seconds):"))
        vbox.addWidget(self.grace_spin)
        vbox.addWidget(QLabel("Alert Target Class:"))
        vbox.addWidget(self.class_combo)

        group.setLayout(vbox)
        layout.addWidget(group)
        
    def setup_run_group(self, layout):
        group = QGroupBox("Controls")
        vbox = QVBoxLayout()
        
        self.start_btn = QPushButton("▶️ START DETECTION")
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn = QPushButton("🛑 STOP DETECTION")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detection)
        
        vbox.addWidget(self.start_btn)
        vbox.addWidget(self.stop_btn)
        
        group.setLayout(vbox)
        layout.addWidget(group)
        
    def setup_alert_list(self, layout):
        # Placeholder for active alert management
        self.alert_group = QGroupBox("🚨 Active Alerts")
        self.alert_layout = QVBoxLayout(self.alert_group)
        self.alert_list_placeholder = QLabel("No active alerts.")
        self.alert_layout.addWidget(self.alert_list_placeholder)
        layout.addWidget(self.alert_group)


    # --- Methods ---

    def init_workers(self):
        """Initializes the worker objects and threads."""
        self.video_thread = QThread()
        self.video_worker = VideoWorker()
        self.video_worker.moveToThread(self.video_thread)
        
        # Connect signals for safe communication
        self.video_thread.started.connect(self.video_worker.run_detection)
        self.video_worker.ImageUpdate.connect(self.image_update_slot)
        self.video_worker.AlertTriggered.connect(self.handle_alert_signal)
        self.video_worker.DebugMessage.connect(self.update_console)
        
        # Clean up when the thread finishes
        self.video_worker.stop_detection() # Ensure initial state is clean
        self.video_worker.destroyed.connect(self.video_thread.quit)
        self.video_thread.finished.connect(self.video_thread.deleteLater)
        self.video_worker.finished.connect(self.video_worker.deleteLater)

    def start_detection(self):
        """Sets configuration and starts the video processing thread."""
        
        # 1. Update Worker Configuration from UI
        self.video_worker.config['conf_threshold'] = self.conf_slider.value() / 100
        self.video_worker.config['alert_min_duration'] = self.duration_spin.value()
        self.video_worker.config['grace_period_seconds'] = self.grace_spin.value()
        self.video_worker.config['max_fps_limit'] = self.fps_spin.value()
        self.video_worker.config['target_alert_class_name'] = self.class_combo.currentText()
        
        # Simplified Class Selection (In a real app, this would be auto-populated after model load)
        self.video_worker.config['selected_classes'] = ["near-drowning", "person"] # Example
        
        # Determine Source
        if self.use_webcam_radio.isChecked():
            self.video_worker.config['video_source'] = self.camera_index_spin.value()
        else:
            self.video_worker.config['video_source'] = self.video_path_input.text()
            
        # 2. Reset UI and State
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.image_label.setText("Detection Running...")
        self.update_console("Starting detection...")
        self.clear_alert_list()
        
        # 3. Start Thread
        self.video_thread.start()

    def stop_detection(self):
        """Stops the video processing thread and all audio workers."""
        if self.video_worker and self.video_thread.isRunning():
            self.video_worker.stop_detection() 
            self.video_thread.quit() # Request thread termination
            self.video_thread.wait() # Wait for thread to finish gracefully
        
        for worker in self.active_audio_workers.values():
            worker.stop_playback()
            
        self.active_audio_workers.clear()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.image_label.setText("Waiting for Video Source...")
        self.update_console("Detection stopped by user.")
        self.clear_alert_list()

    def image_update_slot(self, image):
        """Receives QImage from worker and displays it."""
        pixmap = QPixmap.fromImage(image)
        # Scale the image to fit the label without distortion
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def update_console(self, message):
        """Updates the console output label."""
        self.console_output.setText(f"Console Output: {message}")

    def update_conf_label(self):
        """Updates the confidence slider label."""
        self.conf_label.setText(f"Confidence Threshold: {self.conf_slider.value() / 100:.2f}")

    # --- ALERT and AUDIO Management ---
    
    def handle_alert_signal(self, track_id, elapsed_seconds):
        """
        Receives signal from VideoWorker when an alert is triggered OR reset (-1 duration).
        """
        if elapsed_seconds == -1: # Signal to stop/reset
            self.stop_alert_audio(track_id)
            self.update_alert_list()
            return
            
        if track_id not in self.active_audio_workers:
            # 1. Create a new worker and thread for the unique alert ID
            audio_thread = QThread()
            audio_worker = AudioWorker(AUDIO_FILE_PATH)
            audio_worker.moveToThread(audio_thread)

            # 2. Connect signals
            audio_thread.started.connect(audio_worker.start_playback)
            # When playback is truly finished (either naturally or forced stop)
            audio_worker.playback_finished.connect(audio_thread.quit)
            
            # Optional: Add cleanup on thread finish
            # audio_thread.finished.connect(audio_thread.deleteLater) 
            # audio_worker.playback_finished.connect(audio_worker.deleteLater) 

            # 3. Store and Start
            self.active_audio_workers[track_id] = {'worker': audio_worker, 'thread': audio_thread, 'dismissed': False}
            audio_thread.start()
            self.update_console(f"🚨 New Audio Alert started for ID {track_id}!")

        self.update_alert_list()

    def stop_alert_audio(self, track_id):
        """Stops the audio worker associated with a specific track ID."""
        if track_id in self.active_audio_workers:
            worker_data = self.active_audio_workers[track_id]
            worker_data['worker'].stop_playback() 
            # Give it a moment to quit and remove from dictionary later
            del self.active_audio_workers[track_id]
            self.update_console(f"Audio Alert for ID {track_id} stopped.")
            self.update_alert_list()

    def dismiss_alert_ui(self, track_id):
        """Stops the audio and marks the alert as dismissed."""
        # The worker marks it as dismissed in its tracking state
        self.video_worker.dismissed_alerts.add(track_id)
        self.stop_alert_audio(track_id)
        self.update_console(f"Alert ID {track_id} dismissed.")
        self.update_alert_list()


    def update_alert_list(self):
        """Dynamically updates the alert list UI based on active workers."""
        # Clear existing widgets from the layout
        for i in reversed(range(self.alert_layout.count())): 
            widget = self.alert_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        active_alert_ids = list(self.active_audio_workers.keys())

        if not active_alert_ids:
            self.alert_list_placeholder = QLabel("No active alerts.")
            self.alert_layout.addWidget(self.alert_list_placeholder)
        else:
            for track_id in active_alert_ids:
                h_layout = QHBoxLayout()
                
                # Use a bold label for the alert
                alert_label = QLabel(f"**🚨 ID {track_id} - DROWNING**")
                alert_label.setStyleSheet("font-weight: bold; color: red;")
                
                dismiss_btn = QPushButton("❌ Dismiss")
                dismiss_btn.clicked.connect(lambda _, tid=track_id: self.dismiss_alert_ui(tid))
                
                h_layout.addWidget(alert_label)
                h_layout.addWidget(dismiss_btn)
                
                alert_widget = QWidget()
                alert_widget.setLayout(h_layout)
                self.alert_layout.addWidget(alert_widget)

    def clear_alert_list(self):
         # Clear existing widgets from the layout
        for i in reversed(range(self.alert_layout.count())): 
            widget = self.alert_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        
        self.alert_list_placeholder = QLabel("No active alerts.")
        self.alert_layout.addWidget(self.alert_list_placeholder)
        
    def closeEvent(self, event):
        """Ensures all threads are stopped when the application closes."""
        reply = QMessageBox.question(self, 'Confirm Exit', 
            "Are you sure you want to quit? This will stop all detection and tracking.", 
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.stop_detection() # Cleanly stop all workers and threads
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Ensure necessary widgets are imported for the example code to run if you fill in the blanks
    from PyQt5.QtWidgets import QLineEdit, QRadioButton 
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())