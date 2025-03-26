# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QFileDialog, QLabel, QGraphicsView, QGraphicsScene,
                             QMessageBox, QVBoxLayout, QWidget, QTextEdit,
                             QSpinBox, QDoubleSpinBox, QGridLayout, QPushButton,
                             QSizePolicy) # Added missing imports
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer # Added QTimer
import cv2
import numpy as np
import sys
import os
import time
import json # For saving/loading configuration
from queue import Queue, Empty
from skimage.metrics import structural_similarity as ssim

# --- Constants ---
REF_NORM = "Norm"
REF_SHUTDOWN = "Shutdown"
REF_FAIL = "Fail"
DEFAULT_SSIM_THRESHOLD = 0.90
DEFAULT_ERROR_COOLDOWN = 15 # Default cooldown in seconds
CONFIG_FILE_NAME = "image_checker_config.json"
LOG_FILE_NAME = "activity_log.txt"

# --- Worker Thread for Image Processing ---
class ProcessingWorker(QThread):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str, str) # Message, Background Color
    save_error_signal = pyqtSignal(np.ndarray, str) # Frame, File Path
    ssim_signal = pyqtSignal(float) # Signal to emit the current SSIM score vs Norm

    def __init__(self, frame_queue, ref_images_provider, config_provider, compare_func):
        super().__init__()
        self.frame_queue = frame_queue
        self.get_ref_images = ref_images_provider # Function to get current ref images dict
        self.get_config = config_provider # Function to get current config (threshold, cooldown)
        self.compare_images = compare_func
        self.running = False
        self.last_error_time = 0

    def run(self):
        self.running = True
        self.log_signal.emit("⚙️ Processing thread started.")
        last_status_normal_log_time = 0

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except Empty:
                continue
            except Exception as e:
                self.log_signal.emit(f"❌ Error getting frame from queue: {e}")
                continue

            if not self.running:
                break

            # --- Get current config and references ---
            try:
                current_config = self.get_config()
                ssim_threshold = current_config.get('ssim_threshold', DEFAULT_SSIM_THRESHOLD)
                error_cooldown = current_config.get('error_cooldown', DEFAULT_ERROR_COOLDOWN)
                error_folder = current_config.get('error_folder') # Get error folder from config getter

                ref_images = self.get_ref_images()
                norm_image = ref_images.get(REF_NORM)
                shutdown_image = ref_images.get(REF_SHUTDOWN)
                fail_image = ref_images.get(REF_FAIL)
            except Exception as e:
                 self.log_signal.emit(f"❌ Error reading config/refs in worker: {e}")
                 time.sleep(1) # Avoid spamming if config access fails
                 continue


            # --- Image Comparison Logic ---
            if not norm_image:
                if time.time() % 60 < 1: # Log once a minute approx
                    self.log_signal.emit("⚠️ Norm reference image not set. Skipping comparison.")
                time.sleep(0.1)
                continue

            try:
                # Compare with Norm image first
                is_match_norm, score_norm = self.compare_images(frame, norm_image, ssim_threshold)
                self.ssim_signal.emit(score_norm if score_norm is not None else -1.0) # Emit SSIM score vs Norm

                if is_match_norm:
                    # Status: Normal
                    current_time = time.time()
                    # Log "Normal" status only periodically to avoid flooding logs
                    if current_time - last_status_normal_log_time > 60: # Log every 60 seconds
                        self.status_signal.emit("Normal", "lightgreen")
                        # self.log_signal.emit("✅ Status: Normal") # Optional detailed log
                        last_status_normal_log_time = current_time
                    continue # Go to next frame

                # --- Mismatch with Norm ---
                # Reset periodic normal logging time if mismatch detected
                last_status_normal_log_time = 0

                status_msg = "Unknown Mismatch!"
                status_color = "orange"
                save_image = True
                error_subfolder = "unknown"
                log_msg = f"⚠️ Mismatch vs Norm (SSIM: {score_norm:.3f})."

                # Compare with Shutdown image if available
                if shutdown_image:
                    is_match_shutdown, score_shutdown = self.compare_images(frame, shutdown_image, ssim_threshold)
                    if is_match_shutdown:
                        status_msg = "Shutdown Detected"
                        status_color = "lightblue"
                        save_image = True # Optionally save shutdown state images
                        error_subfolder = "shutdown"
                        log_msg = f"ℹ️ Detected Shutdown state (SSIM vs Shutdown: {score_shutdown:.3f})."
                        self.log_signal.emit(log_msg)
                        self.status_signal.emit(status_msg, status_color)
                        # Continue to saving logic below if save_image is True

                # Compare with Fail image if available AND not already matched Shutdown
                if fail_image and error_subfolder == "unknown": # Only check Fail if not Shutdown
                    is_match_fail, score_fail = self.compare_images(frame, fail_image, ssim_threshold)
                    if is_match_fail:
                        status_msg = "FAIL State Detected!"
                        status_color = "red"
                        save_image = True # Definitely save Fail state images
                        error_subfolder = "fail"
                        log_msg = f"❌ Detected FAIL state (SSIM vs Fail: {score_fail:.3f})."
                        self.log_signal.emit(log_msg)
                        self.status_signal.emit(status_msg, status_color)
                         # Continue to saving logic below

                # If still unknown after checking Shutdown/Fail
                if error_subfolder == "unknown":
                     self.log_signal.emit(log_msg) # Log the initial Norm mismatch message
                     self.status_signal.emit(status_msg, status_color)

                # --- Save Error Image Logic ---
                current_time = time.time()
                if save_image and error_folder and (current_time - self.last_error_time > error_cooldown):
                    try:
                        specific_error_folder = os.path.join(error_folder, error_subfolder)
                        os.makedirs(specific_error_folder, exist_ok=True) # Create subfolder if needed

                        file_name = f"{error_subfolder}_{time.strftime('%Y%m%d_%H%M%S')}.png"
                        file_path = os.path.join(specific_error_folder, file_name)

                        self.save_error_signal.emit(frame.copy(), file_path) # Send a copy
                        self.last_error_time = current_time
                        self.log_signal.emit(f"📸 Saved image: {file_path}")
                        # Update status briefly to show save action?
                        # self.status_signal.emit(f"Saved: {file_name}", "pink")

                    except Exception as e:
                        self.log_signal.emit(f"❌ Error saving image to '{error_subfolder}': {e}")
                elif not error_folder and save_image:
                     if time.time() % 60 < 1:
                         self.log_signal.emit("⚠️ Error folder not set. Cannot save mismatch.")

            except Exception as e:
                self.log_signal.emit(f"❌ Error during image comparison/logic: {e}")
                time.sleep(0.5)

        self.log_signal.emit("⚙️ Processing thread finished.")

    def stop(self):
        self.running = False
        self.log_signal.emit("⚙️ Stopping processing thread...")


# --- Main Application Window ---
class ImageCheckerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        self.ref_images = {REF_NORM: None, REF_SHUTDOWN: None, REF_FAIL: None}
        self.webcam_roi = None # Stores (x, y, w, h) tuple or None
        self.processing = False
        self.error_folder = None
        self.log_file_path = None

        # Configuration values (will be loaded)
        self.config = {
            'ssim_threshold': DEFAULT_SSIM_THRESHOLD,
            'error_cooldown': DEFAULT_ERROR_COOLDOWN,
            'error_folder': None,
            'ref_paths': {REF_NORM: None, REF_SHUTDOWN: None, REF_FAIL: None},
            'webcam_roi': None,
        }

        self.frame_queue = Queue(maxsize=10)

        # --- Worker Thread Setup ---
        # Pass functions (lambdas) to provide current data to the worker safely
        self.processing_worker = ProcessingWorker(
            self.frame_queue,
            lambda: self.ref_images,       # Provides current reference images
            lambda: self.get_current_config_for_worker(), # Provides current config
            self.compare_images            # Provides the comparison method
        )
        self.processing_worker.log_signal.connect(self.log_activity)
        self.processing_worker.status_signal.connect(self.update_status_label)
        self.processing_worker.save_error_signal.connect(self.save_error_image_from_thread)
        self.processing_worker.ssim_signal.connect(self.update_ssim_display)

        self.init_ui()
        self.load_config() # Load config after UI is initialized
        self.log_activity("Ứng dụng khởi động.")
        self.update_all_ui_elements() # Update UI based on loaded config

    def get_current_config_for_worker(self):
        # Function called by worker to get a snapshot of relevant config
        return {
            'ssim_threshold': self.ssimThresholdSpinBox.value(),
            'error_cooldown': self.cooldownSpinBox.value(),
            'error_folder': self.error_folder, # Use the internal state variable
        }

    def init_ui(self):
        self.setWindowTitle("Hệ thống kiểm tra hình ảnh - Webcam v2.0")
        self.setGeometry(100, 100, 1300, 760) # Slightly larger window

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QGridLayout(central_widget)
        main_layout.setSpacing(10) # Add spacing between widgets

        # --- Left Panel: Webcam and Controls ---
        left_panel_layout = QVBoxLayout()
        main_layout.addLayout(left_panel_layout, 0, 0) # Row 0, Col 0

        # Webcam View
        self.scene = QGraphicsScene(self)
        self.graphicsView = QGraphicsView(self.scene)
        self.graphicsView.setMinimumSize(640, 360)
        self.graphicsView.setMaximumSize(640, 360)
        self.graphicsView.setStyleSheet("border: 1px solid black;")
        self.graphicsView.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed) # Fix size
        self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray)) # Placeholder background
        left_panel_layout.addWidget(self.graphicsView)

        # Webcam Control Buttons
        webcam_button_layout = QGridLayout()
        left_panel_layout.addLayout(webcam_button_layout)
        self.ONCam = self.create_button(" Bật Webcam", icon="📷")
        self.ONCam.clicked.connect(self.start_webcam)
        webcam_button_layout.addWidget(self.ONCam, 0, 0)

        self.OFFCam = self.create_button(" Tắt Webcam", icon="🚫")
        self.OFFCam.clicked.connect(self.stop_webcam)
        self.OFFCam.setEnabled(False) # Initially disabled
        webcam_button_layout.addWidget(self.OFFCam, 0, 1)

        # --- Right Panel: Logs, Status, Settings ---
        right_panel_layout = QVBoxLayout()
        main_layout.addLayout(right_panel_layout, 0, 1) # Row 0, Col 1

        # Log Area
        log_label = QLabel("Log Hoạt Động:")
        right_panel_layout.addWidget(log_label)
        self.log_text_edit = QTextEdit(self)
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setStyleSheet("border: 1px solid black; padding: 5px; background-color: white;")
        self.log_text_edit.setMinimumHeight(200)
        self.log_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow expansion
        right_panel_layout.addWidget(self.log_text_edit)

        # Status Label
        status_layout = QGridLayout()
        right_panel_layout.addLayout(status_layout)
        self.process_label = QLabel("Trạng thái: Chờ", self)
        self.process_label.setAlignment(Qt.AlignCenter)
        self.process_label.setStyleSheet("border: 1px solid black; padding: 5px; background-color: lightgray; font-weight: bold;")
        self.process_label.setMinimumHeight(40)
        status_layout.addWidget(self.process_label, 0, 0, 1, 2) # Span 2 columns

        # SSIM Display Label
        self.ssim_label = QLabel("SSIM vs Norm: N/A")
        self.ssim_label.setAlignment(Qt.AlignCenter)
        self.ssim_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;") # Slightly different background
        self.ssim_label.setMinimumHeight(30)
        status_layout.addWidget(self.ssim_label, 1, 0, 1, 2) # Span 2 columns


        # --- Settings Area (Grid Layout) ---
        settings_groupbox = QtWidgets.QGroupBox("Cài đặt và Tham chiếu")
        right_panel_layout.addWidget(settings_groupbox)
        settings_layout = QGridLayout(settings_groupbox)

        # Reference Image Buttons
        self.SettingButton_Norm = self.create_button(" Ảnh Norm", icon="✅")
        self.SettingButton_Norm.clicked.connect(lambda: self.load_reference_image(REF_NORM))
        settings_layout.addWidget(self.SettingButton_Norm, 0, 0)

        self.SettingButton_Shut = self.create_button(" Ảnh Shutdown", icon="⛔️")
        self.SettingButton_Shut.clicked.connect(lambda: self.load_reference_image(REF_SHUTDOWN))
        settings_layout.addWidget(self.SettingButton_Shut, 0, 1)

        self.SettingButton_Fail = self.create_button(" Ảnh Fail", icon="❌")
        self.SettingButton_Fail.clicked.connect(lambda: self.load_reference_image(REF_FAIL))
        settings_layout.addWidget(self.SettingButton_Fail, 0, 2)

        # ROI and Error Folder Buttons
        self.SettingButton_ROI_Webcam = self.create_button(" Chọn ROI", icon="✂️")
        self.SettingButton_ROI_Webcam.clicked.connect(self.select_webcam_roi)
        self.SettingButton_ROI_Webcam.setEnabled(False) # Enable when webcam is on
        settings_layout.addWidget(self.SettingButton_ROI_Webcam, 1, 0)

        self.SaveButton = self.create_button(" Thư mục Lưu lỗi", icon="📁")
        self.SaveButton.clicked.connect(self.select_error_folder)
        settings_layout.addWidget(self.SaveButton, 1, 1)

        # Processing Toggle Button
        self.ToggleProcessingButton = self.create_button(" Bắt đầu Xử lý", icon="▶️")
        self.ToggleProcessingButton.clicked.connect(self.toggle_processing)
        settings_layout.addWidget(self.ToggleProcessingButton, 1, 2)

        # Configuration SpinBoxes
        settings_layout.addWidget(QLabel("Ngưỡng SSIM:"), 2, 0)
        self.ssimThresholdSpinBox = QDoubleSpinBox()
        self.ssimThresholdSpinBox.setRange(0.1, 1.0)
        self.ssimThresholdSpinBox.setSingleStep(0.01)
        self.ssimThresholdSpinBox.setValue(self.config['ssim_threshold'])
        self.ssimThresholdSpinBox.setDecimals(3) # Show more precision
        settings_layout.addWidget(self.ssimThresholdSpinBox, 2, 1)

        settings_layout.addWidget(QLabel("Cooldown Lỗi (giây):"), 3, 0)
        self.cooldownSpinBox = QSpinBox()
        self.cooldownSpinBox.setRange(1, 300) # 1 second to 5 minutes
        self.cooldownSpinBox.setSingleStep(1)
        self.cooldownSpinBox.setValue(self.config['error_cooldown'])
        settings_layout.addWidget(self.cooldownSpinBox, 3, 1)

        # Exit Button
        self.ExitButton = self.create_button(" Thoát", icon="🚪")
        self.ExitButton.clicked.connect(self.close_application)
        settings_layout.addWidget(self.ExitButton, 3, 2) # Place Exit button

        # Set column stretch for settings grid
        settings_layout.setColumnStretch(0, 1)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(2, 1)

        # Set main layout column stretch
        main_layout.setColumnStretch(0, 1) # Left panel takes less space
        main_layout.setColumnStretch(1, 2) # Right panel takes more space

    def create_button(self, text, icon=None):
        """Helper function to create a styled QPushButton."""
        button = QPushButton(text, self)
        if icon:
            button.setText(icon + text)
        button.setStyleSheet("background-color: white; padding: 6px; text-align: left; border: 1px solid #ccc; border-radius: 3px;")
        button.setMinimumHeight(35)
        # Hover effect
        button.enterEvent = lambda event, b=button: b.setStyleSheet(b.styleSheet().replace("white", "#e8f0fe"))
        button.leaveEvent = lambda event, b=button: b.setStyleSheet(b.styleSheet().replace("#e8f0fe", "white"))
        return button

    # --- Configuration Loading/Saving ---

    def save_config(self):
        """Saves current configuration to a JSON file."""
        self.config['ssim_threshold'] = self.ssimThresholdSpinBox.value()
        self.config['error_cooldown'] = self.cooldownSpinBox.value()
        self.config['error_folder'] = self.error_folder
        self.config['webcam_roi'] = self.webcam_roi
        # Save reference image paths, not the images themselves
        self.config['ref_paths'] = {key: path for key, path in self.config['ref_paths'].items()}

        try:
            with open(CONFIG_FILE_NAME, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            self.log_activity(f"💾 Đã lưu cấu hình vào {CONFIG_FILE_NAME}")
        except Exception as e:
            self.log_activity(f"❌ Lỗi khi lưu cấu hình: {e}")

    def load_config(self):
        """Loads configuration from JSON file if it exists."""
        if not os.path.exists(CONFIG_FILE_NAME):
            self.log_activity(f"📄 Không tìm thấy file cấu hình {CONFIG_FILE_NAME}. Sử dụng giá trị mặc định.")
            # Ensure default log path is set if no config and error folder chosen later
            if self.error_folder:
                 self.log_file_path = os.path.join(self.error_folder, LOG_FILE_NAME)
            return

        try:
            with open(CONFIG_FILE_NAME, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            # Update internal config, falling back to defaults if keys are missing
            self.config['ssim_threshold'] = loaded_config.get('ssim_threshold', DEFAULT_SSIM_THRESHOLD)
            self.config['error_cooldown'] = loaded_config.get('error_cooldown', DEFAULT_ERROR_COOLDOWN)
            self.error_folder = loaded_config.get('error_folder') # Load error folder path
            self.config['error_folder'] = self.error_folder # Sync internal config dict too

            # Set log file path based on loaded error folder
            if self.error_folder and os.path.isdir(self.error_folder):
                self.log_file_path = os.path.join(self.error_folder, LOG_FILE_NAME)
            else:
                self.error_folder = None # Reset if path is invalid
                self.log_file_path = None

            self.webcam_roi = loaded_config.get('webcam_roi')
            self.config['webcam_roi'] = self.webcam_roi # Sync internal config dict

            # Load reference images based on saved paths
            loaded_ref_paths = loaded_config.get('ref_paths', {})
            self.config['ref_paths'] = loaded_ref_paths # Store paths in config dict
            for key, path in loaded_ref_paths.items():
                if path and os.path.exists(path):
                    try:
                        img = cv2.imread(path)
                        if img is not None:
                            self.ref_images[key] = img
                            self.log_activity(f"✅ Đã tải lại ảnh tham chiếu '{key}' từ: {os.path.basename(path)}")
                        else:
                             self.log_activity(f"⚠️ Không thể đọc ảnh '{key}' từ: {path}")
                             self.config['ref_paths'][key] = None # Clear invalid path
                    except Exception as e:
                        self.log_activity(f"❌ Lỗi khi tải lại ảnh '{key}' từ {path}: {e}")
                        self.config['ref_paths'][key] = None # Clear invalid path
                elif path:
                    self.log_activity(f"⚠️ Đường dẫn ảnh tham chiếu '{key}' không tồn tại: {path}")
                    self.config['ref_paths'][key] = None # Clear invalid path


            self.log_activity(f"💾 Đã tải cấu hình từ {CONFIG_FILE_NAME}")

        except json.JSONDecodeError as e:
            self.log_activity(f"❌ Lỗi giải mã file cấu hình {CONFIG_FILE_NAME}: {e}. Sử dụng giá trị mặc định.")
            self.reset_to_defaults() # Optionally reset everything
        except Exception as e:
            self.log_activity(f"❌ Lỗi không xác định khi tải cấu hình: {e}. Sử dụng giá trị mặc định.")
            self.reset_to_defaults() # Optionally reset everything

        # Update UI elements after loading
        self.update_all_ui_elements()

    def reset_to_defaults(self):
        """Resets configuration variables to default state."""
        self.config['ssim_threshold'] = DEFAULT_SSIM_THRESHOLD
        self.config['error_cooldown'] = DEFAULT_ERROR_COOLDOWN
        self.error_folder = None
        self.log_file_path = None
        self.config['error_folder'] = None
        self.webcam_roi = None
        self.config['webcam_roi'] = None
        self.ref_images = {REF_NORM: None, REF_SHUTDOWN: None, REF_FAIL: None}
        self.config['ref_paths'] = {REF_NORM: None, REF_SHUTDOWN: None, REF_FAIL: None}
        self.log_activity("🔄 Đã đặt lại cấu hình về mặc định.")
        self.update_all_ui_elements()


    def update_all_ui_elements(self):
        """Updates all relevant UI elements based on current state/config."""
        self.ssimThresholdSpinBox.setValue(self.config['ssim_threshold'])
        self.cooldownSpinBox.setValue(self.config['error_cooldown'])
        self.update_button_colors()
        self.update_toggle_button_text()
        self.update_error_folder_button_style()


    @QtCore.pyqtSlot(str)
    def log_activity(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{timestamp} - {message}"
        self.log_text_edit.append(full_message)
        self.log_text_edit.ensureCursorVisible()

        if self.log_file_path:
            try:
                os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True) # Ensure directory exists
                with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(full_message + "\n")
            except Exception as e:
                error_msg = f"{timestamp} - ❌ ERROR writing to log file '{self.log_file_path}': {e}"
                self.log_text_edit.append(error_msg)
                self.log_text_edit.ensureCursorVisible()
                self.log_file_path = None
                self.update_status_label("Log file error!", "red")

    @QtCore.pyqtSlot(str, str)
    def update_status_label(self, message, background_color="lightgray"):
        self.process_label.setText(f"Trạng thái: {message}")
        self.process_label.setStyleSheet(
            f"border: 1px solid black; padding: 5px; background-color: {background_color}; color: black; font-weight: bold; border-radius: 3px;"
        )

    @QtCore.pyqtSlot(float)
    def update_ssim_display(self, score):
        """Updates the SSIM score label."""
        if score >= 0:
            self.ssim_label.setText(f"SSIM vs Norm: {score:.4f}")
            # Optional: Change background based on score?
            # color = "lightgreen" if score > self.ssimThresholdSpinBox.value() else "lightcoral"
            # self.ssim_label.setStyleSheet(f"padding: 5px; background-color: {color};")
        else:
            self.ssim_label.setText("SSIM vs Norm: N/A")
            # self.ssim_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")


    def update_button_colors(self):
        # Reference image buttons
        for btn, key in zip([self.SettingButton_Norm, self.SettingButton_Shut, self.SettingButton_Fail],
                            [REF_NORM, REF_SHUTDOWN, REF_FAIL]):
            icon = btn.text().split(" ")[0]
            base_text = " ".join(btn.text().split(" ")[1:]).replace(" (Đã chọn)", "") # Clean base text
            if self.ref_images.get(key) is not None:
                btn.setStyleSheet("background-color: lightgreen; color: black; padding: 6px; text-align: left; border: 1px solid #ccc; border-radius: 3px;")
                btn.setText(f"{icon} {base_text} (Đã chọn)")
            else:
                btn.setStyleSheet("background-color: white; padding: 6px; text-align: left; border: 1px solid #ccc; border-radius: 3px;")
                btn.setText(f"{icon} {base_text}")

        # ROI button
        icon_roi = self.SettingButton_ROI_Webcam.text().split(" ")[0]
        base_text_roi = " ".join(self.SettingButton_ROI_Webcam.text().split(" ")[1:]).replace(" (Đã chọn)", "")
        if self.webcam_roi:
             self.SettingButton_ROI_Webcam.setStyleSheet("background-color: lightblue; color: black; padding: 6px; text-align: left; border: 1px solid #ccc; border-radius: 3px;")
             self.SettingButton_ROI_Webcam.setText(f"{icon_roi} {base_text_roi} (Đã chọn)")
        else:
             self.SettingButton_ROI_Webcam.setStyleSheet("background-color: white; padding: 6px; text-align: left; border: 1px solid #ccc; border-radius: 3px;")
             self.SettingButton_ROI_Webcam.setText(f"{icon_roi} {base_text_roi}")

    def update_error_folder_button_style(self):
        """Updates the style of the error folder button."""
        icon = self.SaveButton.text().split(" ")[0]
        base_text = " ".join(self.SaveButton.text().split(" ")[1:]).replace(" (Đã chọn)", "")
        if self.error_folder and os.path.isdir(self.error_folder):
            self.SaveButton.setStyleSheet("background-color: lightblue; color: black; padding: 6px; text-align: left; border: 1px solid #ccc; border-radius: 3px;")
            self.SaveButton.setText(f"{icon} {base_text} (Đã chọn)")
        else:
             self.SaveButton.setStyleSheet("background-color: white; padding: 6px; text-align: left; border: 1px solid #ccc; border-radius: 3px;")
             self.SaveButton.setText(f"{icon} {base_text}")

    def update_toggle_button_text(self):
        if self.processing:
            self.ToggleProcessingButton.setText("⏹ Dừng Xử lý")
            self.ToggleProcessingButton.setStyleSheet("background-color: orange; color: black; padding: 6px; text-align: left; border: 1px solid #ccc; border-radius: 3px;")
        else:
            self.ToggleProcessingButton.setText("▶️ Bắt đầu Xử lý")
            self.ToggleProcessingButton.setStyleSheet("background-color: lightgreen; color: black; padding: 6px; text-align: left; border: 1px solid #ccc; border-radius: 3px;")

    def start_webcam(self):
        if self.cap is None:
            try:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.log_activity("⚠️ CAP_DSHOW failed, trying default backend...")
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        raise IOError("Cannot open webcam")

                # Configure webcam if possible (optional)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Let it use default or adjust later
                # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend = self.cap.getBackendName()

                timer_interval = int(1000 / actual_fps) if actual_fps > 0 else 66

                self.frame_timer.start(timer_interval)
                self.log_activity(f"Webcam đã bật (Backend: {backend}, Res: {w}x{h}, FPS: {actual_fps:.1f}, Interval: {timer_interval}ms)")
                self.ONCam.setEnabled(False)
                self.OFFCam.setEnabled(True)
                self.SettingButton_ROI_Webcam.setEnabled(True)
                self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.black)) # Black background when active

            except Exception as e:
                self.log_activity(f"❌ Lỗi khi bật webcam: {e}")
                QMessageBox.critical(self, "Lỗi Webcam", f"Không thể mở webcam.\n{e}")
                if self.cap: self.cap.release()
                self.cap = None
                self.ONCam.setEnabled(True)
                self.OFFCam.setEnabled(False)
                self.SettingButton_ROI_Webcam.setEnabled(False)
        else:
            self.log_activity("Webcam đã được bật.")

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.log_activity("⚠️ Lỗi đọc khung hình từ webcam.")
            return

        try:
            display_frame = frame.copy()
            processing_frame = frame

            # Draw ROI rectangle persistently on display frame
            if self.webcam_roi:
                x, y, w, h = self.webcam_roi
                fh, fw = frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(fw, x + w), min(fh, y + h)
                if x2 > x1 and y2 > y1:
                    # Crop frame for processing
                    processing_frame = frame[y1:y2, x1:x2]
                    # Draw rectangle on the display frame
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    self.log_activity("⚠️ ROI không hợp lệ, đã đặt lại.")
                    self.webcam_roi = None
                    self.config['webcam_roi'] = None # Update config state
                    self.update_button_colors() # Update ROI button style

            # --- Display Frame ---
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_img).scaled(
                self.graphicsView.width() - 2, self.graphicsView.height() - 2,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            # Update scene efficiently
            items = self.scene.items()
            if not items:
                self.scene.addItem(QtWidgets.QGraphicsPixmapItem(pixmap))
            elif isinstance(items[0], QtWidgets.QGraphicsPixmapItem):
                items[0].setPixmap(pixmap)
            else:
                self.scene.clear()
                self.scene.addItem(QtWidgets.QGraphicsPixmapItem(pixmap))

            # Center the image if scene has items
            if self.scene.items():
                self.graphicsView.centerOn(self.scene.items()[0])

            # --- Queue Frame for Processing ---
            if self.processing and not self.frame_queue.full():
                try:
                    # Put the potentially cropped frame
                    self.frame_queue.put(processing_frame.copy(), block=False)
                except Exception as e:
                     self.log_activity(f"❌ Error putting frame in queue: {e}")
            elif self.processing and self.frame_queue.full():
                # Log queue full periodically to avoid spam
                if time.time() % 5 < 0.1: # Log every 5s approx
                    self.log_activity("⚠️ Hàng đợi xử lý đầy, khung hình bị bỏ qua.")

        except Exception as e:
            self.log_activity(f"❌ Lỗi trong update_frame: {e}")

    def stop_webcam(self):
        if self.cap:
            self.frame_timer.stop()
            time.sleep(0.1)
            self.cap.release()
            self.cap = None
            self.scene.clear()
            self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray)) # Reset background
            self.log_activity("Webcam đã tắt.")
            self.ONCam.setEnabled(True)
            self.OFFCam.setEnabled(False)
            self.SettingButton_ROI_Webcam.setEnabled(False)
        else:
            self.log_activity("Webcam chưa được bật.")

    def load_reference_image(self, img_type):
        options = QFileDialog.Options()
        # Start in the directory of the previously selected image for this type, if available
        start_dir = os.path.dirname(self.config['ref_paths'].get(img_type, "")) if self.config['ref_paths'].get(img_type) else ""
        file_path, _ = QFileDialog.getOpenFileName(self, f"Chọn ảnh {img_type}", start_dir,
                                                   "Images (*.png *.jpg *.jpeg *.bmp *.jfif);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("File không phải là ảnh hợp lệ hoặc không thể đọc.")

                self.ref_images[img_type] = img
                self.config['ref_paths'][img_type] = file_path # Store the path in config

                self.update_button_colors()
                self.log_activity(f"Đã tải ảnh tham chiếu '{img_type}' từ: {os.path.basename(file_path)}")
                # self.save_config() # Optionally save config immediately after selecting an image

            except Exception as e:
                self.log_activity(f"❌ Lỗi khi tải ảnh '{img_type}': {e}")
                QMessageBox.warning(self, "Lỗi Tải Ảnh", f"Không thể tải ảnh: {file_path}\n{e}")

    def select_webcam_roi(self):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "Chưa bật Webcam", "Bạn cần bật webcam để chọn ROI.")
            return

        self.frame_timer.stop()
        ret, frame = self.cap.read()
        if self.cap and self.cap.isOpened():
             self.frame_timer.start() # Restart timer quickly

        if not ret or frame is None:
            QMessageBox.warning(self, "Lỗi Khung hình", "Không thể lấy khung hình từ webcam để chọn ROI.")
            return

        try:
            msg = "Chọn vùng quan tâm (ROI) và nhấn ENTER hoặc SPACE. Nhấn C để hủy."
            # Create a named window to easily destroy it later
            cv2.namedWindow("Chọn ROI", cv2.WINDOW_NORMAL) # Use WINDOW_NORMAL for resizability
            cv2.setWindowTitle("Chọn ROI", msg)
            cv2.resizeWindow("Chọn ROI", frame.shape[1]//2, frame.shape[0]//2) # Start with a smaller window

            roi = cv2.selectROI("Chọn ROI", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Chọn ROI")

            if roi == (0, 0, 0, 0):
                self.log_activity("❌ Người dùng đã hủy chọn ROI.")
                # No need to reset self.webcam_roi here unless you want cancel to clear existing ROI
                return

            if roi[2] > 0 and roi[3] > 0:
                self.webcam_roi = roi
                self.config['webcam_roi'] = roi # Update config state
                self.log_activity(f"✅ Đã chọn ROI cho webcam: {self.webcam_roi}")
                # self.save_config() # Optionally save config immediately
            else:
                self.log_activity("⚠️ ROI không hợp lệ (w/h=0), không thay đổi.")
                # Don't reset self.webcam_roi if selection was invalid

            self.update_button_colors()

        except Exception as e:
            self.log_activity(f"❌ Lỗi trong quá trình chọn ROI: {e}")
            QMessageBox.critical(self, "Lỗi ROI", f"Đã xảy ra lỗi khi chọn ROI:\n{e}")
            cv2.destroyAllWindows()
            if self.cap and self.cap.isOpened() and not self.frame_timer.isActive():
                 self.frame_timer.start()

    def select_error_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu ảnh lỗi",
                                                  self.error_folder if self.error_folder else "",
                                                  options=options)
        if folder:
            self.error_folder = folder
            self.config['error_folder'] = folder # Update config state
            self.log_activity(f"📁 Thư mục lưu ảnh lỗi được đặt thành: {self.error_folder}")
            # Update log file path based on new folder
            self.log_file_path = os.path.join(self.error_folder, LOG_FILE_NAME)
            self.log_activity(f"📄 Log sẽ được lưu tại: {self.log_file_path}")
            # self.save_config() # Optionally save config immediately
        else:
             self.log_activity("⚠️ Người dùng đã hủy chọn thư mục lưu lỗi.")

        self.update_error_folder_button_style()


    def toggle_processing(self):
        # Check prerequisites
        if not self.ref_images.get(REF_NORM):
            QMessageBox.warning(self, "Thiếu ảnh Norm", "Bạn cần chọn ảnh tham chiếu 'Norm' trước.")
            return
        if not self.error_folder or not os.path.isdir(self.error_folder):
            QMessageBox.warning(self, "Thiếu Thư mục Lưu", "Bạn cần chọn một thư mục hợp lệ để lưu ảnh lỗi trước.")
            return
        if not self.cap or not self.cap.isOpened():
             QMessageBox.warning(self, "Webcam chưa bật", "Bạn cần bật webcam để bắt đầu xử lý.")
             return

        self.processing = not self.processing

        if self.processing:
            # Ensure worker uses latest config values from GUI
            self.config['ssim_threshold'] = self.ssimThresholdSpinBox.value()
            self.config['error_cooldown'] = self.cooldownSpinBox.value()

            if not self.processing_worker.isRunning():
                # Clear queue before starting
                while not self.frame_queue.empty():
                    try: self.frame_queue.get_nowait()
                    except Empty: break
                self.processing_worker.start()

            self.processing_worker.running = True # Ensure worker flag is set
            self.update_status_label("🔄 Đang xử lý...", "lightgreen")
            self.log_activity("▶️ Bắt đầu xử lý ảnh.")
        else:
            if self.processing_worker.isRunning():
                 self.processing_worker.stop()
                 # self.processing_worker.wait(1000) # Optional wait

            self.update_status_label("⏹ Đã dừng xử lý", "orange")
            self.log_activity("⏹ Đã dừng xử lý ảnh.")
            self.ssim_label.setText("SSIM vs Norm: N/A") # Reset SSIM display

        self.update_toggle_button_text()

    def compare_images(self, img1, img2, threshold):
        """
        Compares two images using SSIM. Returns (bool: is_match, float: score).
        Score is None if comparison fails.
        """
        if img1 is None or img2 is None:
            return False, None

        try:
            if len(img1.shape) > 2 and img1.shape[2] == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else: img1_gray = img1.copy() # Use copy if already gray

            if len(img2.shape) > 2 and img2.shape[2] == 3:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else: img2_gray = img2.copy()

            h1, w1 = img1_gray.shape
            h2, w2 = img2_gray.shape
            if h1 != h2 or w1 != w2:
                interpolation = cv2.INTER_AREA if h2 > h1 or w2 > w1 else cv2.INTER_LINEAR
                img2_gray = cv2.resize(img2_gray, (w1, h1), interpolation=interpolation)

            # Use win_size appropriate for image dimensions, must be odd and <= min(h, w)
            win_size = min(min(h1, w1), 7) # Default SSIM window size is 7
            if win_size % 2 == 0: win_size -= 1 # Ensure odd size
            if win_size < 3: win_size = 3 # Minimum sensible window size

            # Calculate SSIM
            if h1 < win_size or w1 < win_size:
                 # If image is smaller than window, SSIM might fail or give weird results.
                 # Consider simple MSE or skip comparison for very small ROIs/images.
                 # For now, return False and indicate failure score.
                 # self.log_signal.emit("⚠️ Image too small for SSIM calculation.") # Signal doesn't exist here
                 print(f"Warning: Image size ({w1}x{h1}) is smaller than SSIM window size ({win_size}). Skipping SSIM.")
                 return False, None

            score, _ = ssim(img1_gray, img2_gray, full=True,
                            data_range=img1_gray.max() - img1_gray.min(),
                            win_size=win_size) # Specify window size

            return score > threshold, score

        except Exception as e:
            # Log the error via the main thread's logger if possible,
            # or just print if called from a context without direct logger access.
            print(f"❌ Exception during SSIM comparison: {e}") # Print as fallback
            # self.log_activity(f"❌ Lỗi SSIM: {e}") # This method belongs to the main class
            return False, None # Indicate failure

    @QtCore.pyqtSlot(np.ndarray, str)
    def save_error_image_from_thread(self, frame, file_path):
        try:
             # Directory creation is handled in worker now, but double-check doesn't hurt
             # os.makedirs(os.path.dirname(file_path), exist_ok=True)
             success = cv2.imwrite(file_path, frame)
             if not success:
                 self.log_activity(f"❌ cv2.imwrite failed for: {file_path}")
        except Exception as e:
             self.log_activity(f"❌ Exception during cv2.imwrite for {file_path}: {e}")

    def close_application(self):
        self.log_activity("🚪 Bắt đầu quá trình thoát ứng dụng...")
        self.save_config() # Save config on exit
        self.close() # Triggers closeEvent

    def closeEvent(self, event):
        self.log_activity("🚪 Cửa sổ đang đóng. Dọn dẹp tài nguyên...")

        if self.processing_worker.isRunning():
            self.processing = False
            self.processing_worker.stop()
            self.processing_worker.wait(1500)

        if self.cap and self.cap.isOpened():
            self.stop_webcam()

        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except Empty: break

        self.log_activity("🚪 Dọn dẹp hoàn tất. Thoát.")
        # Final log message write
        if self.log_file_path:
             try:
                 with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                     log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Application Closed\n---\n")
             except Exception: pass # Ignore final log write error

        self.save_config() # Save config one last time
        event.accept()


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    window = ImageCheckerApp()
    window.show()
    sys.exit(app.exec_())