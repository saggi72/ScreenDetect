# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QFileDialog, QLabel, QGraphicsView, QGraphicsScene,
                             QMessageBox, QVBoxLayout, QWidget, QTextEdit,
                             QSpinBox, QDoubleSpinBox, QGridLayout, QPushButton,
                             QSizePolicy, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import cv2
import numpy as np
import sys
import os
import time
import json
from queue import Queue, Empty


# --- Constants ---
REF_NORM = "Norm"
REF_SHUTDOWN = "Shutdown"
REF_FAIL = "Fail"
DEFAULT_SSIM_THRESHOLD = 0.90
DEFAULT_ERROR_COOLDOWN = 15
CONFIG_FILE_NAME = "image_checker_config.json"
LOG_FILE_NAME = "activity_log.txt"

# --- Hàm tính SSIM bằng OpenCV ---
def ssim_opencv(img1, img2, K1=0.01, K2=0.03, win_size=7, data_range=255.0):
    """
    Tính toán chỉ số SSIM giữa hai ảnh bằng OpenCV.
    Args:
        img1 (np.ndarray): Ảnh grayscale thứ nhất (float).
        img2 (np.ndarray): Ảnh grayscale thứ hai (float), cùng kích thước với img1.
        K1 (float): Hằng số SSIM K1.
        K2 (float): Hằng số SSIM K2.
        win_size (int): Kích thước cửa sổ Gaussian (lẻ).
        data_range (float): Phạm vi giá trị pixel (ví dụ: 255.0 cho ảnh 8-bit).
    Returns:
        float: Điểm SSIM trung bình.
               Trả về None nếu có lỗi hoặc kích thước không phù hợp.
    """
    if img1 is None or img2 is None:
        return None
    if img1.shape != img2.shape:
        # Cân nhắc resize hoặc trả về lỗi tùy theo yêu cầu
        # print(f"Warning: SSIM input shapes mismatch: {img1.shape} vs {img2.shape}")
        # Nên xử lý resize trước khi gọi hàm này (như trong compare_images)
        return None # Hoặc raise ValueError

    # Đảm bảo win_size là số lẻ và không lớn hơn kích thước ảnh
    h, w = img1.shape
    win_size = min(win_size, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        # print(f"Warning: SSIM win_size too small ({win_size}), returning None.")
        return None # SSIM không đáng tin cậy với cửa sổ quá nhỏ

    # Chuyển sang float nếu chưa phải
    if img1.dtype != np.float64:
        img1 = img1.astype(np.float64)
    if img2.dtype != np.float64:
        img2 = img2.astype(np.float64)

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    sigma = 1.5 # Sigma cho bộ lọc Gaussian, thường dùng giá trị này

    # Tính trung bình (mean)
    mu1 = cv2.GaussianBlur(img1, (win_size, win_size), sigma)
    mu2 = cv2.GaussianBlur(img2, (win_size, win_size), sigma)

    # Tính bình phương trung bình
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Tính phương sai (variance) và hiệp phương sai (covariance)
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (win_size, win_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (win_size, win_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (win_size, win_size), sigma) - mu1_mu2

    # --- Tính toán SSIM ---
    # Công thức: SSIM(x, y) = [ (2*μx*μy + C1) * (2*σxy + C2) ] / [ (μx² + μy² + C1) * (σx² + σy² + C2) ]
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    # Handle potential division by zero or very small denominator
    # Add a small epsilon to the denominator to prevent NaN/Inf
    epsilon = 1e-8
    ssim_map = numerator / (denominator + epsilon)

    # Clip values to avoid potential floating point issues leading to > 1.0
    ssim_map = np.clip(ssim_map, 0, 1)


    # Điểm SSIM trung bình trên toàn ảnh
    mssim = np.mean(ssim_map)

    return mssim


# --- Worker Thread for Image Processing ---
class ProcessingWorker(QThread):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str, str) # Message, Background Color
    save_error_signal = pyqtSignal(np.ndarray, str) # Frame, File Path
    ssim_signal = pyqtSignal(float) # Signal to emit the current SSIM score vs Norm

    def __init__(self, frame_queue, ref_images_provider, config_provider, compare_func):
        super().__init__()
        self.frame_queue = frame_queue
        # Use .copy() for thread safety snapshot
        self.get_ref_images = lambda: ref_images_provider().copy()
        self.get_config = config_provider # Function to get current config dict
        self.compare_images = compare_func # SẼ SỬ DỤNG compare_images CỦA APP CHÍNH
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
                if not self.running: # Check running flag again after timeout
                    break
                continue
            except Exception as e:
                self.log_signal.emit(f"❌ Error getting frame from queue: {e}")
                continue

            # Double check running flag after getting frame
            if not self.running:
                break

            try:
                current_config = self.get_config() # Get config dict (thread-safe access)
                ssim_threshold = current_config.get('ssim_threshold', DEFAULT_SSIM_THRESHOLD)
                error_cooldown = current_config.get('error_cooldown', DEFAULT_ERROR_COOLDOWN)
                error_folder = current_config.get('error_folder')

                ref_images = self.get_ref_images() # Gets a copy
                norm_image = ref_images.get(REF_NORM)
                shutdown_image = ref_images.get(REF_SHUTDOWN)
                fail_image = ref_images.get(REF_FAIL)

            except Exception as e:
                 self.log_signal.emit(f"❌ Error reading config/refs in worker: {e}")
                 time.sleep(1)
                 continue

            if not isinstance(norm_image, np.ndarray): # Kiểm tra kỹ hơn
                # Log this less frequently
                current_time_check = time.time()
                if not hasattr(self, 'last_norm_warning_time') or current_time_check - self.last_norm_warning_time > 60:
                    self.log_signal.emit("⚠️ Norm reference image not set or invalid. Skipping comparison.")
                    self.last_norm_warning_time = current_time_check
                self.ssim_signal.emit(-1.0) # Gửi tín hiệu SSIM không hợp lệ
                time.sleep(0.1) # Prevent busy-waiting
                continue
            else:
                # Reset warning time if norm image becomes valid
                if hasattr(self, 'last_norm_warning_time'):
                    delattr(self, 'last_norm_warning_time')


            # --- Image Comparison Logic (wrapped in try-except) ---
            try:
                # Gọi hàm compare_images (được truyền vào từ instance của ImageCheckerApp)
                is_match_norm, score_norm = self.compare_images(frame, norm_image, ssim_threshold)
                self.ssim_signal.emit(score_norm if score_norm is not None else -1.0)

                if is_match_norm:
                    current_time = time.time()
                    # Cập nhật trạng thái Normal định kỳ thay vì log
                    if current_time - last_status_normal_log_time > 5: # Update status label every 5s if normal
                        self.status_signal.emit("Normal", "lightgreen")
                        last_status_normal_log_time = current_time
                    # Quan trọng: Ngủ một chút để tránh CPU 100% khi không có gì thay đổi
                    # Tăng nhẹ thời gian ngủ để giảm tải CPU hơn nữa khi bình thường
                    time.sleep(0.1) # 100ms sleep
                    continue

                # --- Mismatch with Norm ---
                last_status_normal_log_time = 0 # Reset periodic normal updating
                status_msg = "Unknown Mismatch!"
                status_color = "orange"
                save_image = True
                error_subfolder = "unknown"
                log_msg = f"⚠️ Mismatch vs Norm (SSIM: {score_norm:.4f})." # Tăng độ chính xác hiển thị

                # So sánh với Shutdown nếu có
                if isinstance(shutdown_image, np.ndarray):
                    is_match_shutdown, score_shutdown = self.compare_images(frame, shutdown_image, ssim_threshold)
                    if is_match_shutdown:
                        status_msg = "Shutdown Detected"
                        status_color = "lightblue"
                        save_image = True # Có thể đặt False nếu không cần lưu ảnh shutdown
                        error_subfolder = "shutdown"
                        log_msg = f"ℹ️ Detected Shutdown state (SSIM vs Shutdown: {score_shutdown:.4f})."
                        # Chỉ log và cập nhật status khi phát hiện Shutdown
                        self.log_signal.emit(log_msg)
                        self.status_signal.emit(status_msg, status_color)
                        # Không cần continue ở đây, vẫn có thể là Fail

                # So sánh với Fail nếu có VÀ chưa xác định là Shutdown
                # (Tránh trường hợp ảnh Fail giống ảnh Shutdown)
                if isinstance(fail_image, np.ndarray) and error_subfolder == "unknown":
                    is_match_fail, score_fail = self.compare_images(frame, fail_image, ssim_threshold)
                    if is_match_fail:
                        status_msg = "FAIL State Detected!"
                        status_color = "red"
                        save_image = True
                        error_subfolder = "fail"
                        log_msg = f"❌ Detected FAIL state (SSIM vs Fail: {score_fail:.4f})."
                        # Chỉ log và cập nhật status khi phát hiện Fail
                        self.log_signal.emit(log_msg)
                        self.status_signal.emit(status_msg, status_color)

                # Nếu vẫn là "unknown" sau khi kiểm tra Shutdown và Fail, log mismatch gốc
                if error_subfolder == "unknown":
                     self.log_signal.emit(log_msg)
                     self.status_signal.emit(status_msg, status_color) # Cập nhật status là Unknown Mismatch

                # --- Save Error Image Logic ---
                current_time = time.time()
                if save_image and error_folder and (current_time - self.last_error_time > error_cooldown):
                    try:
                        specific_error_folder = os.path.join(error_folder, error_subfolder)
                        os.makedirs(specific_error_folder, exist_ok=True)
                        # Add milliseconds to filename for uniqueness
                        timestamp_ms = time.strftime('%Y%m%d_%H%M%S') + f"_{int((current_time - int(current_time)) * 1000):03d}"
                        file_name = f"{error_subfolder}_{timestamp_ms}.png"
                        file_path = os.path.join(specific_error_folder, file_name)
                        # Gửi tín hiệu để luồng chính lưu ảnh (an toàn hơn)
                        self.save_error_signal.emit(frame.copy(), file_path)
                        self.last_error_time = current_time
                        # Log việc *gửi tín hiệu* lưu ảnh, không phải đã lưu xong
                        # self.log_signal.emit(f"📤 Sent request to save image: {file_path}")
                    except Exception as e:
                        self.log_signal.emit(f"❌ Error preparing to save image to '{error_subfolder}': {e}")
                elif not error_folder and save_image:
                     # Log cảnh báo này ít thường xuyên hơn
                     current_time_check = time.time()
                     if not hasattr(self, 'last_save_folder_warning') or current_time_check - self.last_save_folder_warning > 60:
                        self.log_signal.emit("⚠️ Error folder not set. Cannot save mismatch.")
                        self.last_save_folder_warning = current_time_check

                # Thêm một khoảng nghỉ nhỏ sau khi xử lý mismatch để giảm tải CPU
                time.sleep(0.1) # 100ms sleep

            except Exception as e:
                # Catch errors specifically from the comparison/logic block
                self.log_signal.emit(f"❌ Error during image comparison/logic execution: {e}")
                import traceback
                self.log_signal.emit(traceback.format_exc()) # Log traceback để debug
                time.sleep(0.5) # Prevent spamming if error repeats quickly

        self.log_signal.emit("⚙️ Processing thread finished.")

    def stop(self):
        self.running = False
        self.log_signal.emit("⚙️ Stopping processing thread...")
        # No need to join here, wait() in main thread handles that


# --- Main Application Window ---
class ImageCheckerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        self.ref_images = {REF_NORM: None, REF_SHUTDOWN: None, REF_FAIL: None}
        self.webcam_roi = None
        self.processing = False
        self.error_folder = None
        self.log_file_path = None
        self.pixmap_item = None # To hold the graphics pixmap item

        # --- Internal state for thread-safe config access ---
        self._current_ssim_threshold = DEFAULT_SSIM_THRESHOLD
        self._current_error_cooldown = DEFAULT_ERROR_COOLDOWN
        # self.error_folder is already an attribute, safe to read directly by worker

        # Configuration dictionary (loaded/saved)
        self.config = {
            'ssim_threshold': self._current_ssim_threshold,
            'error_cooldown': self._current_error_cooldown,
            'error_folder': None,
            'ref_paths': {REF_NORM: None, REF_SHUTDOWN: None, REF_FAIL: None},
            'webcam_roi': None,
        }

        self.frame_queue = Queue(maxsize=10) # Keep queue size reasonable

        # --- Worker Thread Setup ---
        self.processing_worker = ProcessingWorker(
            self.frame_queue,
            lambda: self.ref_images, # Lambda provides current dict
            lambda: self.get_current_config_for_worker(), # Lambda provides current config dict
            self.compare_images # Pass the compare_images method of this instance
        )
        self.processing_worker.log_signal.connect(self.log_activity)
        self.processing_worker.status_signal.connect(self.update_status_label)
        self.processing_worker.save_error_signal.connect(self.save_error_image_from_thread)
        self.processing_worker.ssim_signal.connect(self.update_ssim_display)

        self.init_ui() # Initialize UI elements
        self.load_config() # Load config after UI elements exist
        self.log_activity("Ứng dụng khởi động.")
        # Update UI elements based on potentially loaded config AFTER load_config
        self.update_all_ui_elements()


    def get_current_config_for_worker(self):
        # **THREAD-SAFE**: Returns config values from internal attributes, not widgets
        return {
            'ssim_threshold': self._current_ssim_threshold,
            'error_cooldown': self._current_error_cooldown,
            'error_folder': self.error_folder,
        }

    @QtCore.pyqtSlot(float)
    def _update_threshold_config(self, value):
        # This slot receives value changes FROM the spinbox
        if self._current_ssim_threshold != value:
             self._current_ssim_threshold = value
             self.log_activity(f"⚙️ Ngưỡng SSIM được cập nhật thành: {value:.3f}")
             # Optional: Save immediately? Or only on exit/explicit save?
             # self.save_config()

    @QtCore.pyqtSlot(int)
    def _update_cooldown_config(self, value):
         # This slot receives value changes FROM the spinbox
        if self._current_error_cooldown != value:
            self._current_error_cooldown = value
            self.log_activity(f"⚙️ Cooldown lỗi được cập nhật thành: {value} giây")
            # self.save_config()

    # --- UI Initialization (using setGeometry) ---
    def init_ui(self):
        self.setWindowTitle("Hệ thống kiểm tra hình ảnh - v2.2 (OpenCV SSIM, Fixed Layout)")
        # Use geometry from the second example
        self.setGeometry(100, 100, 1246, 682)
        # Create a central widget to hold everything (needed even without layout managers)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # --- Webcam View ---
        self.scene = QGraphicsScene(self)
        self.graphicsView = QGraphicsView(self.scene, central_widget) # Parent is central_widget
        # Use geometry from the second example
        self.graphicsView.setGeometry(0, 10, 640, 360)
        self.graphicsView.setStyleSheet("border: 1px solid black;")
        self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray))
        # No size policy needed with setGeometry

        # --- Log Display ---
        # Create QLabel like the first example, then use QTextEdit like the second
        log_label = QLabel("Log Hoạt Động:", central_widget)
        log_label.setGeometry(660, 10, 150, 20) # Position the label slightly above the text edit

        self.log_text_edit = QTextEdit(central_widget) # Use QTextEdit for scrollable logs
        self.log_text_edit.setGeometry(660, 35, 560, 250) # Adjusted Y pos to be below label
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setStyleSheet("border: 1px solid black; padding: 5px; background-color: white; font-family: Consolas, monospace; font-size: 10pt;") # Monospace font

        # --- Status Label ---
        self.process_label = QLabel("Trạng thái: Chờ", central_widget)
        # Use geometry from the second example
        self.process_label.setGeometry(660, 300, 560, 40)
        self.process_label.setAlignment(Qt.AlignCenter)
        self.process_label.setStyleSheet("border: 1px solid black; padding: 5px; background-color: lightgray; font-weight: bold; border-radius: 3px;")

        # --- SSIM Display Label ---
        self.ssim_label = QLabel("SSIM vs Norm: N/A", central_widget)
        # Position it below the status label, adjusting Y coordinate
        self.ssim_label.setGeometry(660, 345, 560, 30)
        self.ssim_label.setAlignment(Qt.AlignCenter)
        self.ssim_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")

        # --- Buttons (using positions from the second example) ---
        # Button Size from second example: (201, 31)
        btn_width = 201
        btn_height = 31

        # Row 1: Webcam
        self.ONCam = self.create_button("📷 Bật Webcam")
        self.ONCam.setGeometry(20, 390, btn_width, btn_height)
        self.ONCam.clicked.connect(self.start_webcam)

        self.OFFCam = self.create_button("🚫 Tắt Webcam")
        self.OFFCam.setGeometry(270, 390, btn_width, btn_height)
        self.OFFCam.clicked.connect(self.stop_webcam)
        self.OFFCam.setEnabled(False)

        # Row 2: Reference Images
        self.SettingButton_Norm = self.create_button("✅ Ảnh Norm")
        self.SettingButton_Norm.setGeometry(20, 430, btn_width, btn_height)
        self.SettingButton_Norm.clicked.connect(lambda: self.load_reference_image(REF_NORM))

        self.SettingButton_Shut = self.create_button("⛔️ Ảnh Shutdown")
        self.SettingButton_Shut.setGeometry(270, 430, btn_width, btn_height)
        self.SettingButton_Shut.clicked.connect(lambda: self.load_reference_image(REF_SHUTDOWN))

        self.SettingButton_Fail = self.create_button("❌ Ảnh Fail")
        self.SettingButton_Fail.setGeometry(520, 430, btn_width, btn_height) # Adjusted X pos slightly
        self.SettingButton_Fail.clicked.connect(lambda: self.load_reference_image(REF_FAIL))

        # Row 3: ROI and Processing Toggle
        self.SettingButton_ROI_Webcam = self.create_button("✂️ Chọn ROI")
        self.SettingButton_ROI_Webcam.setGeometry(20, 470, btn_width, btn_height)
        self.SettingButton_ROI_Webcam.clicked.connect(self.select_webcam_roi)
        self.SettingButton_ROI_Webcam.setEnabled(False)

        self.ToggleProcessingButton = self.create_button("▶️ Bắt đầu Xử lý")
        # Position based on second example's "ToggleROI"
        self.ToggleProcessingButton.setGeometry(520, 470, btn_width, btn_height)
        self.ToggleProcessingButton.clicked.connect(self.toggle_processing)

        # Row 4: Save Folder and Exit
        self.SaveButton = self.create_button("📁 Thư mục Lưu lỗi")
        # Position based on second example's "SaveButton"
        self.SaveButton.setGeometry(520, 510, btn_width, btn_height)
        self.SaveButton.clicked.connect(self.select_error_folder)

        self.ExitButton = self.create_button("🚪 Thoát")
        # Position based on second example's "ExitButton"
        self.ExitButton.setGeometry(20, 510, btn_width, btn_height)
        self.ExitButton.clicked.connect(self.close_application)


        # --- SpinBoxes for Settings ---
        # Place these manually in the remaining space on the right
        label_ssim = QLabel("Ngưỡng SSIM:", central_widget)
        label_ssim.setGeometry(660, 400, 110, 31)

        self.ssimThresholdSpinBox = QDoubleSpinBox(central_widget)
        self.ssimThresholdSpinBox.setGeometry(780, 400, 100, 31) # Position next to label
        self.ssimThresholdSpinBox.setRange(0.1, 1.0)
        self.ssimThresholdSpinBox.setSingleStep(0.01)
        self.ssimThresholdSpinBox.setValue(self._current_ssim_threshold) # Init with internal value
        self.ssimThresholdSpinBox.setDecimals(3)
        self.ssimThresholdSpinBox.valueChanged.connect(self._update_threshold_config) # Connect signal

        label_cooldown = QLabel("Cooldown Lỗi (s):", central_widget)
        label_cooldown.setGeometry(660, 440, 110, 31)

        self.cooldownSpinBox = QSpinBox(central_widget)
        self.cooldownSpinBox.setGeometry(780, 440, 100, 31) # Position next to label
        self.cooldownSpinBox.setRange(1, 300)
        self.cooldownSpinBox.setSingleStep(1)
        self.cooldownSpinBox.setValue(self._current_error_cooldown) # Init with internal value
        self.cooldownSpinBox.valueChanged.connect(self._update_cooldown_config) # Connect signal

        # --- Initial Button States ---
        self.update_button_styles()
        self.update_toggle_button_text()


    def create_button(self, text):
        # Simplified create_button, relies on setGeometry being called externally
        # But keeps the styling from the first example
        button = QPushButton(text, self.centralWidget()) # Parent is central widget
        # CSS styling for buttons
        style_sheet = """
            QPushButton {
                background-color: white;
                padding: 6px;
                text-align: center; /* Center text for fixed size buttons */
                border: 1px solid #ccc;
                border-radius: 3px;
                /* min-height: 35px; Let setGeometry control height */
            }
            QPushButton:hover {
                background-color: #e8f0fe; /* Light blue on hover */
            }
            QPushButton:pressed {
                background-color: #d0e0f8; /* Slightly darker blue when pressed */
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #a0a0a0;
                border-color: #d0d0d0;
            }
        """
        button.setStyleSheet(style_sheet)
        return button


    # --- Configuration Loading/Saving ---
    def save_config(self):
        # Save internal state values, not directly from widgets
        self.config['ssim_threshold'] = self._current_ssim_threshold
        self.config['error_cooldown'] = self._current_error_cooldown
        self.config['error_folder'] = self.error_folder
        # Convert tuple ROI to list for JSON serialization
        self.config['webcam_roi'] = list(self.webcam_roi) if isinstance(self.webcam_roi, tuple) else None
        # Correctly save ref_paths from self.config['ref_paths']
        self.config['ref_paths'] = {key: path for key, path in self.config['ref_paths'].items()}

        try:
            # Ensure directory exists before writing config
            config_dir = os.path.dirname(CONFIG_FILE_NAME)
            if config_dir and not os.path.exists(config_dir):
                 os.makedirs(config_dir, exist_ok=True)

            with open(CONFIG_FILE_NAME, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            # Log saving only if it wasn't triggered by exiting
            # self.log_activity(f"💾 Đã lưu cấu hình vào {CONFIG_FILE_NAME}")
        except Exception as e:
            self.log_activity(f"❌ Lỗi khi lưu cấu hình: {e}")
            QMessageBox.critical(self, "Lỗi Lưu Cấu Hình", f"Không thể lưu cấu hình vào {CONFIG_FILE_NAME}\n{e}")


    def load_config(self):
        if not os.path.exists(CONFIG_FILE_NAME):
            self.log_activity(f"📄 Không tìm thấy file cấu hình {CONFIG_FILE_NAME}. Sử dụng giá trị mặc định.")
            # Initialize internal state from defaults
            self._current_ssim_threshold = DEFAULT_SSIM_THRESHOLD
            self._current_error_cooldown = DEFAULT_ERROR_COOLDOWN
            self.error_folder = None
            self.webcam_roi = None
            self.log_file_path = None

            # Ensure config dictionary matches defaults
            self.config['ssim_threshold'] = self._current_ssim_threshold
            self.config['error_cooldown'] = self._current_error_cooldown
            self.config['error_folder'] = self.error_folder
            self.config['webcam_roi'] = self.webcam_roi
            self.config['ref_paths'] = {REF_NORM: None, REF_SHUTDOWN: None, REF_FAIL: None}
            return # Exit loading process

        try:
            with open(CONFIG_FILE_NAME, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            # --- Load values and validate ---
            # SSIM Threshold
            try:
                threshold = float(loaded_config.get('ssim_threshold', DEFAULT_SSIM_THRESHOLD))
                self._current_ssim_threshold = max(0.1, min(1.0, threshold)) # Clamp to valid range
            except (ValueError, TypeError):
                 self._current_ssim_threshold = DEFAULT_SSIM_THRESHOLD
                 self.log_activity(f"⚠️ Giá trị ssim_threshold không hợp lệ trong config, dùng mặc định {DEFAULT_SSIM_THRESHOLD}")

            # Error Cooldown
            try:
                cooldown = int(loaded_config.get('error_cooldown', DEFAULT_ERROR_COOLDOWN))
                self._current_error_cooldown = max(1, min(300, cooldown)) # Clamp to valid range
            except (ValueError, TypeError):
                self._current_error_cooldown = DEFAULT_ERROR_COOLDOWN
                self.log_activity(f"⚠️ Giá trị error_cooldown không hợp lệ trong config, dùng mặc định {DEFAULT_ERROR_COOLDOWN}")

            # Error Folder
            loaded_folder = loaded_config.get('error_folder')
            if loaded_folder and isinstance(loaded_folder, str) and os.path.isdir(loaded_folder):
                 # Check write access
                 if os.access(loaded_folder, os.W_OK):
                     self.error_folder = loaded_folder
                 else:
                     self.log_activity(f"⚠️ Không có quyền ghi vào thư mục lỗi đã lưu: {loaded_folder}. Đặt lại.")
                     self.error_folder = None
            elif loaded_folder: # Path existed but wasn't a valid directory or not accessible
                 self.log_activity(f"⚠️ Thư mục lỗi đã lưu '{loaded_folder}' không hợp lệ hoặc không truy cập được. Đặt lại.")
                 self.error_folder = None
            else:
                 self.error_folder = None # Not set in config

            # Webcam ROI
            loaded_roi = loaded_config.get('webcam_roi')
            if isinstance(loaded_roi, list) and len(loaded_roi) == 4:
                try:
                     # Ensure all are positive integers
                     roi_tuple = tuple(int(x) for x in loaded_roi)
                     if all(isinstance(v, int) and v >= 0 for v in roi_tuple) and roi_tuple[2] > 0 and roi_tuple[3] > 0:
                         self.webcam_roi = roi_tuple
                     else:
                          self.log_activity(f"⚠️ Giá trị ROI không hợp lệ trong config: {loaded_roi}. Đặt lại.")
                          self.webcam_roi = None
                except (ValueError, TypeError):
                     self.log_activity(f"⚠️ Giá trị ROI không hợp lệ trong config: {loaded_roi}. Đặt lại.")
                     self.webcam_roi = None
            elif loaded_roi is not None: # Exists but not a list of 4 elements
                 self.log_activity(f"⚠️ Định dạng ROI không hợp lệ trong config: {loaded_roi}. Đặt lại.")
                 self.webcam_roi = None
            else:
                 self.webcam_roi = None # Not set in config


            # Update the main config dictionary (for saving later)
            self.config['ssim_threshold'] = self._current_ssim_threshold
            self.config['error_cooldown'] = self._current_error_cooldown
            self.config['error_folder'] = self.error_folder
            self.config['webcam_roi'] = list(self.webcam_roi) if self.webcam_roi else None # Save as list

            # Set Log File Path (only if error folder is valid)
            if self.error_folder:
                self.log_file_path = os.path.join(self.error_folder, LOG_FILE_NAME)
            else:
                self.log_file_path = None


            # --- Load reference image paths and images ---
            loaded_ref_paths = loaded_config.get('ref_paths', {})
            self.config['ref_paths'] = loaded_ref_paths # Store paths in config dict
            # Clear existing images before loading new ones
            self.ref_images = {REF_NORM: None, REF_SHUTDOWN: None, REF_FAIL: None}
            for key, path in loaded_ref_paths.items():
                if key not in self.ref_images: continue # Skip unknown keys
                if path and isinstance(path, str) and os.path.exists(path) and os.path.isfile(path):
                    try:
                        # Use imdecode to handle potential path issues with special chars
                        img_bytes = np.fromfile(path, dtype=np.uint8)
                        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                        if img is not None:
                            self.ref_images[key] = img
                            # Logged later by update_all_ui_elements
                        else:
                             self.log_activity(f"⚠️ Không thể đọc ảnh '{key}' từ: {path} (có thể file bị hỏng hoặc không phải ảnh).")
                             self.config['ref_paths'][key] = None # Clear invalid path
                    except Exception as e:
                        self.log_activity(f"❌ Lỗi khi tải lại ảnh '{key}' từ {path}: {e}")
                        self.config['ref_paths'][key] = None
                elif path: # Path exists in config but is invalid (doesn't exist, not a file, etc.)
                    self.log_activity(f"⚠️ Đường dẫn ảnh tham chiếu '{key}' không hợp lệ hoặc không tồn tại: {path}")
                    self.config['ref_paths'][key] = None

            self.log_activity(f"💾 Đã tải cấu hình từ {CONFIG_FILE_NAME}")

        except json.JSONDecodeError as e:
             self.log_activity(f"❌ Lỗi giải mã JSON trong {CONFIG_FILE_NAME}: {e}. Đặt lại về mặc định.")
             self.reset_to_defaults()
        except Exception as e: # Catch other potential errors during loading/validation
            self.log_activity(f"❌ Lỗi không xác định khi tải cấu hình: {e}. Đặt lại về mặc định.")
            import traceback
            self.log_activity(traceback.format_exc()) # Log traceback for debugging
            self.reset_to_defaults()

        # Explicitly call UI update after loading or resetting
        # This ensures spinboxes etc. show the loaded/default values
        self.update_all_ui_elements()


    def reset_to_defaults(self):
        """Resets internal state and config dictionary to default values."""
        self._current_ssim_threshold = DEFAULT_SSIM_THRESHOLD
        self._current_error_cooldown = DEFAULT_ERROR_COOLDOWN
        self.error_folder = None
        self.log_file_path = None
        self.webcam_roi = None
        self.ref_images = {REF_NORM: None, REF_SHUTDOWN: None, REF_FAIL: None}
        # Reset config dictionary too
        self.config = {
            'ssim_threshold': self._current_ssim_threshold,
            'error_cooldown': self._current_error_cooldown,
            'error_folder': None,
            'ref_paths': {REF_NORM: None, REF_SHUTDOWN: None, REF_FAIL: None},
            'webcam_roi': None,
        }
        self.log_activity("🔄 Đã đặt lại cấu hình về mặc định.")
        # No need to call update_all_ui_elements here, it's called by load_config after reset


    def update_all_ui_elements(self):
        """Updates UI widgets based on current internal state/config."""
        # Block signals while setting values to prevent immediate config updates/logging
        self.ssimThresholdSpinBox.blockSignals(True)
        self.cooldownSpinBox.blockSignals(True)

        self.ssimThresholdSpinBox.setValue(self._current_ssim_threshold)
        self.cooldownSpinBox.setValue(self._current_error_cooldown)

        self.ssimThresholdSpinBox.blockSignals(False)
        self.cooldownSpinBox.blockSignals(False)

        self.update_button_styles() # Update styles for ref, roi, folder buttons
        self.update_toggle_button_text() # Update start/stop button

        # Log loaded references after UI updates button text
        logged_keys = set() # Track logged keys to avoid duplicates if path exists but load failed
        for key, path in self.config['ref_paths'].items():
             if key not in self.ref_images: continue
             if path and self.ref_images.get(key) is None and key not in logged_keys:
                 # Logged during load_config if decode failed
                 # self.log_activity(f"⚠️ Không tải được ảnh tham chiếu '{key}' từ: {path}")
                 logged_keys.add(key)
             elif path and self.ref_images.get(key) is not None and key not in logged_keys:
                 # Use basename for cleaner log
                 self.log_activity(f"✅ Đã tải ảnh tham chiếu '{key}': {os.path.basename(path)}")
                 logged_keys.add(key)
             elif key not in logged_keys: # No path set
                 # Optionally log that ref image is not set
                 # self.log_activity(f"ℹ️ Ảnh tham chiếu '{key}' chưa được đặt.")
                 logged_keys.add(key)


        if self.config.get('error_folder') and not self.error_folder:
            # Logged during load_config if folder was invalid/inaccessible
            # self.log_activity(f"⚠️ Thư mục lỗi '{self.config.get('error_folder')}' không hợp lệ.")
            pass # Already logged
        elif self.error_folder:
             self.log_activity(f"📁 Thư mục lỗi: {self.error_folder}")
             if self.log_file_path:
                self.log_activity(f"📄 Log sẽ lưu tại: {self.log_file_path}")

        if self.webcam_roi:
            self.log_activity(f"✂️ ROI Webcam: {self.webcam_roi}")
        # else:
            # self.log_activity(f"✂️ ROI Webcam: Chưa đặt") # Maybe too verbose


    @QtCore.pyqtSlot(str)
    def log_activity(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{timestamp} - {message}"

        # Ensure UI update happens on the main thread
        if self.log_text_edit.thread() != QtCore.QThread.currentThread():
             QtCore.QMetaObject.invokeMethod(self.log_text_edit, "append", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, full_message))
             QtCore.QMetaObject.invokeMethod(self.log_text_edit, "ensureCursorVisible", QtCore.Qt.QueuedConnection)
        else:
             self.log_text_edit.append(full_message)
             self.log_text_edit.ensureCursorVisible() # Scroll to bottom


        if self.log_file_path:
            try:
                # Ensure directory exists before writing
                log_dir = os.path.dirname(self.log_file_path)
                if log_dir and not os.path.exists(log_dir):
                     try:
                         os.makedirs(log_dir, exist_ok=True)
                         # Log directory creation to UI only
                         self.log_text_edit.append(f"{timestamp} - ℹ️ Created log directory: {log_dir}")
                         self.log_text_edit.ensureCursorVisible()
                     except Exception as mkdir_e:
                          # Log dir creation error to UI
                          err_msg = f"{timestamp} - ❌ ERROR creating log directory '{log_dir}': {mkdir_e}"
                          print(err_msg) # Console fallback
                          self.log_text_edit.append(err_msg)
                          self.log_text_edit.ensureCursorVisible()
                          self.log_file_path = None # Stop trying to write if dir fails
                          return # Don't attempt to write file

                if self.log_file_path: # Check again, might be None if mkdir failed
                    with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                        log_file.write(full_message + "\n")
            except Exception as e:
                # Avoid infinite loop if logging the error itself fails
                error_msg = f"{timestamp} - ❌ ERROR writing to log file '{self.log_file_path}': {e}"
                print(error_msg) # Print to console as fallback
                # Append error to UI log
                if self.log_text_edit.thread() != QtCore.QThread.currentThread():
                    QtCore.QMetaObject.invokeMethod(self.log_text_edit, "append", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, error_msg))
                    QtCore.QMetaObject.invokeMethod(self.log_text_edit, "ensureCursorVisible", QtCore.Qt.QueuedConnection)
                else:
                    self.log_text_edit.append(error_msg)
                    self.log_text_edit.ensureCursorVisible()
                # Disable further file logging for this session to prevent spam
                self.log_file_path = None
                self.update_status_label("Log file error!", "red")


    @QtCore.pyqtSlot(str, str)
    def update_status_label(self, message, background_color="lightgray"):
        # Ensure UI update happens on the main thread
        def _update():
            self.process_label.setText(f"Trạng thái: {message}")
            self.process_label.setStyleSheet(
                f"border: 1px solid black; padding: 5px; background-color: {background_color}; color: black; font-weight: bold; border-radius: 3px;"
            )
        if self.process_label.thread() != QtCore.QThread.currentThread():
             QtCore.QMetaObject.invokeMethod(self, "_update", QtCore.Qt.QueuedConnection)
        else:
             _update()


    @QtCore.pyqtSlot(float)
    def update_ssim_display(self, score):
         # Ensure UI update happens on the main thread
        def _update():
            if score is not None and score >= 0:
                self.ssim_label.setText(f"SSIM vs Norm: {score:.4f}") # Show 4 decimal places
            else:
                self.ssim_label.setText("SSIM vs Norm: N/A")

        if self.ssim_label.thread() != QtCore.QThread.currentThread():
             QtCore.QMetaObject.invokeMethod(self, "_update", QtCore.Qt.QueuedConnection)
        else:
             _update()


    # --- Button Style Updates ---
    def _set_button_style(self, button, base_text, icon, state_text="", background_color="white", text_color="black"):
         """Helper to set button text and style"""
         full_text = f"{icon} {base_text}"
         if state_text:
             full_text += f" ({state_text})"
         button.setText(full_text)

         # Base style sheet from create_button
         style_sheet = """
             QPushButton {{ background-color: {bg}; color: {fg}; padding: 6px; text-align: center; border: 1px solid #ccc; border-radius: 3px; }}
             QPushButton:hover {{ background-color: #e8f0fe; }}
             QPushButton:pressed {{ background-color: #d0e0f8; }}
             QPushButton:disabled {{ background-color: #f0f0f0; color: #a0a0a0; border-color: #d0d0d0; }}
         """.format(bg=background_color, fg=text_color)
         button.setStyleSheet(style_sheet)

    def update_button_styles(self):
        """Updates appearance of buttons based on current state."""
        # Reference image buttons
        base_texts_refs = {"Norm": "Ảnh Norm", "Shutdown": "Ảnh Shutdown", "Fail": "Ảnh Fail"}
        icons_refs = {"Norm": "✅", "Shutdown": "⛔️", "Fail": "❌"}
        buttons_refs = {REF_NORM: self.SettingButton_Norm, REF_SHUTDOWN: self.SettingButton_Shut, REF_FAIL: self.SettingButton_Fail}

        for key, btn in buttons_refs.items():
            base_text = base_texts_refs[key]
            icon = icons_refs[key]
            if self.ref_images.get(key) is not None:
                self._set_button_style(btn, base_text, icon, state_text="Đã chọn", background_color="lightgreen")
            else:
                self._set_button_style(btn, base_text, icon, background_color="white")

        # ROI button
        icon_roi = "✂️"
        base_text_roi = "Chọn ROI"
        if self.webcam_roi:
             self._set_button_style(self.SettingButton_ROI_Webcam, base_text_roi, icon_roi, state_text="Đã chọn", background_color="lightblue")
        else:
             self._set_button_style(self.SettingButton_ROI_Webcam, base_text_roi, icon_roi, background_color="white")
        # Enable/disable ROI button based on webcam status (and processing status)
        self.SettingButton_ROI_Webcam.setEnabled(self.cap is not None and self.cap.isOpened() and not self.processing)


        # Error folder button
        icon_folder = "📁"
        base_text_folder = "Thư mục Lưu lỗi"
        if self.error_folder and os.path.isdir(self.error_folder):
            self._set_button_style(self.SaveButton, base_text_folder, icon_folder, state_text="Đã chọn", background_color="lightblue")
        else:
            self._set_button_style(self.SaveButton, base_text_folder, icon_folder, background_color="white")


    def update_toggle_button_text(self):
        """Updates the Start/Stop processing button."""
        if self.processing:
            self._set_button_style(self.ToggleProcessingButton, "Dừng Xử lý", "⏹", background_color="orange")
        else:
            self._set_button_style(self.ToggleProcessingButton, "Bắt đầu Xử lý", "▶️", background_color="lightgreen")


    # --- Webcam Handling ---
    def start_webcam(self):
        if self.cap is not None and self.cap.isOpened():
            self.log_activity("⚠️ Webcam đã được bật.")
            return

        try:
            # Try common backends
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None] # None is default
            selected_backend = None
            for backend in backends:
                api = f" (API: {backend})" if backend is not None else " (API: Default)"
                self.log_activity(f"ℹ️ Đang thử mở webcam với backend {api}...")
                if backend is not None:
                    self.cap = cv2.VideoCapture(0, backend)
                else:
                    self.cap = cv2.VideoCapture(0)

                if self.cap and self.cap.isOpened():
                    selected_backend = backend
                    self.log_activity(f"✅ Webcam mở thành công với backend {api}.")
                    break # Exit loop on success
                else:
                    if self.cap: self.cap.release() # Release if open failed
                    self.cap = None
                    # self.log_activity(f"⚠️ Mở webcam thất bại với backend {api}.") # Reduce verbosity

            if self.cap is None or not self.cap.isOpened():
                raise IOError("Không thể mở webcam với bất kỳ backend nào được thử.")

            # --- Configure Opened Webcam ---
            # Set desired FPS (e.g., 10-15 FPS is often enough)
            fps_request = 15.0
            set_fps_ok = self.cap.set(cv2.CAP_PROP_FPS, fps_request)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            if actual_fps <= 0:
                 # self.log_activity("⚠️ Không thể lấy FPS thực tế, sử dụng interval mặc định 100ms.")
                 actual_fps = 10 # Fallback FPS if get fails
            timer_interval = max(33, int(1000 / actual_fps)) # Ensure reasonable minimum interval (e.g., ~30 FPS max)

            # Log details
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            backend_name = self.cap.getBackendName()
            self.log_activity(f"🚀 Webcam đã bật (Backend: {backend_name}, Res: {w}x{h}, FPS: {actual_fps:.1f}, Interval: {timer_interval}ms)")

            self.frame_timer.start(timer_interval)
            self.ONCam.setEnabled(False)
            self.OFFCam.setEnabled(True)
            # Enable ROI button only if not currently processing
            self.SettingButton_ROI_Webcam.setEnabled(not self.processing)
            self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.black)) # Black background when running

        except Exception as e:
            error_msg = f"❌ Lỗi nghiêm trọng khi bật webcam: {e}"
            self.log_activity(error_msg)
            QMessageBox.critical(self, "Lỗi Webcam", f"Không thể mở webcam.\nChi tiết: {e}")
            if self.cap: self.cap.release()
            self.cap = None
            self.ONCam.setEnabled(True)
            self.OFFCam.setEnabled(False)
            self.SettingButton_ROI_Webcam.setEnabled(False)
            self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray)) # Gray background on error/off


    def update_frame(self):
        if self.cap is None or not self.cap.isOpened(): return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            current_time_check = time.time()
            if not hasattr(self, 'last_frame_read_error_time') or current_time_check - self.last_frame_read_error_time > 5:
                self.log_activity("⚠️ Lỗi đọc khung hình từ webcam (ret=False or frame=None).")
                self.last_frame_read_error_time = current_time_check
            return

        try:
            # --- Prepare Frames ---
            display_frame = frame.copy()
            processing_frame = frame # Frame to be processed (might be ROI)

            # --- Apply and Draw ROI ---
            if self.webcam_roi:
                x, y, w, h = self.webcam_roi
                fh, fw = frame.shape[:2]
                # Clamp ROI coordinates to be within frame boundaries
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(fw, x + w), min(fh, y + h)

                if x2 > x1 and y2 > y1: # Check if ROI has valid dimensions after clamping
                    # Get the ROI slice for processing
                    processing_frame = frame[y1:y2, x1:x2]
                    # Draw the rectangle on the display frame
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                     # Log invalid ROI less frequently
                     current_time_check = time.time()
                     if not hasattr(self, 'last_invalid_roi_warn_time') or current_time_check - self.last_invalid_roi_warn_time > 10:
                         self.log_activity(f"⚠️ ROI {self.webcam_roi} không hợp lệ hoặc nằm ngoài khung hình {fw}x{fh}. Sử dụng toàn bộ khung hình.")
                         self.last_invalid_roi_warn_time = current_time_check
                     # Option: Reset invalid ROI automatically?
                     # self.webcam_roi = None
                     # self.config['webcam_roi'] = None
                     # self.update_button_styles()
                     processing_frame = frame # Fallback to full frame

            # --- Display Frame ---
            # Convert color for Qt
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h_disp, w_disp, ch = frame_rgb.shape
            bytes_per_line = ch * w_disp
            qt_img = QtGui.QImage(frame_rgb.data, w_disp, h_disp, bytes_per_line, QtGui.QImage.Format_RGB888)

            # Scale pixmap to fit the graphicsView while keeping aspect ratio
            # Use graphicsView's viewport size for accurate scaling
            view_size = self.graphicsView.viewport().size()
            pixmap = QtGui.QPixmap.fromImage(qt_img).scaled(
                view_size - QtCore.QSize(2,2), # Subtract border allowance
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            # Update QGraphicsScene efficiently
            if self.pixmap_item is None:
                self.pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
                self.scene.addItem(self.pixmap_item)
            else:
                self.pixmap_item.setPixmap(pixmap)

            # Ensure the view is centered/fitted (might only be needed once or after resize)
            # self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

            # --- Queue Frame for Processing ---
            if self.processing:
                if not self.frame_queue.full():
                    try:
                        # Put a *copy* of the processing_frame into the queue
                        # This is crucial if processing_frame is an ROI slice (a view)
                        self.frame_queue.put(processing_frame.copy(), block=False)
                    except Exception as e:
                         current_time_check = time.time()
                         if not hasattr(self, 'last_queue_put_error_time') or current_time_check - self.last_queue_put_error_time > 5:
                             self.log_activity(f"❌ Error putting frame in queue: {e}")
                             self.last_queue_put_error_time = current_time_check
                else:
                    # Log queue full less frequently
                    current_time_check = time.time()
                    if not hasattr(self, 'last_queue_full_warn_time') or current_time_check - self.last_queue_full_warn_time > 5:
                        self.log_activity("⚠️ Hàng đợi xử lý đầy, khung hình bị bỏ qua.")
                        self.last_queue_full_warn_time = current_time_check

        except Exception as e:
            current_time_check = time.time()
            if not hasattr(self, 'last_update_frame_error_time') or current_time_check - self.last_update_frame_error_time > 5:
                 self.log_activity(f"❌ Lỗi trong update_frame: {e}")
                 import traceback
                 print(traceback.format_exc()) # Console for debug
                 self.last_update_frame_error_time = current_time_check


    def stop_webcam(self):
        if self.cap and self.cap.isOpened():
            self.frame_timer.stop()
            # Short delay seems unnecessary if cap.release() is blocking enough
            # time.sleep(0.1)
            self.cap.release()
            self.cap = None
            self.scene.clear()
            self.pixmap_item = None # Reset pixmap item holder
            self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray)) # Reset background
            self.log_activity("🚫 Webcam đã tắt.")
            self.ONCam.setEnabled(True)
            self.OFFCam.setEnabled(False)
            self.SettingButton_ROI_Webcam.setEnabled(False) # ROI button requires webcam
            # If processing was running, stop it too
            if self.processing:
                self.toggle_processing() # Call toggle to stop processing cleanly
        elif self.cap is None:
             self.log_activity("ℹ️ Webcam chưa được bật hoặc đã tắt.")
        else: # cap exists but isOpened()=False (shouldn't happen often with checks)
             self.log_activity("ℹ️ Webcam không trong trạng thái hoạt động.")
             self.cap = None # Reset just in case
             self.ONCam.setEnabled(True)
             self.OFFCam.setEnabled(False)
             self.SettingButton_ROI_Webcam.setEnabled(False)


    # --- Reference Image and ROI/Folder Selection ---
    def load_reference_image(self, img_type):
        options = QFileDialog.Options()
        # Determine starting directory
        start_dir = ""
        saved_path = self.config['ref_paths'].get(img_type)
        if saved_path and os.path.exists(saved_path) and os.path.isfile(saved_path):
             start_dir = os.path.dirname(saved_path)
        elif self.error_folder and os.path.isdir(self.error_folder): # Fallback to error folder
             start_dir = self.error_folder
        else:
             start_dir = os.path.expanduser("~") # Fallback to home

        file_path, _ = QFileDialog.getOpenFileName(self, f"Chọn ảnh {img_type}", start_dir,
                                                   "Images (*.png *.jpg *.jpeg *.bmp *.jfif *.webp);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                # Use np.fromfile and cv2.imdecode for robust path handling
                img_bytes = np.fromfile(file_path, dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR) # Load as color image

                if img is None: raise ValueError("File không phải là ảnh hợp lệ hoặc không thể đọc.")

                self.ref_images[img_type] = img
                self.config['ref_paths'][img_type] = file_path # Store absolute path
                self.update_button_styles() # Update button appearance
                self.log_activity(f"✅ Đã tải ảnh tham chiếu '{img_type}' từ: {os.path.basename(file_path)}")
                self.save_config() # Save config immediately after successful load

            except Exception as e:
                self.log_activity(f"❌ Lỗi khi tải ảnh '{img_type}': {e}")
                QMessageBox.warning(self, "Lỗi Tải Ảnh", f"Không thể tải ảnh: {file_path}\n{e}")
                # Clear reference if loading failed
                self.ref_images[img_type] = None
                self.config['ref_paths'][img_type] = None
                self.update_button_styles()
                # Save config even on failure to clear the invalid path
                self.save_config()


    def select_webcam_roi(self):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "Chưa bật Webcam", "Bạn cần bật webcam để chọn ROI.")
            return

        # Temporarily stop frame updates for ROI selection
        was_running = self.frame_timer.isActive()
        if was_running: self.frame_timer.stop()
        time.sleep(0.1) # Short pause

        ret, frame = self.cap.read()

        # Restart timer immediately after getting the frame, before ROI window
        if was_running and self.cap and self.cap.isOpened():
             self.frame_timer.start()

        if not ret or frame is None:
            QMessageBox.warning(self, "Lỗi Khung hình", "Không thể lấy khung hình từ webcam để chọn ROI.")
            return

        try:
            window_name = "Chon ROI - Keo chuot, roi ENTER/SPACE (C=Huy)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Make window resizable
            cv2.resizeWindow(window_name, 800, 600) # Start with a decent size
            cv2.setWindowTitle(window_name, "Select ROI - Drag mouse, then ENTER/SPACE (C=Cancel)")

            # Freeze the frame for selection
            frozen_frame = frame.copy()

            # Add instructions overlay on the frozen frame
            cv2.putText(frozen_frame, "Drag mouse to select ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frozen_frame, "Press ENTER or SPACE to confirm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frozen_frame, "Press C to cancel", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            roi = cv2.selectROI(window_name, frozen_frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window_name)

            # roi is tuple (x, y, w, h)
            if roi == (0, 0, 0, 0): # If 'C' or window closed without selection
                self.log_activity("ℹ️ Người dùng đã hủy chọn ROI.")
                return

            if roi[2] > 0 and roi[3] > 0: # Check for valid width and height
                # Ensure ROI values are non-negative (should be by default, but good practice)
                self.webcam_roi = tuple(max(0, v) for v in roi) # Store ROI as tuple
                # Store in config as list (more JSON friendly)
                self.config['webcam_roi'] = list(self.webcam_roi)
                self.log_activity(f"✅ Đã chọn ROI cho webcam: {self.webcam_roi}")
                self.save_config()
            else:
                self.log_activity("⚠️ ROI không hợp lệ (chiều rộng hoặc cao <= 0). Không thay đổi.")

            self.update_button_styles() # Update button appearance

        except Exception as e:
            self.log_activity(f"❌ Lỗi trong quá trình chọn ROI: {e}")
            QMessageBox.critical(self, "Lỗi ROI", f"Đã xảy ra lỗi khi chọn ROI:\n{e}")
            cv2.destroyAllWindows() # Ensure all OpenCV windows are closed on error
            # Timer was already restarted


    def select_error_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        # Use current error folder as start, else home dir
        start_dir = self.error_folder if (self.error_folder and os.path.isdir(self.error_folder)) else os.path.expanduser("~")

        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu ảnh lỗi và log",
                                                  start_dir,
                                                  options=options)
        if folder:
            # --- Check Write Permissions ---
            if not os.access(folder, os.W_OK):
                 QMessageBox.warning(self, "Quyền Truy Cập", f"Không có quyền ghi vào thư mục:\n{folder}")
                 self.log_activity(f"⚠️ Không có quyền ghi vào thư mục đã chọn: {folder}")
                 return # Do not set the folder

            # --- Set Folder and Log Path ---
            self.error_folder = folder
            self.config['error_folder'] = folder
            self.log_activity(f"📁 Thư mục lưu ảnh lỗi được đặt thành: {self.error_folder}")

            # Update log file path based on the new folder
            self.log_file_path = os.path.join(self.error_folder, LOG_FILE_NAME)
            self.log_activity(f"📄 Log sẽ được lưu tại: {self.log_file_path}")
            # Try writing a test log entry to confirm
            self.log_activity("📝 Kiểm tra ghi log vào thư mục mới.")

            self.save_config() # Save config with the new valid folder
        else:
             self.log_activity("ℹ️ Người dùng đã hủy chọn thư mục lưu lỗi.")

        self.update_button_styles() # Update button appearance regardless


    # --- Toggle Processing ---
    def toggle_processing(self):
        # --- Pre-checks ---
        if not isinstance(self.ref_images.get(REF_NORM), np.ndarray):
            QMessageBox.warning(self, "Thiếu ảnh Norm", "Bạn cần chọn ảnh tham chiếu 'Norm' trước khi xử lý.")
            return
        if not self.error_folder or not os.path.isdir(self.error_folder):
            # Double check write access again, although select_error_folder should ensure it
            if not self.error_folder or not os.access(self.error_folder, os.W_OK):
                 QMessageBox.warning(self, "Thư mục Lưu không hợp lệ", "Bạn cần chọn một thư mục hợp lệ (và có quyền ghi) để lưu ảnh lỗi và log.")
                 return
            # If folder exists but is not writable (unlikely if selected via dialog), log and return
        if self.cap is None or not self.cap.isOpened():
             QMessageBox.warning(self, "Webcam chưa bật", "Bạn cần bật webcam để bắt đầu xử lý.")
             return

        # --- Toggle State ---
        self.processing = not self.processing

        if self.processing:
            # --- Start Processing ---
            # Update internal config from UI *just before* starting worker
            # Block signals temporarily to avoid recursive updates/logging
            self.ssimThresholdSpinBox.blockSignals(True)
            self.cooldownSpinBox.blockSignals(True)
            self._update_threshold_config(self.ssimThresholdSpinBox.value())
            self._update_cooldown_config(self.cooldownSpinBox.value())
            self.ssimThresholdSpinBox.blockSignals(False)
            self.cooldownSpinBox.blockSignals(False)

            # Ensure worker isn't already running (shouldn't happen with toggle logic)
            if self.processing_worker.isRunning():
                 self.log_activity("⚠️ Worker đang chạy? Đang cố gắng dừng trước khi khởi động lại...")
                 self.processing_worker.stop()
                 if not self.processing_worker.wait(1000): # Wait up to 1 sec
                      self.log_activity("⚠️ Worker cũ không dừng kịp thời. Có thể gây lỗi.")
                 else:
                      self.log_activity("✅ Worker cũ đã dừng.")


            # Clear frame queue before starting
            cleared_count = 0
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                    cleared_count += 1
                except Empty:
                    break
            if cleared_count > 0:
                self.log_activity(f"ℹ️ Đã dọn sạch {cleared_count} khung hình khỏi hàng đợi.")

            # Start the worker thread
            self.processing_worker.last_error_time = 0 # Reset error cooldown timer
            # self.processing_worker.running = True # Handled inside worker's run()
            self.processing_worker.start() # Starts the run() method

            self.update_status_label("🔄 Đang xử lý...", "lightgreen")
            self.log_activity("▶️ Bắt đầu xử lý ảnh.")
            # Disable settings while processing
            self.disable_settings_while_processing(True)
        else:
            # --- Stop Processing ---
            # Stop the worker thread
            if self.processing_worker.isRunning():
                 self.processing_worker.stop()
                 # No need to wait here extensively, closeEvent will handle final wait
                 # self.processing_worker.wait(500) # Optional short wait

            self.update_status_label("⏹ Đã dừng xử lý", "orange")
            self.log_activity("⏹ Đã dừng xử lý ảnh.")
            self.ssim_label.setText("SSIM vs Norm: N/A") # Reset SSIM display
            # Re-enable settings
            self.disable_settings_while_processing(False)

        # Update the toggle button's appearance
        self.update_toggle_button_text()


    def disable_settings_while_processing(self, disable):
        """Disables/Enables setting-related widgets during processing."""
        # Reference Image Buttons
        self.SettingButton_Norm.setEnabled(not disable)
        self.SettingButton_Shut.setEnabled(not disable)
        self.SettingButton_Fail.setEnabled(not disable)

        # ROI Button (only enable if webcam is on AND not processing)
        self.SettingButton_ROI_Webcam.setEnabled(self.cap is not None and self.cap.isOpened() and not disable)

        # Save Folder Button
        self.SaveButton.setEnabled(not disable)

        # Spin Boxes
        self.ssimThresholdSpinBox.setEnabled(not disable)
        self.cooldownSpinBox.setEnabled(not disable)

        # Webcam Buttons (prevent turning off/on during processing)
        self.ONCam.setEnabled(not disable and (self.cap is None or not self.cap.isOpened()))
        self.OFFCam.setEnabled(not disable and (self.cap is not None and self.cap.isOpened()))


    # --- Image Comparison (Called by Worker Thread) ---
    def compare_images(self, img1, img2, threshold):
        """
        Compares two images using SSIM (OpenCV). Handles grayscale conversion and resizing.
        img1: Current frame (BGR or Grayscale, NumPy array).
        img2: Reference image (BGR or Grayscale, NumPy array).
        threshold: SSIM threshold for matching.
        Returns: (bool: is_match, float: score or None)
        """
        if img1 is None or not isinstance(img1, np.ndarray) or img1.size == 0:
            # print("Warning: img1 is invalid in compare_images")
            return False, None
        if img2 is None or not isinstance(img2, np.ndarray) or img2.size == 0:
            # print("Warning: img2 is invalid in compare_images")
            return False, None

        try:
            # --- Convert to Grayscale ---
            if len(img1.shape) > 2 and img1.shape[2] == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            elif len(img1.shape) == 2:
                img1_gray = img1 # Already grayscale (or treat single channel as gray)
            else:
                # print(f"Warning: Invalid shape for img1: {img1.shape}")
                return False, None

            if len(img2.shape) > 2 and img2.shape[2] == 3:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            elif len(img2.shape) == 2:
                img2_gray = img2
            else:
                # print(f"Warning: Invalid shape for img2: {img2.shape}")
                return False, None

            h1, w1 = img1_gray.shape
            h2, w2 = img2_gray.shape

            # --- Resize Reference Image (img2) to Match Current Frame (img1) ---
            # This is crucial if ROI is used or webcam resolution differs from ref image.
            if h1 != h2 or w1 != w2:
                # print(f"Info: Resizing reference image from {w2}x{h2} to {w1}x{h1} for comparison.")
                # Choose interpolation: INTER_AREA for shrinking, INTER_LINEAR for enlarging
                interpolation = cv2.INTER_AREA if (w1 < w2 or h1 < h2) else cv2.INTER_LINEAR
                try:
                    # Ensure img2_gray is contiguous before resize if needed, though often not required
                    # if not img2_gray.flags['C_CONTIGUOUS']:
                    #     img2_gray = np.ascontiguousarray(img2_gray)
                    img2_gray_resized = cv2.resize(img2_gray, (w1, h1), interpolation=interpolation)
                    if img2_gray_resized is None or img2_gray_resized.shape != (h1, w1):
                         raise ValueError("Resize failed or returned incorrect shape")
                    img2_gray = img2_gray_resized # Use the resized image
                except Exception as resize_err:
                    # print(f"Error resizing reference image: {resize_err}")
                    # Log error less frequently
                    current_time_check = time.time()
                    if not hasattr(self, 'last_resize_error_time') or current_time_check - self.last_resize_error_time > 10:
                         self.log_activity(f"❌ Lỗi resize ảnh tham chiếu từ {w2}x{h2} sang {w1}x{h1}: {resize_err}")
                         self.last_resize_error_time = current_time_check
                    return False, None # Cannot compare if resize fails


            # --- Calculate Dynamic win_size for SSIM ---
            # Based on the (potentially resized) image dimensions (h1, w1)
            # Keep max win_size relatively small (e.g., 7 or 11) for performance and local features
            # Original code used max 7. Let's stick with that.
            win_size = min(min(h1, w1), 7)
            if win_size % 2 == 0: win_size -= 1 # Ensure odd
            win_size = max(3, win_size) # Ensure minimum size is 3

            # Check if image dimensions are sufficient for the chosen win_size
            if h1 < win_size or w1 < win_size:
                 current_time_check = time.time()
                 if not hasattr(self, 'last_small_img_warn_time') or current_time_check - self.last_small_img_warn_time > 10:
                     print(f"Warning: Image/ROI size ({w1}x{h1}) is too small for SSIM win_size ({win_size}). Comparison skipped.")
                     self.log_activity(f"⚠️ Kích thước ảnh/ROI ({w1}x{h1}) quá nhỏ cho SSIM (win={win_size}). Bỏ qua so sánh.")
                     self.last_small_img_warn_time = current_time_check
                 return False, None


            # --- Calculate SSIM using the dedicated function ---
            # Ensure images are float64 for ssim_opencv function
            # Use .copy() if needed to ensure float conversion doesn't affect original arrays outside scope
            score = ssim_opencv(img1_gray.astype(np.float64),
                                img2_gray.astype(np.float64), # Use potentially resized img2_gray
                                win_size=win_size,
                                data_range=255.0) # Assuming 8-bit grayscale images

            if score is None:
                 # ssim_opencv function handles internal warnings
                 # print("Warning: ssim_opencv returned None.")
                 return False, None

            # print(f"Debug SSIM Score: {score:.4f} vs Threshold: {threshold:.3f}") # Useful debug line
            is_match = score >= threshold
            return is_match, score

        except cv2.error as cv_err:
             # Catch specific OpenCV errors
             current_time_check = time.time()
             if not hasattr(self, 'last_cv_compare_error_time') or current_time_check - self.last_cv_compare_error_time > 5:
                 print(f"❌ OpenCV Exception during SSIM comparison: {cv_err}")
                 self.log_activity(f"❌ Lỗi OpenCV khi so sánh ảnh: {cv_err.msg}")
                 self.last_cv_compare_error_time = current_time_check
             return False, None
        except Exception as e:
            # Catch any other unexpected errors during comparison
            current_time_check = time.time()
            if not hasattr(self, 'last_compare_error_time') or current_time_check - self.last_compare_error_time > 5:
                 print(f"❌ Unexpected Exception during SSIM comparison: {e}")
                 import traceback
                 print(traceback.format_exc())
                 self.log_activity(f"❌ Lỗi không xác định khi so sánh ảnh: {e}")
                 self.last_compare_error_time = current_time_check
            return False, None


    # --- Save Error Image (Slot for Worker Signal, runs on Main Thread) ---
    @QtCore.pyqtSlot(np.ndarray, str)
    def save_error_image_from_thread(self, frame_copy, file_path):
        # This method runs in the main GUI thread because it's a slot
        # connected to a signal from the worker thread.
        try:
             # Use imencode to handle different image types and special characters in path
             # Specify PNG for lossless compression
             success, img_encoded = cv2.imencode('.png', frame_copy, [cv2.IMWRITE_PNG_COMPRESSION, 3]) # Moderate compression

             if not success or img_encoded is None:
                 raise ValueError("cv2.imencode failed to encode image.")

             # Write the encoded image bytes to the file
             with open(file_path, "wb") as f:
                 f.write(img_encoded.tobytes())

             # Log success *after* successful write
             self.log_activity(f"💾 Đã lưu ảnh lỗi: {os.path.basename(file_path)}")

        except Exception as e:
             self.log_activity(f"❌ Lỗi khi lưu ảnh từ luồng chính vào '{os.path.basename(file_path)}': {e}")
             # Optionally, provide more context if possible
             # print(f"Full path attempted: {file_path}")


    # --- Application Closing ---
    def close_application(self):
        self.log_activity("🚪 Bắt đầu quá trình thoát ứng dụng...")
        # Don't save config here, closeEvent will handle it
        self.close() # Triggers the closeEvent


    def closeEvent(self, event):
        # Override the default close event handler
        self.log_activity("🚪 Cửa sổ đang đóng. Dọn dẹp tài nguyên...")

        # 1. Stop Processing Thread
        if self.processing or self.processing_worker.isRunning(): # Check both flags
            self.processing = False # Ensure flag is set
            if self.processing_worker.isRunning():
                self.log_activity("⚙️ Yêu cầu dừng worker thread...")
                self.processing_worker.stop()
                # Wait for the thread to finish properly
                if not self.processing_worker.wait(3000): # Wait up to 3 seconds
                     self.log_activity("⚠️ Worker thread không dừng kịp thời! Có thể bị treo.")
                else:
                     self.log_activity("✅ Worker thread đã dừng.")
            # Update UI state after stopping
            self.disable_settings_while_processing(False) # Re-enable buttons (optional)
            self.update_toggle_button_text()
            self.update_status_label("Đã thoát", "lightgray")


        # 2. Stop Webcam
        if self.cap and self.cap.isOpened():
            self.stop_webcam() # stop_webcam already logs

        # 3. Clear Frame Queue (just in case)
        q_size = self.frame_queue.qsize()
        if q_size > 0:
             self.log_activity(f"ℹ️ Dọn dẹp {q_size} khung hình còn lại trong queue...")
             while not self.frame_queue.empty():
                 try: self.frame_queue.get_nowait()
                 except Empty: break

        # 4. Save Final Configuration
        self.log_activity("💾 Lưu cấu hình lần cuối...")
        self.save_config()

        # 5. Final Log Message (to UI and File)
        final_log_msg = "🚪 Dọn dẹp hoàn tất. Thoát."
        self.log_activity(final_log_msg)

        # Write final marker to log file
        if self.log_file_path:
             try:
                 # Ensure directory exists one last time
                 log_dir = os.path.dirname(self.log_file_path)
                 if log_dir and not os.path.exists(log_dir):
                      os.makedirs(log_dir, exist_ok=True)
                 with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                     log_file.write(f"---\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Application Closed\n---\n")
             except Exception as e:
                 # Log error to console only to avoid issues during shutdown
                 print(f"Error writing final log entry to '{self.log_file_path}': {e}")

        # 6. Accept the close event
        event.accept()
        # Optionally force exit if resources might hang (use with caution)
        # QtWidgets.QApplication.instance().quit()
        # sys.exit(0)


# --- Main Execution ---
if __name__ == "__main__":
    # Enable High DPI Scaling for better visuals on high-resolution displays
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    window = ImageCheckerApp()
    window.show()
    sys.exit(app.exec_())