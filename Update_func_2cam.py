# ScreenDetect Yolo _ Open CV
# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QFileDialog, QLabel, QGraphicsView, QGraphicsScene,
                             QMessageBox, QWidget, QTextEdit, QSizePolicy,
                             QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
                             QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import cv2
import numpy as np
import sys
import os
import time
import json
from queue import Queue, Empty
import traceback
import serial
import serial.tools.list_ports
from enum import Enum, auto

# --- Thử import YOLOv8 ---
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False
    # In cảnh báo ra console một lần khi khởi động
    print("WARNING: Thư viện 'ultralytics' (YOLOv8) chưa được cài đặt. Chức năng YOLO sẽ bị vô hiệu hóa.")
    print("         Vui lòng cài đặt bằng lệnh: pip install ultralytics")

# --- Hằng số ---
METHOD_SSIM = "OpenCV SSIM"
METHOD_YOLO = "YOLOv8 Detection"

class ComparisonStatus(Enum):
    NORMAL = auto()
    SHUTDOWN = auto()
    FAIL = auto()
    UNKNOWN = auto()
    ERROR = auto()

STATUS_MAP = {
    ComparisonStatus.NORMAL: {"label": "Normal", "color": "lightgreen", "log_prefix": "✅", "serial": "Norm"},
    ComparisonStatus.SHUTDOWN: {"label": "Shutdown", "color": "lightblue", "log_prefix": "ℹ️", "serial": "Shutdown"},
    ComparisonStatus.FAIL: {"label": "FAIL!", "color": "red", "log_prefix": "❌", "serial": "Fail"},
    ComparisonStatus.UNKNOWN: {"label": "Unknown Mismatch", "color": "orange", "log_prefix": "⚠️", "serial": None},
    ComparisonStatus.ERROR: {"label": "Comparison Error", "color": "magenta", "log_prefix": "💥", "serial": None},
}

REF_NORM = "Norm"
REF_SHUTDOWN = "Shutdown"
REF_FAIL = "Fail"
DEFAULT_SSIM_THRESHOLD = 0.90
DEFAULT_ERROR_COOLDOWN = 15
DEFAULT_RUNTIME_MINUTES = 0
DEFAULT_RECORD_ON_ERROR = False
DEFAULT_SERIAL_ENABLED = False
DEFAULT_BAUD_RATE = 9600
CONFIG_FILE_NAME = "image_checker_config.json"
LOG_FILE_NAME = "activity_log.txt"
VIDEO_SUBFOLDER = "error_videos"
COMMON_BAUD_RATES = [9600, 19200, 38400, 57600, 115200]
DEFAULT_COMPARISON_METHOD = METHOD_SSIM
DEFAULT_YOLO_CONFIDENCE = 0.5

# --- Hàm SSIM ---
def ssim_opencv(img1, img2, K1=0.01, K2=0.03, win_size=7, data_range=255.0):
    if img1 is None or img2 is None: return None
    try:
        if len(img1.shape) > 2: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) > 2: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if img1.shape != img2.shape:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            interpolation = cv2.INTER_AREA if (w2 > w1 or h2 > h1) else cv2.INTER_LINEAR
            img2 = cv2.resize(img2, (w1, h1), interpolation=interpolation)
            if img2 is None or img2.shape != (h1, w1):
                print("Warning: SSIM resize failed")
                return None

        h, w = img1.shape
        win_size = min(win_size, h, w)
        if win_size % 2 == 0: win_size -= 1
        win_size = max(3, win_size)
        if h < win_size or w < win_size: return None

        if img1.dtype != np.float64: img1 = img1.astype(np.float64)
        if img2.dtype != np.float64: img2 = img2.astype(np.float64)

        C1 = (K1 * data_range)**2
        C2 = (K2 * data_range)**2
        sigma = 1.5
        mu1 = cv2.GaussianBlur(img1, (win_size, win_size), sigma)
        mu2 = cv2.GaussianBlur(img2, (win_size, win_size), sigma)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(img1 * img1, (win_size, win_size), sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, (win_size, win_size), sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (win_size, win_size), sigma) - mu1_mu2
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / (denominator + 1e-8)
        ssim_map = np.clip(ssim_map, 0, 1)
        mssim = np.mean(ssim_map)
        if not np.isfinite(mssim): return None
        return mssim

    except cv2.error as cv_err:
        print(f"OpenCV Error in SSIM: {cv_err}")
        return None
    except Exception as e:
        print(f"General Error in SSIM: {e}")
        traceback.print_exc()
        return None

# --- Worker Thread ---
class ProcessingWorker(QThread):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(ComparisonStatus, object)
    save_error_signal = pyqtSignal(np.ndarray, str)
    comparison_details_signal = pyqtSignal(dict)
    error_detected_signal = pyqtSignal()
    serial_command_signal = pyqtSignal(str)

    def __init__(self, frame_queue, ref_data_provider, config_provider, compare_function):
        super().__init__()
        self.frame_queue = frame_queue
        self.get_ref_data = ref_data_provider
        self.get_config = config_provider
        self.compare_images_func = compare_function
        self.running = False
        self.last_error_time = 0
        self.last_emitted_serial_state = None
        self.last_status = None

    def run(self):
        self.running = True
        self.log_signal.emit("⚙️ Worker started.")
        last_status_log_time = 0
        error_signaled_this_session = False
        self.last_emitted_serial_state = None
        self.last_status = None

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except Empty:
                if not self.running: break
                continue
            except Exception as e:
                self.log_signal.emit(f"❌ Error getting frame from queue: {e}")
                continue

            if not self.running: break

            try:
                cfg = self.get_config()
                err_cd = cfg.get('error_cooldown', DEFAULT_ERROR_COOLDOWN)
                err_f = cfg.get('error_folder')
                ref_data = self.get_ref_data()

                current_status, details = self.compare_images_func(frame, ref_data, cfg)
                self.comparison_details_signal.emit(details or {})

                status_info = STATUS_MAP.get(current_status, STATUS_MAP[ComparisonStatus.ERROR])
                status_label = status_info["label"]
                # status_color = status_info["color"] # Không dùng trực tiếp trong worker nữa
                log_prefix = status_info["log_prefix"]
                serial_cmd = status_info["serial"]

                log_msg = f"{log_prefix} Status: {status_label}"
                detail_str = ""
                if details:
                    if 'detected' in details:
                        det_items = sorted([f"{k}:{v}" for k,v in details['detected'].items()])
                        detail_str = f"Detect: {', '.join(det_items) if det_items else 'None'}"
                        if 'count' in details: detail_str += f" (Total: {details['count']})"
                    elif 'ssim_norm' in details:
                        detail_str = f"SSIM: {details['ssim_norm']:.4f}"
                    if 'reason' in details: detail_str += f" (Reason: {details['reason']})"
                    elif 'error' in details: detail_str += f" (Error: {details['error']})"
                    if detail_str: log_msg += f" ({detail_str})"

                needs_logging = True
                current_time_log = time.time()
                if current_status == ComparisonStatus.NORMAL:
                    if current_time_log - last_status_log_time < 5.0: needs_logging = False
                    else: last_status_log_time = current_time_log
                else:
                    last_status_log_time = 0
                    # Chỉ log lại trạng thái lỗi nếu nó thay đổi so với lần cuối cùng
                    # Hoặc nếu đây là lần đầu tiên sau 1 khoảng thời gian? (Hiện tại chỉ log khi đổi)
                    if self.last_status == current_status: needs_logging = False

                if needs_logging: self.log_signal.emit(log_msg)

                if self.last_status != current_status:
                    self.status_signal.emit(current_status, details or {})
                    self.last_status = current_status

                is_problem_state = current_status in [ComparisonStatus.FAIL, ComparisonStatus.UNKNOWN, ComparisonStatus.ERROR]
                should_save_img = is_problem_state
                should_record = is_problem_state # Dùng cờ này để quyết định video

                if should_record and not error_signaled_this_session:
                    self.error_detected_signal.emit()
                    error_signaled_this_session = True # Đánh dấu đã báo lỗi cho phiên ghi này

                # Gửi lệnh serial nếu trạng thái cần gửi thay đổi
                if serial_cmd and self.last_emitted_serial_state != serial_cmd:
                    self.serial_command_signal.emit(serial_cmd)
                    self.last_emitted_serial_state = serial_cmd
                elif not serial_cmd and self.last_emitted_serial_state: # Trạng thái không cần gửi nữa
                    self.last_emitted_serial_state = None

                # Lưu ảnh lỗi (nếu cần và cooldown đã hết)
                current_time_save = time.time()
                if should_save_img and err_f and (current_time_save - self.last_error_time > err_cd):
                    try:
                        err_sub = status_label.lower().replace("!", "").replace(" ", "_").replace(":", "")
                        save_folder = os.path.join(err_f, err_sub)
                        # Đảm bảo thư mục tồn tại (an toàn hơn)
                        os.makedirs(save_folder, exist_ok=True)
                        timestamp = time.strftime('%Y%m%d_%H%M%S') + f"_{int((current_time_save - int(current_time_save)) * 1000):03d}"
                        filename = f"{err_sub}_{timestamp}.png"
                        filepath = os.path.join(save_folder, filename)
                        # Gửi frame và đường dẫn đến luồng chính để lưu
                        self.save_error_signal.emit(frame.copy(), filepath)
                        self.last_error_time = current_time_save
                    except Exception as e:
                        self.log_signal.emit(f"❌ Lỗi khi chuẩn bị lưu ảnh lỗi: {e}")
                elif not err_f and should_save_img:
                    # self.log_signal.emit("⚠️ Chưa cấu hình thư mục lỗi, không thể lưu ảnh.")
                    pass # Không cần log liên tục

                sleep_time = 0.05 if current_status == ComparisonStatus.NORMAL else 0.1 # Ngủ ít hơn nếu Normal
                time.sleep(sleep_time)

            except Exception as e:
                self.log_signal.emit(f"💥 Lỗi nghiêm trọng trong worker logic: {e}")
                self.log_signal.emit(traceback.format_exc())
                try:
                    # Cố gắng phát tín hiệu lỗi lên UI
                    self.status_signal.emit(ComparisonStatus.ERROR, {"error": str(e)})
                except Exception as sig_e:
                    # Lỗi rất nghiêm trọng nếu cả việc phát tín hiệu cũng thất bại
                    print(f"CRITICAL: Failed to emit error status signal: {sig_e}")
                self.last_status = ComparisonStatus.ERROR # Ghi nhớ trạng thái lỗi
                time.sleep(0.5) # Chờ một chút trước khi thử lại

        self.log_signal.emit("⚙️ Worker finished.")
        self.last_emitted_serial_state = None # Reset trạng thái serial cuối
        error_signaled_this_session = False # Reset cờ báo lỗi video
        self.last_status = None # Reset trạng thái cuối


    def stop(self):
        self.running = False
        self.log_signal.emit("⚙️ Đang yêu cầu dừng worker...")


# --- Main Application Window ---
class ImageCheckerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # --- State Variables ---
        self.cap = None
        self.webcam_fps = 15.0
        self.frame_timer = QTimer(self)
        self.ref_data = {k: None for k in [REF_NORM, REF_SHUTDOWN, REF_FAIL]}
        self.yolo_model = None
        self.webcam_roi = None
        self.processing = False
        self.error_folder = None
        self.log_file_path = None
        self.pixmap_item = None
        self.runtime_timer = QTimer(self)
        self._current_runtime_minutes = DEFAULT_RUNTIME_MINUTES
        self._current_ssim_threshold = DEFAULT_SSIM_THRESHOLD
        self._current_error_cooldown = DEFAULT_ERROR_COOLDOWN
        self._record_on_error_enabled = DEFAULT_RECORD_ON_ERROR
        self.video_writer = None
        self.current_video_path = None
        self.error_occurred_during_recording = False
        self.serial_port = None
        self.serial_port_name = None
        self.serial_baud_rate = DEFAULT_BAUD_RATE
        self.serial_enabled = DEFAULT_SERIAL_ENABLED
        self.current_comparison_method = DEFAULT_COMPARISON_METHOD
        # --- Biến trạng thái mới cho Camera ---
        self.selected_camera_index = 0 # Chỉ số camera đang chọn, mặc định là 0
        self.available_cameras = {} # Dict để lưu {index: "Tên hiển thị"}

        self.comparison_functions = {
            METHOD_SSIM: self.compare_ssim_strategy,
            METHOD_YOLO: self.compare_yolo_strategy,
        }

        # Dictionary cấu hình chính
        self.config = {
            'comparison_method': self.current_comparison_method,
            'ssim_threshold': self._current_ssim_threshold,
            'error_cooldown': self._current_error_cooldown,
            'runtime_duration_minutes': self._current_runtime_minutes,
            'record_on_error': self._record_on_error_enabled,
            'error_folder': None,
            'ref_paths': {k: None for k in [REF_NORM, REF_SHUTDOWN, REF_FAIL]},
            'webcam_roi': None,
            'serial_port': self.serial_port_name,
            'serial_baud': self.serial_baud_rate,
            'serial_enabled': self.serial_enabled,
            'selected_camera_index': self.selected_camera_index, # Thêm camera index vào config
            'yolo_model_path': None,
            'yolo_confidence': DEFAULT_YOLO_CONFIDENCE,
        }

        # Kết nối tín hiệu
        self.frame_timer.timeout.connect(self.update_frame)
        self.runtime_timer.setSingleShot(True)
        self.runtime_timer.timeout.connect(self._runtime_timer_timeout)

        # Queue và Worker
        self.frame_queue = Queue(maxsize=10)
        self.processing_worker = None

        # Khởi tạo UI
        self.init_ui()

        # Đặt sau init_ui để đảm bảo các widget đã tồn tại
        QTimer.singleShot(100, self._refresh_camera_list) # Delay quét camera lần đầu

        # Tải config (sẽ cập nhật các giá trị state)
        self.load_config()

        # Làm mới cổng COM (tải lại danh sách)
        self._refresh_com_ports()

        # Ghi log và cập nhật UI ban đầu
        self.log_activity("Ứng dụng khởi động.")
        self.update_all_ui_elements() # Cập nhật giá trị & trạng thái các nút

        # Thử tải model YOLO nếu có cấu hình sẵn và phương thức là YOLO
        if self.current_comparison_method == METHOD_YOLO and self.config.get('yolo_model_path'):
            # Delay một chút để UI có thời gian hiển thị trước khi tải model (có thể mất thời gian)
            QTimer.singleShot(250, self._load_yolo_model)

    # --- Hàm Provider cho Worker ---
    def get_current_config_for_worker(self):
        # Trả về một bản sao các config cần thiết cho worker
        return {
            'error_cooldown': self._current_error_cooldown,
            'error_folder': self.error_folder,
            'comparison_method': self.current_comparison_method,
            'ssim_threshold': self._current_ssim_threshold,
            'yolo_confidence': self.config.get('yolo_confidence', DEFAULT_YOLO_CONFIDENCE),
        }

    def get_reference_data_for_worker(self):
        # Trả về dữ liệu tham chiếu dựa trên phương thức so sánh
        if self.current_comparison_method == METHOD_SSIM:
            # Trả về bản sao của ảnh để tránh vấn đề đa luồng
            return {k: (img.copy() if isinstance(img, np.ndarray) else None) for k, img in self.ref_data.items()}
        elif self.current_comparison_method == METHOD_YOLO:
            # --- Định nghĩa quy tắc YOLO cố định tại đây (hoặc tải từ file config/UI sau) ---
            return {
                # Ví dụ quy tắc:
                REF_NORM: {"required_objects": ["person"], "min_counts": {"person": 1}, "exact_total_objects": 1, "forbidden_objects": ["alert", "warning"]},
                REF_SHUTDOWN: {"forbidden_objects": ["person", "car"], "max_total_objects": 0},
                REF_FAIL: {"any_of": ["alert", "warning", "error_sign"]},
                # Thêm các quy tắc khác nếu cần
            }
        else:
            return {} # Trường hợp phương thức không xác định

    # --- Slots cập nhật Config ---
    @QtCore.pyqtSlot(float)
    def _update_threshold_config(self, value):
        if abs(self._current_ssim_threshold - value) > 1e-4: # So sánh số thực
            self._current_ssim_threshold = value
            self.log_activity(f"⚙️ Ngưỡng SSIM: {value:.3f}")
            self.config['ssim_threshold'] = value
            # self.save_config() # Tùy chọn: Lưu ngay hoặc chờ

    @QtCore.pyqtSlot(int)
    def _update_cooldown_config(self, value):
        if self._current_error_cooldown != value:
            self._current_error_cooldown = value
            self.log_activity(f"⚙️ Cooldown Lỗi: {value}s")
            self.config['error_cooldown'] = value
            # self.save_config()

    @QtCore.pyqtSlot(int)
    def _update_runtime_config(self, value):
        if self._current_runtime_minutes != value:
            self._current_runtime_minutes = value
            log_msg = f"⚙️ Thời gian chạy: {'Vô hạn' if value == 0 else f'{value} phút'}"
            self.log_activity(log_msg)
            self.config['runtime_duration_minutes'] = value
            # self.save_config()

    @QtCore.pyqtSlot()
    def _toggle_record_on_error(self):
        if self.processing:
             QMessageBox.warning(self,"Đang Xử Lý","Không thể thay đổi cấu hình ghi video khi đang xử lý.")
             # Khôi phục trạng thái nút nếu cần (dù nút thường bị disable)
             self.update_record_button_style()
             return
        self._record_on_error_enabled = not self._record_on_error_enabled
        self.config['record_on_error'] = self._record_on_error_enabled
        self.update_record_button_style()
        self.log_activity(f"⚙️ Ghi video lỗi: {'Bật' if self._record_on_error_enabled else 'Tắt'}")
        # self.save_config()

    @QtCore.pyqtSlot(str)
    def _update_serial_port_config(self, port_name):
        # Được gọi khi user chọn từ ComboBox COM ports
        new_port = port_name if port_name and "Không tìm thấy" not in port_name else None
        if self.serial_port_name != new_port:
             self.serial_port_name = new_port
             self.config['serial_port'] = self.serial_port_name
             self.log_activity(f"⚙️ Cổng COM đã chọn: {self.serial_port_name or 'Chưa chọn'}")
             # Nếu đang kết nối, cảnh báo cần kết nối lại
             if self.serial_enabled: self.log_activity("⚠️ Thay đổi cổng COM yêu cầu Kết nối lại thủ công.")
             # self.save_config()

    @QtCore.pyqtSlot(str)
    def _update_serial_baud_config(self, baud_str):
        # Được gọi khi user chọn từ ComboBox Baudrate
        try:
            bd = int(baud_str)
            if self.serial_baud_rate != bd:
                if bd in COMMON_BAUD_RATES:
                    self.serial_baud_rate = bd
                    self.config['serial_baud'] = bd
                    self.log_activity(f"⚙️ Baud rate đã chọn: {bd}")
                    if self.serial_enabled: self.log_activity("⚠️ Thay đổi Baud rate yêu cầu Kết nối lại thủ công.")
                    # self.save_config()
                else: # Giá trị lạ (không nên xảy ra với ComboBox cố định)
                     self.log_activity(f"⚠️ Baud rate không hợp lệ: {bd}. Sử dụng giá trị cũ: {self.serial_baud_rate}")
                     # Tìm và đặt lại index cũ trong ComboBox
                     idx = self.baudRateComboBox.findText(str(self.serial_baud_rate))
                     if idx >= 0:
                         self.baudRateComboBox.blockSignals(True); self.baudRateComboBox.setCurrentIndex(idx); self.baudRateComboBox.blockSignals(False)
        except ValueError: # Không thể chuyển sang int (không nên xảy ra)
            self.log_activity(f"⚠️ Giá trị Baud rate nhập vào không phải số: {baud_str}")
            # Đặt lại giá trị cũ
            idx = self.baudRateComboBox.findText(str(self.serial_baud_rate))
            if idx >= 0: self.baudRateComboBox.blockSignals(True); self.baudRateComboBox.setCurrentIndex(idx); self.baudRateComboBox.blockSignals(False)

    @QtCore.pyqtSlot(str)
    def _update_comparison_method_config(self, method_name):
        # Được gọi khi user chọn phương thức so sánh từ ComboBox
        if self.processing:
            self.log_activity("⚠️ Không thể thay đổi phương thức khi đang xử lý.")
            # Đặt lại lựa chọn về giá trị cũ
            self.comparisonMethodComboBox.blockSignals(True)
            self.comparisonMethodComboBox.setCurrentText(self.current_comparison_method)
            self.comparisonMethodComboBox.blockSignals(False)
            return

        if method_name in self.comparison_functions and self.current_comparison_method != method_name:
            # Kiểm tra đặc biệt nếu chọn YOLO mà thư viện chưa cài
            if method_name == METHOD_YOLO and not YOLO_AVAILABLE:
                 QMessageBox.critical(self, "Lỗi Thiếu Thư Viện", "Không tìm thấy thư viện YOLOv8 (ultralytics).\nVui lòng cài đặt: pip install ultralytics")
                 # Đặt lại lựa chọn về giá trị cũ
                 self.comparisonMethodComboBox.blockSignals(True)
                 self.comparisonMethodComboBox.setCurrentText(self.current_comparison_method)
                 self.comparisonMethodComboBox.blockSignals(False)
                 return

            # Cập nhật trạng thái và config
            self.current_comparison_method = method_name
            self.config['comparison_method'] = method_name
            self.log_activity(f"⚙️ Phương thức so sánh: {method_name}")

            # Cập nhật UI để ẩn/hiện các phần cấu hình tương ứng
            self._update_method_specific_ui()

            # Nếu chuyển sang YOLO và đã có đường dẫn model, thử tải ngay
            if method_name == METHOD_YOLO and self.config.get('yolo_model_path'):
                QTimer.singleShot(50, self._load_yolo_model) # Delay nhẹ

            # self.save_config() # Lưu thay đổi phương thức

    @QtCore.pyqtSlot(float)
    def _update_yolo_confidence_config(self, value):
        # Được gọi khi giá trị spinbox YOLO Confidence thay đổi
        current_conf = self.config.get('yolo_confidence', DEFAULT_YOLO_CONFIDENCE)
        if abs(current_conf - value) > 1e-4: # So sánh số thực
            # Đảm bảo giá trị trong khoảng hợp lệ (dù SpinBox đã giới hạn)
            clamped_value = max(0.01, min(1.0, value))
            if abs(clamped_value - value) > 1e-4: # Nếu giá trị bị kẹp -> cập nhật lại UI
                self.yoloConfidenceSpinBox.blockSignals(True)
                self.yoloConfidenceSpinBox.setValue(clamped_value)
                self.yoloConfidenceSpinBox.blockSignals(False)
            # Cập nhật config và log
            self.config['yolo_confidence'] = clamped_value
            self.log_activity(f"⚙️ Ngưỡng tin cậy YOLO: {clamped_value:.2f}")
            # self.save_config()

    # --- init_ui ---
    def init_ui(self):
        """Khởi tạo giao diện người dùng."""
        self.setWindowTitle("Image Checker v3.1 (Camera Selection)")
        self.setGeometry(100, 100, 1350, 840) # Tăng chiều cao cửa sổ một chút
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # --- Panel Trái: Hiển thị Webcam và Nút điều khiển ---
        self.scene = QGraphicsScene(self)
        self.graphicsView = QGraphicsView(self.scene, central_widget)
        self.graphicsView.setGeometry(10, 10, 640, 360) # Kích thước khu vực webcam
        self.graphicsView.setStyleSheet("border: 1px solid black;")
        self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray)) # Nền xám khi chưa có ảnh

        # Kích thước và vị trí cơ bản của nút
        bw, bh, vs, bx = 201, 31, 38, 20 # width, height, vertical_space, x_pos
        yp = 380 # Y-start pos for controls below webcam view

        # --- Webcam Controls ---
        self.ONCam = self.create_button("📷 Bật Webcam"); self.ONCam.setGeometry(bx, yp, bw, bh); self.ONCam.clicked.connect(self.start_webcam)
        self.OFFCam = self.create_button("🚫 Tắt Webcam"); self.OFFCam.setGeometry(bx + bw + 10, yp, bw, bh); self.OFFCam.clicked.connect(self.stop_webcam); self.OFFCam.setEnabled(False);
        # Nút bật webcam ban đầu bị disable cho đến khi quét xong camera
        self.ONCam.setEnabled(False)
        yp += vs # Tăng y cho hàng tiếp theo

        # --- THÊM PHẦN CHỌN CAMERA ---
        self.cameraSelectionLabel = QLabel("Chọn Camera:", central_widget)
        self.cameraSelectionLabel.setGeometry(bx, yp + 5, 90, 31) # Nhãn "Chọn Camera:"
        cam_combo_w = (bw * 2 + 10) - 95 - 45 # Chiều rộng cho combobox
        self.cameraSelectionComboBox = QComboBox(central_widget)
        self.cameraSelectionComboBox.setGeometry(bx + 95, yp, cam_combo_w, 31) # ComboBox chọn cam
        self.cameraSelectionComboBox.setEnabled(False) # Disable ban đầu
        self.refreshCamerasButton = QPushButton("🔄", central_widget) # Nút Refresh danh sách cam
        self.refreshCamerasButton.setGeometry(bx + 95 + cam_combo_w + 5, yp, 40, 31) # Vị trí nút Refresh
        self.refreshCamerasButton.setToolTip("Làm mới danh sách camera")
        self.refreshCamerasButton.setEnabled(False) # Disable ban đầu
        # Kết nối tín hiệu
        self.refreshCamerasButton.clicked.connect(self._refresh_camera_list)
        self.cameraSelectionComboBox.currentIndexChanged.connect(self._on_camera_selection_changed)
        # --- KẾT THÚC PHẦN CHỌN CAMERA ---
        yp += vs # Tăng y cho hàng tiếp theo (cho các nút SSIM)

        # --- Load/Capture Reference Image Buttons (SSIM specific) ---
        # Cập nhật vị trí Y của tất cả các nút bên dưới
        self.SettingButton_Norm = self.create_button("📂 Ảnh Norm (SSIM)"); self.SettingButton_Norm.setGeometry(bx, yp, bw, bh); self.SettingButton_Norm.clicked.connect(lambda: self.load_reference_image(REF_NORM))
        self.SettingButton_Shutdown = self.create_button("📂 Ảnh Shutdown (SSIM)"); self.SettingButton_Shutdown.setGeometry(bx + bw + 10, yp, bw, bh); self.SettingButton_Shutdown.clicked.connect(lambda: self.load_reference_image(REF_SHUTDOWN))
        self.SettingButton_Fail = self.create_button("📂 Ảnh Fail (SSIM)"); self.SettingButton_Fail.setGeometry(bx + 2 * (bw + 10), yp, bw, bh); self.SettingButton_Fail.clicked.connect(lambda: self.load_reference_image(REF_FAIL)); yp += vs

        self.CaptureButton_Norm = self.create_button("📸 Chụp Norm (SSIM)"); self.CaptureButton_Norm.setGeometry(bx, yp, bw, bh); self.CaptureButton_Norm.clicked.connect(lambda: self.capture_reference_from_webcam(REF_NORM)); self.CaptureButton_Norm.setEnabled(False)
        self.CaptureButton_Shut = self.create_button("📸 Chụp Shutdown (SSIM)"); self.CaptureButton_Shut.setGeometry(bx + bw + 10, yp, bw, bh); self.CaptureButton_Shut.clicked.connect(lambda: self.capture_reference_from_webcam(REF_SHUTDOWN)); self.CaptureButton_Shut.setEnabled(False)
        self.CaptureButton_Fail = self.create_button("📸 Chụp Fail (SSIM)"); self.CaptureButton_Fail.setGeometry(bx + 2 * (bw + 10), yp, bw, bh); self.CaptureButton_Fail.clicked.connect(lambda: self.capture_reference_from_webcam(REF_FAIL)); self.CaptureButton_Fail.setEnabled(False); yp += vs

        # --- ROI, Save Folder, Start/Stop, Exit Buttons ---
        self.SettingButton_ROI_Webcam = self.create_button("✂️ Chọn ROI"); self.SettingButton_ROI_Webcam.setGeometry(bx, yp, bw, bh); self.SettingButton_ROI_Webcam.clicked.connect(self.select_webcam_roi); self.SettingButton_ROI_Webcam.setEnabled(False)
        self.SaveButton = self.create_button("📁 Thư mục lỗi"); self.SaveButton.setGeometry(bx + bw + 10, yp, bw, bh); self.SaveButton.clicked.connect(self.select_error_folder)
        self.ToggleProcessingButton = self.create_button("▶️ Bắt đầu"); self.ToggleProcessingButton.setGeometry(bx + 2 * (bw + 10), yp, bw, bh); self.ToggleProcessingButton.clicked.connect(self.toggle_processing); yp += vs

        self.ExitButton = self.create_button("🚪 Thoát"); self.ExitButton.setGeometry(bx, yp, bw, bh); self.ExitButton.clicked.connect(self.close_application); # yp += vs # Dòng cuối

        # --- Panel Phải: Log, Trạng thái, Cấu hình ---
        rx = 670 # X-start for right panel
        # Tự động tính chiều rộng panel phải dựa trên kích thước cửa sổ
        self.right_panel_width = self.geometry().width() - rx - 20
        lw = self.right_panel_width

        # --- Log Area ---
        log_label = QLabel("Log Hoạt Động:", central_widget); log_label.setGeometry(rx, 10, 150, 20)
        self.log_text_edit = QTextEdit(central_widget); self.log_text_edit.setGeometry(rx, 35, lw, 250); self.log_text_edit.setReadOnly(True); self.log_text_edit.setStyleSheet("border:1px solid black; padding:5px; background-color:white; font-family:Consolas,monospace; font-size:10pt;")

        # --- Status Labels ---
        status_y = 300
        self.process_label = QLabel("Trạng thái: Chờ", central_widget); self.process_label.setGeometry(rx, status_y, lw, 40); self.process_label.setAlignment(Qt.AlignCenter); self.process_label.setStyleSheet("border:1px solid black; padding:5px; background-color:lightgray; font-weight:bold; border-radius:3px;")
        self.details_label = QLabel("Details: N/A", central_widget); self.details_label.setGeometry(rx, status_y + 45, lw, 30); self.details_label.setAlignment(Qt.AlignCenter); self.details_label.setStyleSheet("padding:5px; background-color:#f0f0f0; border-radius:3px;")

        # --- Configuration Controls ---
        # Vị trí và kích thước cho các control cấu hình
        sx_lbl = rx + 10      # X-pos for labels
        lbl_w = 140           # Width for labels
        sx_ctrl = sx_lbl + lbl_w + 5 # X-pos for controls
        ctrl_w_main = lw - (sx_ctrl - rx) - 10 # Width for most controls (tự tính toán)
        ctrl_w_sm_btn = 40     # Width for small buttons (like refresh COM)
        s_vs_cfg = 38         # Vertical spacing for settings rows
        sy_cfg = status_y + 45 + 30 + 15 # Y-start position for settings

        # --- Method Selection ---
        lm = QLabel("Phương thức:", central_widget); lm.setGeometry(sx_lbl, sy_cfg, lbl_w, 31)
        self.comparisonMethodComboBox = QComboBox(central_widget); self.comparisonMethodComboBox.setGeometry(sx_ctrl, sy_cfg, ctrl_w_main, 31)
        self.comparisonMethodComboBox.addItems([METHOD_SSIM, METHOD_YOLO])
        self.comparisonMethodComboBox.setToolTip("Chọn thuật toán so sánh ảnh")
        self.comparisonMethodComboBox.currentTextChanged.connect(self._update_comparison_method_config); sy_cfg += s_vs_cfg

        # --- SSIM Specific Group ---
        self.ssimGroup = QGroupBox("Cấu hình SSIM", central_widget)
        self.ssimGroup.setGeometry(rx, sy_cfg, lw, s_vs_cfg + 5)
        self.ssimGroup.setVisible(False) # Ẩn ban đầu
        self.ssimThresholdLabel = QLabel("Ngưỡng SSIM:", self.ssimGroup); self.ssimThresholdLabel.setGeometry(10, 10, lbl_w - 10, 31) # Vị trí bên trong group
        self.ssimThresholdSpinBox = QDoubleSpinBox(self.ssimGroup); self.ssimThresholdSpinBox.setGeometry(sx_ctrl - rx, 10, ctrl_w_main - 10, 31) # Vị trí bên trong group
        self.ssimThresholdSpinBox.setRange(0.1, 1.0); self.ssimThresholdSpinBox.setSingleStep(0.01); self.ssimThresholdSpinBox.setDecimals(3)
        self.ssimThresholdSpinBox.valueChanged.connect(self._update_threshold_config);
        # Chỉ tăng sy_cfg nếu group được hiển thị, nhưng để đơn giản, tính toán trước vị trí y tiếp theo
        sy_after_ssim = sy_cfg + s_vs_cfg + 15

        # --- YOLOv8 Specific Group ---
        self.yoloGroup = QGroupBox("Cấu hình YOLOv8", central_widget)
        yolo_group_height = s_vs_cfg * 2 + 15 # Cao hơn SSIM group
        self.yoloGroup.setGeometry(rx, sy_cfg, lw, yolo_group_height) # Đặt ở cùng Y với SSIM group
        self.yoloGroup.setVisible(False) # Ẩn ban đầu
        # --- Model Path Row (bên trong YOLO group) ---
        lyp_y = 10
        lyp = QLabel("Model Path:", self.yoloGroup); lyp.setGeometry(10, lyp_y, lbl_w - 10, 31)
        yolo_btn_w = 150
        self.yoloModelPathButton = QPushButton("📁 Chọn Model (.pt)", self.yoloGroup); self.yoloModelPathButton.setGeometry(sx_ctrl - rx, lyp_y, yolo_btn_w, 31); self.yoloModelPathButton.clicked.connect(self._select_yolo_model_path)
        # Label hiển thị tên file model (nằm bên phải nút chọn)
        yolo_lbl_x = sx_ctrl - rx + yolo_btn_w + 5
        yolo_lbl_w = lw - yolo_lbl_x - 15 # Chiều rộng còn lại
        self.yoloModelPathLabel = QLabel("Chưa chọn model", self.yoloGroup); self.yoloModelPathLabel.setGeometry(yolo_lbl_x, lyp_y, yolo_lbl_w, 31); self.yoloModelPathLabel.setStyleSheet("font-style: italic; color: gray;")
        # --- Confidence Row (bên trong YOLO group) ---
        conf_y = lyp_y + s_vs_cfg
        self.yoloConfidenceLabel = QLabel("Ngưỡng Conf:", self.yoloGroup); self.yoloConfidenceLabel.setGeometry(10, conf_y, lbl_w - 10, 31)
        self.yoloConfidenceSpinBox = QDoubleSpinBox(self.yoloGroup); self.yoloConfidenceSpinBox.setGeometry(sx_ctrl - rx, conf_y, ctrl_w_main - 10, 31) # Kéo dài control
        self.yoloConfidenceSpinBox.setRange(0.01, 1.0); self.yoloConfidenceSpinBox.setSingleStep(0.05); self.yoloConfidenceSpinBox.setDecimals(2)
        self.yoloConfidenceSpinBox.setToolTip("Ngưỡng tin cậy tối thiểu cho YOLO detection (0.01-1.0)")
        self.yoloConfidenceSpinBox.valueChanged.connect(self._update_yolo_confidence_config);
        # Y position sau group YOLO
        sy_after_yolo = sy_cfg + yolo_group_height + 10
        # Y position cuối cùng cho các cài đặt chung là vị trí lớn nhất của 2 group
        sy_cfg = max(sy_after_ssim, sy_after_yolo)

        # --- Common Settings ---
        lc = QLabel("Cooldown Lỗi (s):", central_widget); lc.setGeometry(sx_lbl, sy_cfg, lbl_w, 31)
        self.cooldownSpinBox = QSpinBox(central_widget); self.cooldownSpinBox.setGeometry(sx_ctrl, sy_cfg, ctrl_w_main, 31); self.cooldownSpinBox.setRange(1, 300); self.cooldownSpinBox.setSingleStep(1); self.cooldownSpinBox.valueChanged.connect(self._update_cooldown_config); sy_cfg += s_vs_cfg

        lr = QLabel("Thời gian chạy (phút):", central_widget); lr.setGeometry(sx_lbl, sy_cfg, lbl_w, 31)
        self.runtimeSpinBox = QSpinBox(central_widget); self.runtimeSpinBox.setGeometry(sx_ctrl, sy_cfg, ctrl_w_main, 31); self.runtimeSpinBox.setRange(0, 1440); self.runtimeSpinBox.setSingleStep(10); self.runtimeSpinBox.setToolTip("0 = Chạy vô hạn"); self.runtimeSpinBox.valueChanged.connect(self._update_runtime_config); sy_cfg += s_vs_cfg

        self.ToggleRecordOnErrorButton = self.create_button("🎥 Quay video lỗi: Tắt"); self.ToggleRecordOnErrorButton.setGeometry(sx_lbl, sy_cfg, ctrl_w_main + (sx_ctrl-sx_lbl), 31) # Kéo dài nút
        self.ToggleRecordOnErrorButton.clicked.connect(self._toggle_record_on_error); sy_cfg += s_vs_cfg + 5 # Thêm khoảng cách nhỏ

        # --- Serial Port Settings Group ---
        serial_group_height = s_vs_cfg * 3 + 15
        self.serialGroup = QGroupBox("Cấu hình Serial COM", central_widget)
        self.serialGroup.setGeometry(rx, sy_cfg, lw, serial_group_height) # Đặt group
        sy_serial = 10 # Y tương đối bên trong groupbox này
        # COM Port Row
        lcp = QLabel("Cổng COM:", self.serialGroup); lcp.setGeometry(10, sy_serial, lbl_w - 10, 31)
        com_combo_w = ctrl_w_main - ctrl_w_sm_btn - 5 # Width cho combo COM
        self.comPortComboBox = QComboBox(self.serialGroup); self.comPortComboBox.setGeometry(sx_ctrl - rx, sy_serial, com_combo_w, 31); self.comPortComboBox.currentTextChanged.connect(self._update_serial_port_config)
        self.refreshComButton = QPushButton("🔄", self.serialGroup); self.refreshComButton.setGeometry(sx_ctrl - rx + com_combo_w + 5, sy_serial, ctrl_w_sm_btn, 31); self.refreshComButton.clicked.connect(self._refresh_com_ports); sy_serial += s_vs_cfg
        # Baud Rate Row
        lbr = QLabel("Baud Rate:", self.serialGroup); lbr.setGeometry(10, sy_serial, lbl_w - 10, 31)
        self.baudRateComboBox = QComboBox(self.serialGroup); self.baudRateComboBox.setGeometry(sx_ctrl - rx, sy_serial, ctrl_w_main - 10, 31); self.baudRateComboBox.addItems([str(br) for br in COMMON_BAUD_RATES]); self.baudRateComboBox.currentTextChanged.connect(self._update_serial_baud_config); sy_serial += s_vs_cfg
        # Toggle Connect/Disconnect Button
        self.ToggleSerialPortButton = self.create_button("🔌 Kết nối COM"); self.ToggleSerialPortButton.setGeometry(10, sy_serial, ctrl_w_main + (sx_ctrl-rx-10), 31) # Kéo dài nút
        self.ToggleSerialPortButton.clicked.connect(self._toggle_serial_port)
        # sy_cfg += serial_group_height + 10 # Y position sau group Serial (nếu cần thêm gì bên dưới)

        # Cập nhật UI ẩn/hiện group SSIM/YOLO ban đầu
        self._update_method_specific_ui()

    def create_button(self, text):
        """Hàm trợ giúp tạo QPushButton với style chuẩn."""
        button = QPushButton(text, self.centralWidget())
        # Có thể thêm style mặc định ở đây nếu muốn
        # button.setStyleSheet("padding: 5px;")
        return button

    # --- Config Save/Load/Reset ---
    def save_config(self):
        """Lưu cấu hình hiện tại vào file JSON."""
        self.config['comparison_method'] = self.current_comparison_method
        self.config['ssim_threshold'] = self._current_ssim_threshold
        self.config['error_cooldown'] = self._current_error_cooldown
        self.config['runtime_duration_minutes'] = self._current_runtime_minutes
        self.config['record_on_error'] = self._record_on_error_enabled
        self.config['error_folder'] = self.error_folder
        self.config['webcam_roi'] = list(self.webcam_roi) if self.webcam_roi else None
        # Lấy giá trị YOLO conf từ UI (an toàn hơn)
        if hasattr(self, 'yoloConfidenceSpinBox'):
             self.config['yolo_confidence'] = self.yoloConfidenceSpinBox.value()
        # yolo_model_path đã được cập nhật khi chọn
        self.config['selected_camera_index'] = self.selected_camera_index # Lưu camera đã chọn

        # Chỉ lưu đường dẫn ảnh SSIM nếu hợp lệ
        valid_ref_paths = {}
        for k, img in self.ref_data.items():
            path_in_config = self.config['ref_paths'].get(k)
            if isinstance(img, np.ndarray) and img.size > 0 and isinstance(path_in_config, str) and os.path.isfile(path_in_config):
                valid_ref_paths[k] = path_in_config
        self.config['ref_paths'] = valid_ref_paths

        # Cấu hình Serial
        self.config['serial_port'] = self.serial_port_name
        self.config['serial_baud'] = self.serial_baud_rate
        # Không lưu serial_enabled là True, để người dùng tự kết nối lại
        self.config['serial_enabled'] = False # Luôn lưu là False

        try:
            config_dir = os.path.dirname(CONFIG_FILE_NAME) or '.'
            os.makedirs(config_dir, exist_ok=True) # Đảm bảo thư mục tồn tại
            with open(CONFIG_FILE_NAME, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            # self.log_activity(f"💾 Đã lưu cấu hình vào '{CONFIG_FILE_NAME}'") # Log nếu cần
        except Exception as e:
            self.log_activity(f"❌ Lỗi khi lưu cấu hình: {e}")
            # Hiển thị lỗi cho người dùng
            QMessageBox.critical(self, "Lỗi Lưu Config", f"Không thể lưu cấu hình vào file:\n{CONFIG_FILE_NAME}\n\nLỗi: {e}")

    def load_config(self):
        """Tải cấu hình từ file JSON, xử lý lỗi và giá trị mặc định."""
        if not os.path.exists(CONFIG_FILE_NAME):
            self.log_activity(f"📄 Không tìm thấy file config '{CONFIG_FILE_NAME}'. Sử dụng mặc định.")
            self.reset_to_defaults() # Reset sẽ cập nhật UI và lưu config mặc định
            return

        try:
            with open(CONFIG_FILE_NAME, 'r', encoding='utf-8') as f:
                lcfg = json.load(f) # Loaded config

            # 1. Phương thức so sánh
            loaded_method = lcfg.get('comparison_method', DEFAULT_COMPARISON_METHOD)
            if loaded_method in self.comparison_functions:
                if loaded_method == METHOD_YOLO and not YOLO_AVAILABLE:
                    self.log_activity(f"⚠️ YOLO được chọn nhưng chưa cài đặt. Đổi về {DEFAULT_COMPARISON_METHOD}.")
                    self.current_comparison_method = DEFAULT_COMPARISON_METHOD
                else:
                    self.current_comparison_method = loaded_method
            else:
                self.log_activity(f"⚠️ Phương thức '{loaded_method}' không hợp lệ. Dùng mặc định.")
                self.current_comparison_method = DEFAULT_COMPARISON_METHOD
            self.config['comparison_method'] = self.current_comparison_method

            # 2. Các giá trị số và boolean
            try: self._current_ssim_threshold = max(0.1, min(1.0, float(lcfg.get('ssim_threshold', DEFAULT_SSIM_THRESHOLD))))
            except (ValueError, TypeError): self._current_ssim_threshold = DEFAULT_SSIM_THRESHOLD
            self.config['ssim_threshold'] = self._current_ssim_threshold

            try: self._current_error_cooldown = max(1, min(300, int(lcfg.get('error_cooldown', DEFAULT_ERROR_COOLDOWN))))
            except (ValueError, TypeError): self._current_error_cooldown = DEFAULT_ERROR_COOLDOWN
            self.config['error_cooldown'] = self._current_error_cooldown

            try: self._current_runtime_minutes = max(0, min(1440, int(lcfg.get('runtime_duration_minutes', DEFAULT_RUNTIME_MINUTES))))
            except (ValueError, TypeError): self._current_runtime_minutes = DEFAULT_RUNTIME_MINUTES
            self.config['runtime_duration_minutes'] = self._current_runtime_minutes

            lrec = lcfg.get('record_on_error', DEFAULT_RECORD_ON_ERROR)
            self._record_on_error_enabled = bool(lrec) if isinstance(lrec, bool) else DEFAULT_RECORD_ON_ERROR
            self.config['record_on_error'] = self._record_on_error_enabled

            # 3. Camera Index đã chọn
            try:
                loaded_idx = int(lcfg.get('selected_camera_index', 0))
                self.selected_camera_index = loaded_idx
            except (ValueError, TypeError):
                self.log_activity("⚠️ Giá trị 'selected_camera_index' không hợp lệ. Dùng mặc định 0.")
                self.selected_camera_index = 0
            self.config['selected_camera_index'] = self.selected_camera_index

            # 4. Serial Port (Luôn đặt enabled=False khi tải)
            self.serial_port_name = lcfg.get('serial_port', None)
            if not isinstance(self.serial_port_name, (str, type(None))): self.serial_port_name = None
            try:
                baud = int(lcfg.get('serial_baud', DEFAULT_BAUD_RATE))
                self.serial_baud_rate = baud if baud in COMMON_BAUD_RATES else DEFAULT_BAUD_RATE
            except (ValueError, TypeError): self.serial_baud_rate = DEFAULT_BAUD_RATE
            self.serial_enabled = False # Không tự động kết nối lại khi khởi động
            # Cập nhật config dict
            self.config['serial_port'] = self.serial_port_name
            self.config['serial_baud'] = self.serial_baud_rate
            self.config['serial_enabled'] = False

            # 5. Thư mục lỗi và ROI
            lfold = lcfg.get('error_folder')
            self.error_folder = None
            if lfold and isinstance(lfold, str):
                 # Kiểm tra sự tồn tại và quyền ghi chỉ khi cần dùng
                 # Chúng ta có thể kiểm tra ở đây nếu muốn, nhưng sẽ làm chậm quá trình tải
                 # if os.path.isdir(lfold) and os.access(lfold, os.W_OK):
                 self.error_folder = lfold
                 # else: self.log_activity(f"⚠️ Thư mục lỗi '{lfold}' không hợp lệ/không ghi được.")
            self.config['error_folder'] = self.error_folder

            lroi = lcfg.get('webcam_roi'); self.webcam_roi = None
            if isinstance(lroi, list) and len(lroi) == 4:
                 try:
                     rt = tuple(int(x) for x in lroi)
                     if all(v >= 0 for v in rt) and rt[2] > 0 and rt[3] > 0: self.webcam_roi = rt
                 except (ValueError, TypeError): pass
            self.config['webcam_roi'] = list(self.webcam_roi) if self.webcam_roi else None

            # 6. Tải đường dẫn ảnh SSIM và load ảnh
            lrefs = lcfg.get('ref_paths', {})
            self.config['ref_paths'] = {k: None for k in self.ref_data.keys()} # Reset trong config
            self.ref_data = {k: None for k in self.ref_data.keys()} # Reset ảnh đã tải
            loaded_image_keys = []
            for k in self.ref_data.keys():
                p = lrefs.get(k)
                if p and isinstance(p, str) and os.path.isfile(p):
                    try:
                        img_bytes = np.fromfile(p, dtype=np.uint8) # Xử lý Unicode path
                        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                        if img is not None:
                            self.ref_data[k] = img
                            self.config['ref_paths'][k] = p # Lưu lại đường dẫn đã tải thành công
                            loaded_image_keys.append(k)
                        # else: self.log_activity(f"⚠️ Không thể decode ảnh SSIM '{k}' từ '{p}'")
                    except Exception as e:
                        self.log_activity(f"⚠️ Lỗi khi tải ảnh SSIM '{k}' từ '{p}': {e}")
            if loaded_image_keys: self.log_activity(f"✅ Tải ảnh SSIM: {', '.join(loaded_image_keys)}")

            # 7. Cấu hình YOLO
            self.config['yolo_model_path'] = lcfg.get('yolo_model_path', None)
            if self.config['yolo_model_path'] and not isinstance(self.config['yolo_model_path'], str):
                 self.config['yolo_model_path'] = None # Validate type
            try:
                yolo_conf = float(lcfg.get('yolo_confidence', DEFAULT_YOLO_CONFIDENCE))
                # Kẹp giá trị trong khoảng hợp lệ
                self.config['yolo_confidence'] = max(0.01, min(1.0, yolo_conf))
            except (ValueError, TypeError):
                self.config['yolo_confidence'] = DEFAULT_YOLO_CONFIDENCE

            # 8. Cập nhật đường dẫn file log (dựa trên error_folder vừa tải)
            self.log_file_path = os.path.join(self.error_folder, LOG_FILE_NAME) if self.error_folder else None
            self.log_activity(f"💾 Đã tải cấu hình từ '{CONFIG_FILE_NAME}'.")

        except json.JSONDecodeError as e:
             self.log_activity(f"❌ Lỗi JSON trong file config: {e}. Sử dụng mặc định.")
             self.reset_to_defaults()
        except Exception as e:
             self.log_activity(f"❌ Lỗi nghiêm trọng khi tải config: {e}. Sử dụng mặc định.")
             self.log_activity(traceback.format_exc())
             self.reset_to_defaults()

        # Lưu ý: self.update_all_ui_elements() nên được gọi sau load_config ở __init__

    def reset_to_defaults(self):
        """Reset cấu hình về giá trị mặc định."""
        self.log_activity("🔄 Reset về mặc định...")

        # Reset State Variables
        self.current_comparison_method = DEFAULT_COMPARISON_METHOD
        self._current_ssim_threshold = DEFAULT_SSIM_THRESHOLD
        self._current_error_cooldown = DEFAULT_ERROR_COOLDOWN
        self._current_runtime_minutes = DEFAULT_RUNTIME_MINUTES
        self._record_on_error_enabled = DEFAULT_RECORD_ON_ERROR
        self.error_folder = None
        self.log_file_path = None # Sẽ được đặt lại dựa trên error_folder (hiện là None)
        self.webcam_roi = None
        self.ref_data = {k: None for k in [REF_NORM, REF_SHUTDOWN, REF_FAIL]}
        self.selected_camera_index = 0 # Reset camera index về 0
        self.available_cameras = {}    # Xóa list camera cũ

        # Giải phóng model YOLO
        if self.yolo_model is not None:
            try: del self.yolo_model; self.yolo_model = None
            except Exception: pass # Ignore error on deletion

        # Reset Serial State
        self.serial_port_name = None
        self.serial_baud_rate = DEFAULT_BAUD_RATE
        self.serial_enabled = False
        if self.serial_port and self.serial_port.is_open:
            try: self.serial_port.close()
            except Exception: pass
        self.serial_port = None

        # Reset Config Dictionary về mặc định
        self.config = {
            'comparison_method': DEFAULT_COMPARISON_METHOD,
            'ssim_threshold': DEFAULT_SSIM_THRESHOLD,
            'error_cooldown': DEFAULT_ERROR_COOLDOWN,
            'runtime_duration_minutes': DEFAULT_RUNTIME_MINUTES,
            'record_on_error': DEFAULT_RECORD_ON_ERROR,
            'error_folder': None,
            'ref_paths': {k: None for k in [REF_NORM, REF_SHUTDOWN, REF_FAIL]},
            'webcam_roi': None,
            'serial_port': None,
            'serial_baud': DEFAULT_BAUD_RATE,
            'serial_enabled': False,
            'selected_camera_index': 0, # Reset trong config
            'yolo_model_path': None,
            'yolo_confidence': DEFAULT_YOLO_CONFIDENCE,
        }

        # Cập nhật lại giao diện người dùng và refresh lists
        if hasattr(self, 'comparisonMethodComboBox'): # Kiểm tra UI đã init chưa
            self.update_all_ui_elements() # Cập nhật tất cả các control
            # Refresh các danh sách động sau khi update UI chính
            QTimer.singleShot(50, self._refresh_camera_list) # Làm mới camera list
            QTimer.singleShot(100, self._refresh_com_ports) # Làm mới COM port list

        # Lưu lại file config với giá trị mặc định mới
        self.save_config()
        self.log_activity("🔄 Hoàn tất reset về mặc định.")

    # --- Camera Detection and Selection ---
    def _detect_available_cameras(self):
        """
        Quét các chỉ số camera và trả về dict {index: "Tên hiển thị"}.
        Sử dụng cách thử mở để kiểm tra, cẩn thận hơn để tránh treo.
        """
        detected_cameras = {}
        self.log_activity("🔄 Đang quét tìm camera...")
        # Ưu tiên các backend API hoạt động tốt hơn trên Windows
        preferred_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] # cv2.CAP_ANY là mặc định
        max_test_index = 8 # Giới hạn số lượng camera kiểm tra để tránh quá lâu

        for index in range(max_test_index):
            temp_cap = None
            opened_success = False
            backend_used_name = "N/A"

            for backend in preferred_backends:
                try:
                    # Cố gắng mở camera với API cụ thể
                    temp_cap = cv2.VideoCapture(index, backend)
                    # Kiểm tra xem đã mở được chưa và có đọc được frame không
                    if temp_cap and temp_cap.isOpened():
                        # Đọc thử 1 frame để chắc chắn camera hoạt động
                        ret_test, _ = temp_cap.read()
                        temp_cap.release() # Giải phóng ngay sau khi kiểm tra
                        if ret_test: # Nếu đọc frame thành công
                            backend_used_name = "Unknown"
                            try: # Thử lấy tên backend API (có thể thất bại)
                                # Cần tạo lại capture để lấy tên? Khá tốn kém.
                                # Tạm thời bỏ qua lấy tên backend để đơn giản và nhanh hơn.
                                pass # backend_used_name = temp_cap.getBackendName()
                            except: pass

                            display_name = f"Camera {index}"
                            # if backend_used_name != "N/A": display_name += f" ({backend_used_name})"
                            detected_cameras[index] = display_name
                            opened_success = True
                            break # Tìm thấy backend hoạt động cho index này, chuyển sang index tiếp theo
                    elif temp_cap:
                         temp_cap.release() # Đóng nếu chỉ isOpened() trả về False

                except Exception:
                    # Lỗi khi thử mở camera (bỏ qua và thử backend/index khác)
                    if temp_cap: # Đảm bảo đóng nếu lỗi xảy ra sau khi tạo object
                        try: temp_cap.release()
                        except Exception: pass

            # Optional: Nếu không mở được index > 0, có thể dừng sớm
            # if index > 0 and not opened_success:
            #     break

        if detected_cameras:
             log_msg = "✅ Tìm thấy camera: " + ", ".join(f"{name} (Index {idx})" for idx, name in sorted(detected_cameras.items()))
             self.log_activity(log_msg)
        else:
             self.log_activity("🟠 Không tìm thấy camera nào hoạt động.")
        return detected_cameras

    @QtCore.pyqtSlot()
    def _refresh_camera_list(self):
        """Quét lại camera và cập nhật ComboBox camera."""
        if self.processing or (self.cap and self.cap.isOpened()):
            self.log_activity("ℹ️ Không thể làm mới danh sách khi webcam đang bật hoặc đang xử lý.")
            return

        # Quét lại các camera hiện có
        self.available_cameras = self._detect_available_cameras()

        # Chặn tín hiệu của ComboBox để tránh kích hoạt _on_camera_selection_changed
        self.cameraSelectionComboBox.blockSignals(True)
        self.cameraSelectionComboBox.clear() # Xóa các mục cũ

        # Thêm các camera tìm thấy vào ComboBox
        if self.available_cameras:
            for index, name in sorted(self.available_cameras.items()):
                # Thêm tên hiển thị và lưu chỉ số thực vào userData
                self.cameraSelectionComboBox.addItem(name, userData=index)

            # Cố gắng khôi phục lựa chọn camera đã lưu trong config
            restored = False
            if self.selected_camera_index in self.available_cameras: # Nếu index đã lưu còn tồn tại
                for i in range(self.cameraSelectionComboBox.count()):
                    if self.cameraSelectionComboBox.itemData(i) == self.selected_camera_index:
                        self.cameraSelectionComboBox.setCurrentIndex(i) # Chọn lại mục đó
                        restored = True
                        break
            if not restored and self.cameraSelectionComboBox.count() > 0:
                # Nếu không khôi phục được (index cũ không còn), chọn camera đầu tiên trong danh sách
                self.cameraSelectionComboBox.setCurrentIndex(0)
                new_default_index = self.cameraSelectionComboBox.itemData(0)
                # Chỉ log và cập nhật nếu index thực sự thay đổi
                if self.selected_camera_index != new_default_index:
                     self.log_activity(f"⚠️ Camera đã chọn (Index {self.selected_camera_index}) không còn. Chọn mặc định: {self.cameraSelectionComboBox.currentText()}")
                     self.selected_camera_index = new_default_index
                     self.config['selected_camera_index'] = self.selected_camera_index # Cập nhật config

            # Bật các control liên quan đến camera
            self.cameraSelectionComboBox.setEnabled(True)
            self.refreshCamerasButton.setEnabled(True)
            self.ONCam.setEnabled(True) # Bật nút Bật Webcam
        else:
            # Nếu không tìm thấy camera nào
            self.cameraSelectionComboBox.addItem("Không tìm thấy camera")
            self.cameraSelectionComboBox.setEnabled(False)
            self.refreshCamerasButton.setEnabled(True) # Vẫn cho phép refresh lại
            self.ONCam.setEnabled(False) # Tắt nút Bật Webcam
            # Đặt index về giá trị không hợp lệ nếu không có camera
            if self.selected_camera_index != -1:
                 self.selected_camera_index = -1
                 self.config['selected_camera_index'] = -1

        # Bỏ chặn tín hiệu ComboBox
        self.cameraSelectionComboBox.blockSignals(False)
        # Cập nhật trạng thái enable/disable chung (quan trọng sau khi refresh)
        self._update_controls_state()


    @QtCore.pyqtSlot(int)
    def _on_camera_selection_changed(self, index):
        """Xử lý khi người dùng chọn camera khác từ ComboBox."""
        # index là vị trí trong combobox, không phải index camera thực
        if index < 0: return # Xảy ra khi clear combobox

        selected_data = self.cameraSelectionComboBox.itemData(index)
        if selected_data is not None and isinstance(selected_data, int):
            new_index = selected_data
            # Chỉ cập nhật nếu index thực sự thay đổi so với trạng thái hiện tại
            if self.selected_camera_index != new_index:
                self.selected_camera_index = new_index
                self.config['selected_camera_index'] = self.selected_camera_index # Cập nhật config
                self.log_activity(f"📹 Đã chọn camera: {self.cameraSelectionComboBox.currentText()} (Index: {self.selected_camera_index})")
                # self.save_config() # Có thể lưu ngay
        else:
            # Trường hợp lỗi: userData không phải là số nguyên
            self.log_activity(f"⚠️ Lỗi: Không lấy được index camera từ lựa chọn '{self.cameraSelectionComboBox.currentText()}'")


    # --- UI Update and State Management ---
    def update_all_ui_elements(self):
        """Cập nhật tất cả các control trên UI để phản ánh trạng thái/config hiện tại."""
        # self.log_activity("ℹ️ Cập nhật giao diện người dùng...") # Có thể bỏ log này nếu quá nhiều
        # Danh sách các control cần chặn tín hiệu khi cập nhật giá trị từ code
        controls_to_block = [
            self.comparisonMethodComboBox, self.ssimThresholdSpinBox, self.yoloConfidenceSpinBox,
            self.cooldownSpinBox, self.runtimeSpinBox, self.comPortComboBox, self.baudRateComboBox,
            self.cameraSelectionComboBox # Thêm combobox camera
        ]
        # Chặn tín hiệu
        for control in controls_to_block:
            if hasattr(self, control.objectName()): # Kiểm tra control tồn tại
                 control.blockSignals(True)

        # Cập nhật giá trị cho từng control từ self.config hoặc biến trạng thái
        try:
            if hasattr(self, 'comparisonMethodComboBox'): self.comparisonMethodComboBox.setCurrentText(self.current_comparison_method)
            if hasattr(self, 'ssimThresholdSpinBox'): self.ssimThresholdSpinBox.setValue(self._current_ssim_threshold)
            if hasattr(self, 'yoloConfidenceSpinBox'): self.yoloConfidenceSpinBox.setValue(self.config.get('yolo_confidence', DEFAULT_YOLO_CONFIDENCE))
            if hasattr(self, 'cooldownSpinBox'): self.cooldownSpinBox.setValue(self._current_error_cooldown)
            if hasattr(self, 'runtimeSpinBox'): self.runtimeSpinBox.setValue(self._current_runtime_minutes)
            if hasattr(self, 'baudRateComboBox'): self.baudRateComboBox.setCurrentText(str(self.serial_baud_rate))

            # Cập nhật ComboBox cổng COM
            if hasattr(self, 'comPortComboBox'):
                # Danh sách này được refresh bởi _refresh_com_ports, chỉ cần set index
                com_index = self.comPortComboBox.findText(self.serial_port_name if self.serial_port_name else "")
                if self.serial_port_name and com_index >= 0:
                    self.comPortComboBox.setCurrentIndex(com_index)
                elif self.comPortComboBox.count() > 0:
                    # Nếu cổng lưu không có, không tự động chọn cổng khác ở đây
                    # _refresh_com_ports sẽ xử lý việc chọn cổng mặc định
                    pass

            # Cập nhật ComboBox Camera
            if hasattr(self, 'cameraSelectionComboBox') and self.cameraSelectionComboBox.count() > 0:
                 found_cam_idx_in_combo = -1
                 for i in range(self.cameraSelectionComboBox.count()):
                     if self.cameraSelectionComboBox.itemData(i) == self.selected_camera_index:
                         found_cam_idx_in_combo = i
                         break
                 if found_cam_idx_in_combo != -1:
                      self.cameraSelectionComboBox.setCurrentIndex(found_cam_idx_in_combo)
                 # else: Index đã lưu không có trong list hiện tại, giữ nguyên lựa chọn mặc định của _refresh

            # Cập nhật label đường dẫn model YOLO
            if hasattr(self, 'yoloModelPathLabel'):
                model_path = self.config.get('yolo_model_path')
                if model_path and isinstance(model_path, str):
                    # Chỉ hiển thị tên file, tooltip là đường dẫn đầy đủ
                    base_name = os.path.basename(model_path)
                    self.yoloModelPathLabel.setText(base_name)
                    self.yoloModelPathLabel.setStyleSheet("font-style: normal; color: black;")
                    self.yoloModelPathLabel.setToolTip(model_path)
                else:
                    self.yoloModelPathLabel.setText("Chưa chọn model")
                    self.yoloModelPathLabel.setStyleSheet("font-style: italic; color: gray;")
                    self.yoloModelPathLabel.setToolTip("")

        except Exception as ui_update_err:
            self.log_activity(f"❌ Lỗi khi cập nhật giá trị UI: {ui_update_err}")
            self.log_activity(traceback.format_exc()) # Log chi tiết lỗi
        finally:
            # Bỏ chặn tín hiệu (luôn thực hiện)
            for control in controls_to_block:
                if hasattr(self, control.objectName()): control.blockSignals(False)

        # Cập nhật style các nút và hiển thị/ẩn group
        self.update_button_styles()       # Các nút ảnh tham chiếu, ROI, Folder
        self.update_toggle_button_text()  # Nút Start/Stop processing
        self.update_record_button_style() # Nút Record on error
        self.update_serial_button_style() # Nút Connect COM
        self._update_method_specific_ui() # Ẩn/hiện group SSIM/YOLO

        # Bật/tắt control dựa trên trạng thái tổng thể (quan trọng)
        self._update_controls_state()
        # self.log_activity("ℹ️ Cập nhật giao diện hoàn tất.")

    def _update_method_specific_ui(self):
        """Hiển thị/ẩn các group cấu hình và nút liên quan dựa trên phương thức."""
        is_ssim = (self.current_comparison_method == METHOD_SSIM)
        is_yolo = (self.current_comparison_method == METHOD_YOLO)

        # Ẩn/hiện các group config
        if hasattr(self, 'ssimGroup'): self.ssimGroup.setVisible(is_ssim)
        if hasattr(self, 'yoloGroup'): self.yoloGroup.setVisible(is_yolo)

        # Cập nhật trạng thái và tooltip của các nút liên quan đến SSIM
        # (Vì các nút này có thể bị ảnh hưởng bởi việc chọn phương thức)
        self.update_button_styles()

        # Cập nhật trạng thái enable/disable tổng thể
        self._update_controls_state()

    def _update_controls_state(self):
        """
        Hàm tập trung cập nhật trạng thái Enabled/Disabled của các control
        dựa trên các trạng thái: processing, webcam_running, serial_connected, etc.
        """
        webcam_is_running = self.cap is not None and self.cap.isOpened()
        is_busy_processing = self.processing
        is_busy = is_busy_processing or webcam_is_running # Busy nếu đang xử lý HOẶC webcam đang chạy
        can_config_general = not is_busy_processing # Có thể config nếu không xử lý (cho phép config khi webcam chạy)
        can_interact_webcam = not is_busy_processing # Có thể bật/tắt/chọn webcam nếu không xử lý

        is_ssim = self.current_comparison_method == METHOD_SSIM
        is_yolo = self.current_comparison_method == METHOD_YOLO
        has_cameras = bool(self.available_cameras)
        has_com_ports = hasattr(self, 'comPortComboBox') and self.comPortComboBox.count() > 0 and "Không tìm thấy" not in self.comPortComboBox.itemText(0)


        # --- Left Panel Controls ---
        if hasattr(self, 'ONCam'): self.ONCam.setEnabled(can_interact_webcam and has_cameras and not webcam_is_running)
        if hasattr(self, 'OFFCam'): self.OFFCam.setEnabled(can_interact_webcam and webcam_is_running)
        if hasattr(self, 'cameraSelectionComboBox'): self.cameraSelectionComboBox.setEnabled(can_interact_webcam and has_cameras and not webcam_is_running)
        if hasattr(self, 'refreshCamerasButton'): self.refreshCamerasButton.setEnabled(can_interact_webcam and not webcam_is_running)

        # Nút ảnh tham chiếu SSIM (Tải/Chụp)
        can_config_ssim_refs = can_config_general # Hiện tại cho phép load/chụp khi webcam chạy nhưng ko xử lý
        if hasattr(self, 'SettingButton_Norm'):
            for btn in [self.SettingButton_Norm, self.SettingButton_Shutdown, self.SettingButton_Fail]:
                btn.setEnabled(can_config_ssim_refs)
        if hasattr(self, 'CaptureButton_Norm'):
             for btn in [self.CaptureButton_Norm, self.CaptureButton_Shut, self.CaptureButton_Fail]:
                 # Chỉ bật chụp nếu: không xử lý, webcam đang chạy, VÀ là mode SSIM
                 btn.setEnabled(can_config_ssim_refs and webcam_is_running and is_ssim)

        # ROI Button
        if hasattr(self, 'SettingButton_ROI_Webcam'):
             # Bật nếu không xử lý và webcam đang chạy
             self.SettingButton_ROI_Webcam.setEnabled(can_config_ssim_refs and webcam_is_running)

        # Nút thư mục lỗi
        if hasattr(self, 'SaveButton'): self.SaveButton.setEnabled(can_config_general)
        # Nút Start/Stop processing - Logic này đã có trong update_toggle_button_text, nhưng setEnabled ở đây
        # Nút Start chỉ bật khi không xử lý và các điều kiện cần đã đủ (có cam, folder, ảnh ref/model...)
        # Tạm thời chỉ bật khi không processing
        if hasattr(self, 'ToggleProcessingButton'): self.ToggleProcessingButton.setEnabled(True) # Bật/tắt chủ yếu dựa vào self.processing

        # --- Right Panel Controls ---
        if hasattr(self, 'comparisonMethodComboBox'): self.comparisonMethodComboBox.setEnabled(can_config_general)
        if hasattr(self, 'ssimGroup'): self.ssimGroup.setEnabled(can_config_general and is_ssim)
        if hasattr(self, 'yoloGroup'): self.yoloGroup.setEnabled(can_config_general and is_yolo)
        # Model path button trong YOLO group
        if hasattr(self, 'yoloModelPathButton'): self.yoloModelPathButton.setEnabled(can_config_general and is_yolo)

        # Common settings
        if hasattr(self, 'cooldownSpinBox'): self.cooldownSpinBox.setEnabled(can_config_general)
        if hasattr(self, 'runtimeSpinBox'): self.runtimeSpinBox.setEnabled(can_config_general)
        if hasattr(self, 'ToggleRecordOnErrorButton'): self.ToggleRecordOnErrorButton.setEnabled(can_config_general)

        # Serial configuration group
        can_config_serial = can_config_general and not self.serial_enabled # Config nếu ko xử lý VÀ chưa kết nối COM
        if hasattr(self, 'serialGroup'): self.serialGroup.setEnabled(can_config_general) # Group có thể bật ngay cả khi đã kết nối (để thấy nút Disconnect)
        if hasattr(self, 'comPortComboBox'): self.comPortComboBox.setEnabled(can_config_serial and has_com_ports)
        if hasattr(self, 'baudRateComboBox'): self.baudRateComboBox.setEnabled(can_config_serial and has_com_ports)
        if hasattr(self, 'refreshComButton'): self.refreshComButton.setEnabled(can_config_general and not self.serial_enabled) # Refresh chỉ khi ko xử lý và chưa kết nối
        if hasattr(self, 'ToggleSerialPortButton'): self.ToggleSerialPortButton.setEnabled(can_config_general and has_com_ports) # Nút kết nối/ngắt kết nối bật khi ko xử lý và có cổng


    # --- Logging and Status Updates ---
    @QtCore.pyqtSlot(str)
    def log_activity(self, message):
        """Ghi log vào UI và file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{timestamp} - {message}"

        # Cập nhật UI (đảm bảo thread-safe)
        if hasattr(self, 'log_text_edit'):
            log_widget = self.log_text_edit
            if QtCore.QThread.currentThread() != log_widget.thread():
                # Gọi hàm append và ensureCursorVisible trên luồng của widget
                QtCore.QMetaObject.invokeMethod(log_widget, "append", Qt.QueuedConnection, QtCore.Q_ARG(str, full_message))
                QtCore.QMetaObject.invokeMethod(log_widget, "ensureCursorVisible", Qt.QueuedConnection)
            else:
                # Nếu đang ở luồng chính, gọi trực tiếp
                log_widget.append(full_message)
                log_widget.ensureCursorVisible()

        # Ghi vào file log
        if self.log_file_path:
            try:
                # Kiểm tra và tạo thư mục nếu chưa tồn tại
                log_dir = os.path.dirname(self.log_file_path)
                if log_dir and not os.path.exists(log_dir):
                     try:
                         os.makedirs(log_dir, exist_ok=True)
                     except OSError as e:
                         # Lỗi khi tạo thư mục, vô hiệu hóa ghi log và báo lỗi 1 lần
                         print(f"CRITICAL: Lỗi tạo thư mục log '{log_dir}': {e}. Vô hiệu hóa ghi file log.")
                         self.log_file_path = None
                         return # Không ghi file nữa

                # Mở file để ghi thêm (append)
                with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(full_message + "\n")
            except Exception as e:
                # Lỗi nghiêm trọng khi ghi log, thông báo trên console và vô hiệu hóa ghi file
                print(f"CRITICAL: Lỗi ghi file log '{self.log_file_path}': {e}. Vô hiệu hóa ghi log.")
                self.log_file_path = None # Dừng ghi file nếu có lỗi

    @QtCore.pyqtSlot(ComparisonStatus, object)
    def update_status_label(self, status_enum, details_dict):
        """Cập nhật QLabel hiển thị trạng thái chính (Normal, Fail, etc.)."""
        status_info = STATUS_MAP.get(status_enum, STATUS_MAP[ComparisonStatus.ERROR])
        message = status_info["label"]
        background_color = status_info["color"]

        # Hàm lambda để cập nhật UI
        _update = lambda: (
            self.process_label.setText(f"Trạng thái: {message}"),
            self.process_label.setStyleSheet(f"border:1px solid black; padding:5px; background-color:{background_color}; color:black; font-weight:bold; border-radius:3px;")
        )
        # Đảm bảo cập nhật trên luồng chính
        if hasattr(self, 'process_label'):
            if self.process_label.thread() != QtCore.QThread.currentThread():
                 # Sử dụng helper slot để gọi lambda trên luồng chính
                 QtCore.QMetaObject.invokeMethod(self, "_call_lambda_slot", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(object, _update))
            else:
                _update() # Gọi trực tiếp nếu đang ở luồng chính

    @QtCore.pyqtSlot(dict)
    def update_details_display(self, details):
         """Cập nhật QLabel hiển thị chi tiết (SSIM score, YOLO detects)."""
         display_text = "Details: N/A" # Mặc định
         if details: # Nếu dictionary details không rỗng
             # Ưu tiên hiển thị lỗi nếu có
             if 'error' in details:
                 display_text = f"Error: {details['error']}"
             # Hiển thị kết quả YOLO
             elif 'detected' in details:
                 # Sắp xếp theo tên class để hiển thị nhất quán
                 det_items = sorted([f"{k}:{v}" for k,v in details['detected'].items()])
                 display_text = f"Detect: {', '.join(det_items) if det_items else 'None'}"
                 if 'count' in details: display_text += f" (Total: {details['count']})"
             # Hiển thị kết quả SSIM
             elif 'ssim_norm' in details:
                 # Hiển thị các score SSIM có sẵn
                 scores = []
                 if 'ssim_norm' in details and details['ssim_norm'] is not None: scores.append(f"N:{details['ssim_norm']:.4f}")
                 if 'ssim_shutdown' in details and details['ssim_shutdown'] is not None: scores.append(f"S:{details['ssim_shutdown']:.4f}")
                 if 'ssim_fail' in details and details['ssim_fail'] is not None: scores.append(f"F:{details['ssim_fail']:.4f}")
                 display_text = f"SSIM: {', '.join(scores)}" if scores else "SSIM: (Error)"

             # Thêm lý do (reason) nếu có (từ logic check rule YOLO)
             if 'reason' in details:
                 display_text += f" [{details['reason']}]"

         # Hàm lambda để cập nhật text của label
         _update = lambda: self.details_label.setText(display_text)
         # Đảm bảo cập nhật trên luồng chính
         if hasattr(self, 'details_label'):
            if self.details_label.thread() != QtCore.QThread.currentThread():
                 QtCore.QMetaObject.invokeMethod(self, "_call_lambda_slot", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(object, _update))
            else:
                _update()

    @QtCore.pyqtSlot(object)
    def _call_lambda_slot(self, f):
        """Slot trợ giúp để thực thi một hàm lambda được truyền từ luồng khác."""
        try:
            f()
        except Exception as e:
             self.log_activity(f"❌ Lỗi thực thi lambda trên luồng chính: {e}")
             self.log_activity(traceback.format_exc())

    # --- Button Style Updates ---
    def _set_button_style(self, button, base_text, icon="", state_text="", background_color="white", text_color="black"):
        """Helper function để đặt text và style cho nút."""
        if not hasattr(self, button.objectName()): return # Bỏ qua nếu nút chưa được tạo
        full_text = f"{icon} {base_text}".strip() + (f" ({state_text})" if state_text else "")
        button.setText(full_text)
        style = f"""
            QPushButton {{
                background-color: {background_color}; color: {text_color};
                border: 1px solid #ccc; border-radius: 3px; padding: 5px; /* Giảm padding */
                text-align: center;
            }}
            QPushButton:hover {{ background-color: #e8f0fe; }} /* Màu nhạt khi hover */
            QPushButton:pressed {{ background-color: #d0e0f8; }} /* Màu đậm hơn khi nhấn */
            QPushButton:disabled {{
                background-color: #f0f0f0; color: #a0a0a0; border-color: #d0d0d0;
            }}
        """
        button.setStyleSheet(style)

    def update_button_styles(self):
        """Cập nhật style cho các nút ảnh tham chiếu, ROI, thư mục lỗi."""
        # Nút ảnh tham chiếu SSIM
        is_ssim = self.current_comparison_method == METHOD_SSIM
        button_map = {
            REF_NORM: (getattr(self, 'SettingButton_Norm', None), getattr(self, 'CaptureButton_Norm', None), "Ảnh Norm", "Chụp Norm"),
            REF_SHUTDOWN: (getattr(self, 'SettingButton_Shutdown', None), getattr(self, 'CaptureButton_Shut', None), "Ảnh Shutdown", "Chụp Shutdown"),
            REF_FAIL: (getattr(self, 'SettingButton_Fail', None), getattr(self, 'CaptureButton_Fail', None), "Ảnh Fail", "Chụp Fail"),
        }
        icon_load = "📂"
        icon_capture = "📸"

        for key, (load_btn, cap_btn, load_txt, cap_txt) in button_map.items():
            if not load_btn or not cap_btn: continue # Bỏ qua nếu nút chưa tồn tại

            has_image = isinstance(self.ref_data.get(key), np.ndarray) and self.ref_data[key].size > 0
            # Ảnh được coi là từ file nếu có ảnh VÀ có đường dẫn hợp lệ trong config
            is_from_file = has_image and isinstance(self.config['ref_paths'].get(key), str) and os.path.isfile(self.config['ref_paths'][key])
            # Thêm "(SSIM)" nếu phương thức hiện tại không phải SSIM
            hint = "" if is_ssim else "(SSIM)"

            # Đặt tooltip
            load_tooltip = f"Tải ảnh tham chiếu {key} từ file" + (" (Chỉ dùng cho SSIM)" if not is_ssim else "")
            cap_tooltip = f"Chụp ảnh tham chiếu {key} từ webcam" + (" (Chỉ dùng cho SSIM)" if not is_ssim else "")
            load_btn.setToolTip(load_tooltip)
            cap_btn.setToolTip(cap_tooltip)

            # Đặt style và màu nền
            load_state_text = "File" if is_from_file else ""
            load_bg = "lightgreen" if is_from_file and is_ssim else ("lightgray" if is_from_file else "white")
            self._set_button_style(load_btn, f"{load_txt} {hint}", icon_load, load_state_text, background_color=load_bg)

            cap_state_text = "Webcam" if has_image and not is_from_file else ""
            cap_bg = "lightblue" if has_image and not is_from_file and is_ssim else ("lightgray" if has_image and not is_from_file else "white")
            self._set_button_style(cap_btn, f"{cap_txt} {hint}", icon_capture, cap_state_text, background_color=cap_bg)

        # Nút ROI và Thư mục lỗi
        if hasattr(self, 'SettingButton_ROI_Webcam'):
            roi_state = "Đã chọn" if self.webcam_roi else ""
            roi_bg = "lightblue" if self.webcam_roi else "white"
            self._set_button_style(self.SettingButton_ROI_Webcam, "Chọn ROI", "✂️", roi_state, background_color=roi_bg)

        if hasattr(self, 'SaveButton'):
            folder_state = "Đã chọn" if self.error_folder else ""
            folder_bg = "lightblue" if self.error_folder else "white"
            self._set_button_style(self.SaveButton, "Thư mục lỗi", "📁", folder_state, background_color=folder_bg)

    def update_toggle_button_text(self):
        """Cập nhật text và màu nút Start/Stop processing."""
        if hasattr(self, 'ToggleProcessingButton'):
            if self.processing:
                self._set_button_style(self.ToggleProcessingButton, "Dừng Xử lý", "⏹", background_color="orange")
            else:
                self._set_button_style(self.ToggleProcessingButton, "Bắt đầu", "▶️", background_color="lightgreen")

    def update_record_button_style(self):
        """Cập nhật text và màu nút Record on Error."""
        if hasattr(self, 'ToggleRecordOnErrorButton'):
            state = "Bật" if self._record_on_error_enabled else "Tắt"
            color = "lightcoral" if self._record_on_error_enabled else "lightgray"
            self._set_button_style(self.ToggleRecordOnErrorButton, "Quay video lỗi", "🎥", state, background_color=color)

    def update_serial_button_style(self):
        """Cập nhật text và màu nút Kết nối/Ngắt kết nối COM."""
        if hasattr(self, 'ToggleSerialPortButton'):
            if self.serial_enabled:
                # Đang kết nối -> nút để Ngắt kết nối
                self._set_button_style(self.ToggleSerialPortButton, "Ngắt kết nối COM", "🔌", "Đang kết nối", background_color="lightcoral")
            else:
                # Chưa kết nối -> nút để Kết nối
                has_ports = hasattr(self,'comPortComboBox') and self.comPortComboBox.count() > 0 and "Không tìm thấy" not in self.comPortComboBox.itemText(0)
                state = "Chưa kết nối" if has_ports else "Không có cổng"
                color = "lightgreen" if has_ports else "lightgray"
                self._set_button_style(self.ToggleSerialPortButton, "Kết nối COM", "🔌", state, background_color=color)


    # --- Webcam Handling ---
    def start_webcam(self):
        """Bật webcam đã chọn."""
        if self.cap is not None and self.cap.isOpened():
            self.log_activity("⚠️ Webcam đã được bật.")
            return

        # Kiểm tra xem camera hợp lệ đã được chọn chưa
        if self.selected_camera_index < 0 or self.selected_camera_index not in self.available_cameras:
             msg = "Vui lòng chọn một camera hợp lệ từ danh sách."
             if not self.available_cameras: msg = "Không tìm thấy camera nào. Hãy thử làm mới danh sách (🔄)."
             self.log_activity(f"❌ {msg}")
             QMessageBox.warning(self, "Chưa Chọn Camera", msg)
             # Tự động làm mới danh sách nếu chưa có lựa chọn hợp lệ
             if not self.available_cameras:
                 QTimer.singleShot(50, self._refresh_camera_list)
             return

        cam_index_to_use = self.selected_camera_index # Lấy index đã chọn
        cam_display_name = self.cameraSelectionComboBox.currentText() # Lấy tên hiển thị

        # Tắt các control chọn camera trước khi thử mở
        if hasattr(self,'cameraSelectionComboBox'): self.cameraSelectionComboBox.setEnabled(False)
        if hasattr(self,'refreshCamerasButton'): self.refreshCamerasButton.setEnabled(False)
        if hasattr(self,'ONCam'): self.ONCam.setEnabled(False) # Tạm thời disable nút Bật

        try:
            # Ưu tiên các backend API
            preferred_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] # CAP_ANY là mặc định
            self.cap = None
            opened_backend_name = "N/A"
            capture_opened_successfully = False

            self.log_activity(f"ℹ️ Đang thử mở '{cam_display_name}' (Index: {cam_index_to_use})...")
            for backend in preferred_backends:
                temp_cap = None # Reset temp_cap cho mỗi lần thử backend
                try:
                    # Thử mở camera với index và backend cụ thể
                    temp_cap = cv2.VideoCapture(cam_index_to_use, backend)

                    # Kiểm tra xem có mở được và đọc được frame không
                    if temp_cap and temp_cap.isOpened():
                        # Quan trọng: Đọc thử một frame để đảm bảo hoạt động
                        ret_test, frame_test = temp_cap.read()
                        if ret_test and frame_test is not None and frame_test.size > 0:
                            # Mở thành công VÀ đọc được frame
                            self.cap = temp_cap # Giữ lại capture object này
                            capture_opened_successfully = True
                            # Cố gắng lấy tên backend (có thể thất bại)
                            try: opened_backend_name = self.cap.getBackendName()
                            except: opened_backend_name = f"API_{backend}" # Tên thay thế nếu getBackendName lỗi
                            self.log_activity(f"✅ Mở thành công '{cam_display_name}' với backend: {opened_backend_name}")
                            break # Thoát vòng lặp backend khi đã thành công
                        else:
                             # Mở được nhưng không đọc được frame -> giải phóng và thử backend khác
                             if temp_cap: temp_cap.release()
                             self.log_activity(f"ℹ️ Backend {backend} cho '{cam_display_name}' mở được nhưng không đọc được frame.")
                    elif temp_cap:
                        # isOpened() trả về False -> giải phóng
                        temp_cap.release()

                except Exception as cam_err:
                    # Lỗi trong quá trình thử mở (ví dụ: backend không hỗ trợ)
                    self.log_activity(f"ℹ️ Lỗi khi thử '{cam_display_name}' với backend {backend}: {cam_err}")
                    if temp_cap: # Đảm bảo đóng nếu lỗi xảy ra sau khi tạo
                        try: temp_cap.release()
                        except Exception: pass

            # Kiểm tra sau khi thử hết các backend
            if not capture_opened_successfully or self.cap is None or not self.cap.isOpened():
                 raise IOError(f"Không thể mở '{cam_display_name}' hoặc đọc frame ban đầu sau khi thử các backend.")

            # --- Webcam đã mở thành công ---
            # Lấy thông tin kích thước và FPS
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Cố gắng đặt FPS mong muốn (không phải lúc nào cũng thành công)
            requested_fps = 15.0
            set_fps_success = self.cap.set(cv2.CAP_PROP_FPS, requested_fps)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            # Nếu không lấy được FPS, FPS=0 hoặc quá thấp, dùng giá trị mặc định
            if not actual_fps or actual_fps < 1: actual_fps = requested_fps
            self.webcam_fps = actual_fps
            # Tính khoảng thời gian timer (ms), tối thiểu ~30fps, tối đa 1s
            timer_interval = max(33, min(1000, int(1000 / self.webcam_fps)))

            self.log_activity(f"🚀 Webcam '{cam_display_name}' đã bật (Backend: {opened_backend_name}, Res: {w}x{h}, FPS: {self.webcam_fps:.1f}, Interval: {timer_interval}ms)")

            # Bắt đầu timer đọc frame
            self.frame_timer.start(timer_interval)

            # Cập nhật trạng thái nút Bật/Tắt webcam
            if hasattr(self,'ONCam'): self.ONCam.setEnabled(False)
            if hasattr(self,'OFFCam'): self.OFFCam.setEnabled(True)

            # Đặt nền đen cho khu vực hiển thị video
            if hasattr(self,'graphicsView'): self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.black))

            # Cập nhật trạng thái enable/disable tổng thể
            self._update_controls_state()

        except Exception as e:
            # Bắt lỗi tổng thể khi bật webcam
            emsg = f"❌ Lỗi nghiêm trọng khi bật webcam '{cam_display_name}': {e}"
            self.log_activity(emsg)
            self.log_activity(traceback.format_exc())
            QMessageBox.critical(self, "Lỗi Webcam", f"Không thể khởi động camera đã chọn.\nChi tiết: {e}\n\nVui lòng kiểm tra kết nối, driver hoặc thử chọn camera khác.")
            # Đảm bảo dọn dẹp nếu có lỗi
            if self.cap:
                 try: self.cap.release()
                 except: pass
                 self.cap = None

            # Bật lại các control chọn camera để người dùng thử lại
            has_cameras_after_error = bool(self.available_cameras)
            if hasattr(self,'cameraSelectionComboBox'): self.cameraSelectionComboBox.setEnabled(has_cameras_after_error)
            if hasattr(self,'refreshCamerasButton'): self.refreshCamerasButton.setEnabled(True)
            if hasattr(self,'ONCam'): self.ONCam.setEnabled(has_cameras_after_error) # Bật lại nút Bật nếu có cam
            if hasattr(self,'OFFCam'): self.OFFCam.setEnabled(False)
            if hasattr(self,'graphicsView'): self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray))

            # Cập nhật trạng thái enable/disable tổng thể
            self._update_controls_state()

            # Có thể cân nhắc tự động làm mới danh sách camera ở đây
            QTimer.singleShot(50, self._refresh_camera_list)


    def update_frame(self):
        """Đọc frame từ webcam, hiển thị và đưa vào queue xử lý."""
        if self.cap is None or not self.cap.isOpened(): return # Chưa có webcam hoặc đã bị lỗi

        # Đọc frame từ webcam
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            # Log lỗi đọc frame định kỳ để tránh spam console/log file
            current_time = time.time()
            if not hasattr(self, '_last_read_error_log_time') or current_time - getattr(self, '_last_read_error_log_time', 0) > 10:
                self.log_activity("⚠️ Lỗi đọc frame từ webcam.")
                setattr(self, '_last_read_error_log_time', current_time)
            # Cân nhắc dừng webcam nếu lỗi đọc liên tục? Hiện tại chỉ bỏ qua frame lỗi.
            return

        # Xóa cờ lỗi nếu đọc thành công
        if hasattr(self, '_last_read_error_log_time'): delattr(self, '_last_read_error_log_time')

        try:
            display_frame = frame.copy() # Frame để hiển thị (có thể vẽ ROI)
            processing_frame = frame     # Frame để xử lý (có thể bị crop bởi ROI)

            # 1. Áp dụng ROI nếu có
            if self.webcam_roi:
                x, y, w, h = self.webcam_roi
                fh_orig, fw_orig = frame.shape[:2] # Kích thước frame gốc
                # Đảm bảo ROI nằm trong giới hạn frame gốc
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(fw_orig, int(x + w)), min(fh_orig, int(y + h))
                # Chỉ crop và vẽ nếu ROI hợp lệ (có chiều rộng và cao > 0)
                if x2 > x1 and y2 > y1:
                    # Crop frame cho việc xử lý
                    processing_frame = frame[y1:y2, x1:x2]
                    # Vẽ hình chữ nhật ROI lên frame hiển thị
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Màu xanh lá, độ dày 2
                else:
                     # ROI không hợp lệ, sử dụng frame đầy đủ để xử lý
                     processing_frame = frame
                     # Log lỗi ROI định kỳ
                     current_time = time.time()
                     if not hasattr(self,'_last_roi_error_log_time') or current_time - getattr(self,'_last_roi_error_log_time', 0) > 15:
                         self.log_activity(f"⚠️ ROI {self.webcam_roi} không hợp lệ. Sử dụng frame đầy đủ.")
                         setattr(self,'_last_roi_error_log_time', current_time)
            # else: # Không có ROI -> không cần làm gì

            # 2. Hiển thị frame (display_frame) lên QGraphicsView
            # Chuyển đổi màu từ BGR (OpenCV) sang RGB (Qt)
            try:
                 frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                 h_disp, w_disp, ch_disp = frame_rgb.shape
                 bytes_per_line = ch_disp * w_disp
                 # Tạo QImage từ dữ liệu numpy
                 qt_image = QtGui.QImage(frame_rgb.data, w_disp, h_disp, bytes_per_line, QtGui.QImage.Format_RGB888)
                 # Tạo QPixmap và scale để vừa với view
                 view_w = self.graphicsView.viewport().width()
                 view_h = self.graphicsView.viewport().height()
                 # Trừ viền 1px mỗi bên để không bị tràn
                 pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(max(1, view_w - 2), max(1, view_h - 2),
                                                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)

                 # Cập nhật QGraphicsPixmapItem (tạo nếu chưa có)
                 if self.pixmap_item is None:
                     self.pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
                     self.scene.addItem(self.pixmap_item)
                     # Tự động căn giữa và scale phù hợp lần đầu
                     self.graphicsView.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
                 else:
                     self.pixmap_item.setPixmap(pixmap)
                     # Không cần fitInView liên tục trừ khi kích thước view thay đổi lớn
                     # Căn chỉnh vị trí nếu cần (ví dụ, nếu scale thay đổi nhiều)
                     # Có thể tính toán để luôn giữ ở giữa view
                     px = (self.graphicsView.viewport().width() - pixmap.width()) / 2
                     py = (self.graphicsView.viewport().height() - pixmap.height()) / 2
                     self.pixmap_item.setPos(px, py)

            except cv2.error as cv_err:
                 # Log lỗi chuyển đổi màu
                 current_time = time.time()
                 if not hasattr(self,'_last_cvt_error_log') or current_time-getattr(self,'_last_cvt_error_log',0) > 10:
                     self.log_activity(f"⚠️ Lỗi chuyển đổi màu frame hiển thị: {cv_err}")
                     setattr(self,'_last_cvt_error_log', current_time)
            except Exception as qt_img_err:
                 # Log lỗi tạo QImage/QPixmap
                 current_time = time.time()
                 if not hasattr(self,'_last_qtimg_error_log') or current_time-getattr(self,'_last_qtimg_error_log',0) > 10:
                      self.log_activity(f"⚠️ Lỗi tạo ảnh Qt để hiển thị: {qt_img_err}")
                      setattr(self,'_last_qtimg_error_log', current_time)


            # 3. Xử lý ghi video (nếu đang bật chế độ ghi lỗi VÀ đang xử lý)
            if self.processing and self._record_on_error_enabled:
                # Khởi tạo VideoWriter nếu chưa có VÀ đã có thư mục lỗi
                if self.video_writer is None:
                    if self.error_folder and os.path.isdir(self.error_folder):
                        try:
                            vid_h, vid_w = processing_frame.shape[:2] # Kích thước video = frame xử lý
                            # Kiểm tra kích thước hợp lệ
                            if vid_w <= 0 or vid_h <= 0:
                                raise ValueError("Kích thước frame xử lý không hợp lệ để ghi video.")

                            # Tạo thư mục con cho video nếu chưa có
                            video_dir = os.path.join(self.error_folder, VIDEO_SUBFOLDER)
                            os.makedirs(video_dir, exist_ok=True) # Tạo nếu chưa có

                            timestamp = time.strftime('%Y%m%d_%H%M%S')
                            video_filename = f"error_rec_{timestamp}.mp4" # Dùng mp4 làm mặc định
                            self.current_video_path = os.path.join(video_dir, video_filename)

                            # Chọn codec (thử mp4v, nếu không được có thể dùng avc1 hoặc XVID cho avi)
                            # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Hoặc 'avc1', 'XVID'
                            fourcc = cv2.VideoWriter_fourcc(*'avc1') # Thường tương thích tốt hơn trên nhiều nền tảng
                            if fourcc == -1: # Nếu codec không tồn tại, thử codec khác
                                self.log_activity("⚠️ Codec 'avc1' không khả dụng, thử 'mp4v'.")
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                if fourcc == -1:
                                    self.log_activity("⚠️ Codec 'mp4v' cũng không khả dụng. Ghi video có thể thất bại.")

                            # Đảm bảo FPS hợp lệ (ít nhất 1 FPS)
                            record_fps = max(1.0, min(30.0, self.webcam_fps)) # Giới hạn FPS ghi

                            self.video_writer = cv2.VideoWriter(self.current_video_path, fourcc, record_fps, (vid_w, vid_h))

                            if self.video_writer.isOpened():
                                self.log_activity(f"🔴 Bắt đầu ghi video lỗi: {video_filename} ({vid_w}x{vid_h} @{record_fps:.1f}fps)")
                                self.error_occurred_during_recording = False # Reset cờ lỗi cho file mới
                            else:
                                self.log_activity(f"❌ Không thể khởi tạo VideoWriter cho {video_filename}")
                                self.video_writer = None; self.current_video_path = None
                        except ValueError as ve:
                            self.log_activity(f"❌ Lỗi tạo VideoWriter: {ve}")
                            self.video_writer = None; self.current_video_path = None
                        except Exception as e_vid:
                            self.log_activity(f"❌ Lỗi nghiêm trọng khi tạo VideoWriter: {e_vid}")
                            self.log_activity(traceback.format_exc())
                            self.video_writer = None; self.current_video_path = None
                    else:
                        # Chưa cấu hình thư mục lỗi, log cảnh báo định kỳ
                        current_time = time.time()
                        if not hasattr(self,'_last_vid_folder_err_log') or current_time - getattr(self,'_last_vid_folder_err_log', 0) > 30:
                            self.log_activity("⚠️ Chưa đặt thư mục lỗi hợp lệ để ghi video.")
                            setattr(self,'_last_vid_folder_err_log', current_time)

                # Ghi frame vào video nếu VideoWriter đã được mở
                if self.video_writer and self.video_writer.isOpened():
                    try:
                        self.video_writer.write(processing_frame)
                    except Exception as e_write:
                        # Log lỗi ghi frame, có thể dừng ghi nếu lỗi liên tục
                        current_time = time.time()
                        if not hasattr(self,'_last_vid_write_err_log') or current_time - getattr(self,'_last_vid_write_err_log',0) > 10:
                           self.log_activity(f"❌ Lỗi ghi frame video: {e_write}")
                           setattr(self,'_last_vid_write_err_log', current_time)
                        # Cân nhắc đóng VideoWriter nếu lỗi ghi? Hiện tại chỉ log.

            # 4. Đưa frame vào queue cho worker (nếu đang xử lý)
            if self.processing:
                # Tạo bản sao frame để gửi đi, đảm bảo thread-safe
                # Gửi frame đã được crop ROI (nếu có)
                frame_to_process = processing_frame.copy()
                try:
                    # Đặt vào queue, không chặn nếu đầy
                    self.frame_queue.put(frame_to_process, block=False, timeout=0.01) # Timeout nhỏ phòng trường hợp hiếm
                    if hasattr(self, '_last_queue_full_log'): delattr(self,'_last_queue_full_log') # Reset cờ log queue đầy
                except Full:
                    # Queue đầy, frame bị bỏ qua, log định kỳ
                    current_time = time.time()
                    if not hasattr(self, '_last_queue_full_log') or current_time - getattr(self, '_last_queue_full_log', 0) > 5:
                         self.log_activity("⚠️ Queue xử lý đầy, frame bị bỏ qua.")
                         setattr(self, '_last_queue_full_log', current_time)
                except Exception as q_put_err:
                     # Lỗi khác khi đưa vào queue (hiếm gặp)
                     current_time = time.time()
                     if not hasattr(self,'_last_q_put_err_log') or current_time - getattr(self,'_last_q_put_err_log', 0) > 10:
                        self.log_activity(f"❌ Lỗi đưa frame vào queue: {q_put_err}")
                        setattr(self,'_last_q_put_err_log', current_time)

        except Exception as e:
            # Bắt lỗi chung trong quá trình update_frame (ví dụ: lỗi numpy, ROI,...)
            current_time = time.time()
            if not hasattr(self,'_last_update_frame_err_log') or current_time - getattr(self,'_last_update_frame_err_log', 0) > 10:
                 self.log_activity(f"❌ Lỗi trong update_frame: {e}")
                 self.log_activity(traceback.format_exc())
                 setattr(self,'_last_update_frame_err_log', current_time)


    def stop_webcam(self):
        """Dừng webcam đang chạy và dọn dẹp tài nguyên."""
        # Lấy tên camera đang chạy để log
        current_cam_text = "Unknown Camera"
        if self.cap and hasattr(self,'cameraSelectionComboBox'):
            # Tìm item trong combobox khớp với index đang dùng (selected_camera_index)
            for i in range(self.cameraSelectionComboBox.count()):
                 item_data = self.cameraSelectionComboBox.itemData(i)
                 if item_data is not None and item_data == self.selected_camera_index:
                      current_cam_text = self.cameraSelectionComboBox.itemText(i)
                      break # Tìm thấy

        if self.cap and self.cap.isOpened():
            try:
                self.frame_timer.stop() # Dừng timer đọc frame trước
                # Chờ một chút để frame đang xử lý (nếu có) hoàn tất đọc
                # QtCore.QCoreApplication.processEvents() # Có thể không cần
                time.sleep(0.05) # Chờ ngắn
                self.cap.release() # Giải phóng thiết bị camera
            except Exception as e:
                 self.log_activity(f"⚠️ Lỗi khi dừng webcam '{current_cam_text}': {e}")
            finally:
                 self.cap = None # Đặt lại cờ cap
                 # Dọn dẹp scene và hiển thị
                 if hasattr(self, 'scene'): self.scene.clear(); self.pixmap_item = None
                 if hasattr(self, 'graphicsView'): self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray))
                 self.log_activity(f"🚫 Webcam '{current_cam_text}' đã tắt.")

                 # Cập nhật trạng thái nút và control chọn camera
                 has_cameras = bool(self.available_cameras)
                 if hasattr(self,'ONCam'): self.ONCam.setEnabled(has_cameras) # Bật nút Bật nếu có cam
                 if hasattr(self,'OFFCam'): self.OFFCam.setEnabled(False)
                 if hasattr(self,'cameraSelectionComboBox'): self.cameraSelectionComboBox.setEnabled(has_cameras) # Bật lại chọn cam
                 if hasattr(self,'refreshCamerasButton'): self.refreshCamerasButton.setEnabled(True) # Bật lại nút refresh

                 # Tự động dừng xử lý nếu đang chạy (quan trọng)
                 if self.processing:
                     self.log_activity("ℹ️ Tự động dừng xử lý do webcam tắt.")
                     self.toggle_processing() # Gọi hàm dừng chuẩn

                 # Cập nhật trạng thái enable/disable tổng thể
                 self._update_controls_state()

        elif self.cap is None:
             self.log_activity("ℹ️ Webcam chưa được bật để dừng.")
             # Đảm bảo các nút ở trạng thái đúng
             has_cameras = bool(self.available_cameras)
             if hasattr(self,'ONCam'): self.ONCam.setEnabled(has_cameras)
             if hasattr(self,'OFFCam'): self.OFFCam.setEnabled(False)
             if hasattr(self,'cameraSelectionComboBox'): self.cameraSelectionComboBox.setEnabled(has_cameras)
             if hasattr(self,'refreshCamerasButton'): self.refreshCamerasButton.setEnabled(True)
             self._update_controls_state()
        # else: self.cap tồn tại nhưng not isOpened() -> trạng thái lỗi, không cần làm gì thêm


    # --- Reference Image Handling (SSIM) ---
    def load_reference_image(self, img_type):
        """Tải ảnh tham chiếu (Norm, Shutdown, Fail) cho SSIM từ file."""
        if self.processing:
             QMessageBox.warning(self, "Đang xử lý", "Không thể thay đổi ảnh tham chiếu khi đang xử lý.")
             return
        # Luôn cho phép tải ảnh, nhưng cảnh báo nếu không ở mode SSIM
        if self.current_comparison_method != METHOD_SSIM:
             QMessageBox.information(self, "Thông tin", f"Ảnh tham chiếu '{img_type}' hiện chỉ được sử dụng cho phương thức {METHOD_SSIM}.")

        opts = QFileDialog.Options()
        # Thư mục gợi ý: thư mục của ảnh cũ -> thư mục lỗi -> thư mục người dùng
        suggested_dir = os.path.expanduser("~") # Mặc định
        current_path = self.config['ref_paths'].get(img_type)
        if current_path and os.path.exists(os.path.dirname(current_path)):
            suggested_dir = os.path.dirname(current_path)
        elif self.error_folder and os.path.exists(self.error_folder):
            suggested_dir = self.error_folder

        # Mở dialog chọn file ảnh
        fp, _ = QFileDialog.getOpenFileName(
            self, f"Chọn ảnh tham chiếu '{img_type}' (cho SSIM)", suggested_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)", options=opts)

        if fp: # Nếu người dùng chọn file
            try:
                # Đọc ảnh bằng cách an toàn với đường dẫn Unicode
                img_bytes = np.fromfile(fp, dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Không thể giải mã file ảnh hoặc định dạng không được hỗ trợ.")

                # Cập nhật dữ liệu ảnh và đường dẫn trong config
                self.ref_data[img_type] = img
                self.config['ref_paths'][img_type] = fp # Lưu đường dẫn mới
                self.update_button_styles() # Cập nhật giao diện nút
                self.log_activity(f"✅ Đã tải ảnh '{img_type}' (SSIM) từ: {os.path.basename(fp)}")
                # self.save_config() # Lưu config nếu muốn

            except Exception as e:
                # Xử lý lỗi khi tải hoặc giải mã ảnh
                self.log_activity(f"❌ Lỗi tải ảnh {img_type} từ '{fp}': {e}")
                QMessageBox.warning(self, "Lỗi Tải Ảnh", f"Không thể tải ảnh:\n{fp}\n\nLỗi: {e}")
                # Cân nhắc xóa ảnh và đường dẫn cũ nếu tải lỗi?
                # self.config['ref_paths'][img_type] = None
                # self.ref_data[img_type] = None
                # self.update_button_styles()


    def capture_reference_from_webcam(self, img_type):
        """Chụp ảnh tham chiếu (Norm, Shutdown, Fail) cho SSIM từ webcam đang chạy."""
        # Kiểm tra điều kiện
        if not self.cap or not self.cap.isOpened():
             QMessageBox.warning(self,"Webcam Chưa Bật","Vui lòng bật webcam trước khi chụp.")
             return
        if self.processing:
             QMessageBox.warning(self,"Đang Xử Lý","Không thể chụp ảnh khi đang xử lý.")
             return
        # Vẫn cho phép chụp ảnh, nhưng cảnh báo nếu không ở mode SSIM
        if self.current_comparison_method != METHOD_SSIM:
             QMessageBox.information(self, "Thông tin", f"Chụp ảnh tham chiếu '{img_type}' hiện chỉ áp dụng cho phương thức {METHOD_SSIM}.")

        # Tạm dừng timer đọc frame để lấy ảnh ổn định (nếu đang chạy)
        was_timer_active = self.frame_timer.isActive()
        if was_timer_active: self.frame_timer.stop(); time.sleep(0.1) # Chờ chút xíu

        # Đọc frame hiện tại từ webcam
        ret, frame = self.cap.read()

        # Khởi động lại timer ngay sau khi đọc frame (nếu nó đang chạy trước đó)
        if was_timer_active and self.cap and self.cap.isOpened():
            self.frame_timer.start()

        # Kiểm tra frame đọc được
        if not ret or frame is None or frame.size == 0:
            QMessageBox.warning(self,"Lỗi Đọc Frame","Không thể lấy ảnh từ webcam để chụp.")
            return

        try:
            # Quyết định lưu frame gốc hay frame đã crop ROI?
            # Hiện tại: Lưu frame đã áp dụng ROI (nếu có), tương tự frame đi vào xử lý
            frame_to_save = frame.copy()
            if self.webcam_roi:
                x, y, w, h = self.webcam_roi
                fh_orig, fw_orig = frame.shape[:2]
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(fw_orig, int(x + w)), min(fh_orig, int(y + h))
                if x2 > x1 and y2 > y1:
                     frame_to_save = frame[y1:y2, x1:x2].copy() # Lưu phần ROI
                # else: Lưu frame gốc nếu ROI không hợp lệ

            # Lưu ảnh vào dữ liệu và xóa đường dẫn file cũ trong config
            self.ref_data[img_type] = frame_to_save
            self.config['ref_paths'][img_type] = None # Đánh dấu là chụp từ webcam
            self.log_activity(f"📸 Đã chụp ảnh '{img_type}' (SSIM) từ webcam" + (" (Đã áp dụng ROI)" if self.webcam_roi else ""))
            self.update_button_styles() # Cập nhật giao diện nút
            # self.save_config() # Lưu config nếu muốn

        except Exception as e:
            # Xử lý lỗi khi xử lý hoặc lưu ảnh chụp
            self.log_activity(f"❌ Lỗi khi lưu ảnh chụp '{img_type}': {e}")
            QMessageBox.critical(self,"Lỗi Chụp Ảnh",f"Đã xảy ra lỗi khi chụp và lưu ảnh: {e}")


    # --- ROI and Error Folder ---
    def select_webcam_roi(self):
        """Mở cửa sổ cho phép người dùng chọn vùng quan tâm (ROI) trên ảnh webcam."""
        # Kiểm tra điều kiện
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "Webcam Chưa Bật", "Vui lòng bật webcam trước khi chọn ROI.")
            return
        if self.processing:
            QMessageBox.warning(self, "Đang Xử Lý", "Không thể chọn ROI khi đang xử lý.")
            return

        # Tạm dừng timer webcam để ảnh tĩnh khi chọn ROI
        was_timer_active = self.frame_timer.isActive()
        if was_timer_active: self.frame_timer.stop(); time.sleep(0.1)

        # Đọc một frame hiện tại
        ret, frame = self.cap.read()

        # Nếu đọc lỗi, bật lại timer (nếu cần) và thoát
        if not ret or frame is None or frame.size == 0:
            if was_timer_active and self.cap and self.cap.isOpened(): self.frame_timer.start()
            QMessageBox.warning(self, "Lỗi Đọc Frame", "Không thể lấy ảnh từ webcam để chọn ROI.")
            return

        try:
            # Đặt tên cửa sổ và tạo cửa sổ có thể thay đổi kích thước
            window_name = "Chon ROI (Keo chuot -> Enter/Space | Huy -> C/ESC | Reset -> R)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Cho phép resize
            cv2.resizeWindow(window_name, 800, 600) # Kích thước ban đầu hợp lý
            cv2.setWindowTitle(window_name, window_name) # Đặt title bar (trên Windows)

            # Clone ảnh để vẽ hướng dẫn (tùy chọn)
            frame_roi_select = frame.copy()
            # (Có thể thêm cv2.putText để hướng dẫn rõ hơn trên ảnh)

            # --- Sử dụng cv2.selectROI ---
            # showCrosshair=True: Hiển thị dấu + ở tâm
            # fromCenter=False: Vẽ từ góc trên trái (phổ biến hơn)
            roi_tuple = cv2.selectROI(window_name, frame_roi_select, showCrosshair=True, fromCenter=False)
            # --- selectROI trả về (x, y, w, h) hoặc (0,0,0,0) nếu hủy ---
            cv2.destroyWindow(window_name) # Đóng cửa sổ chọn ROI ngay sau khi hoàn tất

            # Bật lại timer webcam NGAY SAU KHI cửa sổ ROI đóng
            if was_timer_active and self.cap and self.cap.isOpened():
                self.frame_timer.start()

            # Xử lý kết quả ROI
            if roi_tuple == (0, 0, 0, 0): # Người dùng hủy (ESC, C, đóng cửa sổ)
                self.log_activity("ℹ️ Đã hủy chọn ROI.")
                # Không thay đổi ROI hiện có
            elif roi_tuple[2] > 0 and roi_tuple[3] > 0: # ROI hợp lệ (có chiều rộng và cao)
                # Lưu ROI mới dưới dạng tuple số nguyên không âm
                self.webcam_roi = tuple(max(0, int(v)) for v in roi_tuple)
                self.config['webcam_roi'] = list(self.webcam_roi) # Cập nhật config
                self.log_activity(f"✅ Đã chọn ROI mới: {self.webcam_roi}")
                # self.save_config() # Lưu config
            else:
                 # Người dùng chọn nhưng w=0 hoặc h=0 (nhấp chuột không kéo?) -> Coi như reset
                 self.log_activity("⚠️ Đã reset ROI (chọn vùng không hợp lệ).")
                 self.webcam_roi = None # Xóa ROI
                 self.config['webcam_roi'] = None
                 # self.save_config()

            # Cập nhật giao diện nút ROI
            self.update_button_styles()

        except Exception as e:
            # Bắt lỗi không mong muốn trong quá trình chọn ROI
            self.log_activity(f"❌ Lỗi trong quá trình chọn ROI: {e}")
            self.log_activity(traceback.format_exc())
            QMessageBox.critical(self, "Lỗi Chọn ROI", f"Đã xảy ra lỗi khi chọn ROI:\n{e}")
            cv2.destroyAllWindows() # Đảm bảo đóng mọi cửa sổ OpenCV nếu có lỗi
            # Bật lại timer nếu cần và chưa được bật lại do lỗi
            if was_timer_active and self.cap and self.cap.isOpened() and not self.frame_timer.isActive():
                try: self.frame_timer.start()
                except Exception: pass


    def select_error_folder(self):
        """Mở dialog cho phép người dùng chọn thư mục lưu lỗi, video và log."""
        if self.processing:
            QMessageBox.warning(self, "Đang Xử Lý", "Không thể thay đổi thư mục khi đang xử lý.")
            return

        opts = QFileDialog.Options() | QFileDialog.ShowDirsOnly
        # Thư mục gợi ý: thư mục cũ -> thư mục người dùng
        suggested_dir = self.error_folder or os.path.expanduser("~")

        # Mở dialog chọn thư mục
        folder = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục lưu ảnh lỗi, video và log", suggested_dir, opts)

        if folder: # Nếu người dùng chọn thư mục
            try:
                # Kiểm tra quyền ghi vào thư mục đã chọn
                if not os.access(folder, os.W_OK):
                    raise PermissionError(f"Không có quyền ghi vào thư mục '{folder}'.")

                # Chỉ cập nhật nếu thư mục thực sự thay đổi
                if self.error_folder != folder:
                    self.error_folder = folder
                    self.config['error_folder'] = folder
                    self.log_activity(f"📁 Đã chọn thư mục lỗi: {self.error_folder}")
                    # Cập nhật đường dẫn file log dựa trên thư mục mới
                    old_log_path = self.log_file_path
                    self.log_file_path = os.path.join(self.error_folder, LOG_FILE_NAME)
                    # Ghi log về sự thay đổi (vào file log cũ nếu có)
                    if old_log_path and old_log_path != self.log_file_path:
                         # Ghi tạm vào log cũ về việc đổi đường dẫn
                         try:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            with open(old_log_path,"a",encoding="utf-8") as lf_old:
                                lf_old.write(f"{timestamp} - INFO - Đường dẫn log đổi thành: {self.log_file_path}\n")
                         except Exception: pass # Bỏ qua nếu không ghi được vào file cũ

                    self.log_activity(f"📄 File log sẽ ghi tại: {self.log_file_path}")
                    # Thử ghi một dòng vào log mới để xác nhận
                    self.log_activity("📝 (Thử ghi log vào đường dẫn mới)")
                    # self.save_config() # Lưu config
                    self.update_button_styles() # Cập nhật style nút

            except PermissionError as pe:
                QMessageBox.warning(self, "Lỗi Quyền Ghi", f"{pe}\n\nVui lòng chọn thư mục khác hoặc kiểm tra quyền truy cập.")
                self.log_activity(f"⚠️ {pe}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi Thư Mục", f"Đã xảy ra lỗi khi đặt thư mục lỗi:\n{e}")
                self.log_activity(f"❌ Lỗi khi đặt thư mục lỗi '{folder}': {e}")


    # --- Serial Port Handling ---
    @QtCore.pyqtSlot()
    def _refresh_com_ports(self):
        """Quét lại các cổng COM có sẵn và cập nhật ComboBox."""
        # Không cho refresh nếu đang kết nối COM hoặc đang xử lý
        if self.serial_enabled:
            self.log_activity("ℹ️ Ngắt kết nối COM trước khi làm mới danh sách.")
            return
        if self.processing:
            self.log_activity("ℹ️ Không thể làm mới cổng COM khi đang xử lý.")
            return

        self.comPortComboBox.blockSignals(True)
        current_selection = self.comPortComboBox.currentText() # Lưu lại lựa chọn hiện tại
        self.comPortComboBox.clear() # Xóa list cũ
        port_names = []
        try:
            # Liệt kê các cổng COM
            ports = serial.tools.list_ports.comports()
            # Lọc và sắp xếp tên thiết bị (ví dụ: COM1, COM3)
            port_names = sorted([port.device for port in ports])
            if port_names: self.log_activity(f"🔄 Tìm thấy cổng COM: {', '.join(port_names)}")
            else: self.log_activity("🔄 Không tìm thấy cổng COM nào.")
        except Exception as e:
            self.log_activity(f"❌ Lỗi khi liệt kê cổng COM: {e}")

        # Điền vào ComboBox
        if not port_names:
             self.comPortComboBox.addItem("Không tìm thấy cổng")
             self.comPortComboBox.setEnabled(False) # Vô hiệu hóa chọn
             # Nếu trước đó có chọn cổng, giờ không còn -> cập nhật state
             if self.serial_port_name is not None:
                  self.serial_port_name = None
                  self.config['serial_port'] = None
                  # self.save_config()
        else:
            self.comPortComboBox.addItems(port_names)
            self.comPortComboBox.setEnabled(True) # Cho phép chọn
            # Cố gắng khôi phục lựa chọn cũ nếu nó còn trong danh sách mới
            if current_selection in port_names:
                 self.comPortComboBox.setCurrentText(current_selection)
                 # Đảm bảo self.serial_port_name khớp với lựa chọn này
                 if self.serial_port_name != current_selection:
                     self.serial_port_name = current_selection
                     self.config['serial_port'] = self.serial_port_name
            else:
                # Nếu cổng cũ không còn, tự động chọn cổng đầu tiên
                 new_selection = self.comPortComboBox.itemText(0)
                 self.comPortComboBox.setCurrentIndex(0)
                 # Cập nhật state và log nếu lựa chọn thay đổi
                 if self.serial_port_name != new_selection:
                    if self.serial_port_name and "Không tìm thấy" not in self.serial_port_name :
                         self.log_activity(f"⚠️ COM '{self.serial_port_name}' không còn. Chọn mặc định '{new_selection}'.")
                    self.serial_port_name = new_selection
                    self.config['serial_port'] = self.serial_port_name
                    # self.save_config()

        # Bỏ chặn tín hiệu và cập nhật các control liên quan
        self.comPortComboBox.blockSignals(False)
        self._update_controls_state() # Cập nhật trạng thái enabled của các nút/combo

    @QtCore.pyqtSlot()
    def _toggle_serial_port(self):
        """Kết nối hoặc ngắt kết nối cổng Serial đã chọn."""
        if self.processing:
            QMessageBox.warning(self, "Đang Xử Lý", "Không thể thay đổi kết nối COM khi đang xử lý.")
            return

        if not self.serial_enabled: # --- Logic KẾT NỐI ---
            # Lấy thông tin cổng và baud rate từ UI/state
            port_to_connect = self.comPortComboBox.currentText() # Lấy từ UI để đảm bảo khớp
            baud_rate_to_use = self.serial_baud_rate

            # Kiểm tra xem cổng có hợp lệ không
            if not port_to_connect or "Không tìm thấy" in port_to_connect:
                QMessageBox.warning(self, "Chưa Chọn Cổng", "Vui lòng chọn cổng COM hợp lệ từ danh sách.")
                return

            # Đóng cổng cũ (nếu vô tình còn mở) trước khi mở cổng mới
            if self.serial_port and self.serial_port.is_open:
                try: self.serial_port.close()
                except Exception: pass
            self.serial_port = None # Reset

            try:
                self.log_activity(f"🔌 Đang kết nối {port_to_connect} @ {baud_rate_to_use} baud...")
                # Đặt timeout khi đọc và ghi để tránh treo nếu thiết bị không phản hồi
                self.serial_port = serial.Serial(port_to_connect, baud_rate_to_use,
                                                  timeout=0.1,       # Timeout đọc (s)
                                                  write_timeout=1.0) # Timeout ghi (s)

                # ---- Kết nối thành công ----
                self.serial_enabled = True
                self.serial_port_name = port_to_connect # Cập nhật state khớp với cổng đã kết nối
                self.serial_baud_rate = baud_rate_to_use # Cập nhật state baudrate
                # Cập nhật config (chỉ port và baud, enabled luôn là False khi lưu)
                self.config['serial_port'] = port_to_connect
                self.config['serial_baud'] = baud_rate_to_use
                self.log_activity(f"✅ Đã kết nối COM: {port_to_connect}")

            except serial.SerialException as e:
                # Lỗi khi mở cổng (vd: bị dùng bởi chương trình khác, không tồn tại)
                self.log_activity(f"❌ Lỗi mở cổng COM '{port_to_connect}': {e}")
                QMessageBox.critical(self, "Lỗi Kết Nối COM", f"Không thể mở cổng {port_to_connect}.\nLỗi: {e}\n\nKiểm tra driver, kết nối hoặc cổng có đang bị sử dụng không?")
                self.serial_port = None; self.serial_enabled = False
            except Exception as e_unk:
                # Lỗi không xác định khác
                self.log_activity(f"❌ Lỗi không xác định khi mở COM '{port_to_connect}': {e_unk}")
                self.log_activity(traceback.format_exc())
                QMessageBox.critical(self, "Lỗi Nghiêm Trọng", f"Lỗi không mong muốn khi kết nối COM.\nLỗi: {e_unk}")
                self.serial_port = None; self.serial_enabled = False

        else: # --- Logic NGẮT KẾT NỐI ---
            port_to_close = self.serial_port_name or "N/A" # Lấy tên cổng đang kết nối
            try:
                if self.serial_port and self.serial_port.is_open:
                    self.log_activity(f"🔌 Đang ngắt kết nối COM: {port_to_close}...")
                    self.serial_port.close()
                    self.log_activity(f"🔌 Đã ngắt kết nối COM.")
            except serial.SerialException as e:
                 self.log_activity(f"⚠️ Lỗi khi đóng cổng COM '{port_to_close}': {e}")
            except Exception as e_unk:
                 self.log_activity(f"⚠️ Lỗi không xác định khi đóng COM '{port_to_close}': {e_unk}")
            finally:
                 # Luôn dọn dẹp trạng thái sau khi cố gắng đóng
                 self.serial_port = None
                 self.serial_enabled = False
                 # self.config['serial_enabled'] đã là False

        # Cập nhật trạng thái enable/disable của các control và style nút
        self._update_controls_state()
        self.update_serial_button_style() # Cập nhật riêng style nút Connect/Disconnect
        # self.save_config() # Lưu trạng thái mới (port, baud)


    @QtCore.pyqtSlot(str)
    def _send_serial_command(self, command):
        """Gửi lệnh (chuỗi string) qua cổng serial đã mở (được gọi từ worker)."""
        if not self.serial_enabled or not self.serial_port or not self.serial_port.is_open:
            # self.log_activity(f"⚠️ Bỏ qua gửi COM (chưa kết nối): {command}") # Log nếu cần debug
            return

        try:
            # Đảm bảo lệnh kết thúc bằng ký tự xuống dòng (thường cần thiết cho Arduino/ESP)
            cmd_with_newline = command if command.endswith('\n') else command + '\n'
            # Chuyển chuỗi thành bytes (thường dùng utf-8)
            byte_command = cmd_with_newline.encode('utf-8')
            # Ghi dữ liệu ra cổng serial
            bytes_written = self.serial_port.write(byte_command)

            # Kiểm tra xem tất cả bytes đã được ghi chưa (thường không cần nếu có write_timeout)
            if bytes_written != len(byte_command):
                 self.log_activity(f"⚠️ Gửi COM không đủ byte: {command} ({bytes_written}/{len(byte_command)} bytes)")
            # else:
            #      self.log_activity(f"➡️ Gửi COM [{self.serial_port_name}]: {command}") # Log thành công (có thể gây nhiều log)

        except serial.SerialTimeoutException:
            # Lỗi timeout khi ghi (thiết bị không nhận kịp?)
            self.log_activity(f"⚠️ Timeout khi gửi lệnh COM tới '{self.serial_port_name}': {command}")
            # Cân nhắc tự động ngắt kết nối nếu lỗi này xảy ra liên tục?
        except serial.SerialException as e:
            # Lỗi serial khác trong quá trình ghi
            self.log_activity(f"❌ Lỗi nghiêm trọng khi gửi lệnh COM: {e}. Tự động ngắt kết nối.")
            QMessageBox.critical(self, "Lỗi Gửi COM", f"Không thể gửi dữ liệu tới {self.serial_port_name}.\nKết nối sẽ bị đóng.\nLỗi: {e}")
            # Gọi hàm ngắt kết nối một cách an toàn từ luồng chính
            # Không gọi trực tiếp _toggle_serial_port từ slot này nếu nó đang ở luồng worker
            # Thay vào đó, nên có một tín hiệu/cơ chế để yêu cầu luồng chính ngắt kết nối
            # Hoặc đơn giản là ghi nhận lỗi và đợi người dùng xử lý.
            # Hiện tại: Log lỗi và không tự ngắt kết nối từ đây. Worker có thể ngừng gửi lệnh.
            self.serial_enabled = False # Cập nhật trạng thái ngay lập tức để worker không gửi nữa
            if self.serial_port:
                try: self.serial_port.close()
                except Exception: pass
                self.serial_port = None
            # Cập nhật UI từ luồng chính
            QtCore.QMetaObject.invokeMethod(self, "_update_controls_state", QtCore.Qt.QueuedConnection)
            QtCore.QMetaObject.invokeMethod(self, "update_serial_button_style", QtCore.Qt.QueuedConnection)

        except Exception as e_unk:
             # Lỗi không xác định khác
             self.log_activity(f"❌ Lỗi không xác định khi gửi lệnh COM: {e_unk}. Ngắt kết nối.")
             self.log_activity(traceback.format_exc())
             QMessageBox.critical(self, "Lỗi Gửi COM", f"Lỗi không mong muốn khi gửi dữ liệu.\nKết nối sẽ bị đóng.\nLỗi: {e_unk}")
             # Tương tự như trên, cần xử lý ngắt kết nối cẩn thận
             self.serial_enabled = False
             if self.serial_port:
                 try: self.serial_port.close()
                 except Exception: pass
                 self.serial_port = None
             QtCore.QMetaObject.invokeMethod(self, "_update_controls_state", QtCore.Qt.QueuedConnection)
             QtCore.QMetaObject.invokeMethod(self, "update_serial_button_style", QtCore.Qt.QueuedConnection)


    # --- YOLOv8 Model Handling ---
    @QtCore.pyqtSlot()
    def _select_yolo_model_path(self):
        """Mở dialog cho phép người dùng chọn file model YOLOv8 (.pt)."""
        if self.processing:
             QMessageBox.warning(self, "Đang Xử Lý", "Không thể thay đổi model khi đang xử lý.")
             return
        # Chỉ cho phép chọn nếu YOLO khả dụng
        if not YOLO_AVAILABLE:
            QMessageBox.warning(self, "Thiếu Thư Viện", "Chức năng YOLO không khả dụng do thiếu thư viện 'ultralytics'.")
            return

        opts = QFileDialog.Options()
        # Thư mục gợi ý: thư mục model cũ -> thư mục người dùng
        current_path = self.config.get('yolo_model_path')
        suggested_dir = os.path.dirname(current_path) if current_path and os.path.isdir(os.path.dirname(current_path)) else os.path.expanduser("~")

        # Mở dialog chọn file
        fp, _ = QFileDialog.getOpenFileName(self, "Chọn Model YOLOv8 (.pt)", suggested_dir,
                                            "PyTorch Models (*.pt);;All Files (*)", options=opts)
        if fp: # Nếu người dùng chọn file
            # Chỉ xử lý nếu đường dẫn thực sự thay đổi
            if self.config.get('yolo_model_path') != fp:
                self.config['yolo_model_path'] = fp
                self.log_activity(f"📁 Chọn model YOLO mới: {os.path.basename(fp)}")

                # Cập nhật label hiển thị đường dẫn model
                if hasattr(self, 'yoloModelPathLabel'):
                    base_name = os.path.basename(fp)
                    self.yoloModelPathLabel.setText(base_name)
                    self.yoloModelPathLabel.setStyleSheet("font-style: normal; color: black;")
                    self.yoloModelPathLabel.setToolTip(fp)

                # Giải phóng model cũ nếu có
                if self.yolo_model is not None:
                    self.log_activity("🧠 Giải phóng model YOLO cũ...")
                    try: del self.yolo_model; self.yolo_model = None
                    except Exception: pass
                    # Optional: GPU memory cleanup if using PyTorch/CUDA explicitly
                    # try: import torch; torch.cuda.empty_cache(); except: pass

                # Thử tải model mới ngay lập tức (có thể delay nếu cần)
                self._load_yolo_model() # Hàm này sẽ log kết quả và xử lý lỗi
                # self.save_config() # Lưu config

    def _load_yolo_model(self):
        """Tải model YOLO từ đường dẫn trong config. Chạy trên luồng chính."""
        # Kiểm tra điều kiện tiên quyết
        if not YOLO_AVAILABLE:
            # Đã có cảnh báo ở init, không cần log lại
            if hasattr(self,'details_label'): self.details_label.setText("Details: YOLO không khả dụng (thiếu ultralytics)")
            return False
        if not self.yolo_model is None and self.current_comparison_method != METHOD_YOLO:
             self.log_activity("ℹ️ Không phải chế độ YOLO, bỏ qua tải model.")
             return False # Không cần tải nếu không dùng

        model_path = self.config.get('yolo_model_path')
        if not model_path or not isinstance(model_path, str) or not os.path.isfile(model_path):
            msg = "Chưa cấu hình hoặc đường dẫn model YOLO không hợp lệ."
            if model_path: msg = f"Đường dẫn model YOLO không hợp lệ: {model_path}"
            self.log_activity(f"⚠️ {msg}")
            if hasattr(self,'details_label'): self.details_label.setText(f"Details: {msg}")
            if hasattr(self, 'yoloModelPathLabel'):
                self.yoloModelPathLabel.setText("Model không hợp lệ!")
                self.yoloModelPathLabel.setStyleSheet("font-style: italic; color: red;")
            self.yolo_model = None # Đảm bảo model là None
            return False

        # --- Bắt đầu quá trình tải model ---
        # Kiểm tra xem model có cần tải lại không (đã tải và cùng đường dẫn?)
        if self.yolo_model is not None:
             try:
                 # Thử truy cập thuộc tính lưu đường dẫn (có thể thay đổi tùy phiên bản ultralytics)
                 current_model_path_attr = getattr(self.yolo_model, 'ckpt_path', None) or \
                                           getattr(self.yolo_model, 'model', {}).get('yaml',{}).get('yaml_file',None) or \
                                           getattr(self.yolo_model,'cfg', None) # Các cách lấy path cũ/mới
                 if current_model_path_attr and os.path.normpath(current_model_path_attr) == os.path.normpath(model_path):
                     self.log_activity("ℹ️ Model YOLO đã được tải (không đổi).")
                     if hasattr(self,'details_label'): self.details_label.setText("Details: Model YOLO đã tải.")
                     return True # Model đúng đã được tải
             except Exception as e_check:
                self.log_activity(f"ℹ️ Không thể kiểm tra path model cũ, sẽ tải lại. Lỗi: {e_check}")

            # Nếu có model cũ khác -> giải phóng trước khi tải mới

                self.log_activity("🧠 Giải phóng model YOLO cũ...")
                try: del self.yolo_model; self.yolo_model = None
                except Exception: pass
            # try: import torch; torch.cuda.empty_cache(); except: pass

        # Bắt đầu tải model mới
        model_basename = os.path.basename(model_path)
        self.log_activity(f"⏳ Đang tải model YOLO: {model_basename}...")
        # Cập nhật UI để báo đang tải
        if hasattr(self,'details_label'): self.details_label.setText("Details: Đang tải model YOLO...")
        QtWidgets.QApplication.processEvents() # Buộc UI cập nhật ngay (có thể làm UI hơi lag)

        try:
            # === TẢI MODEL (có thể mất thời gian) ===
            start_time = time.time()
            # device='cpu' có thể giúp tránh lỗi CUDA nếu có vấn đề với GPU setup
            # Hoặc để trống ('') để tự động chọn (GPU nếu có, CPU nếu không)
            self.yolo_model = YOLO(model_path) # device='' hoặc device='cpu'
            load_time = time.time() - start_time
            # === TẢI XONG ===

            # Log thiết bị mà model đang chạy trên đó (CPU hoặc GPU)
            device_used = "CPU" # Mặc định
            try: device_used = str(next(self.yolo_model.parameters()).device).upper()
            except: pass
            self.log_activity(f"✅ Model YOLO '{model_basename}' đã tải thành công sau {load_time:.2f}s (Device: {device_used}).")
            if hasattr(self,'details_label'): self.details_label.setText(f"Details: Model YOLO đã tải ({device_used})")

            # Optional: Warm-up model bằng ảnh giả để giảm độ trễ ở lần predict đầu tiên
            try:
                dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
                self.yolo_model.predict(dummy_img, verbose=False, imgsz=64)
                # self.log_activity("ℹ️ Model YOLO đã được warm-up.")
            except Exception as wu_err:
                 self.log_activity(f"⚠️ Lỗi nhỏ khi warm-up model YOLO: {wu_err}") # Không nghiêm trọng
            return True  # Trả về True khi thành công

        except Exception as e:
            # Lỗi nghiêm trọng khi tải model
            load_error_msg = f"Không thể tải model YOLO:\n{model_path}\n\nLỗi: {e}"
            self.log_activity(f"❌ Lỗi nghiêm trọng tải model YOLO: {e}")
            self.log_activity(traceback.format_exc()) # Ghi traceback chi tiết
            QMessageBox.critical(self, "Lỗi Tải Model YOLO", load_error_msg)

            # Đảm bảo model là None nếu lỗi
            self.yolo_model = None
            # Cập nhật UI báo lỗi
            if hasattr(self,'details_label'): self.details_label.setText("Details: Lỗi tải model YOLO.")
            if hasattr(self, 'yoloModelPathLabel'):
                self.yoloModelPathLabel.setText("Lỗi tải model!")
                self.yoloModelPathLabel.setStyleSheet("font-style: italic; color: red;")
                self.yoloModelPathLabel.setToolTip(f"Lỗi: {e}")
            # Có thể cân nhắc xóa đường dẫn lỗi khỏi config?
            # self.config['yolo_model_path'] = None
            return False # Trả về False khi thất bại


    # --- Processing Control ---
    def toggle_processing(self):
        """Bắt đầu hoặc dừng luồng xử lý hình ảnh."""

        # --- KIỂM TRA ĐIỀU KIỆN TRƯỚC KHI BẮT ĐẦU ---
        if not self.processing:
            start_error = None # Lưu trữ thông báo lỗi đầu tiên gặp phải

            # 1. Kiểm tra Webcam
            if not self.cap or not self.cap.isOpened():
                 start_error = "Vui lòng bật webcam trước khi bắt đầu."
            # 2. Kiểm tra Thư mục lỗi
            elif not self.error_folder or not os.path.isdir(self.error_folder) or not os.access(self.error_folder, os.W_OK):
                 # Thử tạo thư mục nếu chỉ không tồn tại
                 if self.error_folder and not os.path.exists(self.error_folder):
                     try:
                         os.makedirs(self.error_folder, exist_ok=True)
                         self.log_activity(f"ℹ️ Tự động tạo thư mục lỗi: {self.error_folder}")
                         # Kiểm tra lại quyền ghi sau khi tạo
                         if not os.access(self.error_folder, os.W_OK):
                              start_error = f"Đã tạo thư mục '{self.error_folder}' nhưng không có quyền ghi."
                     except Exception as e_mkdir:
                         start_error = f"Không thể tạo thư mục lỗi '{self.error_folder}': {e_mkdir}"
                 elif self.error_folder and not os.access(self.error_folder, os.W_OK):
                      start_error = f"Không có quyền ghi vào thư mục lỗi: {self.error_folder}"
                 else: # Chưa chọn thư mục
                      start_error = "Vui lòng chọn thư mục hợp lệ (có quyền ghi) để lưu lỗi/log/video."
            # 3. Kiểm tra theo phương thức
            elif self.current_comparison_method == METHOD_SSIM:
                 # Bắt buộc phải có ảnh Norm cho SSIM
                 if not isinstance(self.ref_data.get(REF_NORM), np.ndarray) or self.ref_data[REF_NORM].size == 0:
                    start_error = f"Vui lòng tải hoặc chụp ảnh '{REF_NORM}' (cho SSIM) trước."
                 # Có thể thêm cảnh báo nếu thiếu ảnh Shutdown/Fail nhưng vẫn cho chạy
            elif self.current_comparison_method == METHOD_YOLO:
                 # Kiểm tra model YOLO
                 if not YOLO_AVAILABLE:
                      start_error = "Thư viện YOLO (ultralytics) chưa được cài đặt."
                 elif not self.yolo_model: # Model chưa được tải thành công
                      # Thử tải lại model nếu có đường dẫn hợp lệ
                      if self.config.get('yolo_model_path') and os.path.isfile(self.config['yolo_model_path']):
                           self.log_activity("ℹ️ Model YOLO chưa tải, đang thử tải lại trước khi bắt đầu...")
                           if not self._load_yolo_model(): # Nếu tải lại vẫn lỗi
                                start_error = "Không thể tải model YOLO. Vui lòng kiểm tra đường dẫn và file model."
                      else: # Không có đường dẫn hoặc đường dẫn không hợp lệ
                           start_error = "Vui lòng chọn đường dẫn model YOLO hợp lệ trong cấu hình."
                 # Thêm kiểm tra quy tắc YOLO nếu cần (ví dụ: đã định nghĩa chưa?)
            else: # Phương thức không xác định
                 start_error = f"Phương thức so sánh '{self.current_comparison_method}' không hợp lệ."

            # Nếu có lỗi khi kiểm tra điều kiện -> Hiển thị và không bắt đầu
            if start_error:
                 QMessageBox.warning(self, "Chưa Sẵn Sàng", start_error)
                 self.log_activity(f"⚠️ Không thể bắt đầu xử lý: {start_error}")
                 return # Không tiếp tục

        # --- CHUYỂN ĐỔI TRẠNG THÁI processing ---
        self.processing = not self.processing

        if self.processing:
            # --- LOGIC BẮT ĐẦU XỬ LÝ ---
            self.log_activity(f"▶️ Bắt đầu xử lý (Phương thức: {self.current_comparison_method})...")
            # Lưu cấu hình hiện tại trước khi chạy (đảm bảo config worker dùng là mới nhất)
            self.save_config()

            # Lấy hàm so sánh tương ứng với phương thức đã chọn
            compare_func = self.comparison_functions.get(self.current_comparison_method)
            if not compare_func: # Không nên xảy ra nếu check ở trên đã đúng
                self.log_activity(f"❌ Lỗi nghiêm trọng: Không tìm thấy hàm xử lý cho '{self.current_comparison_method}'.")
                self.processing = False # Hủy bắt đầu
                self.update_toggle_button_text()
                return

            # --- Chuẩn bị và khởi chạy Worker Thread ---
            # Dọn dẹp worker cũ (nếu có và đang chạy - trường hợp hiếm)
            if self.processing_worker and self.processing_worker.isRunning():
                self.log_activity("⚙️ Đang dừng worker cũ...")
                self.processing_worker.stop()
                if not self.processing_worker.wait(1500): # Chờ tối đa 1.5s
                    self.log_activity("⚠️ Worker cũ không dừng kịp thời!")
                # Ngắt kết nối tín hiệu cũ để tránh lỗi (an toàn hơn)
                try: self.processing_worker.log_signal.disconnect()
                except TypeError: pass
                try: self.processing_worker.status_signal.disconnect()
                except TypeError: pass
                # ... ngắt kết nối các tín hiệu khác ...
            self.processing_worker = None # Reset worker cũ

            # Dọn sạch frame cũ trong queue trước khi bắt đầu worker mới
            cleared_count = 0
            while not self.frame_queue.empty():
                try: self.frame_queue.get_nowait(); cleared_count += 1
                except Empty: break
            if cleared_count > 0: self.log_activity(f"ℹ️ Đã dọn {cleared_count} frame cũ khỏi queue.")

            # Tạo và khởi chạy worker MỚI
            self.processing_worker = ProcessingWorker(
                self.frame_queue,
                self.get_reference_data_for_worker, # Func cung cấp ảnh SSIM/rules YOLO
                self.get_current_config_for_worker, # Func cung cấp config cần thiết
                compare_func                        # Hàm so sánh cụ thể (SSIM/YOLO)
            )
            # Kết nối tín hiệu từ worker mới đến các slot của Main thread (self)
            self.processing_worker.log_signal.connect(self.log_activity)
            self.processing_worker.status_signal.connect(self.update_status_label)
            self.processing_worker.save_error_signal.connect(self.save_error_image_from_thread)
            self.processing_worker.comparison_details_signal.connect(self.update_details_display)
            self.processing_worker.error_detected_signal.connect(self._mark_error_occurred)
            self.processing_worker.serial_command_signal.connect(self._send_serial_command)

            # Bắt đầu luồng worker
            self.processing_worker.last_error_time = 0 # Reset cooldown lỗi trong worker
            self.processing_worker.start() # Chạy hàm run() của worker

            # Cập nhật UI trạng thái bắt đầu
            # Đặt trạng thái ban đầu là Unknown hoặc trạng thái chờ xử lý frame đầu tiên
            self.update_status_label(ComparisonStatus.UNKNOWN, {"status": "Starting..."})
            self.details_label.setText("Details: Waiting for first frame...")

            # Bắt đầu timer hẹn giờ chạy (nếu có cấu hình > 0 phút)
            if self._current_runtime_minutes > 0:
                duration_ms = self._current_runtime_minutes * 60 * 1000
                self.runtime_timer.start(duration_ms)
                self.log_activity(f"⏱️ Hẹn giờ tự động dừng sau {self._current_runtime_minutes} phút.")
            else: # Đảm bảo timer đã dừng nếu không dùng
                if self.runtime_timer.isActive(): self.runtime_timer.stop()

            # Reset trạng thái ghi video
            self.video_writer = None; self.current_video_path = None
            self.error_occurred_during_recording = False # Reset cờ lỗi video

        else:
            # --- LOGIC DỪNG XỬ LÝ ---
            self.log_activity("⏹ Đang yêu cầu dừng xử lý...")
            # Gửi tín hiệu dừng cho worker (nếu đang chạy)
            if self.processing_worker and self.processing_worker.isRunning():
                self.processing_worker.stop()
                # Worker sẽ tự kết thúc khi biến self.running=False
                # Không cần chờ (wait) ở đây để tránh treo UI nếu worker có vấn đề

            # Dừng timer hẹn giờ nếu đang chạy
            if self.runtime_timer.isActive():
                self.runtime_timer.stop()
                self.log_activity("⏱️ Đã hủy hẹn giờ dừng.")

            # --- Hoàn tất việc ghi video (nếu đang ghi) ---
            self._finalize_video_recording()

            # Cập nhật UI trạng thái dừng
            self.update_status_label(ComparisonStatus.UNKNOWN, {"status": "Stopped"}) # Đặt về trạng thái không xác định
            self.details_label.setText("Details: N/A") # Reset chi tiết

            self.log_activity("⏹ Quá trình xử lý đã được yêu cầu dừng.")

        # Cập nhật trạng thái enable/disable của các nút và style nút Start/Stop
        self._update_controls_state()
        self.update_toggle_button_text()


    @QtCore.pyqtSlot()
    def _mark_error_occurred(self):
        """Slot được gọi bởi worker khi phát hiện lỗi đầu tiên trong phiên ghi video."""
        # Chỉ log lần đầu tiên đánh dấu lỗi
        if self._record_on_error_enabled and not self.error_occurred_during_recording:
            self.log_activity("❗️ Phát hiện lỗi đầu tiên trong phiên xử lý. Video sẽ được lưu khi dừng.")
        # Đánh dấu là đã có lỗi (dùng để quyết định xóa hay giữ video cuối)
        self.error_occurred_during_recording = True

    def _finalize_video_recording(self):
        """Đóng file video đang ghi và xử lý (lưu hoặc xóa)."""
        if self.video_writer is not None:
            vp = self.current_video_path # Lưu đường dẫn trước khi release
            try:
                self.video_writer.release() # Đóng file video
                self.log_activity("⚪️ Đã dừng ghi video.")
                # Xử lý file video cuối cùng
                if vp and os.path.exists(vp):
                    if not self.error_occurred_during_recording: # Không có lỗi nào xảy ra -> xóa
                         try:
                             os.remove(vp); self.log_activity(f"🗑️ Đã xóa video (vì không có lỗi): {os.path.basename(vp)}")
                         except Exception as e_rem:
                             self.log_activity(f"⚠️ Lỗi khi xóa video không lỗi '{os.path.basename(vp)}': {e_rem}")
                    else: # Có lỗi đã xảy ra -> giữ lại file video
                         self.log_activity(f"💾 Đã lưu video có lỗi: {os.path.basename(vp)}")
            except Exception as e_vid_rel:
                self.log_activity(f"❌ Lỗi khi giải phóng VideoWriter: {e_vid_rel}")
            finally:
                # Reset trạng thái ghi video dù thành công hay lỗi
                self.video_writer = None; self.current_video_path = None
                self.error_occurred_during_recording = False # Reset cờ


    @QtCore.pyqtSlot()
    def _runtime_timer_timeout(self):
        """Slot được gọi khi timer hẹn giờ chạy kết thúc."""
        if self._current_runtime_minutes > 0:
             self.log_activity(f"⏱️ Đã hết thời gian chạy ({self._current_runtime_minutes} phút).")
             QMessageBox.information(self,"Hết Giờ",f"Đã chạy đủ {self._current_runtime_minutes} phút. Ứng dụng sẽ tự động dừng xử lý.")
             # Tự động dừng xử lý nếu đang chạy
             if self.processing:
                  self.toggle_processing() # Gọi hàm dừng chuẩn
             # Cân nhắc có nên tự động đóng ứng dụng không?
             # Hiện tại: Chỉ dừng xử lý.
             # self.close_application() # Bỏ comment dòng này nếu muốn tự đóng
        else:
             # Trường hợp lạ: timer timeout nhưng runtime là 0
             self.log_activity("ℹ️ Timer hẹn giờ timeout nhưng không có thời gian chạy được cấu hình.")

    # --- Comparison Strategies ---
    def compare_ssim_strategy(self, frame, ref_images, config):
        """
        Chiến lược so sánh dùng SSIM.
        So sánh frame với ảnh Norm, Shutdown, Fail.
        Trả về: (ComparisonStatus, details_dict)
        """
        ssim_th = config.get('ssim_threshold', DEFAULT_SSIM_THRESHOLD) # Lấy ngưỡng từ config
        n_img = ref_images.get(REF_NORM)
        s_img = ref_images.get(REF_SHUTDOWN)
        f_img = ref_images.get(REF_FAIL)
        details = {} # Dictionary để trả về các score SSIM

        # 1. Bắt buộc phải có ảnh Norm
        if not isinstance(n_img, np.ndarray) or n_img.size == 0:
            return ComparisonStatus.ERROR, {"error": "Ảnh Norm (SSIM) không hợp lệ"}

        # 2. Tính SSIM với ảnh Norm
        score_n = ssim_opencv(frame, n_img)
        details["ssim_norm"] = score_n # Luôn thêm score Norm vào details (kể cả khi None)

        if score_n is None: # Lỗi khi tính SSIM với Norm
            return ComparisonStatus.ERROR, {"error": "Lỗi tính SSIM với ảnh Norm", **details}
        elif score_n >= ssim_th: # Nếu khớp ảnh Norm -> Trạng thái NORMAL
            return ComparisonStatus.NORMAL, details

        # 3. Nếu không khớp Norm, tính SSIM với ảnh Shutdown (nếu có)
        score_s = None
        if isinstance(s_img, np.ndarray) and s_img.size > 0:
            score_s = ssim_opencv(frame, s_img)
            details["ssim_shutdown"] = score_s # Thêm score Shutdown
            if score_s is not None and score_s >= ssim_th:
                # Khớp ảnh Shutdown -> Trạng thái SHUTDOWN
                return ComparisonStatus.SHUTDOWN, details

        # 4. Nếu không khớp Norm/Shutdown, tính SSIM với ảnh Fail (nếu có)
        score_f = None
        if isinstance(f_img, np.ndarray) and f_img.size > 0:
            score_f = ssim_opencv(frame, f_img)
            details["ssim_fail"] = score_f # Thêm score Fail
            if score_f is not None and score_f >= ssim_th:
                # Khớp ảnh Fail -> Trạng thái FAIL
                return ComparisonStatus.FAIL, details

        # 5. Nếu không khớp bất kỳ ảnh nào -> Trạng thái UNKNOWN
        return ComparisonStatus.UNKNOWN, details


    def compare_yolo_strategy(self, frame, yolo_rules, config):
        """
        Chiến lược so sánh dùng YOLOv8.
        Dự đoán đối tượng trên frame và kiểm tra với bộ quy tắc (yolo_rules).
        Trả về: (ComparisonStatus, details_dict)
        """
        # Kiểm tra điều kiện tiên quyết
        if not YOLO_AVAILABLE or self.yolo_model is None:
            return ComparisonStatus.ERROR, {"error": "YOLO không sẵn sàng (chưa cài/chưa tải model)"}
        conf_threshold = config.get('yolo_confidence', DEFAULT_YOLO_CONFIDENCE) # Lấy ngưỡng conf từ config

        try:
            # === Thực hiện dự đoán YOLO ===
            # verbose=False: Tắt log mặc định của ultralytics
            # imgsz có thể giúp chuẩn hóa kích thước đầu vào model
            predict_start = time.time()
            results = self.yolo_model.predict(frame, conf=conf_threshold, verbose=False, imgsz=640) # imgsz=640 là kích thước phổ biến
            predict_time = time.time() - predict_start

            # Kiểm tra kết quả trả về
            if not results or len(results) == 0:
                 return ComparisonStatus.ERROR, {"error": "YOLO predict không trả về kết quả"}

            # === Trích xuất thông tin dự đoán ===
            detections = results[0] # Lấy kết quả cho ảnh đầu tiên (và duy nhất trong batch=1)
            detected_objects = {}   # Dictionary đếm số lượng {class_name: count}
            # object_details_list = [] # Tùy chọn: list chi tiết từng bounding box
            obj_count = 0           # Tổng số đối tượng phát hiện được

            # Kiểm tra xem có hộp giới hạn nào được phát hiện không
            if detections.boxes is not None and detections.names is not None:
                 obj_count = len(detections.boxes) # Số lượng hộp giới hạn
                 class_names_map = detections.names # Mapping từ class_id sang class_name

                 # Lặp qua từng hộp giới hạn phát hiện được
                 for i in range(obj_count):
                     try:
                         box = detections.boxes[i] # Lấy đối tượng box theo index
                         # Lấy class_id và confidence, đảm bảo chuyển sang kiểu Python chuẩn
                         class_id = int(box.cls.item())
                         confidence = float(box.conf.item())
                         # Lấy tên class từ map, nếu không có dùng ID
                         class_name = class_names_map.get(class_id, f"ID_{class_id}")

                         # --- Đếm số lượng theo class name ---
                         detected_objects[class_name] = detected_objects.get(class_name, 0) + 1

                         # --- Tùy chọn: Trích xuất thêm chi tiết (ví dụ: tọa độ box) ---
                         # bbox_xywhn = box.xywhn.cpu().numpy()[0] # Tọa độ chuẩn hóa [x_center, y_center, width, height]
                         # object_details_list.append({
                         #     "class": class_name, "conf": confidence,
                         #     "box_norm": [round(c, 4) for c in bbox_xywhn]
                         # })

                     except Exception as box_err:
                          # Log lỗi xử lý một box cụ thể mà không dừng toàn bộ
                          # Dùng print tạm thời, hoặc log_signal nếu muốn đưa vào log chính
                          print(f"Warning: Lỗi xử lý YOLO box thứ {i}: {box_err}")
                          continue # Bỏ qua box bị lỗi này

            # Tạo dictionary chi tiết kết quả để trả về
            details = {"detected": detected_objects, "count": obj_count}
            # Thêm thời gian dự đoán vào details nếu muốn theo dõi hiệu năng
            # details["predict_time_ms"] = round(predict_time * 1000)
            # if object_details_list: details["boxes"] = object_details_list # Thêm chi tiết box nếu cần

            # === Áp dụng quy tắc đã định nghĩa trong yolo_rules ===
            # Lấy các bộ quy tắc tương ứng với từng trạng thái
            norm_rules = yolo_rules.get(REF_NORM, {})
            shut_rules = yolo_rules.get(REF_SHUTDOWN, {})
            fail_rules = yolo_rules.get(REF_FAIL, {})

            # --- Ưu tiên kiểm tra: FAIL -> SHUTDOWN -> NORMAL ---
            # Kiểm tra quy tắc FAIL
            is_fail, fail_reason = self._check_yolo_rule(detected_objects, fail_rules)
            if is_fail:
                details["reason"] = fail_reason # Thêm lý do khớp FAIL
                return ComparisonStatus.FAIL, details

            # Kiểm tra quy tắc SHUTDOWN
            is_shut, shut_reason = self._check_yolo_rule(detected_objects, shut_rules)
            if is_shut:
                details["reason"] = shut_reason # Thêm lý do khớp SHUTDOWN
                return ComparisonStatus.SHUTDOWN, details

            # Kiểm tra quy tắc NORMAL
            is_norm, norm_reason = self._check_yolo_rule(detected_objects, norm_rules)
            if is_norm:
                details["reason"] = norm_reason # Thêm lý do khớp NORMAL
                return ComparisonStatus.NORMAL, details

            # --- Nếu không khớp với bất kỳ quy tắc nào ---
            details["reason"] = "Không khớp quy tắc (Fail/Shutdown/Norm)"
            return ComparisonStatus.UNKNOWN, details

        except Exception as e:
            # Bắt lỗi chung trong quá trình dự đoán hoặc xử lý kết quả YOLO
            err_msg = f"Lỗi xử lý YOLO: {type(e).__name__}: {e}"
            print(f"💥 {err_msg}") # In ra console để debug nhanh
            print(traceback.format_exc())    # In traceback đầy đủ
            # Trả về trạng thái ERROR
            return ComparisonStatus.ERROR, {"error": err_msg}


    def _check_yolo_rule(self, detected_objects, rules):
        """
        Kiểm tra xem các đối tượng phát hiện (detected_objects dict) có khớp với
        bộ quy tắc (rules dict) không.
        Trả về: (bool: True nếu khớp, False nếu không, str: Lý do khớp/không khớp)
        """
        # Nếu không có quy tắc nào -> coi như không khớp
        if not rules: return False, "Không có quy tắc để kiểm tra"

        reasons_violated = [] # Chỉ lưu lý do vi phạm quy tắc
        match_overall = True # Giả định là khớp ban đầu

        # --- Kiểm tra các loại quy tắc (có thể kết hợp) ---

        # 1. "any_of": Yêu cầu có mặt ÍT NHẤT MỘT đối tượng trong danh sách này.
        #    Thường dùng cho trạng thái FAIL (phát hiện bất kỳ dấu hiệu lỗi nào).
        any_of_list = rules.get("any_of")
        if isinstance(any_of_list, list) and any_of_list:
            found_any = False
            matching_obj = None
            for obj_name in any_of_list:
                if detected_objects.get(obj_name, 0) > 0: # Kiểm tra số lượng > 0
                    found_any = True
                    matching_obj = obj_name # Ghi lại đối tượng đầu tiên khớp
                    break # Tìm thấy 1 là đủ
            if not found_any:
                # Nếu "any_of" là quy tắc duy nhất -> không khớp
                # Nếu kết hợp với quy tắc khác, chưa chắc đã vi phạm tổng thể
                 if len(rules) == 1:
                      match_overall = False
                      reasons_violated.append(f"Thiếu một trong 'any_of': {', '.join(any_of_list)}")
                 # else: không vi phạm ngay, để các quy tắc khác quyết định
            else:
                # Nếu tìm thấy 1 đối tượng trong "any_of"
                # và "any_of" là quy tắc DUY NHẤT -> Khớp ngay lập tức
                 if len(rules) == 1:
                      return True, f"Tìm thấy '{matching_obj}' (từ 'any_of')"
                # Nếu còn quy tắc khác, ghi nhận là đã khớp phần "any_of" (không thêm vào reasons_violated)

        # Nếu chỉ có 'any_of' và nó không khớp, trả về False luôn
        if len(rules) == 1 and 'any_of' in rules and not match_overall:
             return False, "; ".join(reasons_violated)

        # 2. "required_objects" / "min_counts": Các đối tượng BẮT BUỘC phải có và đủ số lượng tối thiểu.
        required_list = rules.get("required_objects")
        min_counts_dict = rules.get("min_counts", {}) # Có thể kết hợp với required_list hoặc dùng độc lập
        check_targets = set() # Tập hợp các đối tượng cần kiểm tra min_count
        if isinstance(required_list, list): check_targets.update(required_list)
        check_targets.update(min_counts_dict.keys()) # Thêm các key từ min_counts

        if check_targets:
            rule_violated = False
            temp_reasons = []
            for req_obj in check_targets:
                 req_count = min_counts_dict.get(req_obj, 1 if req_obj in (required_list or []) else 0) # Cần >= 1 nếu trong required_list, ngược lại tùy min_counts
                 actual_count = detected_objects.get(req_obj, 0)
                 if actual_count < req_count:
                     rule_violated = True
                     temp_reasons.append(f"Thiếu '{req_obj}' (cần {req_count}, có {actual_count})")
            if rule_violated:
                match_overall = False
                reasons_violated.extend(temp_reasons)

        # 3. "forbidden_objects": Các đối tượng KHÔNG ĐƯỢC PHÉP xuất hiện.
        forbidden_list = rules.get("forbidden_objects")
        if isinstance(forbidden_list, list) and forbidden_list:
            rule_violated = False
            temp_reasons = []
            for fob_obj in forbidden_list:
                if detected_objects.get(fob_obj, 0) > 0:
                    rule_violated = True
                    temp_reasons.append(f"Phát hiện đối tượng cấm '{fob_obj}'")
            if rule_violated:
                match_overall = False
                reasons_violated.extend(temp_reasons)

        # 4. "max_total_objects": Giới hạn TỔNG số đối tượng TỐI ĐA được phép.
        max_total = rules.get("max_total_objects")
        # Chỉ kiểm tra nếu max_total là số không âm
        if isinstance(max_total, (int, float)) and max_total >= 0:
            current_total = sum(detected_objects.values()) # Tổng tất cả đối tượng phát hiện
            if current_total > max_total:
                match_overall = False
                reasons_violated.append(f"Quá nhiều đối tượng (tối đa {int(max_total)}, có {current_total})")

        # 5. "exact_total_objects": Yêu cầu CHÍNH XÁC TỔNG số đối tượng.
        exact_total = rules.get("exact_total_objects")
        # Chỉ kiểm tra nếu exact_total là số không âm
        if isinstance(exact_total, (int, float)) and exact_total >= 0:
            current_total = sum(detected_objects.values())
            if current_total != exact_total:
                match_overall = False
                reasons_violated.append(f"Sai tổng số đối tượng (cần {int(exact_total)}, có {current_total})")

        # --- Kết luận cuối cùng ---
        if match_overall:
             # Nếu không có lý do vi phạm nào được ghi lại -> khớp
             if not reasons_violated:
                  # Lý do khớp có thể là "Tất cả quy tắc khớp" hoặc lý do từ 'any_of' nếu đó là quy tắc duy nhất
                  match_reason = "Tất cả quy tắc khớp"
                  if len(rules) == 1 and 'any_of' in rules and isinstance(any_of_list, list):
                      # Tìm lại đối tượng đã khớp 'any_of'
                      for obj_name in any_of_list:
                           if detected_objects.get(obj_name, 0) > 0:
                                match_reason = f"Tìm thấy '{obj_name}' (từ 'any_of')"
                                break
                  return True, match_reason
             else:
                 # Trường hợp lạ: match_overall là True nhưng lại có reasons_violated? -> Coi như không khớp
                  return False, "; ".join(reasons_violated) or "Không khớp quy tắc không xác định"
        else:
            # Nếu match_overall là False -> không khớp, trả về lý do vi phạm
            return False, "; ".join(reasons_violated) or "Không khớp quy tắc không xác định"


    # --- Image Saving ---
    @QtCore.pyqtSlot(np.ndarray, str)
    def save_error_image_from_thread(self, frame_copy, file_path):
        """Lưu ảnh lỗi (được gửi từ worker) vào file trong luồng chính."""
        try:
            # Lấy thư mục từ đường dẫn file
            save_dir = os.path.dirname(file_path)
            # Đảm bảo thư mục tồn tại trước khi lưu (an toàn hơn)
            if save_dir and not os.path.exists(save_dir):
                 os.makedirs(save_dir, exist_ok=True)

            # Sử dụng imencode để xử lý định dạng PNG và tùy chọn nén
            # Mức nén PNG từ 0 (không nén) đến 9 (nén cao nhất, chậm hơn)
            encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 3] # Mức nén 3 (cân bằng tốt)
            success, buf = cv2.imencode('.png', frame_copy, encode_param)
            if not success or buf is None:
                raise ValueError("cv2.imencode thất bại khi mã hóa ảnh PNG.")

            # Ghi buffer đã mã hóa vào file (an toàn với đường dẫn Unicode)
            with open(file_path, "wb") as f:
                f.write(buf.tobytes())
            # Log thành công
            self.log_activity(f"💾 Lưu ảnh lỗi: {os.path.basename(file_path)}")

        except Exception as e:
            # Log lỗi nếu lưu thất bại
            self.log_activity(f"❌ Lỗi khi lưu ảnh lỗi '{os.path.basename(file_path)}': {e}")
            # Ghi traceback để debug dễ hơn (vì đây là luồng chính)
            self.log_activity(traceback.format_exc())


    # --- Application Closing ---
    def close_application(self):
        """Đóng ứng dụng một cách an toàn (được gọi bởi nút Exit)."""
        self.log_activity("🚪 Đang yêu cầu đóng ứng dụng...")
        self.close() # Kích hoạt sự kiện closeEvent của QMainWindow

    def closeEvent(self, event):
        """Xử lý sự kiện đóng cửa sổ, dọn dẹp tài nguyên trước khi thoát."""
        # Xác nhận lần cuối trước khi đóng? (Tùy chọn)
        # reply = QMessageBox.question(self, 'Xác nhận thoát', 'Bạn có chắc muốn đóng ứng dụng?',
        #                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # if reply == QMessageBox.No:
        #     event.ignore() # Hủy sự kiện đóng
        #     return

        self.log_activity("🚪 Bắt đầu quá trình dọn dẹp trước khi đóng...")

        # 1. Dừng các Timers
        if self.runtime_timer.isActive(): self.runtime_timer.stop(); self.log_activity("⏱️ Timer hẹn giờ đã dừng.")
        if self.frame_timer.isActive(): self.frame_timer.stop(); self.log_activity("⏱️ Timer đọc frame đã dừng.")

        # 2. Dừng Worker thread (nếu đang chạy)
        worker_stopped_cleanly = True
        if self.processing_worker and self.processing_worker.isRunning():
            self.log_activity("⚙️ Yêu cầu dừng luồng xử lý...")
            self.processing_worker.stop()
            # Chờ worker dừng một chút (không nên chờ quá lâu để tránh treo UI)
            if not self.processing_worker.wait(2000): # Chờ tối đa 2 giây
                self.log_activity("⚠️ Luồng xử lý không dừng kịp thời! Có thể cần đóng cứng.")
                worker_stopped_cleanly = False
            else: self.log_activity("✅ Luồng xử lý đã dừng.")
        # Đặt lại cờ processing dù worker có dừng hẳn hay không
        self.processing = False

        # 3. Giải phóng Webcam
        if self.cap and self.cap.isOpened():
            self.log_activity("🚫 Đang giải phóng webcam...")
            try: self.cap.release()
            except Exception as e_cap: self.log_activity(f"⚠️ Lỗi khi giải phóng webcam: {e_cap}")
            finally: self.cap = None; self.log_activity("🚫 Webcam đã giải phóng.")

        # 4. Hoàn tất ghi Video (nếu đang ghi)
        self._finalize_video_recording() # Gọi hàm dọn dẹp video

        # 5. Đóng cổng Serial (nếu đang mở)
        if self.serial_port and self.serial_port.is_open:
            port_name = self.serial_port_name or self.serial_port.name or "N/A"
            self.log_activity(f"🔌 Đang đóng cổng COM {port_name}...")
            try: self.serial_port.close(); self.log_activity(f"🔌 Đã đóng cổng COM {port_name}.")
            except Exception as e_com: self.log_activity(f"⚠️ Lỗi đóng COM {port_name}: {e_com}")
            finally: self.serial_port = None; self.serial_enabled = False

        # 6. Dọn sạch Queue (nếu còn frame)
        q_size = self.frame_queue.qsize()
        if q_size > 0: self.log_activity(f"ℹ️ Dọn {q_size} frame còn lại trong queue...")
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except Empty: break

        # 7. Giải phóng Model YOLO (nếu đã tải)
        if self.yolo_model is not None:
             self.log_activity("🧠 Giải phóng model YOLO...")
             try:
                 del self.yolo_model; self.yolo_model = None
                 # Optional: Clean GPU memory if using CUDA explicitly
                 # import torch; torch.cuda.empty_cache()
             except Exception as e_yolo_del: self.log_activity(f"ℹ️ Lỗi nhỏ khi giải phóng model YOLO: {e_yolo_del}")

        # 8. Lưu cấu hình lần cuối
        self.log_activity("💾 Lưu cấu hình cuối cùng...")
        try:
            self.save_config()
        except Exception as e_save_final:
             self.log_activity(f"❌ Lỗi khi lưu cấu hình cuối cùng: {e_save_final}")

        self.log_activity("🚪 Dọn dẹp hoàn tất. Tạm biệt!")

        # Ghi dòng log cuối vào file (sau khi log cuối cùng hiển thị trên UI)
        if self.log_file_path:
             try:
                 timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                 # Đảm bảo thư mục tồn tại
                 log_dir = os.path.dirname(self.log_file_path)
                 if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
                 # Ghi dòng cuối
                 with open(self.log_file_path, "a", encoding="utf-8") as lf:
                      lf.write(f"---\n{timestamp} - Ứng dụng đã đóng.\n---\n")
             except Exception as e_log_final:
                  # Không làm gì nếu lỗi ghi log cuối
                  print(f"Note: Lỗi ghi log cuối vào file: {e_log_final}")

        event.accept() # Chấp nhận sự kiện đóng cửa sổ -> ứng dụng sẽ thoát

# --- Main Execution ---
if __name__ == "__main__":
    # Cài đặt thuộc tính cho ứng dụng (HiDPI scaling)
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
         QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
         QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    # Khởi tạo QApplication
    app = QtWidgets.QApplication(sys.argv)

    # Tạo và hiển thị cửa sổ chính
    try:
        window = ImageCheckerApp()
        window.show()
    except Exception as e_init:
        # Lỗi nghiêm trọng ngay khi khởi tạo cửa sổ
        print(f"CRITICAL ERROR during initialization: {e_init}")
        print(traceback.format_exc())
        # Hiển thị MessageBox lỗi nếu có thể
        try:
             msgBox = QMessageBox()
             msgBox.setIcon(QMessageBox.Critical)
             msgBox.setWindowTitle("Lỗi Khởi Tạo")
             msgBox.setText(f"Gặp lỗi nghiêm trọng khi khởi tạo ứng dụng:\n\n{e_init}\n\nXem chi tiết lỗi trong console.")
             msgBox.setStandardButtons(QMessageBox.Ok)
             msgBox.exec_()
        except Exception: pass # Bỏ qua nếu cả MessageBox cũng lỗi
        sys.exit(1) # Thoát với mã lỗi

    # Bắt đầu vòng lặp sự kiện chính của Qt
    sys.exit(app.exec_())
