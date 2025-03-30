import sys
import cv2
import serial
import serial.tools.list_ports
import time
import os
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QPushButton, QLabel, QComboBox, QTextEdit,
                             QFileDialog, QGroupBox, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

# --- Worker Thread cho Webcam ---
class WebcamThread(QThread):
    frame_ready = pyqtSignal(object) # Gửi frame (numpy array)
    error = pyqtSignal(str)          # Gửi thông báo lỗi

    def __init__(self, webcam_index):
        super().__init__()
        self.webcam_index = webcam_index
        self.cap = None
        self._is_running = True

    def run(self):
        self.cap = cv2.VideoCapture(self.webcam_index)
        if not self.cap.isOpened():
            self.error.emit(f"Không thể mở webcam {self.webcam_index}")
            self._is_running = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Có thể điều chỉnh độ phân giải
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self._is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                # Có thể do webcam bị ngắt đột ngột
                self.error.emit("Mất kết nối với webcam hoặc đọc frame thất bại.")
                self._is_running = False
                # Tự động thử kết nối lại sau một khoảng thời gian ngắn? (Tùy chọn)
                # time.sleep(1)
                # self.cap.release()
                # self.cap = cv2.VideoCapture(self.webcam_index)
                # if not self.cap.isOpened():
                #     break # Thoát nếu không kết nối lại được

            # Thêm độ trễ nhỏ để tránh CPU load quá cao nếu cần
            # self.msleep(10) # ~100 FPS max theoretical, thực tế thấp hơn nhiều

        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("Webcam thread stopped.")

    def stop(self):
        self._is_running = False
        self.quit() # Yêu cầu thread thoát vòng lặp sự kiện (nếu có)
        self.wait() # Đợi thread kết thúc hoàn toàn
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("Webcam capture released.")

    def get_properties(self):
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            # Một số webcam trả về fps=0, cần đặt giá trị mặc định
            if fps == 0:
                fps = 30.0 # Giá trị mặc định hợp lý
            return width, height, fps
        return None, None, None

# --- Worker Thread cho Serial ---
class SerialThread(QThread):
    data_received = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, port, baudrate=9600, timeout=1):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self._is_running = True

    def run(self):
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"Serial port {self.port} opened.")
            while self._is_running and self.serial_connection and self.serial_connection.isOpen():
                try:
                    if self.serial_connection.in_waiting > 0:
                        # Đọc từng dòng, loại bỏ ký tự thừa và giải mã
                        line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            self.data_received.emit(line)
                except serial.SerialException as e:
                    self.error.emit(f"Lỗi Serial: {e}")
                    self._is_running = False
                except Exception as e:
                     # Bắt các lỗi không mong muốn khác khi đọc
                     self.error.emit(f"Lỗi đọc Serial không xác định: {e}")
                     self._is_running = False
                # Thêm độ trễ nhỏ để không loop quá nhanh khi không có data
                self.msleep(50)

        except serial.SerialException as e:
            self.error.emit(f"Không thể mở cổng Serial {self.port}: {e}")
        except Exception as e:
             self.error.emit(f"Lỗi khởi tạo Serial không xác định: {e}")
        finally:
            if self.serial_connection and self.serial_connection.isOpen():
                self.serial_connection.close()
            print(f"Serial port {self.port} closed.")

    def stop(self):
        self._is_running = False
        if self.serial_connection and self.serial_connection.isOpen():
             # Đặt timeout nhỏ để readline không bị block quá lâu khi dừng
            self.serial_connection.timeout = 0.1
            # Có thể cần gửi một ký tự đặc biệt để đánh thức nếu nó đang đợi đọc
            # self.serial_connection.write(b'\n') # Gửi newline chẳng hạn
            # Hoặc đóng luôn mà không cần đợi readline hoàn thành
            try:
                self.serial_connection.close()
            except Exception as e:
                print(f"Lỗi khi đóng Serial trong stop(): {e}")

        self.quit()
        self.wait()
        print("Serial thread stopped.")


# --- Cửa sổ chính ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Giám sát Webcam và Điều khiển Serial")
        self.setGeometry(100, 100, 900, 700) # Tăng kích thước cửa sổ

        # Biến trạng thái
        self.webcam_thread = None
        self.serial_thread = None
        self.video_writer = None
        self.is_recording = False
        self.is_paused = False
        self.save_directory = ""
        self.current_frame = None
        self.webcam_properties = {'width': None, 'height': None, 'fps': None}

        # Khởi tạo giao diện
        self._init_ui()

        # Scan webcam và cổng COM khi khởi động
        self._scan_webcams()
        self._scan_serial_ports()

        # Timer để cập nhật UI liên tục hơn (ví dụ: trạng thái nhấp nháy) - Tùy chọn
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status_visuals)
        self.recording_flash_state = False


    def _init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- Khu vực hiển thị Video ---
        self.video_frame_label = QLabel("Chưa bật Webcam")
        self.video_frame_label.setAlignment(Qt.AlignCenter)
        self.video_frame_label.setFont(QFont("Arial", 16))
        self.video_frame_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        self.video_frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Cho phép co giãn
        # Thiết lập kích thước tối thiểu để tránh bị quá nhỏ ban đầu
        self.video_frame_label.setMinimumSize(640, 480) # Kích thước ban đầu hợp lý
        self.main_layout.addWidget(self.video_frame_label)

        # --- Tab Widget ---
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        # -- Tab 1: Config --
        self.tab_config = QWidget()
        self.tab_widget.addTab(self.tab_config, "Config")
        self.config_layout = QVBoxLayout(self.tab_config)

        # Webcam Group
        webcam_group = QGroupBox("Cấu hình Webcam")
        webcam_layout = QHBoxLayout()
        self.combo_webcam = QComboBox()
        self.btn_scan_webcam = QPushButton("Quét Webcam")
        self.btn_start_webcam = QPushButton("Bật Webcam")
        self.btn_stop_webcam = QPushButton("Tắt Webcam")
        self.btn_stop_webcam.setEnabled(False)
        webcam_layout.addWidget(QLabel("Chọn Webcam:"))
        webcam_layout.addWidget(self.combo_webcam)
        webcam_layout.addWidget(self.btn_scan_webcam)
        webcam_layout.addWidget(self.btn_start_webcam)
        webcam_layout.addWidget(self.btn_stop_webcam)
        webcam_group.setLayout(webcam_layout)
        self.config_layout.addWidget(webcam_group)

        # Save Directory Group
        save_dir_group = QGroupBox("Thư mục lưu Video")
        save_dir_layout = QHBoxLayout()
        self.lbl_save_dir = QLabel("Chưa chọn thư mục")
        self.lbl_save_dir.setStyleSheet("font-style: italic;")
        self.btn_select_dir = QPushButton("Chọn thư mục")
        save_dir_layout.addWidget(QLabel("Lưu vào:"))
        save_dir_layout.addWidget(self.lbl_save_dir, 1) # Chiếm nhiều không gian hơn
        save_dir_layout.addWidget(self.btn_select_dir)
        save_dir_group.setLayout(save_dir_layout)
        self.config_layout.addWidget(save_dir_group)

        # Serial COM Group
        serial_group = QGroupBox("Cấu hình Cổng Serial (COM)")
        serial_layout_main = QVBoxLayout() # Chia thành 2 hàng ngang
        serial_layout_config = QHBoxLayout() # Hàng cấu hình port
        serial_layout_log = QHBoxLayout() # Hàng log

        self.combo_com_port = QComboBox()
        self.btn_scan_serial = QPushButton("Quét Cổng COM")
        self.btn_connect_serial = QPushButton("Kết nối")
        self.btn_disconnect_serial = QPushButton("Ngắt Kết nối")
        self.btn_disconnect_serial.setEnabled(False)
        serial_layout_config.addWidget(QLabel("Cổng COM:"))
        serial_layout_config.addWidget(self.combo_com_port)
        serial_layout_config.addWidget(self.btn_scan_serial)
        serial_layout_config.addWidget(self.btn_connect_serial)
        serial_layout_config.addWidget(self.btn_disconnect_serial)

        self.serial_log = QTextEdit()
        self.serial_log.setReadOnly(True)
        self.serial_log.setFixedHeight(80) # Giới hạn chiều cao log
        serial_layout_log.addWidget(QLabel("Log Serial:"))
        serial_layout_log.addWidget(self.serial_log)

        serial_layout_main.addLayout(serial_layout_config)
        serial_layout_main.addLayout(serial_layout_log)
        serial_group.setLayout(serial_layout_main)
        self.config_layout.addWidget(serial_group)

        # --- Exit Button --- (Đặt cuối tab Config hoặc ngoài tab đều được)
        self.btn_exit = QPushButton("Thoát Chương Trình")
        self.config_layout.addWidget(self.btn_exit, 0, Qt.AlignRight) # Đẩy nút sang phải

        # -- Tab 2: Manual --
        self.tab_manual = QWidget()
        self.tab_widget.addTab(self.tab_manual, "Manual Control")
        self.manual_layout = QVBoxLayout(self.tab_manual)

        manual_controls_layout = QHBoxLayout()
        self.btn_start_record = QPushButton("Bắt đầu Quay (Start Recording)")
        self.btn_pause_record = QPushButton("Tạm dừng (Pause)")
        self.btn_stop_save_record = QPushButton("Dừng & Lưu (Stop & Save)")

        self.btn_start_record.setEnabled(False) # Chỉ bật khi webcam chạy
        self.btn_pause_record.setEnabled(False)
        self.btn_stop_save_record.setEnabled(False)

        manual_controls_layout.addWidget(self.btn_start_record)
        manual_controls_layout.addWidget(self.btn_pause_record)
        manual_controls_layout.addWidget(self.btn_stop_save_record)
        self.manual_layout.addLayout(manual_controls_layout)

        self.lbl_manual_status = QLabel("Trạng thái: Chưa hoạt động")
        self.lbl_manual_status.setAlignment(Qt.AlignCenter)
        self.lbl_manual_status.setFont(QFont("Arial", 12, QFont.Bold))
        self.manual_layout.addWidget(self.lbl_manual_status)
        self.manual_layout.addStretch() # Đẩy các control lên trên

        # -- Tab 3: Auto --
        self.tab_auto = QWidget()
        self.tab_widget.addTab(self.tab_auto, "Auto Control (Serial)")
        self.auto_layout = QVBoxLayout(self.tab_auto)
        self.lbl_auto_status = QLabel("Trạng thái: Đợi tín hiệu từ Serial...")
        self.lbl_auto_status.setAlignment(Qt.AlignCenter)
        self.lbl_auto_status.setFont(QFont("Arial", 12, QFont.Bold))
        self.auto_layout.addWidget(self.lbl_auto_status)
        self.auto_layout.addStretch() # Đẩy label lên trên

        # --- Thanh Trạng Thái (StatusBar) ---
        self.statusBar = self.statusBar()
        self.status_label = QLabel(" Sẵn sàng")
        self.statusBar.addWidget(self.status_label, 1)

        # Kết nối tín hiệu (Signals & Slots)
        # Config Tab
        self.btn_scan_webcam.clicked.connect(self._scan_webcams)
        self.btn_start_webcam.clicked.connect(self._start_webcam)
        self.btn_stop_webcam.clicked.connect(self._stop_webcam)
        self.btn_select_dir.clicked.connect(self._select_save_directory)
        self.btn_scan_serial.clicked.connect(self._scan_serial_ports)
        self.btn_connect_serial.clicked.connect(self._connect_serial)
        self.btn_disconnect_serial.clicked.connect(self._disconnect_serial)
        self.btn_exit.clicked.connect(self.close) # Kết nối nút Thoát

        # Manual Tab
        self.btn_start_record.clicked.connect(self._manual_start_recording)
        self.btn_pause_record.clicked.connect(self._manual_pause_recording)
        self.btn_stop_save_record.clicked.connect(self._manual_stop_save_recording)


    # --- Các phương thức xử lý ---

    def _log_serial(self, message):
        """Thêm tin nhắn vào log Serial và tự động cuộn xuống."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.serial_log.append(f"[{timestamp}] {message}")
        self.serial_log.ensureCursorVisible() # Auto-scroll

    def _update_status(self, message):
        """Cập nhật thanh trạng thái."""
        self.status_label.setText(f" {message}")
        print(f"Status: {message}") # Cũng in ra console để debug

    def _update_status_visuals(self):
        """Cập nhật trạng thái trực quan (ví dụ: nhấp nháy khi quay)."""
        status_text = ""
        style_sheet = ""
        if self.is_recording:
            self.recording_flash_state = not self.recording_flash_state
            base_text = "ĐANG QUAY "
            style_sheet = "color: red; font-weight: bold;" if self.recording_flash_state else "color: darkred; font-weight: bold;"
            if self.is_paused:
                status_text = "TẠM DỪNG "
                style_sheet = "color: orange; font-weight: bold;"
            else:
                 status_text = base_text
        elif self.webcam_thread and self.webcam_thread.isRunning():
             status_text = "Webcam Bật"
             style_sheet = "color: green;"
        else:
            status_text = "Chưa hoạt động"
            style_sheet = ""

        self.lbl_manual_status.setText(f"Trạng thái Manual: {status_text}")
        self.lbl_manual_status.setStyleSheet(style_sheet)
        self.lbl_auto_status.setText(f"Trạng thái Auto: {status_text}")
        self.lbl_auto_status.setStyleSheet(style_sheet)

    def _scan_webcams(self):
        """Quét và liệt kê các webcam có sẵn."""
        self.combo_webcam.clear()
        available_webcams = []
        index = 0
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) # Sử dụng CAP_DSHOW trên Windows để quét nhanh hơn
            if cap.isOpened():
                # Lấy tên hoặc mô tả nếu có thể (thường chỉ là index)
                backend_name = cap.getBackendName()
                # Lấy thông tin chi tiết hơn đôi khi khó khăn và phụ thuộc OS/Driver
                # Bạn có thể cần thư viện bổ sung như `pygrabber` nếu cần tên thân thiện hơn
                # Ở đây dùng index làm tên tạm
                cam_name = f"Webcam {index}"
                if backend_name:
                     cam_name += f" ({backend_name})"

                available_webcams.append((index, cam_name))
                cap.release()
                index += 1
            else:
                cap.release() # Quan trọng dù không isOpened()
                # Thử thêm 1 vài index nữa phòng trường hợp bị nhảy index
                if index < 5: # Thử tối đa 5 index liên tiếp không tìm thấy
                    index += 1
                else:
                    break # Dừng nếu không tìm thấy liên tiếp

        if not available_webcams:
            self.combo_webcam.addItem("Không tìm thấy webcam")
            self.btn_start_webcam.setEnabled(False)
            self._update_status("Không tìm thấy webcam nào.")
        else:
            for idx, name in available_webcams:
                self.combo_webcam.addItem(name, userData=idx) # Lưu index vào userData
            self.btn_start_webcam.setEnabled(True)
            self._update_status(f"Tìm thấy {len(available_webcams)} webcam.")


    def _start_webcam(self):
        """Bắt đầu hiển thị video từ webcam được chọn."""
        if self.webcam_thread and self.webcam_thread.isRunning():
            QMessageBox.warning(self, "Thông báo", "Webcam đang chạy.")
            return

        selected_index = self.combo_webcam.currentIndex()
        if selected_index < 0 or self.combo_webcam.itemData(selected_index) is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn webcam hợp lệ.")
            return

        webcam_idx = self.combo_webcam.itemData(selected_index)

        self.video_frame_label.setText(f"Đang kết nối Webcam {webcam_idx}...")
        QApplication.processEvents() # Cập nhật giao diện

        self.webcam_thread = WebcamThread(webcam_idx)
        self.webcam_thread.frame_ready.connect(self._update_frame)
        self.webcam_thread.error.connect(self._handle_webcam_error)
        # Đảm bảo thread được dọn dẹp khi kết thúc
        self.webcam_thread.finished.connect(self._on_webcam_thread_finished)

        self.webcam_thread.start()

        # Lấy thông số webcam sau khi thread bắt đầu một chút
        # Dùng QTimer để tránh block GUI nếu việc get_properties lâu
        QTimer.singleShot(500, self._fetch_webcam_properties)

        self.btn_start_webcam.setEnabled(False)
        self.btn_stop_webcam.setEnabled(True)
        self.combo_webcam.setEnabled(False)
        self.btn_scan_webcam.setEnabled(False)

        # Cho phép bắt đầu quay khi webcam chạy
        self.btn_start_record.setEnabled(True)
        self.pause_record.setEnabled(False) # Chỉ bật Pause sau khi Start
        self.stop_save_record.setEnabled(False) # Chỉ bật Stop sau khi Start

        self._update_status(f"Đã bật Webcam {webcam_idx}.")
        self.status_timer.start(500) # Bắt đầu timer nhấp nháy trạng thái (0.5 giây)


    def _fetch_webcam_properties(self):
        """Lấy thông số Width, Height, FPS từ webcam thread."""
        if self.webcam_thread:
            w, h, f = self.webcam_thread.get_properties()
            if w and h and f:
                self.webcam_properties = {'width': w, 'height': h, 'fps': f}
                print(f"Webcam properties: {self.webcam_properties}")
                # Kích thước hiển thị có thể khác kích thước gốc
                # self._update_status(f"Webcam {self.combo_webcam.currentData()} [{w}x{h} @ {f:.2f} FPS].")
            else:
                 # Thử lại sau giây lát nếu lần đầu chưa sẵn sàng
                 QTimer.singleShot(1000, self._fetch_webcam_properties)
                 print("Could not get webcam properties yet, retrying...")

    def _stop_webcam(self):
        """Dừng webcam và dọn dẹp."""
        if self.is_recording:
            # Tự động dừng và lưu video nếu đang quay khi tắt webcam
            self._confirm_and_stop_recording("Webcam đang tắt. Bạn có muốn lưu video đang quay không?")

        if self.webcam_thread:
            self.webcam_thread.stop()
            # self.webcam_thread = None # _on_webcam_thread_finished sẽ set về None

    def _on_webcam_thread_finished(self):
        """Được gọi khi webcam thread thực sự kết thúc."""
        self.webcam_thread = None # Dọn dẹp tham chiếu
        self.video_frame_label.setText("Chưa bật Webcam")
        self.video_frame_label.setPixmap(QPixmap()) # Xóa ảnh cũ
        self.btn_start_webcam.setEnabled(True)
        self.btn_stop_webcam.setEnabled(False)
        self.combo_webcam.setEnabled(True)
        self.btn_scan_webcam.setEnabled(True)

        # Tắt các nút điều khiển quay phim
        self.btn_start_record.setEnabled(False)
        self.btn_pause_record.setEnabled(False)
        self.btn_stop_save_record.setEnabled(False)

        self.is_recording = False
        self.is_paused = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.status_timer.stop() # Dừng timer nhấp nháy
        self._update_status("Webcam đã tắt.")
        self._update_status_visuals() # Cập nhật lại trạng thái lần cuối


    def _handle_webcam_error(self, message):
        """Xử lý lỗi từ webcam thread."""
        QMessageBox.critical(self, "Lỗi Webcam", message)
        # Có thể cố gắng dừng webcam nếu lỗi xảy ra khi đang chạy
        if self.webcam_thread and self.webcam_thread.isRunning():
             self._stop_webcam()
        else: # Nếu lỗi xảy ra ngay khi bắt đầu
            self._on_webcam_thread_finished() # Reset UI về trạng thái ban đầu

    def _update_frame(self, frame):
        """Cập nhật hiển thị frame từ webcam."""
        self.current_frame = frame # Lưu frame hiện tại để ghi video

        # Chuyển đổi frame OpenCV (BGR) sang QImage (RGB)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale ảnh để vừa với QLabel, giữ tỷ lệ khung hình
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_frame_label.setPixmap(scaled_pixmap)

        # Ghi frame vào video nếu đang quay và không tạm dừng
        if self.is_recording and not self.is_paused and self.video_writer:
            try:
                # Ghi frame gốc chưa scale, BGR format
                self.video_writer.write(frame)
            except Exception as e:
                self._log_serial(f"Lỗi ghi frame video: {e}") # Log lỗi
                # Có thể dừng ghi hình tại đây nếu lỗi nghiêm trọng
                # self._stop_save_recording()


    def _scan_serial_ports(self):
        """Quét và liệt kê các cổng COM khả dụng."""
        self.combo_com_port.clear()
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.combo_com_port.addItem("Không tìm thấy cổng COM")
            self.btn_connect_serial.setEnabled(False)
            self._update_status("Không tìm thấy cổng COM nào.")
        else:
            for port in ports:
                self.combo_com_port.addItem(f"{port.device} - {port.description}", userData=port.device)
            self.btn_connect_serial.setEnabled(True)
            self._update_status(f"Tìm thấy {len(ports)} cổng COM.")


    def _connect_serial(self):
        """Kết nối tới cổng Serial được chọn."""
        if self.serial_thread and self.serial_thread.isRunning():
            QMessageBox.warning(self, "Thông báo", "Đã kết nối Serial.")
            return

        selected_index = self.combo_com_port.currentIndex()
        if selected_index < 0 or self.combo_com_port.itemData(selected_index) is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn cổng COM hợp lệ.")
            return

        port_name = self.combo_com_port.itemData(selected_index)

        self._log_serial(f"Đang kết nối tới {port_name}...")
        QApplication.processEvents()

        self.serial_thread = SerialThread(port_name) # Sử dụng baudrate mặc định 9600
        self.serial_thread.data_received.connect(self._handle_serial_data)
        self.serial_thread.error.connect(self._handle_serial_error)
        self.serial_thread.finished.connect(self._on_serial_thread_finished)
        self.serial_thread.start()

        # Tạm thời disable nút connect, enable disconnect (sẽ cập nhật lại trong finished/error)
        self.btn_connect_serial.setEnabled(False)
        self.btn_disconnect_serial.setEnabled(True)
        self.combo_com_port.setEnabled(False)
        self.btn_scan_serial.setEnabled(False)
        self._update_status(f"Đang kết nối {port_name}...")


    def _disconnect_serial(self):
        """Ngắt kết nối Serial."""
        if self.serial_thread:
            self._log_serial("Đang ngắt kết nối Serial...")
            self.serial_thread.stop()
            # self.serial_thread = None # _on_serial_thread_finished xử lý


    def _on_serial_thread_finished(self):
         """Được gọi khi serial thread thực sự kết thúc."""
         self.serial_thread = None # Dọn dẹp tham chiếu
         self.btn_connect_serial.setEnabled(True)
         self.btn_disconnect_serial.setEnabled(False)
         self.combo_com_port.setEnabled(True)
         self.btn_scan_serial.setEnabled(True)
         # Chỉ log khi thực sự đã từng kết nối thành công (tránh log khi lỗi ngay lúc đầu)
         # if self.serial_connection_was_successful: # Cần thêm flag này
         self._log_serial("Đã ngắt kết nối Serial.")
         self._update_status("Đã ngắt kết nối Serial.")


    def _handle_serial_error(self, message):
        """Xử lý lỗi từ Serial thread."""
        self._log_serial(f"LỖI SERIAL: {message}")
        self._update_status(f"Lỗi Serial: {message}")
        # Tự động reset UI về trạng thái disconnected
        self._on_serial_thread_finished()
        # Hiện thông báo lỗi cho người dùng
        QMessageBox.critical(self, "Lỗi Serial", message)


    def _handle_serial_data(self, data):
        """Xử lý dữ liệu nhận được từ Serial."""
        self._log_serial(f"Nhận: '{data}'")

        # --- Xử lý lệnh điều khiển Auto ---
        if not (self.webcam_thread and self.webcam_thread.isRunning()):
            self._log_serial("Cảnh báo: Nhận lệnh Serial nhưng webcam chưa bật.")
            # return # Không xử lý lệnh nếu webcam không chạy? Hoặc thông báo lỗi?

        command = data.strip() # Loại bỏ khoảng trắng thừa

        if command == "0": # Bắt đầu quay
             if self.webcam_thread and self.webcam_thread.isRunning():
                if not self.is_recording:
                    self._start_recording("Serial")
                else:
                     self._log_serial("Lệnh '0' bị bỏ qua: Đang quay hoặc tạm dừng.")
             else:
                  self._log_serial("Lệnh '0' bị bỏ qua: Webcam chưa bật.")

        elif command == "1": # Dừng quay, không lưu
            if self.is_recording:
                self._stop_discard_recording("Serial")
            else:
                self._log_serial("Lệnh '1' bị bỏ qua: Không có gì đang quay.")

        elif command == "2": # Dừng quay và lưu
            if self.is_recording:
                 self._stop_save_recording("Serial")
            else:
                self._log_serial("Lệnh '2' bị bỏ qua: Không có gì đang quay.")

        else:
            self._log_serial(f"Lệnh không xác định từ Serial: '{data}'")


    def _select_save_directory(self):
        """Mở hộp thoại chọn thư mục lưu video."""
        directory = QFileDialog.getExistingDirectory(self, "Chọn Thư mục Lưu Video", self.save_directory or os.getcwd())
        if directory:
            self.save_directory = directory
            # Hiển thị đường dẫn rút gọn nếu quá dài
            max_len = 40
            display_path = directory
            if len(directory) > max_len:
                display_path = "..." + directory[-(max_len-3):]

            self.lbl_save_dir.setText(display_path)
            self.lbl_save_dir.setToolTip(directory) # Tooltip hiển thị đường dẫn đầy đủ
            self._update_status(f"Đã chọn thư mục lưu: {directory}")
        else:
             self._update_status("Việc chọn thư mục đã bị hủy.")


    def _generate_video_filename(self):
        """Tạo tên file video dựa trên thời gian hiện tại."""
        now = datetime.now()
        filename = now.strftime("%H-%M-%S_%d-%m-%Y") + ".mp4" # Sử dụng .mp4 mặc định
        return filename

    def _create_video_writer(self, filepath):
        """Khởi tạo đối tượng cv2.VideoWriter."""
        if not self.webcam_properties['width'] or not self.webcam_properties['height'] or not self.webcam_properties['fps']:
            QMessageBox.critical(self, "Lỗi", "Không thể lấy thông số webcam (kích thước, FPS) để ghi video.")
            self._update_status("Lỗi: Không đủ thông số webcam để ghi hình.")
            return False

        # Chọn codec MP4V (hoạt động tốt trên nhiều nền tảng)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Cho file .mp4
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Hoặc XVID cho .avi (ổn định hơn?) - Nếu dùng XVID, đổi đuôi file
        # filename = os.path.splitext(filepath)[0] + ".avi" # Đổi đuôi nếu dùng XVID
        # filepath = os.path.join(os.path.dirname(filepath), filename) # Cập nhật filepath

        width = self.webcam_properties['width']
        height = self.webcam_properties['height']
        fps = self.webcam_properties['fps']

        # Đảm bảo fps hợp lệ
        if fps <= 0 or fps > 120: # Giới hạn FPS hợp lý
            print(f"Warning: FPS không hợp lệ ({fps}), đặt về 30.0")
            fps = 30.0

        try:
            # Đảm bảo đường dẫn thư mục tồn tại
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            if not self.video_writer.isOpened():
                raise IOError(f"Không thể mở file video để ghi: {filepath}")
            print(f"VideoWriter created for {filepath} [{width}x{h} @ {fps:.2f} FPS]")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Ghi Video", f"Không thể tạo file video: {e}")
            self._update_status(f"Lỗi tạo file video: {e}")
            self.video_writer = None
            return False

    def _start_recording(self, source="Manual"):
        """Bắt đầu quá trình quay video."""
        if not (self.webcam_thread and self.webcam_thread.isRunning()):
             QMessageBox.warning(self, "Cảnh báo", "Webcam chưa bật. Không thể bắt đầu quay.")
             self._log_serial(f"Yêu cầu quay từ {source} thất bại: Webcam chưa bật.")
             return

        if self.is_recording:
             QMessageBox.warning(self, "Cảnh báo", "Đã đang trong quá trình quay phim.")
             self._log_serial(f"Yêu cầu quay từ {source} thất bại: Đang quay.")
             return

        if not self.save_directory:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn thư mục lưu video trong tab Config trước.")
            self._update_status("Cần chọn thư mục lưu trước khi quay.")
            return

        video_filename = self._generate_video_filename()
        full_filepath = os.path.join(self.save_directory, video_filename)

        if self._create_video_writer(full_filepath):
            self.is_recording = True
            self.is_paused = False
            self._update_status(f"Bắt đầu quay: {video_filename}")
            self._log_serial(f"Bắt đầu quay [{source}]: {video_filename}")

            # Cập nhật trạng thái nút Manual
            self.btn_start_record.setEnabled(False)
            self.btn_pause_record.setEnabled(True)
            self.btn_pause_record.setText("Tạm dừng (Pause)")
            self.btn_stop_save_record.setEnabled(True)
            self._update_status_visuals() # Cập nhật label trạng thái


    def _pause_recording(self, source="Manual"):
         """Tạm dừng hoặc tiếp tục quay video."""
         if not self.is_recording:
             self._log_serial(f"Yêu cầu Pause/Resume từ {source} bị bỏ qua: Chưa bắt đầu quay.")
             return

         self.is_paused = not self.is_paused
         if self.is_paused:
             self.btn_pause_record.setText("Tiếp tục (Resume)")
             self._update_status("Đã tạm dừng quay video.")
             self._log_serial(f"Tạm dừng quay [{source}].")
         else:
             self.btn_pause_record.setText("Tạm dừng (Pause)")
             self._update_status("Đã tiếp tục quay video.")
             self._log_serial(f"Tiếp tục quay [{source}].")
         self._update_status_visuals()

    def _stop_save_recording(self, source="Manual"):
        """Dừng quay và lưu file video."""
        if not self.is_recording:
            self._log_serial(f"Yêu cầu Stop&Save từ {source} bị bỏ qua: Chưa bắt đầu quay.")
            return

        saved_filepath = ""
        if self.video_writer:
            saved_filepath = self.video_writer.filename # Lấy tên file đã lưu
            self.video_writer.release()
            self.video_writer = None

        self.is_recording = False
        self.is_paused = False

        self._update_status(f"Đã dừng và lưu video: {os.path.basename(saved_filepath)}")
        self._log_serial(f"Dừng và Lưu [{source}]: {os.path.basename(saved_filepath)}")
        QMessageBox.information(self, "Thông báo", f"Đã lưu video thành công:\n{saved_filepath}")

        # Reset nút Manual
        self.btn_start_record.setEnabled(True) # Cho phép quay lại nếu webcam còn chạy
        self.btn_pause_record.setEnabled(False)
        self.btn_pause_record.setText("Tạm dừng (Pause)")
        self.btn_stop_save_record.setEnabled(False)
        self._update_status_visuals()

    def _stop_discard_recording(self, source="Manual"):
        """Dừng quay nhưng không lưu file video."""
        if not self.is_recording:
            self._log_serial(f"Yêu cầu Stop&Discard từ {source} bị bỏ qua: Chưa bắt đầu quay.")
            return

        # Chỉ cần giải phóng writer mà không lưu frame cuối
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.is_recording = False
        self.is_paused = False

        self._update_status("Đã dừng quay (không lưu).")
        self._log_serial(f"Dừng & Hủy [{source}].")

        # Reset nút Manual
        self.btn_start_record.setEnabled(True)
        self.btn_pause_record.setEnabled(False)
        self.btn_pause_record.setText("Tạm dừng (Pause)")
        self.btn_stop_save_record.setEnabled(False)
        self._update_status_visuals()


    def _manual_start_recording(self):
        self._start_recording("Manual")

    def _manual_pause_recording(self):
        self._pause_recording("Manual")

    def _manual_stop_save_recording(self):
        self._stop_save_recording("Manual")


    def _confirm_and_stop_recording(self, message):
         """Hỏi người dùng có muốn lưu video trước khi dừng không."""
         reply = QMessageBox.question(self, 'Xác nhận',
                                     message,
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                     QMessageBox.Save) # Mặc định chọn Save

         if reply == QMessageBox.Save:
             self._stop_save_recording("Shutdown/WebcamStop")
         elif reply == QMessageBox.Discard:
             self._stop_discard_recording("Shutdown/WebcamStop")
         elif reply == QMessageBox.Cancel:
            return False # Người dùng hủy hành động
         return True # Người dùng chọn Save hoặc Discard


    # --- Xử lý sự kiện đóng cửa sổ ---
    def closeEvent(self, event):
        """Được gọi khi người dùng đóng cửa sổ."""
        print("Close event triggered.")
        # Hỏi lưu video nếu đang quay
        if self.is_recording:
            if not self._confirm_and_stop_recording("Bạn có muốn lưu video đang quay trước khi thoát không?"):
                event.ignore() # Ngăn không cho đóng cửa sổ nếu người dùng chọn Cancel
                return

        # Dừng các thread một cách an toàn
        self._update_status("Đang đóng ứng dụng...")
        QApplication.processEvents() # Cho phép UI cập nhật

        print("Stopping webcam thread...")
        if self.webcam_thread and self.webcam_thread.isRunning():
             self.webcam_thread.stop()
             # Không cần wait ở đây vì thread tự stop() có wait()

        print("Stopping serial thread...")
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            # Không cần wait ở đây

        print("Releasing video writer if any...")
        # Đảm bảo video writer được giải phóng lần cuối (dù đã gọi trong _confirm_and_stop)
        if self.video_writer:
             try:
                 self.video_writer.release()
                 print("Video writer released on exit.")
             except Exception as e:
                 print(f"Error releasing video writer on exit: {e}")
             self.video_writer = None

        self.status_timer.stop() # Dừng timer

        print("Exiting application.")
        event.accept() # Chấp nhận sự kiện đóng cửa sổ

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
