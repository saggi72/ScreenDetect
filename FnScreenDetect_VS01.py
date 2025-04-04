# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QFileDialog, QLabel, QGraphicsView, QGraphicsScene,
                             QMessageBox, QVBoxLayout, QWidget, QTextEdit,
                             QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
                             QSizePolicy, QGroupBox)
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

# --- Constants ---
REF_NORM = "Norm"
REF_SHUTDOWN = "Shutdown" # Giữ nguyên giá trị constant
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

# --- Hàm tính SSIM bằng OpenCV ---
def ssim_opencv(img1, img2, K1=0.01, K2=0.03, win_size=7, data_range=255.0):
    if img1 is None or img2 is None: return None
    if len(img1.shape) > 2: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Tự động resize nếu kích thước khác nhau (ảnh 2 về kích thước ảnh 1)
    if img1.shape != img2.shape:
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            interpolation = cv2.INTER_AREA if (w2 > w1 or h2 > h1) else cv2.INTER_LINEAR
            img2 = cv2.resize(img2, (w1, h1), interpolation=interpolation)
            # print(f"SSIM Resized img2 from {w2}x{h2} to {w1}x{h1}") # Debug log
        except Exception as resize_err:
            print(f"Error resizing image for SSIM: {resize_err}") # Nên dùng logging thay print
            return None
    # Đã resize (nếu cần), tiếp tục tính toán
    h, w = img1.shape; win_size = min(win_size, h, w)
    if win_size % 2 == 0: win_size -= 1
    if win_size < 3: return None # Kích thước cửa sổ quá nhỏ
    if img1.dtype != np.float64: img1 = img1.astype(np.float64)
    if img2.dtype != np.float64: img2 = img2.astype(np.float64)
    C1=(K1*data_range)**2; C2=(K2*data_range)**2; sigma=1.5
    mu1=cv2.GaussianBlur(img1,(win_size,win_size),sigma); mu2=cv2.GaussianBlur(img2,(win_size,win_size),sigma)
    mu1_sq=mu1*mu1; mu2_sq=mu2*mu2; mu1_mu2=mu1*mu2
    sigma1_sq=cv2.GaussianBlur(img1*img1,(win_size,win_size),sigma)-mu1_sq
    sigma2_sq=cv2.GaussianBlur(img2*img2,(win_size,win_size),sigma)-mu2_sq
    sigma12=cv2.GaussianBlur(img1*img2,(win_size,win_size),sigma)-mu1_mu2
    num=(2*mu1_mu2+C1)*(2*sigma12+C2); den=(mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2)
    eps=1e-8; ssim_map=num/(den+eps); ssim_map=np.clip(ssim_map,0,1)
    mssim=np.mean(ssim_map)
    if not np.isfinite(mssim): return None # Xử lý NaN/inf
    return mssim

# --- Worker Thread ---
class ProcessingWorker(QThread):
    log_signal = pyqtSignal(str); status_signal = pyqtSignal(str, str)
    save_error_signal = pyqtSignal(np.ndarray, str); ssim_signal = pyqtSignal(float)
    error_detected_signal = pyqtSignal(); serial_command_signal = pyqtSignal(str)

    def __init__(self, frame_queue, ref_images_provider, config_provider, compare_func):
        super().__init__(); self.frame_queue=frame_queue; self.get_ref_images=lambda: ref_images_provider().copy(); self.get_config=config_provider; self.compare_images=compare_func; self.running=False; self.last_error_time=0; self.last_emitted_serial_state=None

    def run(self):
        self.running=True; self.log_signal.emit("⚙️ Worker started."); last_status_normal_log_time=0; error_signaled_this_session=False; self.last_emitted_serial_state=None
        while self.running:
            try:
                frame=self.frame_queue.get(timeout=0.5)
            except Empty:
                if not self.running: break
                continue
            except Exception as e: # Bắt lỗi khi lấy frame từ queue
                self.log_signal.emit(f"❌ Error getting frame from queue: {e}")
                continue # Bỏ qua frame này và tiếp tục vòng lặp

            if not self.running: break # Kiểm tra lại sau khi lấy frame

            try:
                # Lấy config và ảnh tham chiếu MỚI NHẤT mỗi vòng lặp
                cfg=self.get_config()
                ssim_th=cfg.get('ssim_threshold',DEFAULT_SSIM_THRESHOLD)
                err_cd=cfg.get('error_cooldown',DEFAULT_ERROR_COOLDOWN)
                err_f=cfg.get('error_folder')
                refs=self.get_ref_images()
                n_img,s_img,f_img = refs.get(REF_NORM), refs.get(REF_SHUTDOWN), refs.get(REF_FAIL)

                # --- Kiểm tra ảnh Norm ---
                if not isinstance(n_img,np.ndarray) or n_img.size == 0:
                    ct=time.time();
                    if not hasattr(self,'lnwt') or ct-getattr(self,'lnwt',0)>60: # Log mỗi 60s
                        self.log_signal.emit("⚠️ Ảnh Norm không hợp lệ hoặc chưa được tải. Không thể so sánh."); setattr(self,'lnwt',ct)
                    self.ssim_signal.emit(-1.0) # Gửi tín hiệu không có SSIM
                    time.sleep(0.1); continue # Chờ và bỏ qua vòng lặp này
                else: # Ảnh Norm OK, xóa timer log lỗi nếu có
                    if hasattr(self,'lnwt'): delattr(self,'lnwt')

                # --- So sánh với ảnh Norm ---
                is_match_n, score_n = self.compare_images(frame, n_img, ssim_th)
                self.ssim_signal.emit(score_n if score_n is not None else -1.0) # Gửi điểm SSIM (hoặc -1 nếu lỗi)

                if score_n is None: # Lỗi khi so sánh với Norm
                    ct=time.time();
                    if not hasattr(self,'lcflt') or ct-getattr(self,'lcflt',0)>10: # Log mỗi 10s
                        self.log_signal.emit("⚠️ Lỗi khi so sánh frame với ảnh Norm."); setattr(self,'lcflt',ct)
                    time.sleep(0.1); continue # Chờ và bỏ qua
                if hasattr(self,'lcflt'): delattr(self,'lcflt') # So sánh OK, xóa timer log lỗi

                ser_state=None # Trạng thái serial sẽ gửi (Norm, Shutdown, Fail)

                # --- Trường hợp khớp Norm ---
                if is_match_n:
                    ct=time.time();
                    if ct-last_status_normal_log_time>5: # Cập nhật status "Normal" mỗi 5s
                        self.status_signal.emit("Normal","lightgreen"); last_status_normal_log_time=ct
                    ser_state=REF_NORM;
                    # Chỉ gửi lệnh serial nếu trạng thái thay đổi
                    if self.last_emitted_serial_state != ser_state:
                        self.serial_command_signal.emit(ser_state)
                        self.last_emitted_serial_state = ser_state
                    time.sleep(0.05) # Giảm tải CPU một chút khi normal
                    continue # Ảnh khớp Norm, quay lại đầu vòng lặp chính

                # --- Trường hợp KHÔNG khớp Norm ---
                last_status_normal_log_time=0 # Reset timer log normal
                msg,color,save_img,err_sub="Unknown Mismatch!","orange",True,"unknown" # Giá trị mặc định
                log_m=f"⚠️ Không khớp Norm (SSIM: {score_n:.4f})."
                rec_err=True # Mặc định là lỗi cần ghi hình/lưu ảnh
                ser_state=REF_FAIL # Mặc định gửi Fail nếu không khớp các trạng thái khác

                # --- Kiểm tra khớp Shutdown (chỉ nếu ảnh Shutdown tồn tại) ---
                if isinstance(s_img, np.ndarray) and s_img.size > 0:
                    is_match_s, score_s = self.compare_images(frame, s_img, ssim_th)
                    if score_s is not None and is_match_s:
                        msg,color,err_sub="Shutdown","lightblue","shutdown"
                        log_m=f"ℹ️ Trạng thái Shutdown (SSIM vs Shutdown: {score_s:.4f})."
                        rec_err=False # Shutdown thường không phải lỗi cần ghi lại
                        ser_state=REF_SHUTDOWN
                        self.log_signal.emit(log_m) # Log trạng thái Shutdown
                        self.status_signal.emit(msg,color) # Cập nhật status UI

                # --- Kiểm tra khớp Fail (chỉ nếu ảnh Fail tồn tại và CHƯA khớp Shutdown) ---
                if ser_state != REF_SHUTDOWN and isinstance(f_img, np.ndarray) and f_img.size > 0:
                     is_match_f, score_f = self.compare_images(frame, f_img, ssim_th)
                     if score_f is not None and is_match_f:
                         msg,color,err_sub="FAIL!","red","fail"
                         log_m=f"❌ Trạng thái FAIL (SSIM vs Fail: {score_f:.4f})."
                         rec_err=True # Fail là lỗi cần ghi lại
                         ser_state=REF_FAIL
                         self.log_signal.emit(log_m) # Log trạng thái Fail
                         self.status_signal.emit(msg,color) # Cập nhật status UI

                # Nếu vẫn là unknown (không khớp Norm, Shutdown, Fail), log và cập nhật status
                if err_sub=="unknown":
                    self.log_signal.emit(log_m) # Log mismatch ban đầu
                    self.status_signal.emit(msg,color)

                # --- Xử lý sau khi xác định trạng thái ---
                # Gửi tín hiệu lỗi để bắt đầu ghi video nếu cần (chỉ 1 lần/phiên xử lý)
                if rec_err and not error_signaled_this_session:
                    self.error_detected_signal.emit()
                    error_signaled_this_session = True

                # Gửi lệnh Serial nếu trạng thái thay đổi so với lần gửi trước
                if ser_state and self.last_emitted_serial_state != ser_state:
                     self.serial_command_signal.emit(ser_state)
                     self.last_emitted_serial_state = ser_state

                # Lưu ảnh lỗi nếu cần, có thư mục lỗi và đã hết cooldown
                ct=time.time();
                if save_img and err_f and (ct-self.last_error_time > err_cd):
                    try:
                        save_folder=os.path.join(err_f, err_sub) # Thư mục con theo loại lỗi
                        os.makedirs(save_folder, exist_ok=True) # Tạo nếu chưa có
                        timestamp=time.strftime('%Y%m%d_%H%M%S') + f"_{int((ct-int(ct))*1000):03d}" # Thêm ms
                        filename=f"{err_sub}_{timestamp}.png"
                        filepath=os.path.join(save_folder, filename)
                        # Gửi tín hiệu để luồng chính thực hiện lưu file I/O
                        self.save_error_signal.emit(frame.copy(), filepath)
                        self.last_error_time = ct # Reset cooldown timer
                    except Exception as e:
                        self.log_signal.emit(f"❌ Lỗi khi chuẩn bị lưu ảnh lỗi: {e}")
                elif not err_f and save_img: # Log cảnh báo nếu cần lưu nhưng chưa có folder
                    ct2=time.time()
                    if not hasattr(self,'lsfw') or ct2-getattr(self,'lsfw',0)>60: # Log mỗi 60s
                        self.log_signal.emit("⚠️ Chưa cấu hình thư mục lỗi để lưu ảnh.")
                        setattr(self,'lsfw',ct2)
                # else: # Không cần lưu hoặc đang cooldown
                #     pass

                time.sleep(0.1) # Giảm tải CPU một chút khi có mismatch/lỗi

            except Exception as e: # Bắt lỗi chung trong logic so sánh
                self.log_signal.emit(f"❌ Lỗi nghiêm trọng trong worker logic: {e}")
                self.log_signal.emit(traceback.format_exc())
                time.sleep(0.5) # Chờ một chút nếu có lỗi nghiêm trọng

        # --- Kết thúc vòng lặp while self.running ---
        self.log_signal.emit("⚙️ Worker finished.")
        # Reset trạng thái khi worker dừng
        self.last_emitted_serial_state = None
        error_signaled_this_session = False # Reset cờ để lần chạy sau có thể gửi lại


    def stop(self):
        self.running=False
        self.log_signal.emit("⚙️ Đang yêu cầu dừng worker...") # Log yêu cầu dừng

# --- Main Application Window ---
class ImageCheckerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # --- Khởi tạo State Variables ---
        self.cap=None; self.webcam_fps=15.0; self.frame_timer=QTimer(self); self.ref_images={k:None for k in[REF_NORM,REF_SHUTDOWN,REF_FAIL]}; self.webcam_roi=None; self.processing=False; self.error_folder=None; self.log_file_path=None; self.pixmap_item=None; self.runtime_timer=QTimer(self); self._current_runtime_minutes=DEFAULT_RUNTIME_MINUTES; self._current_ssim_threshold=DEFAULT_SSIM_THRESHOLD; self._current_error_cooldown=DEFAULT_ERROR_COOLDOWN; self._record_on_error_enabled=DEFAULT_RECORD_ON_ERROR; self.video_writer=None; self.current_video_path=None; self.error_occurred_during_recording=False; self.serial_port=None; self.serial_port_name=None; self.serial_baud_rate=DEFAULT_BAUD_RATE; self.serial_enabled=DEFAULT_SERIAL_ENABLED

        # --- Khởi tạo Config Dictionary ---
        self.config={'ssim_threshold':self._current_ssim_threshold,'error_cooldown':self._current_error_cooldown,'runtime_duration_minutes':self._current_runtime_minutes,'record_on_error':self._record_on_error_enabled,'error_folder':None,'ref_paths':{k:None for k in[REF_NORM,REF_SHUTDOWN,REF_FAIL]},'webcam_roi':None,'serial_port':self.serial_port_name,'serial_baud':self.serial_baud_rate,'serial_enabled':self.serial_enabled,}

        # --- Cấu hình Timers và Queue ---
        self.frame_timer.timeout.connect(self.update_frame)
        self.runtime_timer.setSingleShot(True)
        self.runtime_timer.timeout.connect(self._runtime_timer_timeout)
        self.frame_queue=Queue(maxsize=10) # Giới hạn queue để tránh dùng quá nhiều RAM

        # --- Khởi tạo Worker Thread ---
        self.processing_worker=ProcessingWorker(
            self.frame_queue,
            lambda:self.ref_images, # Cung cấp ảnh tham chiếu
            lambda:self.get_current_config_for_worker(), # Cung cấp config cần thiết
            self.compare_images # Cung cấp hàm so sánh
        )
        # Kết nối signals từ worker tới slots của Main thread
        self.processing_worker.log_signal.connect(self.log_activity)
        self.processing_worker.status_signal.connect(self.update_status_label)
        self.processing_worker.save_error_signal.connect(self.save_error_image_from_thread)
        self.processing_worker.ssim_signal.connect(self.update_ssim_display)
        self.processing_worker.error_detected_signal.connect(self._mark_error_occurred)
        self.processing_worker.serial_command_signal.connect(self._send_serial_command)

        # --- Khởi tạo Giao diện và Tải cấu hình ---
        self.init_ui() # Phải gọi trước load_config để các widget tồn tại
        self.load_config() # Tải cấu hình từ file (nếu có)
        self._refresh_com_ports() # Làm mới danh sách cổng COM ban đầu
        self.log_activity("Ứng dụng khởi động.")
        self.update_all_ui_elements() # Cập nhật UI với giá trị đã tải/mặc định

    def get_current_config_for_worker(self):
        # Chỉ gửi những config cần thiết cho worker để giảm overhead
        return {
            'ssim_threshold': self._current_ssim_threshold,
            'error_cooldown': self._current_error_cooldown,
            'error_folder': self.error_folder # Worker cần biết thư mục để lưu ảnh
            }

    # --- Slots Cập nhật Config/State từ UI ---
    @QtCore.pyqtSlot(float)
    def _update_threshold_config(self,value):
        if self._current_ssim_threshold!=value:
            self._current_ssim_threshold=value; self.log_activity(f"⚙️ Ngưỡng SSIM được cập nhật thành: {value:.3f}"); self.config['ssim_threshold']=value
            # Không cần save_config() ngay, sẽ lưu khi bắt đầu xử lý hoặc đóng app

    @QtCore.pyqtSlot(int)
    def _update_cooldown_config(self,value):
        if self._current_error_cooldown!=value:
            self._current_error_cooldown=value; self.log_activity(f"⚙️ Cooldown lưu ảnh lỗi được cập nhật thành: {value} giây"); self.config['error_cooldown']=value

    @QtCore.pyqtSlot(int)
    def _update_runtime_config(self,value):
        if self._current_runtime_minutes!=value:
            self._current_runtime_minutes=value; log_msg=f"⚙️ Thời gian chạy tối đa được cập nhật thành: {'Vô hạn' if value==0 else f'{value} phút'}"; self.log_activity(log_msg); self.config['runtime_duration_minutes']=value

    @QtCore.pyqtSlot()
    def _toggle_record_on_error(self):
        # Không cho phép thay đổi khi đang xử lý
        if self.processing:
            QMessageBox.warning(self,"Đang Xử Lý","Không thể thay đổi cài đặt ghi video khi đang xử lý.")
            return
        self._record_on_error_enabled = not self._record_on_error_enabled
        self.config['record_on_error'] = self._record_on_error_enabled
        self.update_record_button_style() # Cập nhật style nút ngay
        log_msg = f"⚙️ Ghi video khi có lỗi: {'Bật' if self._record_on_error_enabled else 'Tắt'}"
        self.log_activity(log_msg)
        self.save_config() # Lưu thay đổi này ngay lập tức

    @QtCore.pyqtSlot(str)
    def _update_serial_port_config(self,port_name):
        # Chỉ cập nhật nếu tên cổng thực sự thay đổi và không phải placeholder
        new_port = port_name if port_name and "Không tìm thấy" not in port_name else None
        if self.serial_port_name != new_port:
             self.serial_port_name = new_port
             self.config['serial_port']=self.serial_port_name;
             self.log_activity(f"⚙️ Cổng COM được chọn: {self.serial_port_name or 'Chưa chọn'}");
             # Cảnh báo nếu đang kết nối
             if self.serial_enabled:
                 self.log_activity("⚠️ Thay đổi cổng COM yêu cầu Ngắt kết nối và Kết nối lại.")

    @QtCore.pyqtSlot(str)
    def _update_serial_baud_config(self,baud_str):
        try:
            bd=int(baud_str);
            # Chỉ cập nhật nếu baud rate thay đổi
            if self.serial_baud_rate != bd:
                if bd in COMMON_BAUD_RATES: # Chỉ chấp nhận baud rate hợp lệ
                    self.serial_baud_rate=bd; self.config['serial_baud']=bd;
                    self.log_activity(f"⚙️ Baud rate được chọn: {bd}");
                    # Cảnh báo nếu đang kết nối
                    if self.serial_enabled:
                         self.log_activity("⚠️ Thay đổi Baud rate yêu cầu Ngắt kết nối và Kết nối lại.")
                else:
                     self.log_activity(f"⚠️ Baud rate không hợp lệ: {bd}. Sử dụng giá trị cũ: {self.serial_baud_rate}")
                     # Đặt lại ComboBox về giá trị cũ
                     idx=self.baudRateComboBox.findText(str(self.serial_baud_rate));
                     if idx>=0:
                         self.baudRateComboBox.blockSignals(True)
                         self.baudRateComboBox.setCurrentIndex(idx)
                         self.baudRateComboBox.blockSignals(False)

        except ValueError:
            self.log_activity(f"⚠️ Giá trị Baud rate nhập vào không phải số: {baud_str}");
            # Đặt lại ComboBox về giá trị cũ
            idx=self.baudRateComboBox.findText(str(self.serial_baud_rate));
            if idx>=0:
                self.baudRateComboBox.blockSignals(True)
                self.baudRateComboBox.setCurrentIndex(idx)
                self.baudRateComboBox.blockSignals(False)

    # --- init_ui ---
    def init_ui(self):
        self.setWindowTitle("Image Checker v2.8 (Fixed Syntax)"); self.setGeometry(100,100,1350,760);
        central_widget=QWidget(self); self.setCentralWidget(central_widget)

        # --- Graphics View (Hiển thị Webcam) ---
        self.scene=QGraphicsScene(self)
        self.graphicsView=QGraphicsView(self.scene, central_widget)
        self.graphicsView.setGeometry(10,10,640,360)
        self.graphicsView.setStyleSheet("border: 1px solid black;")
        self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray)) # Nền xám khi chưa có video

        # --- Buttons (Cột trái) ---
        bw,bh,vs,bx,yp = 201, 31, 40, 20, 380 # Button width, height, vertical spacing, x-pos, y-start
        # Webcam Controls
        self.ONCam=self.create_button("📷 Bật Webcam"); self.ONCam.setGeometry(bx,yp,bw,bh); self.ONCam.clicked.connect(self.start_webcam)
        self.OFFCam=self.create_button("🚫 Tắt Webcam"); self.OFFCam.setGeometry(bx+bw+10,yp,bw,bh); self.OFFCam.clicked.connect(self.stop_webcam); self.OFFCam.setEnabled(False); yp+=vs
        # Load Reference Image Buttons
        self.SettingButton_Norm=self.create_button("📂 Ảnh Norm"); self.SettingButton_Norm.setGeometry(bx,yp,bw,bh); self.SettingButton_Norm.clicked.connect(lambda: self.load_reference_image(REF_NORM))
        self.SettingButton_Shutdown=self.create_button("📂 Ảnh Shutdown"); self.SettingButton_Shutdown.setGeometry(bx+bw+10,yp,bw,bh); self.SettingButton_Shutdown.clicked.connect(lambda: self.load_reference_image(REF_SHUTDOWN))
        self.SettingButton_Fail=self.create_button("📂 Ảnh Fail"); self.SettingButton_Fail.setGeometry(bx+2*(bw+10),yp,bw,bh); self.SettingButton_Fail.clicked.connect(lambda: self.load_reference_image(REF_FAIL)); yp+=vs
        # Capture Reference Image Buttons
        self.CaptureButton_Norm=self.create_button("📸 Chụp Norm"); self.CaptureButton_Norm.setGeometry(bx,yp,bw,bh); self.CaptureButton_Norm.clicked.connect(lambda: self.capture_reference_from_webcam(REF_NORM)); self.CaptureButton_Norm.setEnabled(False)
        self.CaptureButton_Shut=self.create_button("📸 Chụp Shutdown"); self.CaptureButton_Shut.setGeometry(bx+bw+10,yp,bw,bh); self.CaptureButton_Shut.clicked.connect(lambda: self.capture_reference_from_webcam(REF_SHUTDOWN)); self.CaptureButton_Shut.setEnabled(False)
        self.CaptureButton_Fail=self.create_button("📸 Chụp Fail"); self.CaptureButton_Fail.setGeometry(bx+2*(bw+10),yp,bw,bh); self.CaptureButton_Fail.clicked.connect(lambda: self.capture_reference_from_webcam(REF_FAIL)); self.CaptureButton_Fail.setEnabled(False); yp+=vs
        # ROI and Processing Buttons
        self.SettingButton_ROI_Webcam=self.create_button("✂️ Chọn ROI"); self.SettingButton_ROI_Webcam.setGeometry(bx,yp,bw,bh); self.SettingButton_ROI_Webcam.clicked.connect(self.select_webcam_roi); self.SettingButton_ROI_Webcam.setEnabled(False);
        self.SaveButton=self.create_button("📁 Thư mục lỗi"); self.SaveButton.setGeometry(bx+bw+10,yp,bw,bh); self.SaveButton.clicked.connect(self.select_error_folder) # Đổi vị trí
        self.ToggleProcessingButton=self.create_button("▶️ Bắt đầu"); self.ToggleProcessingButton.setGeometry(bx+2*(bw+10),yp,bw,bh); self.ToggleProcessingButton.clicked.connect(self.toggle_processing); yp+=vs
        # Exit Button
        self.ExitButton=self.create_button("🚪 Thoát"); self.ExitButton.setGeometry(bx,yp,bw,bh); self.ExitButton.clicked.connect(self.close_application)

        # --- Right Panel (Log, Status, Settings) ---
        rx = 670 # Right panel starting x
        lw = self.geometry().width()-rx-20 # Right panel width
        # Log Area
        log_label=QLabel("Log Hoạt Động:",central_widget); log_label.setGeometry(rx,10,150,20);
        self.log_text_edit=QTextEdit(central_widget); self.log_text_edit.setGeometry(rx,35,lw,250); self.log_text_edit.setReadOnly(True); self.log_text_edit.setStyleSheet("border:1px solid black; padding:5px; background-color:white; font-family:Consolas,monospace; font-size:10pt;")
        # Status Labels
        self.process_label=QLabel("Trạng thái: Chờ",central_widget); self.process_label.setGeometry(rx,300,lw,40); self.process_label.setAlignment(Qt.AlignCenter); self.process_label.setStyleSheet("border:1px solid black; padding:5px; background-color:lightgray; font-weight:bold; border-radius:3px;")
        self.ssim_label=QLabel("SSIM: N/A",central_widget); self.ssim_label.setGeometry(rx,345,lw,30); self.ssim_label.setAlignment(Qt.AlignCenter); self.ssim_label.setStyleSheet("padding:5px; background-color:#f0f0f0; border-radius:3px;")

        # --- Settings Controls (Right Panel) ---
        sx_lbl = rx + 20      # Settings label x-pos
        sx_ctrl = sx_lbl + 160 # Settings control x-pos
        sy = 390              # Settings starting y-pos
        s_vs = 40             # Settings vertical spacing
        ctrl_w = 180          # Control width
        btn_w_sm = 40         # Small button width (refresh)

        # SSIM Threshold
        ls=QLabel("Ngưỡng SSIM:",central_widget); ls.setGeometry(sx_lbl,sy,150,31);
        self.ssimThresholdSpinBox=QDoubleSpinBox(central_widget); self.ssimThresholdSpinBox.setGeometry(sx_ctrl,sy,ctrl_w,31); self.ssimThresholdSpinBox.setRange(0.1,1.0); self.ssimThresholdSpinBox.setSingleStep(0.01); self.ssimThresholdSpinBox.setValue(self._current_ssim_threshold); self.ssimThresholdSpinBox.setDecimals(3); self.ssimThresholdSpinBox.valueChanged.connect(self._update_threshold_config); sy+=s_vs
        # Error Cooldown
        lc=QLabel("Cooldown Lỗi (s):",central_widget); lc.setGeometry(sx_lbl,sy,150,31);
        self.cooldownSpinBox=QSpinBox(central_widget); self.cooldownSpinBox.setGeometry(sx_ctrl,sy,ctrl_w,31); self.cooldownSpinBox.setRange(1,300); self.cooldownSpinBox.setSingleStep(1); self.cooldownSpinBox.setValue(self._current_error_cooldown); self.cooldownSpinBox.valueChanged.connect(self._update_cooldown_config); sy+=s_vs
        # Runtime Duration
        lr=QLabel("Thời gian chạy (phút):",central_widget); lr.setGeometry(sx_lbl,sy,150,31);
        self.runtimeSpinBox=QSpinBox(central_widget); self.runtimeSpinBox.setGeometry(sx_ctrl,sy,ctrl_w,31); self.runtimeSpinBox.setRange(0,1440); self.runtimeSpinBox.setSingleStep(10); self.runtimeSpinBox.setValue(self._current_runtime_minutes); self.runtimeSpinBox.setToolTip("0 = Chạy vô hạn"); self.runtimeSpinBox.valueChanged.connect(self._update_runtime_config); sy+=s_vs
        # Record on Error Toggle Button
        self.ToggleRecordOnErrorButton=self.create_button("🎥 Quay video lỗi: Tắt"); self.ToggleRecordOnErrorButton.setGeometry(sx_lbl,sy,ctrl_w+50,31); self.ToggleRecordOnErrorButton.setCheckable(False); self.ToggleRecordOnErrorButton.clicked.connect(self._toggle_record_on_error); sy+=s_vs

        # --- Serial Port Settings ---
        # COM Port Selection
        lcp=QLabel("Cổng COM:",central_widget); lcp.setGeometry(sx_lbl,sy,150,31);
        self.comPortComboBox=QComboBox(central_widget); self.comPortComboBox.setGeometry(sx_ctrl,sy,ctrl_w-btn_w_sm-5,31); self.comPortComboBox.setToolTip("Chọn cổng COM để gửi tín hiệu"); self.comPortComboBox.currentTextChanged.connect(self._update_serial_port_config)
        # Refresh COM Ports Button
        self.refreshComButton=QPushButton("🔄",central_widget); self.refreshComButton.setGeometry(sx_ctrl+ctrl_w-btn_w_sm,sy,btn_w_sm,31); self.refreshComButton.setToolTip("Làm mới danh sách cổng COM"); self.refreshComButton.clicked.connect(self._refresh_com_ports); sy+=s_vs
        # Baud Rate Selection
        lbr=QLabel("Baud Rate:",central_widget); lbr.setGeometry(sx_lbl,sy,150,31);
        self.baudRateComboBox=QComboBox(central_widget); self.baudRateComboBox.setGeometry(sx_ctrl,sy,ctrl_w,31); self.baudRateComboBox.addItems([str(br) for br in COMMON_BAUD_RATES]); self.baudRateComboBox.setToolTip("Chọn tốc độ Baud"); self.baudRateComboBox.setCurrentText(str(self.serial_baud_rate)); self.baudRateComboBox.currentTextChanged.connect(self._update_serial_baud_config); sy+=s_vs
        # Toggle Serial Connection Button
        self.ToggleSerialPortButton=self.create_button("🔌 Kết nối COM"); self.ToggleSerialPortButton.setGeometry(sx_lbl,sy,ctrl_w+50,31); self.ToggleSerialPortButton.setCheckable(False); self.ToggleSerialPortButton.setToolTip("Bật/Tắt gửi tín hiệu qua cổng COM"); self.ToggleSerialPortButton.clicked.connect(self._toggle_serial_port)

        # --- Initial UI Update ---
        # Gọi các hàm cập nhật style ban đầu sau khi tất cả widget đã được tạo
        self.update_button_styles()
        self.update_toggle_button_text()
        self.update_record_button_style()
        self.update_serial_button_style()

    def create_button(self, text): # Helper để tạo button chuẩn
        button = QPushButton(text, self.centralWidget())
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) # Cho phép co giãn ngang
        return button

    def save_config(self):
        # Gom tất cả state hiện tại vào dict config trước khi lưu
        self.config['ssim_threshold']=self._current_ssim_threshold
        self.config['error_cooldown']=self._current_error_cooldown
        self.config['runtime_duration_minutes']=self._current_runtime_minutes
        self.config['record_on_error']=self._record_on_error_enabled
        self.config['error_folder']=self.error_folder
        self.config['webcam_roi']=list(self.webcam_roi) if self.webcam_roi else None
        # Đảm bảo chỉ lưu đường dẫn hợp lệ trong ref_paths
        valid_ref_paths = {}
        for k, p in self.config['ref_paths'].items():
             if k in self.ref_images and isinstance(p, str) and os.path.isfile(p):
                 valid_ref_paths[k] = p
             elif k in self.ref_images and self.ref_images[k] is not None and p is None: # Ảnh từ webcam
                 valid_ref_paths[k] = None # Lưu None để biết là từ webcam
             else: # Không có ảnh hoặc đường dẫn không hợp lệ
                 valid_ref_paths[k] = None
        self.config['ref_paths'] = valid_ref_paths

        self.config['serial_port']=self.serial_port_name
        self.config['serial_baud']=self.serial_baud_rate
        self.config['serial_enabled']=self.serial_enabled # Lưu trạng thái cuối cùng

        try:
            config_dir=os.path.dirname(CONFIG_FILE_NAME)
            # Tạo thư mục config nếu chưa có (chỉ áp dụng nếu CONFIG_FILE_NAME có đường dẫn)
            if config_dir and not os.path.exists(config_dir):
                try:
                    os.makedirs(config_dir, exist_ok=True)
                    self.log_activity(f"ℹ️ Đã tạo thư mục cấu hình: {config_dir}")
                except OSError as e:
                     self.log_activity(f"❌ Lỗi khi tạo thư mục cấu hình '{config_dir}': {e}")
                     QMessageBox.critical(self,"Lỗi Lưu Config",f"Không thể tạo thư mục để lưu cấu hình:\n{config_dir}\nLỗi: {e}")
                     return # Không thử ghi file nếu không tạo được thư mục

            # Ghi file JSON
            with open(CONFIG_FILE_NAME,'w',encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            # self.log_activity("💾 Cấu hình đã được lưu.") # Log khi cần thiết, ví dụ khi đóng app

        except Exception as e:
            self.log_activity(f"❌ Lỗi nghiêm trọng khi lưu cấu hình vào '{CONFIG_FILE_NAME}': {e}")
            self.log_activity(traceback.format_exc())
            QMessageBox.critical(self,"Lỗi Lưu Config",f"Không thể lưu cấu hình vào file:\n{CONFIG_FILE_NAME}\n\nLỗi: {e}")

    def load_config(self):
        if not os.path.exists(CONFIG_FILE_NAME):
            self.log_activity(f"📄 Không tìm thấy file cấu hình '{CONFIG_FILE_NAME}'. Sử dụng cài đặt mặc định."); self.reset_to_defaults(); return
        try:
            with open(CONFIG_FILE_NAME,'r',encoding='utf-8') as f: lcfg=json.load(f)

            # --- Tải các giá trị cơ bản với kiểm tra và giá trị mặc định ---
            try: self._current_ssim_threshold=max(0.1,min(1.0,float(lcfg.get('ssim_threshold',DEFAULT_SSIM_THRESHOLD))))
            except (ValueError, TypeError): self._current_ssim_threshold=DEFAULT_SSIM_THRESHOLD; self.log_activity(f"⚠️ Giá trị ssim_threshold không hợp lệ trong config, dùng mặc định ({DEFAULT_SSIM_THRESHOLD}).")
            try: self._current_error_cooldown=max(1,min(300,int(lcfg.get('error_cooldown',DEFAULT_ERROR_COOLDOWN))))
            except (ValueError, TypeError): self._current_error_cooldown=DEFAULT_ERROR_COOLDOWN; self.log_activity(f"⚠️ Giá trị error_cooldown không hợp lệ trong config, dùng mặc định ({DEFAULT_ERROR_COOLDOWN}).")
            try: self._current_runtime_minutes=max(0,min(1440,int(lcfg.get('runtime_duration_minutes',DEFAULT_RUNTIME_MINUTES))))
            except (ValueError, TypeError): self._current_runtime_minutes=DEFAULT_RUNTIME_MINUTES; self.log_activity(f"⚠️ Giá trị runtime_duration_minutes không hợp lệ trong config, dùng mặc định ({DEFAULT_RUNTIME_MINUTES}).")

            lrec=lcfg.get('record_on_error',DEFAULT_RECORD_ON_ERROR); self._record_on_error_enabled=bool(lrec) if isinstance(lrec,bool) else DEFAULT_RECORD_ON_ERROR

            # --- Tải cài đặt Serial ---
            self.serial_port_name=lcfg.get('serial_port',None)
            if not isinstance(self.serial_port_name, (str, type(None))): # Đảm bảo là string hoặc None
                self.log_activity(f"⚠️ Giá trị serial_port ('{self.serial_port_name}') không hợp lệ, đặt về None."); self.serial_port_name = None

            try:
                baud=int(lcfg.get('serial_baud',DEFAULT_BAUD_RATE));
                self.serial_baud_rate=baud if baud in COMMON_BAUD_RATES else DEFAULT_BAUD_RATE
                if baud not in COMMON_BAUD_RATES and 'serial_baud' in lcfg: # Log nếu baud không chuẩn nhưng có trong config
                     self.log_activity(f"⚠️ Baud rate '{baud}' trong config không nằm trong danh sách phổ biến, dùng mặc định ({DEFAULT_BAUD_RATE}).")
            except (ValueError, TypeError): self.serial_baud_rate=DEFAULT_BAUD_RATE; self.log_activity(f"⚠️ Giá trị serial_baud không hợp lệ trong config, dùng mặc định ({DEFAULT_BAUD_RATE}).")

            # Tải trạng thái serial_enabled, nhưng không tự động kết nối ở đây.
            lse=lcfg.get('serial_enabled',DEFAULT_SERIAL_ENABLED); self.serial_enabled=bool(lse) if isinstance(lse,bool) else DEFAULT_SERIAL_ENABLED
            # Quan trọng: Nếu self.serial_enabled là True ở đây, UI sẽ hiển thị là "Bật" nhưng thực tế chưa kết nối.
            # Người dùng cần bấm "Ngắt kết nối" rồi "Kết nối" lại, hoặc ta phải thêm logic tự kết nối nếu muốn.
            # Để đơn giản, ta sẽ reset self.serial_enabled về False sau khi load, bắt buộc người dùng kết nối lại.
            if self.serial_enabled:
                 self.log_activity("ℹ️ Trạng thái Serial 'Bật' được tải từ config. Vui lòng kết nối lại thủ công nếu cần.")
                 self.serial_enabled = False # Bắt buộc kết nối lại

            # --- Tải thư mục lỗi ---
            lfold=lcfg.get('error_folder'); self.error_folder=None # Reset trước khi load
            if lfold and isinstance(lfold,str):
                if os.path.isdir(lfold):
                    if os.access(lfold,os.W_OK): self.error_folder=lfold # Chỉ chấp nhận nếu tồn tại và ghi được
                    else: self.log_activity(f"⚠️ Không có quyền ghi vào thư mục lỗi đã lưu '{lfold}'. Thư mục lỗi bị vô hiệu hóa.")
                else: self.log_activity(f"⚠️ Đường dẫn thư mục lỗi đã lưu '{lfold}' không tồn tại hoặc không phải thư mục. Thư mục lỗi bị vô hiệu hóa.")
            elif lfold: self.log_activity(f"⚠️ Giá trị error_folder ('{lfold}') trong config không hợp lệ. Thư mục lỗi bị vô hiệu hóa.")

            # --- Tải ROI ---
            lroi=lcfg.get('webcam_roi'); self.webcam_roi=None # Reset trước khi load
            if isinstance(lroi,list) and len(lroi)==4:
                try:
                    rt=tuple(int(x) for x in lroi); # Chuyển sang tuple số nguyên
                    if all(isinstance(v,int) and v>=0 for v in rt) and rt[2]>0 and rt[3]>0: # Kiểm tra hợp lệ (>=0, w>0, h>0)
                        self.webcam_roi=rt
                    else: self.log_activity(f"⚠️ Giá trị ROI trong config không hợp lệ (số âm hoặc w/h=0): {lroi}. ROI bị vô hiệu hóa.")
                except (ValueError, TypeError): self.log_activity(f"⚠️ Giá trị ROI trong config không phải số nguyên: {lroi}. ROI bị vô hiệu hóa.")
            elif lroi is not None: self.log_activity(f"⚠️ Định dạng ROI trong config không hợp lệ (không phải list 4 phần tử): {lroi}. ROI bị vô hiệu hóa.")

            # --- Tải đường dẫn ảnh tham chiếu và load ảnh ---
            lrefs=lcfg.get('ref_paths',{}); self.config['ref_paths']={k:None for k in self.ref_images.keys()} # Reset trong config
            self.ref_images={k:None for k in self.ref_images.keys()} # Reset ảnh đã load
            loaded_image_keys = []
            for k in self.ref_images.keys(): # Duyệt qua các key chuẩn (Norm, Shutdown, Fail)
                p = lrefs.get(k) # Lấy đường dẫn từ config đã tải
                if p and isinstance(p,str): # Nếu có đường dẫn và là string
                    if os.path.isfile(p):
                        try:
                            ib=np.fromfile(p,dtype=np.uint8) # Đọc an toàn với unicode path
                            img=cv2.imdecode(ib,cv2.IMREAD_COLOR)
                            if img is not None:
                                self.ref_images[k]=img # Lưu ảnh đã load
                                self.config['ref_paths'][k] = p # Lưu lại đường dẫn hợp lệ vào config hiện tại
                                loaded_image_keys.append(k)
                            else: # File đọc được nhưng không decode được ảnh
                                self.log_activity(f"⚠️ Không thể giải mã ảnh '{k}' từ file đã lưu: {p}.");
                        except Exception as e: # Lỗi khi đọc file
                            self.log_activity(f"❌ Lỗi khi tải ảnh '{k}' từ file đã lưu {p}: {e}");
                    else: # Đường dẫn có nhưng không phải file
                        self.log_activity(f"⚠️ File ảnh '{k}' đã lưu không tồn tại hoặc không phải file: {p}");
                elif p is None and k in lrefs: # Đường dẫn là None -> có thể là ảnh chụp từ webcam cũ
                    # Không cần load gì cả, chỉ cần biết là không có file
                    pass
                elif p: # Đường dẫn có nhưng không phải string
                     self.log_activity(f"⚠️ Đường dẫn ảnh '{k}' trong config không hợp lệ (không phải string): {p}");

            # Log các ảnh đã tải thành công (sau vòng lặp để tránh spam)
            if loaded_image_keys:
                self.log_activity(f"✅ Đã tải thành công ảnh tham chiếu từ file cho: {', '.join(loaded_image_keys)}")

            # --- Cập nhật dict config cuối cùng và đường dẫn log ---
            # (Đảm bảo config phản ánh đúng state sau khi load và validate)
            self.config.update({
                'ssim_threshold':self._current_ssim_threshold, 'error_cooldown':self._current_error_cooldown,
                'runtime_duration_minutes':self._current_runtime_minutes, 'record_on_error':self._record_on_error_enabled,
                'error_folder':self.error_folder, 'webcam_roi':list(self.webcam_roi) if self.webcam_roi else None,
                'serial_port':self.serial_port_name, 'serial_baud':self.serial_baud_rate,
                'serial_enabled':self.serial_enabled # Sẽ luôn là False sau khi load
                # ref_paths đã được cập nhật trong vòng lặp trên
            })

            self.log_file_path=os.path.join(self.error_folder,LOG_FILE_NAME) if self.error_folder else None
            self.log_activity(f"💾 Đã tải cấu hình từ '{CONFIG_FILE_NAME}'.")

        except json.JSONDecodeError as e:
            self.log_activity(f"❌ Lỗi phân tích JSON trong file cấu hình '{CONFIG_FILE_NAME}': {e}. Sử dụng cài đặt mặc định."); self.reset_to_defaults()
        except Exception as e:
            self.log_activity(f"❌ Lỗi không xác định khi tải cấu hình: {e}."); self.log_activity(traceback.format_exc()); self.reset_to_defaults()

    def reset_to_defaults(self):
        self.log_activity("🔄 Đang reset về cài đặt mặc định...")
        # Reset state variables
        self._current_ssim_threshold=DEFAULT_SSIM_THRESHOLD
        self._current_error_cooldown=DEFAULT_ERROR_COOLDOWN
        self._current_runtime_minutes=DEFAULT_RUNTIME_MINUTES
        self._record_on_error_enabled=DEFAULT_RECORD_ON_ERROR
        self.error_folder=None
        self.log_file_path=None
        self.webcam_roi=None
        self.ref_images={k:None for k in[REF_NORM,REF_SHUTDOWN,REF_FAIL]}
        self.serial_port_name=None
        self.serial_baud_rate=DEFAULT_BAUD_RATE
        self.serial_enabled=False # Luôn tắt khi reset

        # Đóng cổng serial nếu đang mở
        if self.serial_port and self.serial_port.is_open:
            try: self.serial_port.close()
            except: pass # Bỏ qua lỗi khi đóng ở đây
        self.serial_port=None

        # Reset config dictionary
        self.config={
            'ssim_threshold':self._current_ssim_threshold, 'error_cooldown':self._current_error_cooldown,
            'runtime_duration_minutes':self._current_runtime_minutes, 'record_on_error':self._record_on_error_enabled,
            'error_folder':None, 'ref_paths':{k:None for k in[REF_NORM,REF_SHUTDOWN,REF_FAIL]},
            'webcam_roi':None, 'serial_port':self.serial_port_name,
            'serial_baud':self.serial_baud_rate, 'serial_enabled':self.serial_enabled,
            }

        # Sau khi reset, cần cập nhật lại UI và lưu config mặc định mới
        # Cần đảm bảo UI đã được init trước khi gọi update_all_ui_elements
        if hasattr(self, 'ssimThresholdSpinBox'): # Kiểm tra xem init_ui đã chạy chưa
             self.update_all_ui_elements()
             self.save_config() # Lưu lại config mặc định
             self.log_activity("🔄 Đã hoàn tất reset về mặc định và cập nhật giao diện.")
        else:
             self.log_activity("🔄 Đã hoàn tất reset về mặc định (UI chưa sẵn sàng để cập nhật).")


    def update_all_ui_elements(self):
        self.log_activity("ℹ️ Đang cập nhật giao diện người dùng...")
        # Block signals để tránh kích hoạt slot khi set giá trị
        controls_to_block = [
            self.ssimThresholdSpinBox, self.cooldownSpinBox, self.runtimeSpinBox,
            self.comPortComboBox, self.baudRateComboBox
        ]
        for control in controls_to_block: control.blockSignals(True)

        # Cập nhật giá trị từ state/config
        self.ssimThresholdSpinBox.setValue(self._current_ssim_threshold)
        self.cooldownSpinBox.setValue(self._current_error_cooldown)
        self.runtimeSpinBox.setValue(self._current_runtime_minutes)
        self.baudRateComboBox.setCurrentText(str(self.serial_baud_rate))

        # Cập nhật COM port combo box (nên gọi refresh trước đó)
        # self._refresh_com_ports() # Đã gọi ở init, không gọi lại ở đây để tránh xóa log khởi động
        current_com_text = self.comPortComboBox.currentText()
        com_index = self.comPortComboBox.findText(self.serial_port_name if self.serial_port_name else "")
        if self.serial_port_name and com_index >= 0:
             self.comPortComboBox.setCurrentIndex(com_index)
        elif self.comPortComboBox.count() > 0 and "Không tìm thấy" not in self.comPortComboBox.itemText(0):
             # Nếu cổng đã lưu không có nhưng có cổng khác, chọn cổng đầu tiên
             first_available_port = self.comPortComboBox.itemText(0)
             self.comPortComboBox.setCurrentIndex(0)
             if self.serial_port_name != first_available_port:
                  # self.log_activity(f"⚠️ Cổng COM đã lưu '{self.serial_port_name}' không tìm thấy. Chọn cổng '{first_available_port}'.") # Log khi refresh rõ hơn
                  self.serial_port_name = first_available_port
                  self.config['serial_port'] = first_available_port
        # else: Nếu không có cổng nào, comPortComboBox đã bị disable bởi refresh

        # Unblock signals
        for control in controls_to_block: control.blockSignals(False)

        # Cập nhật style các nút
        self.update_button_styles()
        self.update_toggle_button_text()
        self.update_record_button_style()
        self.update_serial_button_style()

        # Kích hoạt/Vô hiệu hóa các control dựa trên trạng thái processing và serial
        self.disable_settings_while_processing(self.processing)

        # Log trạng thái ban đầu (chỉ log 1 lần khi update UI sau load/reset)
        # (Đã chuyển phần log ảnh/folder/ROI/Serial vào load_config và các hàm thay đổi tương ứng)
        self.log_activity("ℹ️ Giao diện người dùng đã được cập nhật.")


    @QtCore.pyqtSlot(str)
    def log_activity(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S"); full_message = f"{timestamp} - {message}"

        # Đảm bảo cập nhật QTextEdit từ đúng luồng
        if hasattr(self, 'log_text_edit'): # Kiểm tra widget tồn tại
            if self.log_text_edit.thread() != QtCore.QThread.currentThread():
                QtCore.QMetaObject.invokeMethod(self.log_text_edit, "append", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, full_message))
                QtCore.QMetaObject.invokeMethod(self.log_text_edit, "ensureCursorVisible", QtCore.Qt.QueuedConnection)
            else:
                self.log_text_edit.append(full_message)
                self.log_text_edit.ensureCursorVisible()

        # Ghi vào file log
        if self.log_file_path:
            try:
                log_dir = os.path.dirname(self.log_file_path)
                # Kiểm tra và tạo thư mục nếu chưa có
                if log_dir and not os.path.exists(log_dir):
                     try:
                         os.makedirs(log_dir, exist_ok=True)
                     except OSError as mkdir_e:
                         print(f"CRITICAL: Cannot create log directory '{log_dir}'. Error: {mkdir_e}. Disabling file logging.")
                         self.log_file_path=None # Vô hiệu hóa ghi file
                         return # Không thử ghi nữa

                # Ghi vào file (chỉ khi log_file_path vẫn hợp lệ)
                if self.log_file_path:
                    with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                        log_file.write(full_message + "\n")

            except Exception as e:
                print(f"CRITICAL: Error writing to log file '{self.log_file_path}'. Error: {e}. Disabling file logging.")
                self.log_file_path = None # Vô hiệu hóa ghi file
        # else: # Không có đường dẫn log (chưa chọn folder lỗi) -> chỉ hiển thị trên UI
             # pass


    @QtCore.pyqtSlot(str, str)
    def update_status_label(self, message, background_color="lightgray"):
        # Sử dụng lambda để tránh lỗi pickling khi dùng invokeMethod với hàm lồng nhau
        _update = lambda: (
            self.process_label.setText(f"Trạng thái: {message}"),
            self.process_label.setStyleSheet(f"border:1px solid black; padding:5px; background-color:{background_color}; color:black; font-weight:bold; border-radius:3px;")
        )
        if self.process_label.thread()!=QtCore.QThread.currentThread():
             # Dùng invokeMethod để gọi lambda _update trên luồng chính
             # Cần một slot trung gian để gọi lambda
             QtCore.QMetaObject.invokeMethod(self, "_call_lambda_slot", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(object, _update))
        else: _update()

    @QtCore.pyqtSlot(float)
    def update_ssim_display(self, score):
         _update = lambda: self.ssim_label.setText(f"SSIM: {score:.4f}" if score is not None and score>=0 else "SSIM: N/A")
         if self.ssim_label.thread()!=QtCore.QThread.currentThread():
             QtCore.QMetaObject.invokeMethod(self, "_call_lambda_slot", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(object, _update))
         else: _update()

    # Slot trung gian để gọi hàm lambda từ luồng khác
    @QtCore.pyqtSlot(object)
    def _call_lambda_slot(self, f):
        f()


    def _set_button_style(self, button, base_text, icon, state_text="", background_color="white", text_color="black"):
        full_text=f"{icon} {base_text}"+ (f" ({state_text})" if state_text else "")
        button.setText(full_text)
        style=f"""
            QPushButton {{
                background-color: {background_color};
                color: {text_color};
                padding: 6px;
                text-align: center;
                border: 1px solid #ccc;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: #e8f0fe; /* Màu xanh nhạt khi hover */
            }}
            QPushButton:pressed {{
                background-color: #d0e0f8; /* Màu xanh đậm hơn khi nhấn */
            }}
            QPushButton:disabled {{
                background-color: #f0f0f0; /* Màu xám khi bị vô hiệu hóa */
                color: #a0a0a0;
                border-color: #d0d0d0;
            }}
        """
        button.setStyleSheet(style)

    def update_button_styles(self):
        # Dict ánh xạ key (Norm, Shutdown, Fail) tới nút Load và Capture tương ứng
        button_load_map={
            REF_NORM: self.SettingButton_Norm,
            REF_SHUTDOWN: self.SettingButton_Shutdown,
            REF_FAIL: self.SettingButton_Fail
        }
        button_capture_map={
            REF_NORM: self.CaptureButton_Norm,
            REF_SHUTDOWN: self.CaptureButton_Shut,
            REF_FAIL: self.CaptureButton_Fail
        }
        # Icons và Text cho các nút
        icon_load="📂"; icon_capture="📸"
        text_load={"Norm":"Ảnh Norm","Shutdown":"Ảnh Shutdown","Fail":"Ảnh Fail"}
        text_capture={"Norm":"Chụp Norm","Shutdown":"Chụp Shutdown","Fail":"Chụp Fail"}

        # Cập nhật style cho từng cặp nút Load/Capture
        for key in self.ref_images.keys():
            load_button = button_load_map[key]
            capture_button = button_capture_map[key]
            has_image = isinstance(self.ref_images.get(key), np.ndarray) and self.ref_images[key].size > 0
            is_from_file = has_image and isinstance(self.config['ref_paths'].get(key), str) # Kiểm tra có đường dẫn file không

            if has_image:
                if is_from_file: # Ảnh từ file
                    self._set_button_style(load_button, text_load[key], icon_load, "File", "lightgreen") # Nút Load nổi bật
                    self._set_button_style(capture_button, text_capture[key], icon_capture) # Nút Capture bình thường
                else: # Ảnh từ webcam capture (path là None)
                    self._set_button_style(load_button, text_load[key], icon_load) # Nút Load bình thường
                    self._set_button_style(capture_button, text_capture[key], icon_capture, "Webcam", "lightblue") # Nút Capture nổi bật
            else: # Chưa có ảnh
                self._set_button_style(load_button, text_load[key], icon_load) # Nút Load bình thường
                self._set_button_style(capture_button, text_capture[key], icon_capture) # Nút Capture bình thường

        # Cập nhật nút ROI
        if self.webcam_roi: self._set_button_style(self.SettingButton_ROI_Webcam,"Chọn ROI","✂️","Đã chọn","lightblue")
        else: self._set_button_style(self.SettingButton_ROI_Webcam,"Chọn ROI","✂️")

        # Cập nhật nút Thư mục lỗi
        if self.error_folder: self._set_button_style(self.SaveButton,"Thư mục lỗi","📁","Đã chọn","lightblue")
        else: self._set_button_style(self.SaveButton,"Thư mục lỗi","📁")

    def update_toggle_button_text(self):
        if self.processing:
            self._set_button_style(self.ToggleProcessingButton,"Dừng Xử lý","⏹", background_color="orange")
        else:
            self._set_button_style(self.ToggleProcessingButton,"Bắt đầu","▶️", background_color="lightgreen")

    def update_record_button_style(self):
        if self._record_on_error_enabled:
            self._set_button_style(self.ToggleRecordOnErrorButton,"Quay video lỗi","🎥","Bật","lightcoral")
        else:
            self._set_button_style(self.ToggleRecordOnErrorButton,"Quay video lỗi","🎥","Tắt","lightgray")

    def update_serial_button_style(self):
        if self.serial_enabled:
            self._set_button_style(self.ToggleSerialPortButton,"Ngắt kết nối COM","🔌","Đang kết nối","lightcoral") # Đỏ khi đang bật
        else:
            # Kiểm tra có cổng COM khả dụng không
            has_ports = self.comPortComboBox.count() > 0 and "Không tìm thấy" not in self.comPortComboBox.itemText(0)
            if has_ports:
                 self._set_button_style(self.ToggleSerialPortButton,"Kết nối COM","🔌","Chưa kết nối","lightgreen") # Xanh nếu có thể kết nối
            else:
                 self._set_button_style(self.ToggleSerialPortButton,"Kết nối COM","🔌","Không có cổng","lightgray") # Xám nếu không có cổng


    def start_webcam(self):
        if self.cap is not None and self.cap.isOpened():
            self.log_activity("⚠️ Webcam đã được bật."); return
        try:
            # Thử các backend phổ biến
            backends=[cv2.CAP_DSHOW, cv2.CAP_MSMF, None] # Ưu tiên DSHOW, MSMF trên Windows
            self.cap=None
            opened_backend = "N/A"

            for bk in backends:
                api_pref = bk if bk is not None else cv2.CAP_ANY
                try:
                    tc=cv2.VideoCapture(0, api_pref)
                    if tc and tc.isOpened():
                        ret_test, _ = tc.read() # Thử đọc 1 frame
                        if ret_test:
                            self.cap=tc
                            opened_backend = self.cap.getBackendName()
                            break # Đã mở thành công, thoát vòng lặp
                        else:
                           tc.release()
                           self.log_activity(f"ℹ️ Webcam backend {api_pref} mở được nhưng không đọc được frame.")
                    elif tc: tc.release()
                except Exception as cam_err:
                    # Log lỗi chi tiết hơn khi thử từng backend
                    self.log_activity(f"ℹ️ Lỗi khi thử mở webcam với backend {api_pref}: {cam_err}")
                    if tc: tc.release()

            if self.cap is None:
                 raise IOError("Không thể mở webcam với bất kỳ backend nào hoặc không đọc được frame ban đầu.")

            # --- Đã mở webcam thành công ---
            fps_req=15.0 # FPS mong muốn
            self.cap.set(cv2.CAP_PROP_FPS, fps_req)
            actual_fps=self.cap.get(cv2.CAP_PROP_FPS)
            if actual_fps <= 0: actual_fps=15.0 # Mặc định nếu không lấy được FPS
            self.webcam_fps = actual_fps
            timer_interval = max(33, int(1000 / self.webcam_fps)) # ms, tối thiểu ~30fps

            w=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.log_activity(f"🚀 Webcam đã bật (Backend: {opened_backend}, Resolution: {w}x{h}, FPS: {self.webcam_fps:.1f}, Interval: {timer_interval}ms)")

            self.frame_timer.start(timer_interval) # Bắt đầu timer đọc frame
            self.ONCam.setEnabled(False); self.OFFCam.setEnabled(True) # Cập nhật nút bật/tắt cam
            self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.black)) # Nền đen khi có video

            # Kích hoạt các nút liên quan đến webcam (nếu không đang xử lý)
            is_processing = self.processing
            self.SettingButton_ROI_Webcam.setEnabled(not is_processing)
            self.CaptureButton_Norm.setEnabled(not is_processing)
            self.CaptureButton_Shut.setEnabled(not is_processing)
            self.CaptureButton_Fail.setEnabled(not is_processing)

        except Exception as e:
            emsg=f"❌ Lỗi nghiêm trọng khi bật webcam: {e}"
            self.log_activity(emsg)
            self.log_activity(traceback.format_exc()) # Log traceback để debug
            QMessageBox.critical(self,"Lỗi Webcam",f"Không thể khởi động webcam.\nChi tiết: {e}");
            if self.cap: self.cap.release(); self.cap=None;
            # Đảm bảo các nút ở trạng thái đúng khi có lỗi
            self.ONCam.setEnabled(True); self.OFFCam.setEnabled(False)
            self.SettingButton_ROI_Webcam.setEnabled(False)
            self.CaptureButton_Norm.setEnabled(False)
            self.CaptureButton_Shut.setEnabled(False)
            self.CaptureButton_Fail.setEnabled(False)
            self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray)) # Nền xám


    def update_frame(self):
        if self.cap is None or not self.cap.isOpened(): return # Thoát nếu cam chưa sẵn sàng

        ret,frame=self.cap.read() # Đọc frame
        if not ret or frame is None: # Kiểm tra đọc lỗi
            current_time = time.time()
            if not hasattr(self,'last_read_error_time') or current_time - getattr(self,'last_read_error_time', 0) > 5: # Log định kỳ
                self.log_activity("⚠️ Lỗi đọc frame từ webcam.")
                setattr(self,'last_read_error_time', current_time)
            return # Không có frame để xử lý

        # Đã có frame, xóa timer lỗi đọc nếu có
        if hasattr(self,'last_read_error_time'): delattr(self, 'last_read_error_time')

        try:
            display_frame=frame.copy() # Frame để hiển thị (có thể vẽ ROI lên)
            processing_frame=frame     # Frame để xử lý (nguyên gốc hoặc đã crop ROI)

            # --- Áp dụng ROI nếu có ---
            if self.webcam_roi:
                x,y,w,h=self.webcam_roi
                frame_h, frame_w = frame.shape[:2]
                x1, y1 = max(0, x), max(0, y) # Góc trên trái
                x2, y2 = min(frame_w, x + w), min(frame_h, y + h) # Góc dưới phải
                if x2 > x1 and y2 > y1: # ROI hợp lệ
                    processing_frame = frame[y1:y2, x1:x2] # Crop frame để xử lý
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Vẽ ROI lên frame hiển thị
                else: # ROI không hợp lệ (ví dụ ngoài khung hình)
                    current_time = time.time()
                    if not hasattr(self,'last_roi_invalid_time') or current_time - getattr(self,'last_roi_invalid_time', 0) > 10: # Log định kỳ
                        self.log_activity(f"⚠️ ROI {self.webcam_roi} không hợp lệ với kích thước frame {frame_w}x{frame_h}. Sử dụng frame đầy đủ.")
                        setattr(self,'last_roi_invalid_time', current_time)
                    processing_frame = frame # Sử dụng frame đầy đủ nếu ROI lỗi
            else: # Không có ROI, xóa timer lỗi nếu có
                 if hasattr(self,'last_roi_invalid_time'): delattr(self, 'last_roi_invalid_time')

            # --- Hiển thị frame ---
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB) # Chuyển BGR sang RGB cho QImage
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            view_size = self.graphicsView.viewport().size() # Kích thước của viewport
            pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(
                view_size - QtCore.QSize(2, 2), # Scale để vừa viewport (trừ viền)
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            # Cập nhật QGraphicsPixmapItem
            if self.pixmap_item is None: # Lần đầu hiển thị
                self.pixmap_item=QtWidgets.QGraphicsPixmapItem(pixmap)
                self.scene.addItem(self.pixmap_item)
                self.graphicsView.fitInView(self.pixmap_item, Qt.KeepAspectRatio) # Căn giữa
            else: # Cập nhật pixmap đã có
                self.pixmap_item.setPixmap(pixmap)
                # Có thể gọi fitInView lại nếu muốn tự động căn giữa khi resize cửa sổ
                # self.graphicsView.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

            # --- Gửi frame cho worker nếu đang xử lý ---
            if self.processing:
                frame_to_process = processing_frame.copy() # Tạo bản sao cho worker

                # --- Xử lý ghi video ---
                if self._record_on_error_enabled:
                    # Khởi tạo VideoWriter nếu chưa có và cần ghi
                    if self.video_writer is None:
                        if self.error_folder and os.path.isdir(self.error_folder): # Cần thư mục lỗi
                            try:
                                vid_h, vid_w = frame_to_process.shape[:2] # Kích thước video = kích thước frame xử lý
                                video_dir = os.path.join(self.error_folder, VIDEO_SUBFOLDER)
                                os.makedirs(video_dir, exist_ok=True) # Tạo thư mục con nếu chưa có
                                timestamp = time.strftime('%Y%m%d_%H%M%S')
                                video_filename = f"error_recording_{timestamp}.mp4"
                                self.current_video_path = os.path.join(video_dir, video_filename)
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec MP4
                                self.video_writer = cv2.VideoWriter(self.current_video_path, fourcc, self.webcam_fps, (vid_w, vid_h))

                                if self.video_writer.isOpened():
                                    self.log_activity(f"🔴 Bắt đầu ghi video lỗi: {video_filename}")
                                    self.error_occurred_during_recording = False # Reset cờ lỗi khi bắt đầu video mới
                                else: # Không mở được writer
                                    self.log_activity(f"❌ Không thể khởi tạo VideoWriter cho file: {video_filename}")
                                    self.video_writer=None; self.current_video_path=None
                            except Exception as e: # Lỗi nghiêm trọng khi tạo writer
                                self.log_activity(f"❌ Lỗi nghiêm trọng khi khởi tạo VideoWriter: {e}")
                                self.log_activity(traceback.format_exc())
                                self.video_writer=None; self.current_video_path=None
                        else: # Chưa có thư mục lỗi
                            current_time = time.time()
                            if not hasattr(self,'last_video_folder_error') or current_time - getattr(self,'last_video_folder_error', 0) > 30: # Log định kỳ
                                self.log_activity("⚠️ Chưa đặt thư mục lỗi hợp lệ để ghi video.")
                                setattr(self,'last_video_folder_error', current_time)

                    # Ghi frame vào video nếu VideoWriter đã sẵn sàng
                    if self.video_writer and self.video_writer.isOpened():
                        try:
                            # Không cần resize vì video tạo theo kích thước frame xử lý
                            self.video_writer.write(frame_to_process)
                        except Exception as e:
                            self.log_activity(f"❌ Lỗi khi ghi frame vào video: {e}")
                            # Cân nhắc dừng ghi nếu lỗi liên tục?

                # --- Đưa frame vào queue cho worker ---
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put(frame_to_process, block=False) # Non-blocking put
                        if hasattr(self,'last_queue_full_time'): delattr(self, 'last_queue_full_time') # Xóa log queue đầy nếu thành công
                        if hasattr(self,'last_queue_put_error_time'): delattr(self, 'last_queue_put_error_time')
                    except Exception as e: # Lỗi khi put (hiếm)
                         current_time = time.time()
                         if not hasattr(self,'last_queue_put_error_time') or current_time - getattr(self,'last_queue_put_error_time', 0) > 5: # Log định kỳ
                             self.log_activity(f"❌ Lỗi khi đưa frame vào queue: {e}")
                             setattr(self,'last_queue_put_error_time', current_time)
                else: # Queue đầy
                    current_time = time.time()
                    if not hasattr(self,'last_queue_full_time') or current_time - getattr(self,'last_queue_full_time', 0) > 5: # Log định kỳ
                        self.log_activity("⚠️ Queue xử lý frame bị đầy, frame hiện tại bị bỏ qua.")
                        setattr(self,'last_queue_full_time', current_time)

        except Exception as e: # Bắt lỗi chung trong update_frame
             current_time = time.time()
             if not hasattr(self,'last_update_frame_error_time') or current_time - getattr(self,'last_update_frame_error_time', 0) > 5: # Log định kỳ
                 self.log_activity(f"❌ Lỗi trong quá trình update_frame: {e}")
                 self.log_activity(traceback.format_exc())
                 setattr(self,'last_update_frame_error_time', current_time)


    def stop_webcam(self):
        if self.cap and self.cap.isOpened():
            self.frame_timer.stop() # Dừng timer đọc frame
            self.cap.release()      # Giải phóng webcam
            self.cap = None
            self.scene.clear()      # Xóa nội dung hiển thị
            self.pixmap_item = None
            self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray)) # Đặt lại nền xám
            self.log_activity("🚫 Webcam đã tắt.")

            # Cập nhật trạng thái nút
            self.ONCam.setEnabled(True); self.OFFCam.setEnabled(False)
            self.SettingButton_ROI_Webcam.setEnabled(False)
            self.CaptureButton_Norm.setEnabled(False)
            self.CaptureButton_Shut.setEnabled(False)
            self.CaptureButton_Fail.setEnabled(False)

            # Tự động dừng xử lý nếu đang chạy
            if self.processing:
                 self.log_activity("ℹ️ Tự động dừng xử lý do webcam tắt.")
                 self.toggle_processing() # Gọi hàm toggle để dừng đúng cách
        elif self.cap is None:
             self.log_activity("ℹ️ Webcam chưa được bật.")
        else: # Trường hợp cap không None nhưng isOpened() là False (lỗi)
             self.log_activity("ℹ️ Webcam đang ở trạng thái lỗi, đã dọn dẹp.")
             if self.frame_timer.isActive(): self.frame_timer.stop()
             self.cap = None
             if self.scene: self.scene.clear()
             self.pixmap_item = None
             if self.graphicsView: self.graphicsView.setBackgroundBrush(QtGui.QBrush(Qt.darkGray))
             # Cập nhật nút
             if hasattr(self, 'ONCam'): self.ONCam.setEnabled(True)
             if hasattr(self, 'OFFCam'): self.OFFCam.setEnabled(False)
             if hasattr(self, 'SettingButton_ROI_Webcam'): self.SettingButton_ROI_Webcam.setEnabled(False)
             if hasattr(self, 'CaptureButton_Norm'): self.CaptureButton_Norm.setEnabled(False)
             if hasattr(self, 'CaptureButton_Shut'): self.CaptureButton_Shut.setEnabled(False)
             if hasattr(self, 'CaptureButton_Fail'): self.CaptureButton_Fail.setEnabled(False)

             if self.processing:
                  self.log_activity("ℹ️ Tự động dừng xử lý do webcam lỗi.")
                  self.toggle_processing()

    def load_reference_image(self, img_type):
        if self.processing: QMessageBox.warning(self,"Đang Xử Lý","Không thể thay đổi ảnh tham chiếu khi đang xử lý."); return

        opts=QFileDialog.Options()
        # Thư mục gợi ý: thư mục của ảnh cũ -> thư mục lỗi -> home
        suggested_dir = ""
        old_path = self.config['ref_paths'].get(img_type)
        if old_path and os.path.exists(os.path.dirname(old_path)):
             suggested_dir = os.path.dirname(old_path)
        elif self.error_folder and os.path.exists(self.error_folder):
             suggested_dir = self.error_folder
        else:
             suggested_dir = os.path.expanduser("~")

        fp,_=QFileDialog.getOpenFileName(self,f"Chọn ảnh tham chiếu cho '{img_type}'",suggested_dir,"Images (*.png *.jpg *.jpeg *.bmp *.jfif *.webp);;All Files (*)",options=opts)

        if fp: # Nếu người dùng chọn file
            try:
                img_bytes = np.fromfile(fp, dtype=np.uint8) # Đọc an toàn
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR) # Decode ảnh
                if img is None: raise ValueError("Không thể giải mã file ảnh hoặc định dạng không được hỗ trợ.")

                # Cập nhật state và config
                self.ref_images[img_type] = img
                self.config['ref_paths'][img_type] = fp
                self.update_button_styles() # Cập nhật UI
                self.log_activity(f"✅ Đã tải ảnh '{img_type}' từ file: {os.path.basename(fp)}")
                self.save_config() # Lưu config mới

            except Exception as e:
                self.log_activity(f"❌ Lỗi khi tải ảnh '{img_type}' từ file '{fp}': {e}")
                QMessageBox.warning(self,"Lỗi Tải Ảnh",f"Không thể tải ảnh:\n{fp}\n\nLỗi: {e}")
                # Không thay đổi ảnh/config nếu tải lỗi

    def capture_reference_from_webcam(self,img_type):
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self,"Webcam Chưa Bật","Vui lòng bật webcam trước khi chụp ảnh."); return
        if self.processing:
            QMessageBox.warning(self,"Đang Xử Lý","Không thể chụp ảnh tham chiếu khi đang xử lý."); return

        # Tạm dừng timer để lấy frame ổn định (tùy chọn)
        was_running = self.frame_timer.isActive()
        if was_running: self.frame_timer.stop(); time.sleep(0.1)

        ret,frame=self.cap.read()

        if was_running and self.cap and self.cap.isOpened(): self.frame_timer.start() # Khởi động lại timer

        if not ret or frame is None:
            QMessageBox.warning(self,"Lỗi Đọc Frame","Không thể lấy frame từ webcam để chụp."); return

        try:
            # Quyết định lưu frame gốc hay frame đã crop ROI?
            # Hiện tại lưu frame gốc (chưa crop)
            frame_to_save = frame.copy()
            # Nếu muốn lưu frame đã crop ROI (nếu ROI tồn tại):
            # if self.webcam_roi:
            #     x,y,w,h=self.webcam_roi
            #     fh, fw = frame.shape[:2]
            #     x1,y1,x2,y2 = max(0,x),max(0,y),min(fw,x+w),min(fh,y+h)
            #     if x2>x1 and y2>y1: frame_to_save = frame[y1:y2,x1:x2].copy()

            self.ref_images[img_type] = frame_to_save
            self.config['ref_paths'][img_type] = None # Đánh dấu là ảnh từ webcam
            self.log_activity(f"📸 Đã chụp ảnh '{img_type}' từ webcam.")
            self.update_button_styles()
            self.save_config() # Lưu config

        except Exception as e:
            self.log_activity(f"❌ Lỗi khi lưu ảnh chụp '{img_type}': {e}")
            QMessageBox.critical(self,"Lỗi Chụp Ảnh",f"Đã xảy ra lỗi khi lưu ảnh chụp:\n{e}")

    def select_webcam_roi(self):
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self,"Webcam Chưa Bật","Vui lòng bật webcam trước khi chọn ROI."); return
        if self.processing:
            QMessageBox.warning(self,"Đang Xử Lý","Không thể chọn ROI khi đang xử lý."); return

        was_running = self.frame_timer.isActive()
        if was_running: self.frame_timer.stop(); time.sleep(0.1) # Dừng timer, chờ frame ổn định

        ret,frame=self.cap.read()

        # Không khởi động lại timer ngay, để cửa sổ ROI hiển thị frame tĩnh

        if not ret or frame is None:
            if was_running and self.cap and self.cap.isOpened(): self.frame_timer.start() # Bật lại timer nếu lỗi
            QMessageBox.warning(self,"Lỗi Đọc Frame","Không thể lấy frame từ webcam để chọn ROI."); return

        try:
            window_name="Chon ROI - Keo chuot, roi ENTER/SPACE (C=Huy, R=Reset)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.setWindowTitle(window_name, window_name)

            frame_for_roi = frame.copy()
            # Vẽ hướng dẫn
            cv2.putText(frame_for_roi,"Keo chuot de chon vung",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame_for_roi,"Nhan ENTER hoac SPACE de xac nhan",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame_for_roi,"Nhan C hoac ESC de huy",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame_for_roi,"Nhan R de chon lai",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,150,0),2)

            roi = cv2.selectROI(window_name, frame_for_roi, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window_name) # Đóng cửa sổ chọn ROI

            # Khởi động lại timer webcam sau khi đóng cửa sổ ROI
            if was_running and self.cap and self.cap.isOpened(): self.frame_timer.start()

            if roi == (0,0,0,0): # Người dùng hủy
                self.log_activity("ℹ️ Người dùng đã hủy chọn ROI.")
                return

            if roi[2] > 0 and roi[3] > 0: # ROI hợp lệ (w>0, h>0)
                self.webcam_roi = tuple(max(0, int(v)) for v in roi) # Đảm bảo số nguyên không âm
                self.config['webcam_roi'] = list(self.webcam_roi)
                self.log_activity(f"✅ Đã chọn ROI mới: {self.webcam_roi}")
                self.save_config() # Lưu ROI mới
            else:
                self.log_activity("⚠️ ROI được chọn không hợp lệ (chiều rộng hoặc chiều cao bằng 0).")

            self.update_button_styles() # Cập nhật UI nút ROI

        except Exception as e:
            self.log_activity(f"❌ Lỗi trong quá trình chọn ROI: {e}")
            self.log_activity(traceback.format_exc())
            QMessageBox.critical(self,"Lỗi Chọn ROI",f"Đã xảy ra lỗi:\n{e}")
            cv2.destroyAllWindows() # Đảm bảo đóng tất cả cửa sổ OpenCV nếu lỗi
            # Khởi động lại timer nếu cần
            if was_running and self.cap and self.cap.isOpened() and not self.frame_timer.isActive():
                self.frame_timer.start()


    def select_error_folder(self):
        if self.processing: QMessageBox.warning(self,"Đang Xử Lý","Không thể thay đổi thư mục lỗi khi đang xử lý."); return

        opts=QFileDialog.Options() | QFileDialog.ShowDirsOnly
        suggested_dir = self.error_folder or os.path.expanduser("~")
        folder=QFileDialog.getExistingDirectory(self,"Chọn thư mục lưu ảnh lỗi, video và log",suggested_dir,opts)

        if folder: # Nếu người dùng chọn thư mục
            # Kiểm tra quyền ghi
            if not os.access(folder,os.W_OK):
                QMessageBox.warning(self,"Không Có Quyền Ghi",f"Không thể ghi vào thư mục đã chọn:\n{folder}\n\nVui lòng chọn thư mục khác hoặc kiểm tra lại quyền truy cập.")
                return # Không thay đổi nếu không ghi được

            # Chỉ cập nhật nếu thư mục thực sự thay đổi
            if self.error_folder != folder:
                self.error_folder = folder
                self.config['error_folder'] = folder
                self.log_activity(f"📁 Đã chọn thư mục lưu lỗi: {self.error_folder}")
                # Cập nhật đường dẫn file log
                self.log_file_path = os.path.join(self.error_folder, LOG_FILE_NAME)
                self.log_activity(f"📄 File log sẽ được ghi tại: {self.log_file_path}")
                # Ghi dòng test vào log mới
                self.log_activity("📝 (Đây là dòng log thử nghiệm sau khi chọn thư mục mới)")
                self.save_config() # Lưu config
                self.update_button_styles() # Cập nhật UI nút
            # else: # Thư mục không đổi
                # self.log_activity(f"ℹ️ Thư mục lỗi không đổi: {self.error_folder}")


    @QtCore.pyqtSlot()
    def _refresh_com_ports(self):
        if self.serial_enabled: # Không cho refresh khi đang kết nối
            self.log_activity("ℹ️ Ngắt kết nối COM trước khi làm mới danh sách.")
            return

        self.comPortComboBox.blockSignals(True) # Chặn tín hiệu
        current_selection = self.comPortComboBox.currentText() # Lưu lựa chọn cũ
        self.comPortComboBox.clear() # Xóa item cũ

        port_names = []
        try:
            ports = serial.tools.list_ports.comports()
            port_names = sorted([port.device for port in ports])
            if port_names:
                 self.log_activity(f"🔄 Làm mới COM. Tìm thấy: {', '.join(port_names)}")
            else:
                 self.log_activity("🔄 Làm mới COM. Không tìm thấy cổng nào.")
        except Exception as e:
            self.log_activity(f"❌ Lỗi khi liệt kê các cổng COM: {e}")

        # Xử lý kết quả
        if not port_names:
             self.comPortComboBox.addItem("Không tìm thấy cổng COM")
             self.comPortComboBox.setEnabled(False)
             self.serial_port_name = None
             # Các nút khác sẽ được disable/enable trong update_serial_button_style
        else:
            self.comPortComboBox.addItems(port_names)
            self.comPortComboBox.setEnabled(True)
            # Cố gắng chọn lại cổng cũ hoặc cổng đầu tiên
            if current_selection in port_names:
                 self.comPortComboBox.setCurrentText(current_selection)
                 self.serial_port_name = current_selection
            else:
                 self.comPortComboBox.setCurrentIndex(0)
                 new_selection = self.comPortComboBox.currentText()
                 self.serial_port_name = new_selection
                 if current_selection and "Không tìm thấy" not in current_selection:
                     self.log_activity(f"⚠️ Cổng COM '{current_selection}' không còn. Đã tự động chọn '{new_selection}'.")


        self.config['serial_port'] = self.serial_port_name # Cập nhật config
        self.comPortComboBox.blockSignals(False) # Bỏ chặn tín hiệu

        # Cập nhật trạng thái các nút liên quan đến serial
        self.update_serial_button_style()
        self.baudRateComboBox.setEnabled(bool(port_names))


    @QtCore.pyqtSlot()
    def _toggle_serial_port(self):
        if self.processing:
            QMessageBox.warning(self, "Đang Xử Lý", "Không thể thay đổi trạng thái kết nối COM khi đang xử lý.")
            return

        if not self.serial_enabled:
            # --- Logic để BẬT kết nối ---
            port = self.comPortComboBox.currentText()
            baud = self.serial_baud_rate

            if not port or "Không tìm thấy" in port:
                QMessageBox.warning(self, "Chưa Chọn Cổng COM", "Vui lòng chọn một cổng COM hợp lệ từ danh sách.")
                return

            # Đóng cổng cũ nếu còn mở (phòng ngừa)
            if self.serial_port and self.serial_port.is_open:
                try: self.serial_port.close()
                except: pass

            # Cố gắng mở cổng mới
            try:
                self.log_activity(f"🔌 Đang cố gắng kết nối cổng {port} @ {baud} baud...")
                # Thêm write_timeout để tránh bị treo nếu thiết bị không nhận dữ liệu
                self.serial_port = serial.Serial(port, baud, timeout=0.1, write_timeout=1)
                # ---- Thành công ----
                self.serial_enabled = True
                self.serial_port_name = port # Cập nhật tên cổng thực tế đã mở
                self.config['serial_enabled'] = True
                self.config['serial_port'] = port
                self.config['serial_baud'] = baud
                self.log_activity(f"✅ Đã kết nối cổng COM: {port} @ {baud} baud")
                # Vô hiệu hóa cấu hình
                self.comPortComboBox.setEnabled(False)
                self.baudRateComboBox.setEnabled(False)
                self.refreshComButton.setEnabled(False)

            except serial.SerialException as e:
                # ---- Lỗi khi mở cổng ----
                self.log_activity(f"❌ Lỗi khi mở cổng COM '{port}': {e}")
                QMessageBox.critical(self, "Lỗi Kết Nối COM", f"Không thể mở cổng {port}.\n\nLỗi: {e}\n\nKiểm tra lại cổng, driver hoặc thiết bị có đang được sử dụng bởi chương trình khác không.")
                self.serial_port=None
                self.serial_enabled=False
                self.config['serial_enabled']=False
                # Cho phép cấu hình lại
                self.comPortComboBox.setEnabled(True)
                self.baudRateComboBox.setEnabled(True)
                self.refreshComButton.setEnabled(True)

            except Exception as e:
                # ---- Lỗi không xác định khác ----
                self.log_activity(f"❌ Lỗi không xác định khi mở cổng COM '{port}': {e}")
                self.log_activity(traceback.format_exc())
                QMessageBox.critical(self, "Lỗi Nghiêm Trọng", f"Đã xảy ra lỗi không mong muốn khi kết nối cổng COM.\n\nLỗi: {e}")
                self.serial_port=None
                self.serial_enabled=False
                self.config['serial_enabled']=False
                # Cho phép cấu hình lại
                self.comPortComboBox.setEnabled(True)
                self.baudRateComboBox.setEnabled(True)
                self.refreshComButton.setEnabled(True)

        else:
            # --- Logic để TẮT kết nối ---
            port_to_close = self.serial_port_name
            try:
                if self.serial_port and self.serial_port.is_open:
                    self.log_activity(f"🔌 Đang ngắt kết nối cổng COM: {port_to_close}...")
                    self.serial_port.close()
                    self.log_activity(f"🔌 Đã ngắt kết nối cổng COM: {port_to_close}")
            except serial.SerialException as e:
                 self.log_activity(f"⚠️ Lỗi khi đóng cổng COM '{port_to_close}': {e}")
            except Exception as e:
                 self.log_activity(f"⚠️ Lỗi không xác định khi đóng cổng COM '{port_to_close}': {e}")
            finally:
                 # Luôn thực hiện cleanup và cập nhật UI
                 self.serial_port = None
                 self.serial_enabled = False
                 self.config['serial_enabled'] = False
                 # Kích hoạt lại các control cấu hình COM
                 # self._refresh_com_ports() # Không cần refresh ngay, chỉ bật lại control
                 has_ports_after_refresh = self.comPortComboBox.count() > 0 and "Không tìm thấy" not in self.comPortComboBox.itemText(0)
                 self.comPortComboBox.setEnabled(has_ports_after_refresh)
                 self.baudRateComboBox.setEnabled(has_ports_after_refresh)
                 self.refreshComButton.setEnabled(True)

        # Cập nhật style nút và lưu trạng thái mới
        self.update_serial_button_style()
        self.save_config()


    @QtCore.pyqtSlot(str)
    def _send_serial_command(self, command):
        if self.serial_enabled and self.serial_port and self.serial_port.is_open:
            try:
                cmd_with_newline = command if command.endswith('\n') else command + '\n'
                byte_command = cmd_with_newline.encode('utf-8')
                bytes_written = self.serial_port.write(byte_command)
                # self.serial_port.flush() # Đảm bảo dữ liệu được gửi đi ngay lập tức (tùy chọn)
                if bytes_written == len(byte_command):
                     self.log_activity(f"➡️ Gửi COM [{self.serial_port_name}]: {command}")
                else: # Trường hợp ghi không đủ byte (ít xảy ra với write_timeout)
                     self.log_activity(f"⚠️ Gửi COM [{self.serial_port_name}] không hoàn chỉnh: {command} (Ghi {bytes_written}/{len(byte_command)} bytes)")

            except serial.SerialTimeoutException: # Lỗi timeout khi ghi
                self.log_activity(f"⚠️ Timeout khi ghi lệnh COM tới '{self.serial_port_name}'. Thiết bị có thể không phản hồi.")
                # Cân nhắc tự động tắt COM nếu timeout liên tục?
            except serial.SerialException as e: # Lỗi nghiêm trọng khác khi ghi
                self.log_activity(f"❌ Lỗi nghiêm trọng khi ghi lệnh COM tới '{self.serial_port_name}': {e}. Tự động ngắt kết nối.")
                QMessageBox.critical(self,"Lỗi Ghi COM",f"Không thể gửi dữ liệu tới cổng {self.serial_port_name}.\nKết nối sẽ bị đóng.\n\nLỗi: {e}")
                self._toggle_serial_port() # Tự động gọi hàm tắt kết nối
            except Exception as e: # Lỗi không xác định
                 self.log_activity(f"❌ Lỗi không xác định khi ghi lệnh COM: {e}. Tự động ngắt kết nối.")
                 self.log_activity(traceback.format_exc())
                 QMessageBox.critical(self,"Lỗi Ghi COM",f"Đã xảy ra lỗi không mong muốn khi gửi dữ liệu.\nKết nối sẽ bị đóng.\n\nLỗi: {e}")
                 self._toggle_serial_port() # Tự động gọi hàm tắt kết nối


    def toggle_processing(self):
        # --- Kiểm tra điều kiện trước khi BẬT ---
        if not self.processing:
            if not isinstance(self.ref_images.get(REF_NORM),np.ndarray) or self.ref_images[REF_NORM].size == 0:
                 QMessageBox.warning(self,"Thiếu Ảnh Tham Chiếu","Vui lòng tải hoặc chụp ảnh 'Norm' trước khi bắt đầu."); return
            if not self.error_folder or not os.path.isdir(self.error_folder) or not os.access(self.error_folder,os.W_OK):
                 QMessageBox.warning(self,"Thiếu Thư Mục Lỗi","Vui lòng chọn một thư mục hợp lệ (có quyền ghi) để lưu lỗi và log."); return
            if not self.cap or not self.cap.isOpened():
                 QMessageBox.warning(self,"Webcam Chưa Bật","Vui lòng bật webcam trước khi bắt đầu xử lý."); return

        # --- Chuyển đổi trạng thái processing ---
        self.processing = not self.processing

        if self.processing:
            # --- Logic khi BẬT xử lý ---
            self.log_activity("▶️ Đang chuẩn bị bắt đầu xử lý...")
            # Cập nhật config từ UI
            self.ssimThresholdSpinBox.blockSignals(True); self.cooldownSpinBox.blockSignals(True); self.runtimeSpinBox.blockSignals(True);
            self._update_threshold_config(self.ssimThresholdSpinBox.value())
            self._update_cooldown_config(self.cooldownSpinBox.value())
            self._update_runtime_config(self.runtimeSpinBox.value())
            self.ssimThresholdSpinBox.blockSignals(False); self.cooldownSpinBox.blockSignals(False); self.runtimeSpinBox.blockSignals(False);
            self.save_config() # Lưu config trước khi bắt đầu

            # Dừng worker cũ nếu đang chạy
            if self.processing_worker.isRunning():
                self.log_activity("⚠️ Worker cũ đang chạy, đang cố gắng dừng...")
                self.processing_worker.stop(); self.processing_worker.wait(1500)

            # Dọn sạch queue - SỬA LỖI SYNTAX
            cleared_count = 0
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                    cleared_count += 1
                except Empty:
                    break # Ngắt vòng lặp khi queue rỗng

            if cleared_count > 0: self.log_activity(f"ℹ️ Đã dọn {cleared_count} frame cũ từ queue.")

            # Khởi động worker mới
            self.processing_worker.last_error_time=0 # Reset cooldown
            self.processing_worker.start()

            self.update_status_label("🔄 Đang xử lý...","lightgreen")
            self.log_activity("▶️ Bắt đầu quá trình xử lý ảnh.")

            # Bắt đầu hẹn giờ runtime nếu cần
            if self._current_runtime_minutes > 0:
                duration_ms = self._current_runtime_minutes * 60 * 1000
                self.runtime_timer.start(duration_ms)
                self.log_activity(f"⏱️ Quá trình xử lý sẽ tự động dừng sau {self._current_runtime_minutes} phút.")
            else:
                 if self.runtime_timer.isActive(): self.runtime_timer.stop()

            # Reset trạng thái ghi video
            self.video_writer,self.current_video_path,self.error_occurred_during_recording = None,None,False

            # Vô hiệu hóa các nút cài đặt
            self.disable_settings_while_processing(True)
        else:
            # --- Logic khi TẮT xử lý ---
            self.log_activity("⏹ Đang dừng quá trình xử lý...")
            # Dừng worker thread
            if self.processing_worker.isRunning():
                self.processing_worker.stop()
                # Chờ worker dừng một chút để xử lý nốt frame cuối cùng? (Tùy chọn)
                # self.processing_worker.wait(500)

            # Dừng runtime timer
            if self.runtime_timer.isActive():
                self.runtime_timer.stop(); self.log_activity("⏱️ Đã hủy hẹn giờ dừng tự động.")

            # Xử lý dừng ghi video
            if self.video_writer is not None:
                video_path_to_check = self.current_video_path
                try:
                    self.video_writer.release(); self.log_activity("⚪️ Đã dừng ghi video.")
                    if video_path_to_check and os.path.exists(video_path_to_check):
                        if not self.error_occurred_during_recording:
                             try: os.remove(video_path_to_check); self.log_activity(f"🗑️ Đã xóa file video không có lỗi: {os.path.basename(video_path_to_check)}")
                             except Exception as e: self.log_activity(f"⚠️ Lỗi khi xóa file video '{os.path.basename(video_path_to_check)}': {e}")
                        else: self.log_activity(f"💾 Đã lưu file video có lỗi: {os.path.basename(video_path_to_check)}")
                except Exception as e: self.log_activity(f"❌ Lỗi khi dừng VideoWriter: {e}")
                finally: self.video_writer, self.current_video_path, self.error_occurred_during_recording = None, None, False

            # Cập nhật UI
            self.update_status_label("⏹ Đã dừng xử lý","orange")
            self.log_activity("⏹ Quá trình xử lý ảnh đã dừng.")
            self.ssim_label.setText("SSIM: N/A") # Reset SSIM
            # Kích hoạt lại các nút cài đặt
            self.disable_settings_while_processing(False)

        # Cập nhật text/style nút Start/Stop
        self.update_toggle_button_text()


    @QtCore.pyqtSlot()
    def _mark_error_occurred(self):
        # Chỉ log lần đầu tiên có lỗi trong phiên ghi video hiện tại
        if self._record_on_error_enabled and not self.error_occurred_during_recording:
            self.log_activity("❗ Phát hiện lỗi đầu tiên trong phiên ghi. Video sẽ được lưu khi dừng.")
        self.error_occurred_during_recording=True

    def disable_settings_while_processing(self, disable):
        # Nút tải/chụp ảnh tham chiếu
        self.SettingButton_Norm.setEnabled(not disable)
        self.SettingButton_Shutdown.setEnabled(not disable)
        self.SettingButton_Fail.setEnabled(not disable)

        webcam_is_on = self.cap is not None and self.cap.isOpened()
        self.CaptureButton_Norm.setEnabled(webcam_is_on and not disable)
        self.CaptureButton_Shut.setEnabled(webcam_is_on and not disable)
        self.CaptureButton_Fail.setEnabled(webcam_is_on and not disable)
        self.SettingButton_ROI_Webcam.setEnabled(webcam_is_on and not disable)

        # Nút cấu hình chung
        self.SaveButton.setEnabled(not disable)
        self.ssimThresholdSpinBox.setEnabled(not disable)
        self.cooldownSpinBox.setEnabled(not disable)
        self.runtimeSpinBox.setEnabled(not disable)
        self.ToggleRecordOnErrorButton.setEnabled(not disable)

        # Nút cấu hình Serial
        can_configure_serial = not disable and not self.serial_enabled
        has_ports = self.comPortComboBox.count() > 0 and "Không tìm thấy" not in self.comPortComboBox.itemText(0)
        self.comPortComboBox.setEnabled(can_configure_serial and has_ports)
        self.baudRateComboBox.setEnabled(can_configure_serial and has_ports)
        # Nút refresh luôn bật trừ khi đang kết nối serial
        self.refreshComButton.setEnabled(not self.serial_enabled)

        # Nút bật/tắt Serial (chỉ enable khi không xử lý *và* có cổng)
        self.ToggleSerialPortButton.setEnabled(not disable and has_ports)

        # Nút bật/tắt Webcam (chỉ enable khi không xử lý)
        self.ONCam.setEnabled(not disable and not webcam_is_on)
        self.OFFCam.setEnabled(not disable and webcam_is_on)


    @QtCore.pyqtSlot()
    def _runtime_timer_timeout(self):
        self.log_activity(f"⏱️ Đã hết thời gian chạy tối đa ({self._current_runtime_minutes} phút).")
        QMessageBox.information(self,"Hết Giờ",f"Quá trình xử lý đã chạy đủ {self._current_runtime_minutes} phút và sẽ tự động thoát.")
        # Tự động dừng xử lý trước khi đóng
        if self.processing:
             self.toggle_processing() # Gọi hàm dừng chuẩn
        # Sau đó đóng ứng dụng
        self.close_application() # Gọi hàm đóng chuẩn


    def compare_images(self, img1, img2, threshold):
        # --- Kiểm tra đầu vào ---
        if img1 is None or not isinstance(img1, np.ndarray) or img1.size==0:
             # self.log_activity("Lỗi compare: img1 không hợp lệ.", level="error") # Nên dùng signal log
             return False, None
        if img2 is None or not isinstance(img2, np.ndarray) or img2.size==0:
             # self.log_activity("Lỗi compare: img2 không hợp lệ.", level="warning") # Ảnh ref có thể chưa load
             return False, None

        try:
            # --- Chuyển sang grayscale ---
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape)>2 else img1
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape)>2 else img2

            # --- Resize nếu kích thước khác nhau ---
            if img1_gray.shape != img2_gray.shape:
                h1, w1 = img1_gray.shape
                h2, w2 = img2_gray.shape
                interpolation = cv2.INTER_AREA if (w2 > w1 or h2 > h1) else cv2.INTER_LINEAR
                try:
                    img2_resized = cv2.resize(img2_gray, (w1, h1), interpolation=interpolation)
                    if img2_resized is None or img2_resized.shape != (h1, w1): raise ValueError("Resize failed")
                    img2_gray = img2_resized
                    # Log định kỳ nếu phải resize
                    ct = time.time();
                    if not hasattr(self,'lrt') or ct-getattr(self,'lrt',0)>15:
                        # Dùng signal để log từ hàm này (an toàn hơn)
                        if hasattr(self, 'log_signal') and isinstance(self.log_signal, pyqtSignal):
                            self.log_signal.emit(f"ℹ️ Resize ảnh ref ({w2}x{h2})->({w1}x{h1}) để so sánh SSIM.")
                        setattr(self,'lrt',ct)
                except Exception as resize_err:
                    ct = time.time();
                    if not hasattr(self,'lre') or ct-getattr(self,'lre',0)>10:
                        if hasattr(self, 'log_signal') and isinstance(self.log_signal, pyqtSignal):
                            self.log_signal.emit(f"❌ Lỗi resize ref {w2}x{h2}->{w1}x{h1}: {resize_err}")
                        setattr(self,'lre',ct)
                    return False,None # Không thể so sánh nếu resize lỗi
            else: # Kích thước khớp, xóa log resize nếu có
                 if hasattr(self,'lrt'): delattr(self,'lrt')
                 if hasattr(self,'lre'): delattr(self,'lre')

            # --- Tính SSIM ---
            h, w = img1_gray.shape
            win_size = min(min(h, w), 7); # Giới hạn max 7x7
            if win_size % 2 == 0: win_size -= 1; win_size = max(3, win_size) # Đảm bảo lẻ >= 3

            if h < win_size or w < win_size: # Kiểm tra ảnh có đủ lớn không
                ct = time.time();
                if not hasattr(self,'lsie') or ct-getattr(self,'lsie',0)>10:
                    if hasattr(self, 'log_signal') and isinstance(self.log_signal, pyqtSignal):
                        self.log_signal.emit(f"⚠️ Ảnh/ROI ({w}x{h}) quá nhỏ cho SSIM (win={win_size}).")
                    setattr(self,'lsie',ct)
                return False,None
            else: # Kích thước OK, xóa log lỗi nếu có
                 if hasattr(self,'lsie'): delattr(self,'lsie')

            # Gọi hàm SSIM đã tối ưu (tự viết hoặc từ thư viện)
            # Sử dụng hàm ssim_opencv đã có ở trên
            score = ssim_opencv(img1_gray.astype(np.float64), img2_gray.astype(np.float64), win_size=win_size, data_range=255.0)

            if score is None or not np.isfinite(score): # Kiểm tra kết quả SSIM
                 ct = time.time();
                 if not hasattr(self,'lssime') or ct-getattr(self,'lssime',0)>10:
                    if hasattr(self, 'log_signal') and isinstance(self.log_signal, pyqtSignal):
                        self.log_signal.emit("⚠️ Tính toán SSIM trả về giá trị không hợp lệ (None, NaN, Inf).")
                    setattr(self,'lssime',ct)
                 return False, None
            else: # Có score hợp lệ, xóa log lỗi nếu có
                 if hasattr(self,'lssime'): delattr(self,'lssime')

            return score >= threshold, score # Trả về kết quả so sánh và điểm số

        except cv2.error as cv_err:
            ct=time.time();
            if not hasattr(self,'lcve') or ct-getattr(self,'lcve',0)>5:
                if hasattr(self, 'log_signal') and isinstance(self.log_signal, pyqtSignal):
                    self.log_signal.emit(f"❌ Lỗi OpenCV compare: {cv_err.msg}")
                setattr(self,'lcve',ct);
            return False,None
        except Exception as e:
            ct=time.time();
            if not hasattr(self,'lce') or ct-getattr(self,'lce',0)>5:
                if hasattr(self, 'log_signal') and isinstance(self.log_signal, pyqtSignal):
                    self.log_signal.emit(f"❌ Lỗi compare: {e}")
                    # self.log_signal.emit(traceback.format_exc()) # Avoid long tracebacks in worker log
                setattr(self,'lce',ct);
            return False,None


    @QtCore.pyqtSlot(np.ndarray, str)
    def save_error_image_from_thread(self, frame_copy, file_path):
        # Hàm này được gọi bởi tín hiệu từ worker, chạy trên Main thread
        try:
            # Tạo thư mục nếu chưa có (an toàn hơn khi làm ở Main thread)
            save_dir = os.path.dirname(file_path)
            if not os.path.exists(save_dir):
                 os.makedirs(save_dir, exist_ok=True)

            # Encode và ghi file (có thể vẫn mất chút thời gian)
            success, img_encoded = cv2.imencode('.png', frame_copy, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not success or img_encoded is None: raise ValueError("imencode failed.")

            with open(file_path, "wb") as f: f.write(img_encoded.tobytes())
            self.log_activity(f"💾 Đã lưu ảnh lỗi: {os.path.basename(file_path)}")

        except Exception as e:
            self.log_activity(f"❌ Lỗi khi lưu ảnh lỗi '{os.path.basename(file_path)}' trên luồng chính: {e}")
            self.log_activity(traceback.format_exc())


    def close_application(self):
        self.log_activity("🚪 Đang yêu cầu đóng ứng dụng...")
        self.close() # Kích hoạt closeEvent


    def closeEvent(self, event):
        self.log_activity("🚪 Bắt đầu quá trình dọn dẹp trước khi đóng...")

        # Dừng các timer
        if self.runtime_timer.isActive(): self.runtime_timer.stop()
        if self.frame_timer.isActive(): self.frame_timer.stop()

        # Dừng luồng xử lý
        worker_stopped = True
        if self.processing or self.processing_worker.isRunning():
            if self.processing_worker.isRunning():
                self.log_activity("⚙️ Đang dừng luồng xử lý...")
                self.processing_worker.stop()
                if not self.processing_worker.wait(2000): # Chờ tối đa 2 giây
                    self.log_activity("⚠️ Luồng xử lý không dừng kịp thời!")
                    worker_stopped = False
                else:
                    self.log_activity("✅ Luồng xử lý đã dừng.")
            self.processing = False

        # Giải phóng webcam
        if self.cap and self.cap.isOpened():
            self.cap.release(); self.cap = None; self.log_activity("🚫 Webcam đã được giải phóng.")

        # Giải phóng video writer và xử lý file cuối
        if self.video_writer is not None:
            vp=self.current_video_path;
            try:
                self.video_writer.release(); self.log_activity("⚪️ Video writer đã được giải phóng.")
                if vp and os.path.exists(vp):
                    if not self.error_occurred_during_recording:
                        try: os.remove(vp); self.log_activity(f"🗑️ Đã xóa video cuối (không lỗi): {os.path.basename(vp)}")
                        except Exception as e: self.log_activity(f"⚠️ Lỗi xóa video '{os.path.basename(vp)}': {e}")
                    else: self.log_activity(f"💾 Đã lưu video cuối (có lỗi): {os.path.basename(vp)}")
            except Exception as e: self.log_activity(f"❌ Lỗi giải phóng VideoWriter: {e}")
            finally: self.video_writer, self.current_video_path = None, None

        # Đóng cổng Serial
        if self.serial_port and self.serial_port.is_open:
            port_name = self.serial_port_name
            try:
                self.serial_port.close(); self.log_activity(f"🔌 Đã đóng cổng COM: {port_name}")
            except Exception as e: self.log_activity(f"⚠️ Lỗi đóng COM {port_name}: {e}")
            finally: self.serial_port = None; self.serial_enabled = False

        # Dọn queue lần cuối - SỬA LỖI SYNTAX
        qs=self.frame_queue.qsize();
        if qs>0: self.log_activity(f"ℹ️ Đang dọn {qs} frame còn lại trong queue...")
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break # Ngắt vòng lặp khi queue rỗng

        # Lưu cấu hình lần cuối
        self.log_activity("💾 Đang lưu cấu hình cuối cùng...")
        self.save_config()

        self.log_activity("🚪 Dọn dẹp hoàn tất. Ứng dụng sẽ đóng.")

        # Ghi dòng cuối vào file log
        if self.log_file_path:
             try:
                 ld=os.path.dirname(self.log_file_path);
                 if ld and not os.path.exists(ld): os.makedirs(ld,exist_ok=True)
                 with open(self.log_file_path,"a",encoding="utf-8") as lf: lf.write(f"---\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Ứng dụng đã đóng\n---\n")
             except Exception as e: print(f"Lỗi ghi log cuối: {e}") # In ra console nếu không ghi được

        event.accept() # Chấp nhận sự kiện đóng

# --- Main Execution ---
if __name__ == "__main__":
    # Bật cài đặt HiDPI nếu có
    if hasattr(QtCore.Qt,'AA_EnableHighDpiScaling'): QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling,True)
    if hasattr(QtCore.Qt,'AA_UseHighDpiPixmaps'): QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps,True)

    app=QtWidgets.QApplication(sys.argv)
    window=ImageCheckerApp()
    window.show()
    sys.exit(app.exec_())
