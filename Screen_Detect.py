from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QLabel, QGraphicsView, QGraphicsScene
from PyQt5.QtWidgets import QMessageBox
import cv2
import numpy as np
import sys
import os
import time
import threading
from collections import deque

from queue import Queue, Empty
from skimage.metrics import structural_similarity as ssim

class ImageCheckerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.ref_images = {"Norm": None, "Shutdown": None, "Fail": None}
        self.webcam_roi = None
        self.show_roi = False
        self.processing = False
        self.last_error_time = 0
        self.error_folder = None

        self.frame_queue = Queue(maxsize=5)
        self.processing_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.process_images, daemon=True)
        self.processing_thread.start()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Hệ thống kiểm tra hình ảnh")
        self.setGeometry(100, 100, 1246, 682)

        self.scene = QGraphicsScene(self)
        self.graphicsView = QGraphicsView(self.scene, self)
        self.graphicsView.setGeometry(0, 10, 640, 360)

        self.logDialog = QLabel("Các hoạt động:", self)
        self.logDialog.setGeometry(660, 10, 560, 250)
        self.logDialog.setStyleSheet("border: 1px solid black; padding: 5px; background-color: white;")

        self.process_label = QLabel("", self)
        self.process_label.setGeometry(660, 300, 560, 40)
        self.process_label.setStyleSheet("border: 1px solid black; padding: 5px; background-color:light blue; color: white;")

        self.init_buttons()

    def init_buttons(self):
        self.ONCam = self.create_button("MỞ WEBCAM", (20, 390), self.start_webcam)
        self.OFFCam = self.create_button("TẮT WEBCAM", (270, 390), self.stop_webcam)
        self.SettingButton_Norm = self.create_button("Chọn ảnh Norm", (20, 430),
                                                     lambda: self.load_reference_image("Norm"))
        self.SettingButton_Shut = self.create_button("Chọn ảnh Shutdown", (270, 430),
                                                     lambda: self.load_reference_image("Shutdown"))
        self.SettingButton_Fail = self.create_button("Chọn ảnh Fail", (520, 430),
                                                     lambda: self.load_reference_image("Fail"))
        self.SettingButton_ROI_Webcam = self.create_button("Chọn ROI Webcam", (20, 470), self.select_webcam_roi)
        self.ToggleROI = self.create_button("Bật/Tắt Xử lý ảnh", (520, 470), self.toggle_processing)
        self.SaveButton = self.create_button("Chọn thư mục lưu ảnh lỗi", (520, 510), self.select_error_folder)

        self.ExitButton = self.create_button("Thoát Ứng Dụng", (20, 510), self.close_application)

        self.update_button_colors()

    def create_button(self, text, position, callback):
        button = QtWidgets.QPushButton(text, self)
        button.setGeometry(*position, 201, 31)
        button.clicked.connect(callback)
        button.setStyleSheet("background-color: white;")
        return button

    def log_activity(self, message):
        QtCore.QMetaObject.invokeMethod(self.logDialog, "setText", QtCore.Qt.QueuedConnection,
                                        QtCore.Q_ARG(str, self.logDialog.text() + "\n" + message))
        QtCore.QMetaObject.invokeMethod(self.process_label, "setText", QtCore.Qt.QueuedConnection,
                                        QtCore.Q_ARG(str, message))

    def update_button_colors(self):
        for btn, key in zip([self.SettingButton_Norm, self.SettingButton_Shut, self.SettingButton_Fail],
                            ["Norm", "Shutdown", "Fail"]):
            btn.setStyleSheet("background-color: green; color: white;" if self.ref_images[
                                                                              key] is not None else "background-color: white;")

    def start_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FPS, 10)  # Giới hạn FPS
            self.timer.start(100)  # Giảm tải CPU (100ms = 10 FPS)
            self.log_activity("Webcam đã bật")

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:  # Kiểm tra nếu khung hình bị lỗi
            return
        if ret:
            if self.webcam_roi:
                x, y, w, h = self.webcam_roi
                frame = frame[y:y + h, x:x + w]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0],
                                  QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_img).scaled(640, 360, QtCore.Qt.KeepAspectRatio)

            if not self.scene.items():  # Chỉ cập nhật nếu chưa có ảnh
                self.scene.addPixmap(pixmap)
            else:
                self.scene.items()[0].setPixmap(pixmap)

            if self.processing:
                self.frame_queue.put(frame)

    def stop_webcam(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.scene.clear()
            self.log_activity("Webcam đã tắt")
        self.export_log()

    def log_activity(self, message):
        timestamp = time.strftime("%H:%M:%S %d-%m-%Y")
        full_message = f"{timestamp} - {message}"

        QtCore.QMetaObject.invokeMethod(self.logDialog, "setText", QtCore.Qt.QueuedConnection,
                                        QtCore.Q_ARG(str, self.logDialog.text() + "\n" + full_message))

        QtCore.QMetaObject.invokeMethod(self.process_label, "setText", QtCore.Qt.QueuedConnection,
                                        QtCore.Q_ARG(str, message))

        # Lưu log ngay lập tức
        if self.error_folder:
            log_path = os.path.join(self.error_folder, "log_activity.txt")
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(full_message + "\n")

    def load_reference_image(self, img_type):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.jfif *.png *.jpg *.jpeg)")
        if file_path:
            self.ref_images[img_type] = cv2.imread(file_path)
            self.update_button_colors()
            self.log_activity(f"Đã chọn ảnh {img_type}")

    def select_webcam_roi(self):
        if self.cap is None:
            self.log_activity("Chưa bật webcam!")
            return
        ret, frame = self.cap.read()
        if ret:
            self.webcam_roi = cv2.selectROI("Chọn ROI", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()
            self.log_activity("Đã chọn ROI cho webcam")

    def select_error_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu ảnh lỗi")
        if folder:
            self.error_folder = folder
            self.log_activity(f"Đã chọn thư mục lưu ảnh: {folder}")

    def toggle_processing(self):
        if not self.error_folder:
            self.log_activity("Chưa chọn thư mục lưu ảnh!")
            return

        self.processing = not self.processing  # Đảo trạng thái bật/tắt

        if self.processing:
            self.processing_thread = threading.Thread(target=self.process_images, daemon=True)
            self.processing_thread.start()
            self.process_label.setStyleSheet("background-color: green; color: white;")
            self.log_activity("🔄 Xử lý ảnh đã bật")
        else:
            self.process_label.setStyleSheet("background-color: red; color: white;")
            self.log_activity("⏹ Xử lý ảnh đã tắt")

    def process_images(self):
        while self.processing:  # Dừng nếu processing = False
            try:
                frame = self.frame_queue.get(timeout=0.5)  # Lấy ảnh từ queue
            except Empty:  # Đổi từ Queue.Empty -> Empty
                continue  # Nếu không có ảnh, tiếp tục vòng lặp

            if self.compare_images(frame, self.ref_images["Norm"]):
                continue

            if self.error_folder and time.time() - self.last_error_time > 30:
                file_path = os.path.join(self.error_folder, f"error_{time.strftime('%H-%M_%d-%m-%Y')}.png")
                cv2.imwrite(file_path, frame)
                self.last_error_time = time.time()
                self.log_activity(f"📸 Lưu ảnh lỗi: {file_path}")

    def compare_images(self, img1, img2):
        if img1 is None or img2 is None:
            return False  # Trả về False nếu thiếu ảnh

        try:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Resize ảnh nếu kích thước khác nhau
            if img1_gray.shape != img2_gray.shape:
                img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

            score, _ = ssim(img1_gray, img2_gray, full=True)
            return score > 0.9
        except Exception as e:
            print(f"❌ Lỗi khi so sánh ảnh: {e}")
            return False

    def save_log_file(self):
        if not self.error_folder:
            print("⚠️ Chưa chọn thư mục lưu ảnh lỗi! Không thể lưu log.")
            return
        log_path = os.path.join(self.error_folder, "log_activity.txt")
        try:
            with open(log_path, "a", encoding="utf-8") as log_file:
                timestamp = time.strftime("%H:%M:%S %d-%m-%Y")
                log_file.write(f"{timestamp} - {self.process_label.text()}\n")
            print(f"✔ Log đã lưu tại: {log_path}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu log: {e}")
    def close_application(self):
        self.save_log_file()
        time.sleep(1)  # Chờ 1 giây trước khi thoát để đảm bảo log được ghi
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageCheckerApp()
    window.show()
    sys.exit(app.exec_())