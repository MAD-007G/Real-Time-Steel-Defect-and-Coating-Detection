import cv2
import numpy as np
import sys
import time
import threading
import queue
import os
import serial
from datetime import datetime
from skimage.feature import local_binary_pattern
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
from ids_peak import ids_peak_ipl_extension

# ----------------------------
# Configuration Constants
# ----------------------------
SERIAL_PORT = "COM16"
BAUD_RATE = 9600
SAVE_DIR = "detected_images"
XRAY_MIN_DEFECT_AREA = 30
XRAY_DEFECT_THRESHOLD = 50
LINE_DETECTION_THRESHOLD = 40
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_PIXEL_FORMAT = ids_peak_ipl.PixelFormatName_BGRa8
FONT = cv2.FONT_HERSHEY_SIMPLEX

# LBP Parameters for Coating detection
RADIUS = 1
N_POINTS = 8 * RADIUS
LBP_MEAN_THRESHOLD = 170  # Tune this after testing

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Serial Communication Handler
# ----------------------------
class SerialHandler:
    def __init__(self, port, baud_rate):
        self.ser = None
        self.last_sent_result = None
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=1)
            time.sleep(2)
            print("✅ Serial port initialized")
        except serial.SerialException:
            print("⚠️ Could not open serial port. Running without relay control.")

    def send_relay_command(self, result):
        if self.ser is None or result == self.last_sent_result:
            return
        try:
            if result == "OK":
                self.ser.write(b"1 1\n")
                self.ser.write(b"2 0\n")
                print("Sent: Relay 1 ON, Relay 2 OFF (OK)")
            elif result == "NG":
                self.ser.write(b"1 0\n")
                self.ser.write(b"2 1\n")
                print("Sent: Relay 1 OFF, Relay 2 ON (NG)")
            self.last_sent_result = result
        except serial.SerialException:
            print("⚠️ Error sending serial command")

    def close(self):
        if self.ser:
            self.ser.close()
            print("Serial port closed")

# ----------------------------
# X-Ray Image Processing
# ----------------------------
class XRayProcessor:
    @staticmethod
    def create_xray_vision(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        xray_image = np.zeros_like(image)
        xray_image[:, :, 0] = enhanced
        xray_image[:, :, 1] = enhanced // 2
        xray_image[:, :, 2] = enhanced // 3
        return xray_image, enhanced

    @staticmethod
    def detect_defects(gray_img, xray_img, contours):
        defect_found = False
        defect_img = xray_img.copy()
        if not contours:
            return defect_img, defect_found
        mask = np.zeros_like(gray_img)
        cv2.drawContours(mask, contours, -1, 255, -1)
        _, defect_binary = cv2.threshold(gray_img, XRAY_DEFECT_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        defect_binary = cv2.bitwise_and(defect_binary, defect_binary, mask=mask)
        defect_contours, _ = cv2.findContours(defect_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in defect_contours:
            area = cv2.contourArea(cnt)
            if area > XRAY_MIN_DEFECT_AREA:
                defect_found = True
                cv2.drawContours(defect_img, [cnt], -1, (0, 0, 255), 2)
        return defect_img, defect_found

# ----------------------------
# LBP / Coating Detection
# ----------------------------
def process_frame(gray):
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, method="uniform")
    lbp_norm = np.uint8((lbp / lbp.max()) * 255 if lbp.max() > 0 else lbp)
    combined = cv2.bitwise_and(thresh, lbp_norm)
    return thresh, lbp_norm, combined

def analyze_coating(lbp_img):
    lbp_mean = np.mean(lbp_img)
    label = "COATED" if lbp_mean < LBP_MEAN_THRESHOLD else "NON-COATED"
    return label, lbp_mean

# ----------------------------
# Steel Inspector
# ----------------------------
class SteelInspector:
    def __init__(self, serial_handler):
        self.serial_handler = serial_handler
        self.checking_active = False
        self.object_was_present = False
        self.final_result = "OK"
        self.result_display_active = False
        self.show_start_label = False
        self.start_display_start_time = 0
        self.start_display_duration = 3
        self.frame_counter = 0

    def detect_and_evaluate(self, image, timestamp):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # LBP coating detection
        thresh, lbp_img, combined = process_frame(gray)
        coating_label, lbp_mean = analyze_coating(lbp_img)
        print(f"[DEBUG] LBP Mean: {lbp_mean:.2f} --> {coating_label}")

        

# ----------------------------
# Main Application
# ----------------------------
class SteelInspectionSystem:
    def __init__(self):
    system = SteelInspectionSystem()
    system.run()

if __name__ == "__main__":
    main()

---

## 📌 Author Notice

This project was developed entirely based on my own knowledge and experience in industrial vision systems.do any need please reach me.
