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

        # X-ray processing
        xray_image, enhanced_gray = XRayProcessor.create_xray_vision(image)

        _, object_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 5000
        max_contour_area = 9000000
        valid_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

        object_found = len(valid_contours) > 0
        result_frame = xray_image.copy() if object_found else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
        cv2.putText(result_frame, timestamp_str, (10, CAMERA_HEIGHT - 10), FONT, 0.5, (255, 255, 255), 1)
        cv2.putText(result_frame, "X-RAY MODE", (CAMERA_WIDTH - 130, 30), FONT, 0.5, (0, 255, 255), 1)

        if object_found:
            if not self.checking_active:
                self.final_result = "OK"
                self.checking_active = True
                self.object_was_present = True
                self.result_display_active = False
                self.show_start_label = True
                self.start_display_start_time = time.time()
                print("Steel Object Detected - Starting Inspection")

            all_points = np.vstack(valid_contours)
            hull = cv2.convexHull(all_points)
            cv2.polylines(result_frame, [hull], True, (0, 255, 0), 2)

            defect_frame, defects_found = XRayProcessor.detect_defects(enhanced_gray, result_frame, valid_contours)
            result_frame = defect_frame

            if coating_label == "NON-COATED":
                self.final_result = "NG"
                print("Non-Coated Steel Detected → Marking as NG")
            elif defects_found:
                self.final_result = "NG"
                print("Defects Found → Marking as NG")

            if self.final_result == "NG":
                filename = os.path.join(SAVE_DIR, f"NG_{timestamp_str.replace(':', '-')}.png")
                cv2.imwrite(filename, result_frame)
                print(f"Saved NG frame: {filename}")

            self.serial_handler.send_relay_command(self.final_result)

            label_color = (0, 0, 255) if coating_label == "NON-COATED" else (0, 255, 0)
            cv2.putText(result_frame, coating_label, (10, 100), FONT, 1, label_color, 2)

            if self.show_start_label:
                current_time = time.time()
                if current_time - self.start_display_start_time <= self.start_display_duration:
                    cv2.putText(result_frame, "START", (10, 30), FONT, 1, (0, 255, 255), 2)
                else:
                    self.show_start_label = False

        else:
            if self.object_was_present:
                color = (0, 255, 0) if self.final_result == "OK" else (0, 0, 255)
                cv2.putText(result_frame, self.final_result, (10, 60), FONT, 1.5, color, 3)
                print(f"Final Inspection Result: {self.final_result}")

                self.checking_active = False
                self.object_was_present = False
                self.result_display_active = True
                self.serial_handler.send_relay_command(self.final_result)

            elif self.result_display_active:
                color = (0, 255, 0) if self.final_result == "OK" else (0, 0, 255)
                cv2.putText(result_frame, self.final_result, (580, 40), FONT, 1.5, color, 3)
                cv2.putText(result_frame, "END", (10, 30), FONT, 1, (0, 255, 255), 2)
            else:
                cv2.putText(result_frame, "WAITING FOR STEEL...", (10, 30), FONT, 1, (255, 0, 0), 2)

        self.frame_counter += 1
        return result_frame


# ----------------------------
# IDS Camera Acquisition
# ----------------------------
class AcquisitionThread(threading.Thread):
    def __init__(self, device, node_map, datastream, image_converter):
        super().__init__()
        self.running = True
        self.device = device
        self.node_map = node_map
        self.datastream = datastream
        self.image_converter = image_converter
        self.frame_buffer = queue.Queue(maxsize=30)

    def run(self):
        while self.running:
            try:
                buffer = self.datastream.WaitForFinishedBuffer(1000)
                ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
                converted_image = self.image_converter.Convert(ipl_image, TARGET_PIXEL_FORMAT)
                img_array = converted_image.get_numpy_1D()
                frame = np.reshape(img_array, (converted_image.Height(), converted_image.Width(), 4))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame = cv2.resize(frame, (480, 320))  # Downsample to reduce lag
                self.frame_buffer.put((frame.copy(), time.time()), block=False)
                self.datastream.QueueBuffer(buffer)
            except queue.Full:
                print("Frame buffer full, dropping frame")
                self.datastream.QueueBuffer(buffer)
            except ids_peak.Exception as e:
                print(f"Acquisition error: {str(e)}")
                time.sleep(0.1)

    def stop(self):
        self.running = False

# ----------------------------
# Frame Processing
# ----------------------------
class ProcessingThread(threading.Thread):
    def __init__(self, inspector, frame_buffer):
        super().__init__()
        self.running = True
        self.inspector = inspector
        self.frame_buffer = frame_buffer
        self.target_fps = 10
        self.frame_delay = 1.0 / self.target_fps

    def run(self):
        last_frame_time = 0
        while self.running:
            now = time.time()
            if now - last_frame_time < self.frame_delay:
                time.sleep(0.001)
                continue
            try:
                while not self.frame_buffer.empty():
                    frame, timestamp = self.frame_buffer.get_nowait()
                if frame is None:
                    continue
                processed_img = self.inspector.detect_and_evaluate(frame, timestamp)
                cv2.imshow("STEEL X-RAY INSPECTION SYSTEM (IDS Camera)", processed_img)
                key = cv2.waitKey(1)
                if key in (ord('q'), 27):
                    self.running = False
                last_frame_time = now
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {str(e)}")

    def stop(self):
        self.running = False

# ----------------------------
# Main Application
# ----------------------------
class SteelInspectionSystem:
    def __init__(self):
        self.serial_handler = SerialHandler(SERIAL_PORT, BAUD_RATE)
        self.inspector = SteelInspector(self.serial_handler)
        self.device = None
        self.node_map = None
        self.datastream = None
        self.image_converter = None
        self.acquisition_thread = None
        self.processing_thread = None

    def initialize_camera(self):
        print("🔍 Initializing IDS camera...")
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()
        if device_manager.Devices().empty():
            raise RuntimeError("❌ No IDS cameras found!")
        self.device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        self.node_map = self.device.RemoteDevice().NodeMaps()[0]
        self.node_map.FindNode("UserSetSelector").SetCurrentEntry("Default")
        self.node_map.FindNode("UserSetLoad").Execute()
        self.node_map.FindNode("UserSetLoad").WaitUntilDone()
        try:
            self.node_map.FindNode("AcquisitionFrameRateEnable").SetValue(True)
            self.node_map.FindNode("AcquisitionFrameRate").SetValue(8.0)
        except ids_peak.Exception as e:
            print(f"Warning: Could not set frame rate: {str(e)}")
        self.datastream = self.device.DataStreams()[0].OpenDataStream()
        payload_size = self.node_map.FindNode("PayloadSize").Value()
        buffer_amount = max(8, self.datastream.NumBuffersAnnouncedMinRequired())
        buffers = [self.datastream.AllocAndAnnounceBuffer(payload_size) for _ in range(buffer_amount)]
        for buffer in buffers:
            self.datastream.QueueBuffer(buffer)
        self.image_converter = ids_peak_ipl.ImageConverter()
        camera_width = self.node_map.FindNode("Width").Value()
        camera_height = self.node_map.FindNode("Height").Value()
        input_pixel_format = ids_peak_ipl.PixelFormat(self.node_map.FindNode("PixelFormat").CurrentEntry().Value())
        self.image_converter.PreAllocateConversion(input_pixel_format, TARGET_PIXEL_FORMAT, camera_width, camera_height)

    def start(self):
        print("\n===== STEEL INSPECTION SYSTEM =====")
        print("Detecting steel objects and defects with convex hull and relay signaling")
        print("Press 'q' or ESC in the image window to stop the inspection system...\n")
        self.datastream.StartAcquisition()
        self.node_map.FindNode("AcquisitionStart").Execute()
        self.node_map.FindNode("AcquisitionStart").WaitUntilDone()
        self.acquisition_thread = AcquisitionThread(self.device, self.node_map, self.datastream, self.image_converter)
        self.processing_thread = ProcessingThread(self.inspector, self.acquisition_thread.frame_buffer)
        self.acquisition_thread.start()
        self.processing_thread.start()
        print("📸 Camera started successfully")

    def run(self):
        try:
            self.initialize_camera()
            self.start()
            while self.acquisition_thread.is_alive() and self.processing_thread.is_alive():
                time.sleep(0.5)
        except Exception as e:
            print(f"System error: {str(e)}")
        finally:
            self.shutdown()

    def shutdown(self):
        print("🛑 Shutting down system...")
        if self.acquisition_thread:
            self.acquisition_thread.stop()
            self.acquisition_thread.join()
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread.join()
        cv2.destroyAllWindows()
        try:
            self.node_map.FindNode("AcquisitionStop").Execute()
            self.datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            self.datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for buffer in self.datastream.AnnouncedBuffers():
                self.datastream.RevokeBuffer(buffer)
        except Exception as e:
            print(f"Error stopping acquisition: {str(e)}")
        ids_peak.Library.Close()
        self.serial_handler.close()
        print("System stopped.")

def main():
    system = SteelInspectionSystem()
    system.run()

if __name__ == "__main__":
    main()