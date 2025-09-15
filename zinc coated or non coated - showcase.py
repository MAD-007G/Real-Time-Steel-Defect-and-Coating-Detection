import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
from ids_peak import ids_peak_ipl_extension

# ----------------------------
# IDS Camera Initialization
# ----------------------------
def initialize_ids_camera():
    try:
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()

        if len(device_manager.Devices()) == 0:
            raise RuntimeError("No IDS camera found.")

        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        nodemap = device.RemoteDevice().NodeMaps()[0]

        pixel_format_node = nodemap.FindNode("PixelFormat")
        pixel_format_node.SetCurrentEntry("Mono8")

        width_node = nodemap.FindNode("Width")
        height_node = nodemap.FindNode("Height")
        width_node.SetValue(480)
        height_node.SetValue(320)

        datastream = device.DataStreams()[0].OpenDataStream()
        payload_size = int(nodemap.FindNode("PayloadSize").Value())
        buffers = []
        for _ in range(4):
            buffer = datastream.AllocAndAnnounceBuffer(payload_size)
            datastream.QueueBuffer(buffer)
            buffers.append(buffer)

        datastream.StartAcquisition()
        nodemap.FindNode("AcquisitionStart").Execute()

        return datastream, device, buffers
    except Exception as e:
        ids_peak.Library.Close()
        raise RuntimeError(f"Camera initialization failed: {e}")

# ----------------------------
# LBP Parameters
# ----------------------------
RADIUS = 1
N_POINTS = 8 * RADIUS
LBP_MEAN_THRESHOLD = 180  # <--- You will tune this value after testing

# ----------------------------
# Process Frame
# ----------------------------
def process_frame(gray):
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, method="uniform")
    lbp_norm = np.uint8((lbp / lbp.max()) * 255 if lbp.max() > 0 else lbp)
    combined = cv2.bitwise_and(thresh, lbp_norm)
    return thresh, lbp_norm, combined

# ----------------------------
# Analyze Coating
# ----------------------------
def analyze_coating(lbp_img):
    lbp_mean = np.mean(lbp_img)
    # Zinc coated --> higher LBP mean (more rough)
    label = "COATED" if lbp_mean < LBP_MEAN_THRESHOLD else "NON-COATED"
    return label, lbp_mean

# ----------------------------
# Main Loop
# ----------------------------
def main():
    datastream, device, buffers = None, None, []
    try:
        datastream, device, buffers = initialize_ids_camera()
        ipl_converter = ids_peak_ipl.ImageConverter()

        while True:
            buffer = datastream.WaitForFinishedBuffer(5000)
            if buffer is None:
                continue

            try:
                image = ids_peak_ipl.Image.CreateFromSizeAndBuffer(
                    buffer.PixelFormat(),
                    buffer.BasePtr(),
                    buffer.Size(),
                    buffer.Width(),
                    buffer.Height()
                )

                image_converted = ipl_converter.Convert(image, ids_peak_ipl.PixelFormatName_Mono8)
                np_image = np.array(image_converted.get_numpy_2D())

                thresh, lbp, combined = process_frame(np_image)
                label, lbp_mean = analyze_coating(lbp)

                # Debug print
                print(f"[DEBUG] LBP Mean: {lbp_mean:.2f} --> {label}")

                # Detect contours and apply convex hull
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                display_img = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(largest_contour)
                    cv2.drawContours(display_img, [hull], -1, (0, 255, 0), 2)

                    # Draw label
                    label_color = (0, 0, 255) if label == "NON-COATED" else (0, 255, 0)
                    cv2.putText(display_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, label_color, 2, cv2.LINE_AA)

                # Display windows
                cv2.imshow("Original", display_img)
                cv2.imshow("Thresholded", thresh)
                cv2.imshow("LBP", lbp)
                cv2.imshow("Combined", combined)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Processing error: {e}")
            finally:
                datastream.QueueBuffer(buffer)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        try:
            if datastream:
                datastream.StopAcquisition()
            if device:
                device.RemoteDevice().NodeMaps()[0].FindNode("AcquisitionStop").Execute()
            cv2.destroyAllWindows()
            ids_peak.Library.Close()
        except Exception as e:
            print(f"Cleanup error: {e}")

if __name__ == "__main__":
    main()
