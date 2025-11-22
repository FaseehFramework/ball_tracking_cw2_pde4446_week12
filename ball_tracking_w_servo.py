from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import serial


# ============================================================
# CONFIGURATION
# ============================================================
SERIAL_PORT = 'COM12'
BAUD_RATE = 115200

# --- PD CONTROLLER TUNING ---
PAN_P_GAIN = 0.00002
TILT_P_GAIN = 0.00002

PAN_D_GAIN = 0.0
TILT_D_GAIN = 0.0

MAX_SPEED = 0.008
DEADBAND = 15
CIRCLE_RADIUS = 20

# --- COMMAND THROTTLING ---
COMMAND_INTERVAL = 0.06

# --- EXPONENTIAL SMOOTHING ---
EMA_ALPHA = 0.25


# ============================================================
# HSV TRACKBAR CALLBACK
# ============================================================
def nothing(x):
    """Dummy callback function for trackbars"""
    pass


# ============================================================
# FUNCTIONS
# ============================================================
def circle_overlaps_rectangle(cx, cy, radius, rect_x1, rect_y1, rect_x2, rect_y2):
    """
    Check if a circle overlaps with a rectangle.
    Returns True if there is ANY overlap, False if completely outside.
    """
    closest_x = max(rect_x1, min(cx, rect_x2))
    closest_y = max(rect_y1, min(cy, rect_y2))
    
    distance_x = cx - closest_x
    distance_y = cy - closest_y
    
    distance_squared = (distance_x ** 2) + (distance_y ** 2)
    return distance_squared <= (radius ** 2)


# ============================================================
# SERIAL CONNECTION
# ============================================================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print(f"✓ Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
except Exception as e:
    print(f"✗ Error connecting to serial: {e}")
    ser = None


# ============================================================
# ARGUMENT PARSING
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")

try:
    args = vars(ap.parse_args())
except:
    args = {'video': None, 'buffer': 64}


# ============================================================
# HSV COLOR DETECTION - INITIAL VALUES
# ============================================================
# Default values for red color detection
initial_h_min = 0
initial_s_min = 55
initial_v_min = 40
initial_h_max = 3
initial_s_max = 255
initial_v_max = 255

pts = deque(maxlen=args["buffer"])


# ============================================================
# CONTROL VARIABLES
# ============================================================
current_pan = 0.0
current_tilt = 0.0

prev_errorPan = 0
prev_errorTilt = 0

smoothed_pan = 0.0
smoothed_tilt = 0.0

last_command_time = 0
frame_count = 0


# ============================================================
# VIDEO STREAM INITIALIZATION
# ============================================================
print("Starting video stream...")
if not args.get("video", False):
    vs = VideoStream(src=1).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])
    time.sleep(1.0)

print("✓ Video stream started.")


# ============================================================
# CREATE TRACKBAR WINDOW
# ============================================================
# Create a window for HSV trackbars
cv2.namedWindow("HSV Tuning", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Tuning", 400, 300)

# Create trackbars for HSV color detection
cv2.createTrackbar("H Min", "HSV Tuning", initial_h_min, 179, nothing)
cv2.createTrackbar("S Min", "HSV Tuning", initial_s_min, 255, nothing)
cv2.createTrackbar("V Min", "HSV Tuning", initial_v_min, 255, nothing)
cv2.createTrackbar("H Max", "HSV Tuning", initial_h_max, 179, nothing)
cv2.createTrackbar("S Max", "HSV Tuning", initial_s_max, 255, nothing)
cv2.createTrackbar("V Max", "HSV Tuning", initial_v_max, 255, nothing)

# Create a toggle for showing the mask
cv2.createTrackbar("Show Mask (0/1)", "HSV Tuning", 0, 1, nothing)

print("✓ HSV Trackbars created. Adjust sliders to tune color detection.")
print("  Press 'q' to quit | Press 's' to save current HSV values")


# ============================================================
# MAIN LOOP
# ============================================================
try:
    while True:
        # --- READ TRACKBAR VALUES ---
        h_min = cv2.getTrackbarPos("H Min", "HSV Tuning")
        s_min = cv2.getTrackbarPos("S Min", "HSV Tuning")
        v_min = cv2.getTrackbarPos("V Min", "HSV Tuning")
        h_max = cv2.getTrackbarPos("H Max", "HSV Tuning")
        s_max = cv2.getTrackbarPos("S Max", "HSV Tuning")
        v_max = cv2.getTrackbarPos("V Max", "HSV Tuning")
        show_mask = cv2.getTrackbarPos("Show Mask (0/1)", "HSV Tuning")

        # Update HSV bounds
        #redLower = (133, 108, 5)
        #redUpper = (179, 255, 245)
        redLower = (h_min, s_min, v_min)
        redUpper = (h_max, s_max, v_max)

        # --- READ FRAME ---
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame

        if frame is None:
            print("Warning: No frame captured. Retrying...")
            time.sleep(0.1)
            continue

        # --- PREPROCESSING ---
        frame = imutils.resize(frame, width=600)
        (H, W) = frame.shape[:2]
        centerX = W // 2
        centerY = H // 2

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # --- COLOR MASK ---
        mask = cv2.inRange(hsv, redLower, redUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # --- SHOW MASK WINDOW IF ENABLED ---
        if show_mask == 1:
            cv2.imshow("Color Mask", mask)

        # --- CONTOUR DETECTION ---
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                # --- DRAW DETECTION ---
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, CIRCLE_RADIUS, (0, 0, 255), -1)

                # --- CALCULATE ERROR ---
                errorPan = center[0] - centerX
                errorTilt = center[1] - centerY

                # --- DEADBAND RECTANGLE BOUNDS ---
                deadband_x1 = centerX - DEADBAND
                deadband_y1 = centerY - DEADBAND
                deadband_x2 = centerX + DEADBAND
                deadband_y2 = centerY + DEADBAND

                # --- CHECK IF CIRCLE IS OUTSIDE DEADBAND ---
                circle_outside_deadband = not circle_overlaps_rectangle(
                    center[0], center[1], CIRCLE_RADIUS,
                    deadband_x1, deadband_y1, deadband_x2, deadband_y2
                )

                should_move = False

                # --- TRACKING LOGIC ---
                if circle_outside_deadband:
                    # PD CONTROL FOR PAN
                    if abs(errorPan) > DEADBAND:
                        derivative_pan = errorPan - prev_errorPan
                        delta_pan = (errorPan * PAN_P_GAIN) + (derivative_pan * PAN_D_GAIN)
                        delta_pan = max(-MAX_SPEED, min(MAX_SPEED, delta_pan))
                        current_pan -= delta_pan #THIS WILL
                        should_move = True
                    
                    prev_errorPan = errorPan

                    # PD CONTROL FOR TILT
                    if abs(errorTilt) > DEADBAND:
                        derivative_tilt = errorTilt - prev_errorTilt
                        delta_tilt = (errorTilt * TILT_P_GAIN) + (derivative_tilt * TILT_D_GAIN)
                        delta_tilt = max(-MAX_SPEED, min(MAX_SPEED, delta_tilt))
                        current_tilt -= delta_tilt #THIS WILL
                        should_move = True
                    
                    prev_errorTilt = errorTilt
                else:
                    prev_errorPan = 0
                    prev_errorTilt = 0

                # --- EXPONENTIAL SMOOTHING ---
                smoothed_pan = EMA_ALPHA * current_pan + (1 - EMA_ALPHA) * smoothed_pan
                smoothed_tilt = EMA_ALPHA * current_tilt + (1 - EMA_ALPHA) * smoothed_tilt

                smoothed_pan = max(-1.0, min(1.0, smoothed_pan))
                smoothed_tilt = max(-1.0, min(1.0, smoothed_tilt))

                # --- SEND SERIAL COMMAND ---
                if should_move and ser is not None and ser.is_open:
                    current_time = time.time()
                    if current_time - last_command_time >= COMMAND_INTERVAL:
                        if ser.out_waiting > 100:
                            ser.reset_output_buffer()
                        
                        command = "{:.3f} {:.3f}\n".format(smoothed_pan, smoothed_tilt)
                        try:
                            ser.write(command.encode('utf-8'))
                            last_command_time = current_time
                        except serial.SerialException as e:
                            print(f"Serial write error: {e}")

                # --- VISUAL FEEDBACK ---
                deadband_color = (0, 255, 0) if circle_outside_deadband else (0, 165, 255)
                cv2.rectangle(frame, (deadband_x1, deadband_y1), 
                              (deadband_x2, deadband_y2), deadband_color, 2)
        
        else:
            prev_errorPan = 0
            prev_errorTilt = 0
            cv2.rectangle(frame, (centerX - DEADBAND, centerY - DEADBAND), 
                          (centerX + DEADBAND, centerY + DEADBAND), (0, 255, 0), 1)

        # --- DRAW CENTER CROSSHAIR ---
        cv2.line(frame, (centerX - 10, centerY), (centerX + 10, centerY), (255, 0, 0), 1)
        cv2.line(frame, (centerX, centerY - 10), (centerX, centerY + 10), (255, 0, 0), 1)

        # --- DRAW HSV VALUES ON FRAME ---
        hsv_text = f"HSV: ({h_min},{s_min},{v_min}) - ({h_max},{s_max},{v_max})"
        cv2.putText(frame, hsv_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # --- DRAW TRACKING TRAIL ---
        pts.appendleft(center)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # --- DISPLAY FRAME ---
        cv2.imshow("Ball Tracking", frame)
        key = cv2.waitKey(1) & 0xFF

        # --- PERIODIC SERIAL BUFFER CLEARING ---
        frame_count += 1
        if frame_count % 100 == 0 and ser is not None and ser.is_open:
            try:
                ser.reset_input_buffer()
            except serial.SerialException:
                pass

        # --- SAVE HSV VALUES ---
        if key == ord("s"):
            print("\n" + "="*50)
            print("SAVED HSV VALUES:")
            print(f"  redLower = ({h_min}, {s_min}, {v_min})")
            print(f"  redUpper = ({h_max}, {s_max}, {v_max})")
            print("="*50 + "\n")

        # --- CHECK FOR QUIT ---
        if key == ord("q"):
            print("\nQuitting...")
            break

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

finally:
    # ============================================================
    # CLEANUP
    # ============================================================
    print("Cleaning up...")
    
    if not args.get("video", False):
        vs.stop()
    else:
        vs.release()
    
    if ser is not None and ser.is_open:
        ser.close()
        print("✓ Serial connection closed")
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    print("✓ Cleanup complete")
