import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Hardcoded model path and video source index
modelpath = r'D:\VIT_D\MDX\S&MC\cw2\project_root\yolo\my_model\my_model.pt'
imgsource = 1  # Camera index or video file path or image source
minthresh = 0.5  # Confidence threshold
userres = None  # Set resolution to None to use default camera resolution
record = False  # True if you want to save output video

# Check model path
if not os.path.exists(modelpath):
    print("ERROR: Model path is invalid or model was not found. Exiting.")
    sys.exit(0)

# Load model and move to GPU if available
model = YOLO(modelpath)
if torch.cuda.is_available():
    model.to("cuda")
    model.half()  # Use FP16 precision to accelerate inference if supported
    print("Using GPU for inference")
else:
    print("Using CPU for inference")

# Initialize video capture for USB camera (adjust if using video/image source)
usbidx = int(imgsource)
cap = cv2.VideoCapture(usbidx)

# Set resolution if specified by userres (format like "1280x720")
if userres:
    resW, resH = map(int, userres.split("x"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# Bounding box colors with Tableau 10 color scheme
bboxcolors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), 
              (88,159,106), (96,202,231), (159,124,168), (169,162,241),
              (98,118,150), (172,176,184)]

avgframerate = 0
frameratebuffer = []
fpsavglen = 200

resize_frames = True  # Enable frame resizing for speed-up
frame_target_width = 640  # Resize width for inference while keeping aspect ratio

while True:
    tstart = time.perf_counter()

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Unable to read frames from the camera. Exiting.")
        break

    # Resize frame for faster inference if needed, keeping aspect ratio
    if resize_frames:
        target_h = int((frame_target_width / frame.shape[1]) * frame.shape[0])
        frame = cv2.resize(frame, (frame_target_width, target_h))

    # Run inference with confidence threshold and no verbose printing
    results = model(frame, imgsz=frame_target_width, conf=minthresh, verbose=False)

    detections = results[0].boxes
    objectcount = 0

    # Draw detections on the frame
    for i in range(len(detections)):
        xyxytensor = detections[i].xyxy.cpu()
        xyxy = xyxytensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = results[0].names[classidx]
        conf = detections[i].conf.item()

        if conf >= minthresh:
            color = bboxcolors[classidx % len(bboxcolors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{classname} {int(conf * 100)}%"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            labelymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(frame, (xmin, labelymin - labelSize[1] - 10), 
                                (xmin + labelSize[0], labelymin + baseLine - 10), (color), cv2.FILLED)
            cv2.putText(frame, label, (xmin, labelymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            objectcount += 1

    # Calculate average FPS over recent frames
    tstop = time.perf_counter()
    frameratecalc = 1 / (tstop - tstart)
    if len(frameratebuffer) >= fpsavglen:
        frameratebuffer.pop(0)
    frameratebuffer.append(frameratecalc)
    avgframerate = np.mean(frameratebuffer)

    # Display FPS and object count on frame
    if cap.isOpened():
        cv2.putText(frame, f"FPS: {avgframerate:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {objectcount}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLO detection results", frame)

    # Press keys q/Q to quit, s/S to pause, p/P to save image snapshot
    key = cv2.waitKey(5)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        print("Paused. Press any key to continue...")
        cv2.waitKey(0)
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite("capture.png", frame)
        print("Saved current frame to capture.png")

cap.release()
cv2.destroyAllWindows()
