# Pan & Tilt Ball Tracking Camera System

## Project Overview

This project implements an **automated pan-tilt camera tracking system** for Sensing and Motion Control applications. The system uses computer vision algorithms to detect and track colored objects (balls) in real-time, automatically adjusting servo motors to keep the target centered in the camera's field of view.

The system features a **web-based control interface** with dual detection modes (HSV color-based and YOLO object detection), real-time video streaming, and configurable PD (Proportional-Derivative) control for smooth servo movements.

---

## Key Features

### 1. **Dual Detection Modes**
- **HSV Color Detection**: Traditional color-based tracking with real-time HSV calibration
- **YOLO Object Detection**: Deep learning-based detection using a custom-trained YOLOv8 model

### 2. **Web-Based Control Interface**
- Live video feed with mask/debug visualization
- Real-time HSV parameter tuning with interactive sliders
- Preset management system (save/load up to 4 HSV configurations)
- Serial port configuration and connection management
- System activity logs

### 3. **Advanced Servo Control**
- **PD Controller**: Proportional-Derivative control for smooth, responsive tracking
- **Exponential Moving Average (EMA)**: Smoothing filter to reduce jitter
- **Deadband Zone**: Prevents unnecessary micro-adjustments
- **Command Throttling**: Prevents serial buffer overflow
- **Automatic Buffer Management**: Periodic flushing to maintain responsiveness

### 4. **Performance Optimizations**
- Multi-threaded architecture for concurrent video processing and web serving
- GPU acceleration support for YOLO inference (CUDA-enabled)
- Frame buffering and non-blocking serial communication
- Configurable inference parameters

---

## Project Structure

```
project_root/
├── app.py                  # Flask web application (main entry point)
├── camera.py               # Video processing and servo control logic
├── config.py               # Centralized configuration parameters
├── presets.json            # Saved HSV presets (auto-generated)
├── templates/
│   └── index.html          # Web interface UI
└── yolo/
    └── my_model/
        ├── best.pt         # Trained YOLO model weights
        └── my_model.pt     # Alternative model weights
```

---

## Hardware Requirements

### Components
1. **Camera**: USB webcam
2. **Microcontroller**: Arduino with serial communication
3. **Servo Motors**: 2x servo motors (pan and tilt axes)
4. **Power Supply**: Adequate power for servos (typically 5-6V)
5. **Pan-Tilt Mechanism**: Mechanical mount for camera and servos

### Arduino Setup
The Arduino should be programmed to:
- Accept serial commands in the format: `pan_value tilt_value\n`
- Values range from `-1.0` to `1.0` (normalized servo positions)
- Parse and apply these values to the pan and tilt servos

---

## Software Requirements

### Python Dependencies
```bash
# Core Libraries
flask>=2.0.0
opencv-python>=4.5.0
imutils>=0.5.4
numpy>=1.21.0
pyserial>=3.5

# YOLO Dependencies
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
```

### Installation

**Install dependencies**
   ```bash
   pip install flask opencv-python imutils numpy pyserial torch torchvision ultralytics
   ```

**Configure the system**
   - Edit `config.py` to match your hardware setup
   - Set correct `SERIAL_PORT` (e.g., `COM12`, `/dev/ttyUSB0`)
   - Adjust `VIDEO_SOURCE` (0 for default camera, 1 for external)
   - Update `YOLO_MODEL_PATH` if using a different model location

---

## Quick Start Guide


### Software Configuration
1. Open `config.py` and verify settings:
   ```python
   SERIAL_PORT = 'COM12'      # Your Arduino port
   VIDEO_SOURCE = 1           # Camera index
   BAUD_RATE = 9600          # Match Arduino baud rate
   ```

2. For HSV mode, calibrate initial color values:
   ```python
   HSV_INITIAL_H_MIN = 0
   HSV_INITIAL_S_MIN = 187
   HSV_INITIAL_V_MIN = 66
   # ... (adjust based on your target object color)
   ```

3. Launch Application
```bash
python app.py
```

The server will start on `http://0.0.0.0:5000`

### Access Web Interface
1. Open browser and navigate to `http://localhost:5000`
2. Click **CONN** button to establish serial connection
3. Select detection mode (HSV or YOLO)
4. Click **START TRACKING** to begin

---

## Using the Web Interface

### Hardware Panel
- **COM Port Input**: Enter your Arduino's serial port
- **CONN Button**: Establish serial connection
- **START/STOP TRACKING**: Toggle tracking on/off

### Algorithm Panel
- **HSV Mode**: Color-based detection (requires calibration)
- **YOLO Mode**: AI-based object detection currently works for MDX red stress ball (requires trained model)

### Vision Feed
- **Live Input**: Raw camera feed with tracking overlays
- **Mask/Debug**: 
  - HSV mode: Shows binary mask of detected color
  - YOLO mode: Duplicate feed with bounding boxes

### HSV Calibration Panel
Adjust six sliders to isolate your target color:
- **H (Hue)**: Color type (0-179)
  - Red: 0-10, 170-179
  - Green: 40-80
  - Blue: 100-130
- **S (Saturation)**: Color intensity (0-255)
- **V (Value)**: Brightness (0-255)

### Presets
- **SAVE**: Store current HSV values to a slot (1-4)
- **LOAD**: Restore previously saved HSV configuration

---

## Configuration Parameters

### Serial Communication
```python
SERIAL_PORT = 'COM12'        # Arduino port
BAUD_RATE = 9600            # Communication speed
```

### Video Settings
```python
VIDEO_SOURCE = 1            # Camera index (0, 1, 2, or URL)
FRAME_WIDTH = 600          # Processing resolution (pixels)
```

### PD Controller Tuning
```python
# Proportional Gains (responsiveness)
PAN_P_GAIN = 0.08          # Higher = faster pan response
TILT_P_GAIN = 0.08         # Higher = faster tilt response

# Derivative Gains (damping)
PAN_D_GAIN = 0.04          # Higher = more damping
TILT_D_GAIN = 0.04         # Higher = more damping

# Constraints
MAX_SPEED = 0.008          # Maximum servo speed per update
DEADBAND = 25              # Pixel tolerance (no movement zone)
```

**Tuning Tips:**
- **Oscillation/Overshoot**: Reduce P gain or increase D gain
- **Slow Response**: Increase P gain
- **Jittery Movement**: Increase `EMA_ALPHA` or `DEADBAND`

### Smoothing & Timing
```python
EMA_ALPHA = 0.25           # Smoothing factor (0-1, lower = smoother)
COMMAND_INTERVAL = 0.1     # Seconds between servo commands
```

### YOLO Configuration
```python
YOLO_MODEL_PATH = r'path\to\best.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_USE_GPU = True        # Enable CUDA acceleration
YOLO_INFERENCE_SIZE = 640  # Input image size
```

---

## System Architecture

### Threading Model
```
Main Thread (Flask)
    ├── HTTP Request Handling
    ├── Video Stream Generators
    └── API Endpoints

Background Thread (VideoCamera)
    ├── Frame Capture
    ├── Object Detection (HSV/YOLO)
    ├── Servo Control (PD Algorithm)
    └── Frame Encoding (JPEG)
```

### Control Flow

#### HSV Mode
```
Camera Frame → Resize → Gaussian Blur → HSV Conversion
    → Color Masking → Morphological Operations → Contour Detection
    → Largest Contour → Centroid Calculation → Deadband Check
    → PD Controller → EMA Smoothing → Serial Command
```

#### YOLO Mode
```
Camera Frame → Resize → YOLO Inference → Bounding Boxes
    → Largest Detection → Center Calculation → Deadband Check
    → PD Controller → EMA Smoothing → Serial Command
```

### PD Control Algorithm
```python
# Error calculation
error = target_position - center_position

# Derivative term
d_error = error - previous_error

# Control signal
delta = (error * P_GAIN) + (d_error * D_GAIN)

# Apply speed limit
delta = clamp(delta, -MAX_SPEED, MAX_SPEED)

# Update servo position
servo_position -= delta

# Smooth output
smoothed_position = EMA_ALPHA * servo_position + (1 - EMA_ALPHA) * previous_smoothed
```

---

## Performance Metrics

### Typical Performance (on mid-range laptop)
- **Frame Rate**: 25-30 FPS (HSV mode), 15-20 FPS (YOLO mode)
- **Latency**: 50-100ms (detection to servo response)
- **Tracking Accuracy**: ±5 pixels at 600px resolution
- **Servo Update Rate**: 10 Hz (configurable)

### Optimization Recommendations
1. **For Speed**: Use HSV mode, reduce frame width, disable debug features
2. **For Accuracy**: Use YOLO mode with GPU, increase frame width
3. **For Smoothness**: Increase EMA_ALPHA, reduce MAX_SPEED, tune PD gains

---

## Educational Context

This project demonstrates key concepts in:
- **Computer Vision**: Color segmentation, object detection, contour analysis
- **Control Systems**: PD control, feedback loops, system stability
- **Embedded Systems**: Serial communication, real-time constraints
- **Signal Processing**: Exponential moving average, noise filtering
- **Web Development**: Flask framework, real-time video streaming, AJAX
- **Machine Learning**: YOLO object detection, model inference

---


## File Descriptions

### `app.py`
Flask web application serving the control interface. Handles:
- Route definitions (`/`, `/video_feed_main`, `/video_feed_mask`)
- API endpoints for tracking control, HSV updates, mode switching
- Video stream generators using multipart responses

### `camera.py`
Core video processing and control logic. Contains:
- `VideoCamera` class with threading support
- HSV and YOLO processing pipelines
- PD controller implementation
- Serial communication handling
- Preset management system

### `config.py`
Centralized configuration file for all tunable parameters:
- Hardware settings (serial, video)
- Control algorithm parameters (PD gains, limits)
- Detection thresholds (HSV ranges, YOLO confidence)

### `templates/index.html`
Web interface with:
- Dual video feed display
- Interactive HSV sliders
- Mode switching and preset management
- Real-time system logs

---

## References & Resources

### Libraries Used
- **OpenCV**: https://opencv.org/
- **Flask**: https://flask.palletsprojects.com/
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **PySerial**: https://pyserial.readthedocs.io/

### Learning Resources
- https://www.youtube.com/watch?v=1x6t3tHBdRY
- https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
- https://github.com/Practical-CV/Color-Based-Ball-Tracking-With-OpenCV

---

## Contributing

This is an academic project for Sensing and Motion Control coursework. For improvements or bug fixes:
1. Test changes thoroughly with actual hardware
2. Document parameter changes in `config.py`
3. Update this README if adding new features

---

## Author

**MISIS**:   M01088120
**Course**: PDE4446 Sensing and Motion Control  
**Institution**: Middlesex University Dubai


---
