# ============================================================
# SERIAL COMMUNICATION
# ============================================================
SERIAL_PORT = 'COM9'
BAUD_RATE = 115200

# ============================================================
# VIDEO STREAM
# ============================================================
VIDEO_SOURCE = 1  
FRAME_WIDTH = 600 

# ============================================================
# PD CONTROLLER TUNING
# ============================================================
PAN_P_GAIN = 0.0008
TILT_P_GAIN = 0.0008

PAN_D_GAIN = 0.0
TILT_D_GAIN = 0.0

MAX_SPEED = 0.008
DEADBAND = 15

# ============================================================
# TRACKING VISUALS
# ============================================================
CIRCLE_RADIUS = 20  # Radius of the center circle drawn on screen
BUFFER_SIZE = 64    # Length of the tracking trail

# ============================================================
# COMMAND THROTTLING
# ============================================================
COMMAND_INTERVAL = 0.08

# ============================================================
# EXPONENTIAL SMOOTHING
# ============================================================
EMA_ALPHA = 0.25

# ============================================================
# HSV COLOR DETECTION DEFAULTS
# ============================================================
HSV_INITIAL_H_MIN = 0
HSV_INITIAL_S_MIN = 187
HSV_INITIAL_V_MIN = 66
HSV_INITIAL_H_MAX = 179
HSV_INITIAL_S_MAX = 255
HSV_INITIAL_V_MAX = 133

# ============================================================
# YOLO MODEL CONFIGURATION
# ============================================================
YOLO_MODEL_PATH = r'D:\VIT_D\MDX\S&MC\cw2\project_root\yolo\my_model\my_model.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_USE_GPU = True  # Will auto-detect if CUDA is available
YOLO_INFERENCE_SIZE = 640  # Image size for inference (matches your frame_target_width)