import cv2
import imutils
import time
import serial
import json
import os
import threading
import numpy as np
import config
import torch
from ultralytics import YOLO

class VideoCamera(object):
    def __init__(self):
        # --- VIDEO INIT ---
        src = config.VIDEO_SOURCE
        if isinstance(src, str) and src.isdigit(): src = int(src)
        self.video = cv2.VideoCapture(src)
        
        # --- THREADING INIT ---
        self.started = False
        self.lock = threading.Lock()
        self.frame_main_bytes = None
        self.frame_mask_bytes = None

        # --- STATE VARIABLES ---
        self.mode = "HSV"  
        self.tracking_active = False 
        self.current_port = config.SERIAL_PORT
        self.frame_counter = 0 # To time buffer flushes
        
        # Servo Vars
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.prev_errorPan = 0
        self.prev_errorTilt = 0
        self.smoothed_pan = 0.0
        self.smoothed_tilt = 0.0
        self.last_command_time = 0
        
        # --- HSV VALUES ---
        self.hsv_values = {
            'h_min': config.HSV_INITIAL_H_MIN, 's_min': config.HSV_INITIAL_S_MIN, 'v_min': config.HSV_INITIAL_V_MIN,
            'h_max': config.HSV_INITIAL_H_MAX, 's_max': config.HSV_INITIAL_S_MAX, 'v_max': config.HSV_INITIAL_V_MAX
        }
        self.update_hsv_bounds()

        # --- SERIAL INIT ---
        self.connect_serial(self.current_port)

        # --- PRESETS ---
        self.presets_file = "presets.json"
        self.presets = self.load_presets_from_disk()

        # --- YOLO MODEL INIT ---
        self.yolo_model = None
        self.yolo_ready = False
        self.load_yolo_model()

    def connect_serial(self, port_name):
        if hasattr(self, 'ser') and self.ser is not None and self.ser.is_open:
            self.ser.close()
        try:
            # ADDED: write_timeout=0 (Non-blocking mode attempt)
            self.ser = serial.Serial(port_name, config.BAUD_RATE, timeout=1, write_timeout=0.1)
            time.sleep(0.5)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.current_port = port_name
            print(f"✓ SERIAL CONNECTED: {port_name}")
            return True, "Connected"
        except Exception as e:
            self.ser = None
            print(f"✗ SERIAL ERROR: {e}")
            return False, str(e)
    
    def load_yolo_model(self):
        """Load YOLO model with error handling"""
        try:
            if not os.path.exists(config.YOLO_MODEL_PATH):
                print(f"✗ YOLO MODEL NOT FOUND: {config.YOLO_MODEL_PATH}")
                return False
            
            self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
            
            # Move to GPU if available and requested
            if config.YOLO_USE_GPU and torch.cuda.is_available():
                self.yolo_model.to("cuda")
                self.yolo_model.half()  # FP16 for speed
                print("✓ YOLO MODEL LOADED (GPU)")
            else:
                print("✓ YOLO MODEL LOADED (CPU)")
            
            self.yolo_ready = True
            return True
            
        except Exception as e:
            print(f"✗ YOLO MODEL ERROR: {e}")
            self.yolo_ready = False
            return False

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True 
        self.thread.start()
        return self

    def update(self):
        while self.started:
            success, frame = self.video.read()
            if not success:
                time.sleep(0.1)
                continue
            
            self.frame_counter += 1
            
            # --- PROCESSING ---
            frame = imutils.resize(frame, width=config.FRAME_WIDTH)
            (H, W) = frame.shape[:2]
            centerX, centerY = W // 2, H // 2
            
            mask_frame = np.zeros((H, W, 3), dtype="uint8")

            if self.mode == "HSV":
                frame, mask_frame = self.process_hsv(frame, centerX, centerY)
            elif self.mode == "YOLO":
                frame = self.process_yolo(frame, centerX, centerY)
                mask_frame = frame 

            # HUD
            cv2.line(frame, (centerX - 10, centerY), (centerX + 10, centerY), (255, 0, 0), 1)
            cv2.line(frame, (centerX, centerY - 10), (centerX, centerY + 10), (255, 0, 0), 1)
            
            status_color = (0, 255, 0) if self.tracking_active else (0, 0, 255)
            status_text = "ACTIVE" if self.tracking_active else "STOPPED"
            cv2.putText(frame, f"TRACK: {status_text}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            # --- ENCODING ---
            ret1, jpeg_main = cv2.imencode('.jpg', frame)
            ret2, jpeg_mask = cv2.imencode('.jpg', mask_frame)

            with self.lock:
                self.frame_main_bytes = jpeg_main.tobytes()
                self.frame_mask_bytes = jpeg_mask.tobytes()
            
            # --- HOUSEKEEPING: Prevent Serial Clog ---
            # Every 100 frames, flush the buffers to prevent lag buildup
            if self.frame_counter % 100 == 0:
                if self.ser and self.ser.is_open:
                    try:
                        self.ser.reset_input_buffer()
                        self.ser.reset_output_buffer()
                    except: pass

    def get_frame_bytes(self, feed_type='main'):
        with self.lock:
            if feed_type == 'mask': return self.frame_mask_bytes
            return self.frame_main_bytes

    # ... [__del__, update_hsv_bounds, set_hsv, load/save presets, load_preset remain SAME] ...
    def __del__(self):
        self.started = False
        self.video.release()
        if self.ser: self.ser.close()

    def update_hsv_bounds(self):
        self.redLower = np.array([int(self.hsv_values['h_min']), int(self.hsv_values['s_min']), int(self.hsv_values['v_min'])], dtype="uint8")
        self.redUpper = np.array([int(self.hsv_values['h_max']), int(self.hsv_values['s_max']), int(self.hsv_values['v_max'])], dtype="uint8")

    def set_hsv(self, key, value):
        self.hsv_values[key] = int(value)
        self.update_hsv_bounds()

    def load_presets_from_disk(self):
        if os.path.exists(self.presets_file):
            with open(self.presets_file, 'r') as f: return json.load(f)
        return [{"name": f"Slot {i+1}", "data": None} for i in range(4)]

    def save_preset(self, slot_index, name):
        if 0 <= slot_index < 4:
            self.presets[slot_index] = {"name": name or f"Preset {slot_index+1}", "data": self.hsv_values.copy()}
            with open(self.presets_file, 'w') as f: json.dump(self.presets, f)
            return True
        return False

    def load_preset(self, slot_index):
        if 0 <= slot_index < 4 and self.presets[slot_index]["data"]:
            self.hsv_values = self.presets[slot_index]["data"].copy()
            self.update_hsv_bounds()
            return self.hsv_values
        return None
    # ... [End of Standard Methods] ...

    def process_hsv(self, frame, centerX, centerY):
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.redLower, self.redUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask_visual = cv2.bitwise_and(frame, frame, mask=mask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        should_move = False

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                db = config.DEADBAND
                if (center[0] < centerX - db or center[0] > centerX + db or 
                    center[1] < centerY - db or center[1] > centerY + db):
                    should_move = True

                if self.tracking_active and should_move:
                    self.update_servos(center[0], center[1], centerX, centerY)

        color = (0, 255, 0) if should_move else (0, 165, 255)
        cv2.rectangle(frame, (centerX - config.DEADBAND, centerY - config.DEADBAND), 
                     (centerX + config.DEADBAND, centerY + config.DEADBAND), color, 2)
        return frame, mask_visual

    def update_servos(self, target_x, target_y, center_x, center_y):
        # PID Calculations
        errorPan = target_x - center_x
        errorTilt = target_y - center_y
        
        d_pan = errorPan - self.prev_errorPan
        delta_pan = (errorPan * config.PAN_P_GAIN) + (d_pan * config.PAN_D_GAIN)
        delta_pan = max(-config.MAX_SPEED, min(config.MAX_SPEED, delta_pan))
        self.current_pan -= delta_pan
        self.prev_errorPan = errorPan

        d_tilt = errorTilt - self.prev_errorTilt
        delta_tilt = (errorTilt * config.TILT_P_GAIN) + (d_tilt * config.TILT_D_GAIN)
        delta_tilt = max(-config.MAX_SPEED, min(config.MAX_SPEED, delta_tilt))
        self.current_tilt += delta_tilt
        self.prev_errorTilt = errorTilt
        
        self.smoothed_pan = max(-1.0, min(1.0, config.EMA_ALPHA * self.current_pan + (1 - config.EMA_ALPHA) * self.smoothed_pan))
        self.smoothed_tilt = max(-1.0, min(1.0, config.EMA_ALPHA * self.current_tilt + (1 - config.EMA_ALPHA) * self.smoothed_tilt))

        if self.ser and self.ser.is_open:
            current_time = time.time()
            if current_time - self.last_command_time >= config.COMMAND_INTERVAL:
                
                # --- CRITICAL FIX: ANTI-BLOCKING CHECK ---
                try:
                    # 1. Check if buffer is getting full (arbitrary limit, e.g., 64 bytes)
                    if self.ser.out_waiting > 64:
                        # Panic mode: Buffer is full, clear it and SKIP this command
                        self.ser.reset_output_buffer()
                        print("Buffer Overflow - Flushing")
                        return 

                    # 2. Write with a timeout (handled by Serial init)
                    cmd = "{:.3f} {:.3f}\n".format(self.smoothed_pan, self.smoothed_tilt)
                    self.ser.write(cmd.encode('utf-8'))
                    self.last_command_time = current_time
                    
                except serial.SerialTimeoutException:
                    print("Serial Write Timeout - Skipping")
                except Exception as e:
                    print(f"Serial Error: {e}")

    def process_yolo(self, frame, centerX, centerY):
        """Process frame using YOLO detection"""
        
        # If model not ready, show error and return
        if not self.yolo_ready or self.yolo_model is None:
            cv2.putText(frame, "YOLO MODEL NOT LOADED", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame
        
        # Run YOLO inference
        try:
            results = self.yolo_model(frame, imgsz=config.YOLO_INFERENCE_SIZE, 
                                      conf=config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)
            
            detections = results[0].boxes
            object_count = 0
            best_detection = None
            best_area = 0
            
            # Process all detections
            for i in range(len(detections)):
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)
                
                classidx = int(detections[i].cls.item())
                classname = results[0].names[classidx]
                conf = detections[i].conf.item()
                
                if conf >= config.YOLO_CONFIDENCE_THRESHOLD:
                    # Draw bounding box
                    color = (0, 255, 255)  # Yellow for detected objects
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    
                    # Draw label with confidence
                    label = f"{classname} {int(conf * 100)}%"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    labelymin = max(ymin, labelSize[1] + 10)
                    
                    cv2.rectangle(frame, (xmin, labelymin - labelSize[1] - 10), 
                                 (xmin + labelSize[0], labelymin + baseLine - 10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, labelymin - 7), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Calculate center of bounding box
                    bbox_center_x = int((xmin + xmax) / 2)
                    bbox_center_y = int((ymin + ymax) / 2)
                    
                    # Draw center point
                    cv2.circle(frame, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)
                    
                    # Track the largest detection (by area)
                    area = (xmax - xmin) * (ymax - ymin)
                    if area > best_area:
                        best_area = area
                        best_detection = (bbox_center_x, bbox_center_y)
                    
                    object_count += 1
            
            # Display object count
            cv2.putText(frame, f"Objects: {object_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # If tracking is active and we have a detection, update servos
            if self.tracking_active and best_detection:
                target_x, target_y = best_detection
                
                # Check if outside deadband
                db = config.DEADBAND
                should_move = (target_x < centerX - db or target_x > centerX + db or 
                              target_y < centerY - db or target_y > centerY + db)
                
                if should_move:
                    # Highlight the tracked object
                    cv2.line(frame, (centerX, centerY), (target_x, target_y), (0, 255, 0), 2)
                    self.update_servos(target_x, target_y, centerX, centerY)
            
        except Exception as e:
            cv2.putText(frame, f"YOLO ERROR: {str(e)[:30]}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"YOLO Processing Error: {e}")
        
        return frame