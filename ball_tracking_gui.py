import customtkinter as ctk
from collections import deque
import numpy as np
import cv2
import imutils
import time
import serial
from PIL import Image
import threading
import config

class BallTrackingGUI:
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Ball Tracking with Servo Control")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.is_tracking = False
        self.vs = None
        self.ser = None
        self.frame = None
        self.mask = None
        
        # Control variables
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.prev_errorPan = 0
        self.prev_errorTilt = 0
        self.smoothed_pan = 0.0
        self.smoothed_tilt = 0.0
        self.last_command_time = 0
        self.frame_count = 0
        self.fps = 0
        self.ball_detected = False
        
        # Trail buffer
        self.pts = deque(maxlen=config.BUFFER_SIZE)
        
        # HSV values
        self.h_min = ctk.IntVar(value=config.HSV_INITIAL_H_MIN)
        self.s_min = ctk.IntVar(value=config.HSV_INITIAL_S_MIN)
        self.v_min = ctk.IntVar(value=config.HSV_INITIAL_V_MIN)
        self.h_max = ctk.IntVar(value=config.HSV_INITIAL_H_MAX)
        self.s_max = ctk.IntVar(value=config.HSV_INITIAL_S_MAX)
        self.v_max = ctk.IntVar(value=config.HSV_INITIAL_V_MAX)
        
        # Display options
        self.show_trail = ctk.BooleanVar(value=True)
        
        # Setup GUI
        self.setup_gui()
        
        # Try to connect to serial
        self.connect_serial()
        
    def setup_gui(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Video feeds
        left_panel = ctk.CTkFrame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Title row
        title_frame = ctk.CTkFrame(left_panel)
        title_frame.pack(fill="x", pady=10)
        
        video_label = ctk.CTkLabel(title_frame, text="📹 Video Feed", 
                                   font=ctk.CTkFont(size=18, weight="bold"))
        video_label.pack(side="left", padx=(10, 150))
        
        mask_label = ctk.CTkLabel(title_frame, text="🎨 Color Mask", 
                                 font=ctk.CTkFont(size=18, weight="bold"))
        mask_label.pack(side="left", padx=(150, 10))
        
        # Video displays side by side
        video_frame = ctk.CTkFrame(left_panel)
        video_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Video display
        self.video_canvas = ctk.CTkLabel(video_frame, text="")
        self.video_canvas.pack(side="left", padx=5)
        
        # Mask display
        self.mask_canvas = ctk.CTkLabel(video_frame, text="")
        self.mask_canvas.pack(side="left", padx=5)
        
        # Control buttons
        btn_frame = ctk.CTkFrame(left_panel)
        btn_frame.pack(pady=10)
        
        self.start_btn = ctk.CTkButton(btn_frame, text="Start Tracking", 
                                       command=self.start_tracking,
                                       font=ctk.CTkFont(size=14, weight="bold"),
                                       width=150, height=40)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ctk.CTkButton(btn_frame, text="Stop Tracking", 
                                      command=self.stop_tracking,
                                      font=ctk.CTkFont(size=14, weight="bold"),
                                      fg_color="red", hover_color="darkred",
                                      width=150, height=40, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # Right panel - Controls (Scrollable)
        right_panel = ctk.CTkScrollableFrame(main_frame, width=400)
        right_panel.grid(row=0, column=1, sticky="nsew")
        
        # HSV Controls
        hsv_frame = ctk.CTkFrame(right_panel)
        hsv_frame.pack(fill="x", padx=10, pady=10)
        
        hsv_title = ctk.CTkLabel(hsv_frame, text="🎨 HSV Color Detection", 
                                font=ctk.CTkFont(size=16, weight="bold"))
        hsv_title.pack(pady=10)
        
        # Create sliders
        self.create_slider(hsv_frame, "H Min", self.h_min, 0, 179)
        self.create_slider(hsv_frame, "S Min", self.s_min, 0, 255)
        self.create_slider(hsv_frame, "V Min", self.v_min, 0, 255)
        self.create_slider(hsv_frame, "H Max", self.h_max, 0, 179)
        self.create_slider(hsv_frame, "S Max", self.s_max, 0, 255)
        self.create_slider(hsv_frame, "V Max", self.v_max, 0, 255)
        
        # Display options
        options_frame = ctk.CTkFrame(right_panel)
        options_frame.pack(fill="x", padx=10, pady=10)
        
        options_title = ctk.CTkLabel(options_frame, text="⚙️ Display Options", 
                                     font=ctk.CTkFont(size=16, weight="bold"))
        options_title.pack(pady=10)
        
        trail_switch = ctk.CTkSwitch(options_frame, text="Show Tracking Trail", 
                                     variable=self.show_trail,
                                     font=ctk.CTkFont(size=13))
        trail_switch.pack(pady=5)
        
        # Action buttons
        action_frame = ctk.CTkFrame(right_panel)
        action_frame.pack(fill="x", padx=10, pady=10)
        
        save_btn = ctk.CTkButton(action_frame, text="Save HSV Values", 
                                command=self.save_hsv,
                                font=ctk.CTkFont(size=13),
                                fg_color="green", hover_color="darkgreen")
        save_btn.pack(fill="x", pady=5)
        
        reset_btn = ctk.CTkButton(action_frame, text="Reset Values", 
                                 command=self.reset_values,
                                 font=ctk.CTkFont(size=13))
        reset_btn.pack(fill="x", pady=5)
        
        # Status panel
        status_frame = ctk.CTkFrame(right_panel)
        status_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        status_title = ctk.CTkLabel(status_frame, text="📊 System Status", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        status_title.pack(pady=10)
        
        # Status items
        self.serial_label = self.create_status_item(status_frame, "Serial Port", "Disconnected")
        self.camera_label = self.create_status_item(status_frame, "Camera Status", "Stopped")
        self.pan_label = self.create_status_item(status_frame, "Pan Position", "0.000")
        self.tilt_label = self.create_status_item(status_frame, "Tilt Position", "0.000")
        self.ball_label = self.create_status_item(status_frame, "Ball Detected", "No")
        self.fps_label = self.create_status_item(status_frame, "FPS", "0")
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
    def create_slider(self, parent, label, variable, from_, to):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=5)
        
        label_frame = ctk.CTkFrame(frame)
        label_frame.pack(fill="x")
        
        name_label = ctk.CTkLabel(label_frame, text=label, 
                                 font=ctk.CTkFont(size=12))
        name_label.pack(side="left", padx=5)
        
        value_label = ctk.CTkLabel(label_frame, text=str(variable.get()),
                                   font=ctk.CTkFont(size=12, weight="bold"),
                                   text_color="#1f6aa5")
        value_label.pack(side="right", padx=5)
        
        slider = ctk.CTkSlider(frame, from_=from_, to=to, variable=variable,
                              command=lambda v: value_label.configure(text=str(int(v))))
        slider.pack(fill="x", padx=5, pady=2)
        
    def create_status_item(self, parent, label, initial_value):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=3, padx=10)
        
        name = ctk.CTkLabel(frame, text=label, 
                          font=ctk.CTkFont(size=11),
                          text_color="#aaa")
        name.pack(side="left", padx=5)
        
        value = ctk.CTkLabel(frame, text=initial_value,
                           font=ctk.CTkFont(size=11, weight="bold"),
                           text_color="#1f6aa5")
        value.pack(side="right", padx=5)
        
        return value
        
    def connect_serial(self):
        try:
            self.ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=1)
            time.sleep(0.5)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.serial_label.configure(text="Connected", text_color="green")
            print(f"✓ Connected to {config.SERIAL_PORT}")
        except Exception as e:
            self.serial_label.configure(text="Disconnected", text_color="red")
            print(f"✗ Serial connection failed: {e}")
            self.ser = None
            
    def circle_overlaps_rectangle(self, cx, cy, radius, rect_x1, rect_y1, rect_x2, rect_y2):
        closest_x = max(rect_x1, min(cx, rect_x2))
        closest_y = max(rect_y1, min(cy, rect_y2))
        distance_x = cx - closest_x
        distance_y = cy - closest_y
        distance_squared = (distance_x ** 2) + (distance_y ** 2)
        return distance_squared <= (radius ** 2)
        
    def start_tracking(self):
        if not self.is_tracking:
            self.is_tracking = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            
            # Initialize video stream
            try:
                self.vs = cv2.VideoCapture(config.VIDEO_SOURCE)
                time.sleep(1.0)
                self.camera_label.configure(text="Running", text_color="green")
                
                # Start tracking thread
                self.tracking_thread = threading.Thread(target=self.tracking_loop, daemon=True)
                self.tracking_thread.start()
            except Exception as e:
                print(f"Error starting video: {e}")
                self.camera_label.configure(text="Error", text_color="red")
                self.is_tracking = False
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                
    def stop_tracking(self):
        self.is_tracking = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.camera_label.configure(text="Stopped", text_color="orange")
        
        if self.vs is not None:
            self.vs.release()
            self.vs = None
            
    def tracking_loop(self):
        fps_start_time = time.time()
        fps_frame_count = 0
        
        while self.is_tracking:
            ret, frame = self.vs.read()
            if not ret or frame is None:
                continue
                
            # Process frame
            frame = imutils.resize(frame, width=config.FRAME_WIDTH)
            (H, W) = frame.shape[:2]
            centerX = W // 2
            centerY = H // 2
            
            # Color detection
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            redLower = (self.h_min.get(), self.s_min.get(), self.v_min.get())
            redUpper = (self.h_max.get(), self.s_max.get(), self.v_max.get())
            
            mask = cv2.inRange(hsv, redLower, redUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            self.mask = mask.copy()
            
            # Find contours
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None
            
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
                if radius > 10:
                    self.ball_detected = True
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, config.CIRCLE_RADIUS, (0, 0, 255), -1)
                    
                    # Calculate error
                    errorPan = center[0] - centerX
                    errorTilt = center[1] - centerY
                    
                    # Deadband
                    deadband_x1 = centerX - config.DEADBAND
                    deadband_y1 = centerY - config.DEADBAND
                    deadband_x2 = centerX + config.DEADBAND
                    deadband_y2 = centerY + config.DEADBAND
                    
                    circle_outside_deadband = not self.circle_overlaps_rectangle(
                        center[0], center[1], config.CIRCLE_RADIUS,
                        deadband_x1, deadband_y1, deadband_x2, deadband_y2
                    )
                    
                    should_move = False
                    
                    if circle_outside_deadband:
                        # PD control for pan
                        if abs(errorPan) > config.DEADBAND:
                            derivative_pan = errorPan - self.prev_errorPan
                            delta_pan = (errorPan * config.PAN_P_GAIN) + (derivative_pan * config.PAN_D_GAIN)
                            delta_pan = max(-config.MAX_SPEED, min(config.MAX_SPEED, delta_pan))
                            self.current_pan -= delta_pan
                            should_move = True
                            self.prev_errorPan = errorPan
                            
                        # PD control for tilt
                        if abs(errorTilt) > config.DEADBAND:
                            derivative_tilt = errorTilt - self.prev_errorTilt
                            delta_tilt = (errorTilt * config.TILT_P_GAIN) + (derivative_tilt * config.TILT_D_GAIN)
                            delta_tilt = max(-config.MAX_SPEED, min(config.MAX_SPEED, delta_tilt))
                            self.current_tilt -= delta_tilt
                            should_move = True
                            self.prev_errorTilt = errorTilt
                    else:
                        self.prev_errorPan = 0
                        self.prev_errorTilt = 0
                        
                    # Smoothing
                    self.smoothed_pan = config.EMA_ALPHA * self.current_pan + (1 - config.EMA_ALPHA) * self.smoothed_pan
                    self.smoothed_tilt = config.EMA_ALPHA * self.current_tilt + (1 - config.EMA_ALPHA) * self.smoothed_tilt
                    self.smoothed_pan = max(-1.0, min(1.0, self.smoothed_pan))
                    self.smoothed_tilt = max(-1.0, min(1.0, self.smoothed_tilt))
                    
                    # Send serial command
                    if should_move and self.ser is not None and self.ser.is_open:
                        current_time = time.time()
                        if current_time - self.last_command_time >= config.COMMAND_INTERVAL:
                            command = "{:.3f} {:.3f}\n".format(self.smoothed_pan, self.smoothed_tilt)
                            try:
                                self.ser.write(command.encode('utf-8'))
                                self.last_command_time = current_time
                            except Exception as e:
                                print(f"Serial write error: {e}")
                                
                    # Draw deadband
                    deadband_color = (0, 255, 0) if circle_outside_deadband else (0, 165, 255)
                    cv2.rectangle(frame, (deadband_x1, deadband_y1),
                                (deadband_x2, deadband_y2), deadband_color, 2)
                else:
                    self.ball_detected = False
            else:
                self.ball_detected = False
                self.prev_errorPan = 0
                self.prev_errorTilt = 0
                cv2.rectangle(frame, (centerX - config.DEADBAND, centerY - config.DEADBAND),
                            (centerX + config.DEADBAND, centerY + config.DEADBAND), (0, 255, 0), 1)
                            
            # Draw crosshair
            cv2.line(frame, (centerX - 10, centerY), (centerX + 10, centerY), (255, 0, 0), 1)
            cv2.line(frame, (centerX, centerY - 10), (centerX, centerY + 10), (255, 0, 0), 1)
            
            # Draw trail
            if self.show_trail.get():
                self.pts.appendleft(center)
                for i in range(1, len(self.pts)):
                    if self.pts[i - 1] is None or self.pts[i] is None:
                        continue
                    thickness = int(np.sqrt(config.BUFFER_SIZE / float(i + 1)) * 2.5)
                    cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)
                    
            # Update display
            self.frame = frame
            self.update_display()
            
            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                self.fps = fps_frame_count
                fps_frame_count = 0
                fps_start_time = time.time()
                
            # Update status
            self.update_status()
            
            time.sleep(0.01)
            
    def update_display(self):
        # Update video feed
        if self.frame is not None:
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((400, 300), Image.LANCZOS)
            
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(400, 300))
            self.video_canvas.configure(image=ctk_img)
            self.video_canvas.image = ctk_img
        
        # Update mask feed - NO POPUP, display in GUI
        if self.mask is not None:
            # Convert grayscale mask to RGB for display
            mask_rgb = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB)
            mask_img = Image.fromarray(mask_rgb)
            mask_img = mask_img.resize((400, 300), Image.LANCZOS)
            
            ctk_mask_img = ctk.CTkImage(light_image=mask_img, dark_image=mask_img, size=(400, 300))
            self.mask_canvas.configure(image=ctk_mask_img)
            self.mask_canvas.image = ctk_mask_img
            
    def update_status(self):
        self.pan_label.configure(text=f"{self.smoothed_pan:.3f}")
        self.tilt_label.configure(text=f"{self.smoothed_tilt:.3f}")
        self.ball_label.configure(
            text="Yes" if self.ball_detected else "No",
            text_color="green" if self.ball_detected else "red"
        )
        self.fps_label.configure(text=str(self.fps))
        
    def save_hsv(self):
        print("\n" + "="*50)
        print("SAVED HSV VALUES:")
        print(f"  redLower = ({self.h_min.get()}, {self.s_min.get()}, {self.v_min.get()})")
        print(f"  redUpper = ({self.h_max.get()}, {self.s_max.get()}, {self.v_max.get()})")
        print("="*50 + "\n")
        
    def reset_values(self):
        self.h_min.set(config.HSV_INITIAL_H_MIN)
        self.s_min.set(config.HSV_INITIAL_S_MIN)
        self.v_min.set(config.HSV_INITIAL_V_MIN)
        self.h_max.set(config.HSV_INITIAL_H_MAX)
        self.s_max.set(config.HSV_INITIAL_S_MAX)
        self.v_max.set(config.HSV_INITIAL_V_MAX)
        
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        self.is_tracking = False
        if self.vs is not None:
            self.vs.release()
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        self.root.destroy()

if __name__ == "__main__":
    app = BallTrackingGUI()
    app.run()
