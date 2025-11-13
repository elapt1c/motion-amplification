import cv2
import numpy as np
from scipy.signal import butter
from concurrent.futures import ThreadPoolExecutor
import time
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, Scale
import threading
import tempfile
from PIL import Image, ImageTk # Used for displaying images in Tkinter

# --- Original Motion Amplification Functions ---

def ComputeLaplacianPyramid(frame, max_levels):
    G = frame.copy()
    gpA = [G]
    for i in range(max_levels):
        G = cv2.pyrDown(G)
        gpA.append(G)
    
    lpA = [gpA[-1]]
    for i in range(len(gpA)-1, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.resize(GE, size)
        L = cv2.subtract(gpA[i-1], GE)
        lpA.append(L)
    
    lpA.reverse()
    return lpA

def ComputeRieszPyramid(frame, max_levels):
    laplacian_pyramid = ComputeLaplacianPyramid(frame, max_levels)
    number_of_levels = len(laplacian_pyramid) - 1

    kernel_x = np.array([[0.0, 0.0, 0.0],
                         [0.5, 0.0, -0.5],
                         [0.0, 0.0, 0.0]], dtype=np.float32)

    kernel_y = np.array([[0.0, 0.5, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, -0.5, 0.0]], dtype=np.float32)

    riesz_x = []
    riesz_y = []

    for k in range(number_of_levels):
        rx = cv2.filter2D(laplacian_pyramid[k], -1, kernel_x, borderType=cv2.BORDER_REFLECT)
        ry = cv2.filter2D(laplacian_pyramid[k], -1, kernel_y, borderType=cv2.BORDER_REFLECT)
        riesz_x.append(rx)
        riesz_y.append(ry)

    return laplacian_pyramid, riesz_x, riesz_y

def ComputePhaseDifferenceAndAmplitude(current_real, current_x, current_y, previous_real, previous_x, previous_y):
    q_conj_prod_real = current_real * previous_real + current_x * previous_x + current_y * previous_y
    q_conj_prod_x = -current_real * previous_x + previous_real * current_x
    q_conj_prod_y = -current_real * previous_y + previous_real * current_y

    q_conj_prod_amplitude = np.sqrt(q_conj_prod_real ** 2 + q_conj_prod_x ** 2 + q_conj_prod_y ** 2) + 1e-8

    phase_difference = np.arccos(np.clip(q_conj_prod_real / q_conj_prod_amplitude, -1, 1))

    denom_orientation = np.sqrt(q_conj_prod_x ** 2 + q_conj_prod_y ** 2) + 1e-8
    cos_orientation = q_conj_prod_x / denom_orientation
    sin_orientation = q_conj_prod_y / denom_orientation

    phase_difference_cos = phase_difference * cos_orientation
    phase_difference_sin = phase_difference * sin_orientation

    amplitude = np.sqrt(q_conj_prod_amplitude)

    return phase_difference_cos, phase_difference_sin, amplitude

def IIRTemporalFilter(B, A, phase, register0, register1):
    temporally_filtered_phase = B[0] * phase + register0
    register0_new = B[1] * phase + register1 - A[1] * temporally_filtered_phase
    register1_new = B[2] * phase - A[2] * temporally_filtered_phase

    return temporally_filtered_phase, register0_new, register1_new

def AmplitudeWeightedBlur(temporally_filtered_phase, amplitude, blur_kernel):
    numerator = cv2.filter2D(temporally_filtered_phase * amplitude, -1, blur_kernel, borderType=cv2.BORDER_REFLECT)
    denominator = cv2.filter2D(amplitude, -1, blur_kernel, borderType=cv2.BORDER_REFLECT) + 1e-8
    spatially_smooth_temporally_filtered_phase = numerator / denominator

    return spatially_smooth_temporally_filtered_phase

def PhaseShiftCoefficientRealPart(riesz_real, riesz_x, riesz_y, phase_cos, phase_sin):
    phase_magnitude = np.sqrt(phase_cos ** 2 + phase_sin ** 2) + 1e-8
    exp_phase_real = np.cos(phase_magnitude)
    sin_phase_magnitude = np.sin(phase_magnitude)
    exp_phase_x = phase_cos / phase_magnitude * sin_phase_magnitude
    exp_phase_y = phase_sin / phase_magnitude * sin_phase_magnitude

    result = exp_phase_real * riesz_real - exp_phase_x * riesz_x - exp_phase_y * riesz_y

    return result

def CollapseLaplacianPyramid(pyramid):
    current = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        upsampled = cv2.pyrUp(current)
        size = (level.shape[1], level.shape[0])
        upsampled = cv2.resize(upsampled, size)
        current = upsampled + level

    return current

# Modified process_pyramid_level to return updated states explicitly
def process_pyramid_level(args):
    (
        k,
        current_laplacian,
        current_riesz_x,
        current_riesz_y,
        previous_laplacian,
        previous_riesz_x,
        previous_riesz_y,
        phase_cos_k_prev, # Renamed to show they are previous states
        phase_sin_k_prev,
        register0_cos_k_prev,
        register1_cos_k_prev,
        register0_sin_k_prev,
        register1_sin_k_prev,
        B, A,
        amplification_factor,
        gaussian_kernel_2d
    ) = args

    phase_difference_cos, phase_difference_sin, amplitude = ComputePhaseDifferenceAndAmplitude(
        current_laplacian,
        current_riesz_x,
        current_riesz_y,
        previous_laplacian,
        previous_riesz_x,
        previous_riesz_y
    )

    # Accumulate phase difference
    phase_cos_k_current = phase_cos_k_prev + phase_difference_cos
    phase_sin_k_current = phase_sin_k_prev + phase_difference_sin

    # Apply IIR Temporal Filter
    phase_filtered_cos, register0_cos_k_new, register1_cos_k_new = IIRTemporalFilter(B, A, phase_cos_k_current, register0_cos_k_prev, register1_cos_k_prev)
    phase_filtered_sin, register0_sin_k_new, register1_sin_k_new = IIRTemporalFilter(B, A, phase_sin_k_current, register0_sin_k_prev, register1_sin_k_prev)

    # Apply Amplitude Weighted Blur
    phase_filtered_cos_smooth = AmplitudeWeightedBlur(phase_filtered_cos, amplitude, gaussian_kernel_2d)
    phase_filtered_sin_smooth = AmplitudeWeightedBlur(phase_filtered_sin, amplitude, gaussian_kernel_2d)

    # Magnify phase
    phase_magnified_filtered_cos = amplification_factor * phase_filtered_cos_smooth
    phase_magnified_filtered_sin = amplification_factor * phase_filtered_sin_smooth

    # Compute phase-shifted Laplacian coefficient
    motion_magnified_coeff = PhaseShiftCoefficientRealPart(
        current_laplacian,
        current_riesz_x,
        current_riesz_y,
        phase_magnified_filtered_cos,
        phase_magnified_filtered_sin
    )

    return {
        'level': k,
        'motion_magnified_coeff': motion_magnified_coeff,
        'phase_cos': phase_cos_k_current, # Return the accumulated phase
        'phase_sin': phase_sin_k_current,
        'register0_cos': register0_cos_k_new,
        'register1_cos': register1_cos_k_new,
        'register0_sin': register0_sin_k_new,
        'register1_sin': register1_sin_k_new
    }


# --- Tkinter GUI Application for Live Amplification ---

class LiveVideoAmplifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Riesz Phase Live Video Magnifier")
        master.geometry("1000x700") # Larger window for side-by-side video

        self.live_running = False
        self.cap_live = None
        self.thread_pool_executor = ThreadPoolExecutor()

        # State for live processing (instance variables)
        self.previous_laplacian_pyramid = None
        self.previous_riesz_x = None
        self.previous_riesz_y = None
        self.phase_cos = None
        self.phase_sin = None
        self.register0_cos = None
        self.register1_cos = None
        self.register0_sin = None
        self.register1_sin = None
        self.B = None
        self.A = None
        self.gaussian_kernel_2d = None
        self.max_levels = 4 # Fixed for now, can be made configurable

        # Camera selection variables
        self.available_cameras = []
        self.selected_camera_idx_var = tk.StringVar(master)
        
        self.setup_gui()
        self._detect_cameras_and_update_dropdown() # Detect cameras after GUI is set up

    def _get_available_cameras(self):
        """Detects available webcam devices by trying common indices."""
        detected_indices = []
        # Use CAP_DSHOW for Windows, often more reliable enumeration
        # For other OS, CAP_ANY (default) should work.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY 
        
        self.update_status("Detecting cameras...")
        # Check a reasonable range of indices (e.g., 0 to 9)
        for i in range(10): 
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Attempt to read a frame to ensure it's a truly functional camera
                    # Some systems might list non-functional or reserved devices.
                    ret, _ = cap.read() 
                    if ret:
                        detected_indices.append(str(i)) # Store as string for Tkinter StringVar
                        self.update_status(f"Found camera: {i}")
                    cap.release()
                else:
                    cap.release() # Ensure release even if it didn't open
            except Exception as e:
                # Catch any unexpected errors during camera access
                self.update_status(f"Error checking camera {i}: {e}")
                pass # Continue to next index
        
        if not detected_indices:
            self.update_status("No cameras detected.")
            return ["No cameras found"] # Provide a default non-functional option
        
        return detected_indices

    def _detect_cameras_and_update_dropdown(self):
        """Called after GUI setup to populate the camera selection dropdown."""
        self.available_cameras = self._get_available_cameras()
        
        # Clear existing options in case of re-detection (though not currently implemented)
        self.camera_option_menu['menu'].delete(0, 'end')

        if self.available_cameras and self.available_cameras[0] != "No cameras found":
            self.selected_camera_idx_var.set(self.available_cameras[0]) # Set default to first detected
            for camera_id in self.available_cameras:
                self.camera_option_menu['menu'].add_command(label=camera_id, 
                                                              command=tk._setit(self.selected_camera_idx_var, camera_id))
            self.start_live_button.config(state=tk.NORMAL) # Enable start button if cameras are found
        else:
            self.selected_camera_idx_var.set("No cameras found")
            self.camera_option_menu['menu'].add_command(label="No cameras found", state=tk.DISABLED)
            self.start_live_button.config(state=tk.DISABLED) # Disable start button if no camera

    def setup_gui(self):
        # Parameters Frame
        params_frame = tk.LabelFrame(self.master, text="Parameters", padx=10, pady=10)
        params_frame.pack(side="top", padx=10, pady=10, fill="x")

        tk.Label(params_frame, text="Amplification Factor:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.amp_factor_var = tk.StringVar(value="20")
        tk.Entry(params_frame, textvariable=self.amp_factor_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(params_frame, text="Low Cutoff (Hz):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.low_cutoff_var = tk.StringVar(value="0.4")
        tk.Entry(params_frame, textvariable=self.low_cutoff_var).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(params_frame, text="High Cutoff (Hz):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.high_cutoff_var = tk.StringVar(value="3.0")
        tk.Entry(params_frame, textvariable=self.high_cutoff_var).grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        
        params_frame.grid_columnconfigure(1, weight=1) # Make the entry fields expand

        # Camera Selection Frame
        camera_frame = tk.LabelFrame(self.master, text="Camera Selection", padx=10, pady=5)
        camera_frame.pack(side="top", padx=10, pady=5, fill="x")

        tk.Label(camera_frame, text="Select Camera:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        
        # OptionMenu for camera selection (initially empty, will be populated by _detect_cameras_and_update_dropdown)
        self.camera_option_menu = tk.OptionMenu(camera_frame, self.selected_camera_idx_var, "")
        self.camera_option_menu.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        camera_frame.grid_columnconfigure(1, weight=1)

        # Control Buttons Frame
        control_frame = tk.Frame(self.master, padx=10, pady=10)
        control_frame.pack(side="top", padx=10, pady=10, fill="x")

        self.start_live_button = tk.Button(control_frame, text="Start Live Amplification", command=self.start_live_amplification, state=tk.DISABLED) # Start disabled
        self.start_live_button.pack(side="left", padx=5, pady=5, expand=True)

        self.stop_live_button = tk.Button(control_frame, text="Stop Live Amplification", command=self.stop_live_amplification, state=tk.DISABLED)
        self.stop_live_button.pack(side="left", padx=5, pady=5, expand=True)

        # Video Display Frames
        video_display_frame = tk.Frame(self.master, padx=10, pady=10)
        video_display_frame.pack(side="top", fill="both", expand=True)

        # Original Feed Label
        original_frame = tk.LabelFrame(video_display_frame, text="Original Feed")
        original_frame.pack(side="left", padx=5, pady=5, fill="both", expand=True)
        self.raw_feed_label = tk.Label(original_frame)
        self.raw_feed_label.pack(fill="both", expand=True)

        # Amplified Feed Label
        amplified_frame = tk.LabelFrame(video_display_frame, text="Amplified Feed")
        amplified_frame.pack(side="right", padx=5, pady=5, fill="both", expand=True)
        self.amplified_feed_label = tk.Label(amplified_frame)
        self.amplified_feed_label.pack(fill="both", expand=True)

        # Status Log
        self.log_frame = tk.LabelFrame(self.master, text="Status Log", padx=10, pady=10)
        self.log_frame.pack(side="bottom", padx=10, pady=10, fill="x")
        self.status_log = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD, height=5)
        self.status_log.pack(padx=5, pady=5, fill="x")
        self.status_log.config(state=tk.DISABLED) # Make it read-only

    def update_status(self, message):
        # Schedule the update on the main Tkinter thread
        self.master.after(0, self._perform_update_status, message)

    def _perform_update_status(self, message):
        self.status_log.config(state=tk.NORMAL)
        self.status_log.insert(tk.END, message + "\n")
        self.status_log.see(tk.END) # Scroll to the end
        self.status_log.config(state=tk.DISABLED)

    def start_live_amplification(self):
        if self.live_running:
            return

        selected_camera_str = self.selected_camera_idx_var.get()
        if selected_camera_str == "No cameras found":
            messagebox.showerror("Camera Error", "No camera selected or found. Cannot start live amplification.")
            self.update_status("Cannot start live amplification: No camera selected or found.")
            return

        try:
            camera_index = int(selected_camera_str)
        except ValueError:
            messagebox.showerror("Camera Error", "Invalid camera selection. Please select a valid camera index.")
            self.update_status("Invalid camera selection.")
            return

        try:
            self.amplification_factor = float(self.amp_factor_var.get())
            self.low_cutoff = float(self.low_cutoff_var.get())
            self.high_cutoff = float(self.high_cutoff_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for amplification, low, and high cutoffs.")
            return

        self.update_status(f"Starting live amplification with camera {camera_index}...")
        # Use CAP_DSHOW for Windows, often more reliable
        self.cap_live = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not self.cap_live.isOpened():
            self.update_status(f"Error: Could not open camera {camera_index}. It might be in use or not accessible by OpenCV.")
            messagebox.showerror("Camera Error", f"Could not open camera {camera_index}.")
            return
        
        # Get camera properties
        self.fps = self.cap_live.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap_live.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap_live.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Fallback for unstable FPS from some webcams
        if self.fps == 0:
            self.fps = 30
            self.update_status("Warning: Camera FPS reported as 0, defaulting to 30 FPS for filter calculation.")
        
        # Initialize filter parameters
        nyquist_frequency = self.fps / 2.0
        temporal_filter_order = 1
        # Ensure cutoff frequencies are within valid range (0 to Nyquist)
        low_norm = np.clip(self.low_cutoff / nyquist_frequency, 0.01, 0.99)
        high_norm = np.clip(self.high_cutoff / nyquist_frequency, 0.01, 0.99)

        if low_norm >= high_norm:
            self.update_status(f"Warning: Low cutoff ({self.low_cutoff} Hz) is not less than high cutoff ({self.high_cutoff} Hz). Adjusting values slightly.")
            # Adjust to ensure valid bandpass if they are too close or inverted
            if low_norm == high_norm:
                low_norm = max(0.01, low_norm - 0.01)
                high_norm = min(0.99, high_norm + 0.01)
            # If low is still greater than high after adjustment (shouldn't happen with clip but as a safeguard)
            if low_norm > high_norm: 
                low_norm, high_norm = high_norm, low_norm

        self.B, self.A = butter(temporal_filter_order, [low_norm, high_norm], btype='bandpass')
        self.B = self.B.astype(np.float32)
        self.A = self.A.astype(np.float32)

        gaussian_kernel_sd = 2
        gaussian_kernel_size = int(gaussian_kernel_sd * 6 + 1)
        if gaussian_kernel_size % 2 == 0: gaussian_kernel_size += 1 # Ensure odd kernel size
        gaussian_kernel_1d = cv2.getGaussianKernel(gaussian_kernel_size, gaussian_kernel_sd)
        self.gaussian_kernel_2d = gaussian_kernel_1d * gaussian_kernel_1d.T

        # Read the first frame to initialize previous pyramid components
        ret, first_frame_raw = self.cap_live.read()
        if not ret:
            self.update_status("Failed to read first frame from camera. Stopping.")
            self.cap_live.release()
            return
        
        first_frame_float = first_frame_raw.astype(np.float32) / 255.0
        self.previous_laplacian_pyramid, self.previous_riesz_x, self.previous_riesz_y = ComputeRieszPyramid(first_frame_float, self.max_levels)

        number_of_levels = len(self.previous_laplacian_pyramid) - 1

        # Initialize phase and registers for all pyramid levels
        self.phase_cos = [None] * number_of_levels
        self.phase_sin = [None] * number_of_levels
        self.register0_cos = [None] * number_of_levels
        self.register1_cos = [None] * number_of_levels
        self.register0_sin = [None] * number_of_levels
        self.register1_sin = [None] * number_of_levels

        for k in range(number_of_levels):
            size = self.previous_laplacian_pyramid[k].shape
            self.phase_cos[k] = np.zeros(size, dtype=np.float32)
            self.phase_sin[k] = np.zeros(size, dtype=np.float32)
            self.register0_cos[k] = np.zeros(size, dtype=np.float32)
            self.register1_cos[k] = np.zeros(size, dtype=np.float32)
            self.register0_sin[k] = np.zeros(size, dtype=np.float32)
            self.register1_sin[k] = np.zeros(size, dtype=np.float32)

        self.live_running = True
        self.start_live_button.config(state=tk.DISABLED)
        self.stop_live_button.config(state=tk.NORMAL)

        # Start the processing loop in a separate thread
        self.live_thread = threading.Thread(target=self._live_capture_and_process_loop)
        self.live_thread.daemon = True
        self.live_thread.start()
        self.update_status("Live amplification started.")
    
    def _live_capture_and_process_loop(self):
        while self.live_running:
            ret, frame_raw = self.cap_live.read()
            if not ret:
                self.update_status("Failed to read frame from camera. Stopping live amplification.")
                self.stop_live_amplification()
                break

            # Convert frame for processing (0-1 float, BGR)
            current_frame_float = frame_raw.astype(np.float32) / 255.0

            # Compute pyramids for current frame
            current_laplacian_pyramid, current_riesz_x, current_riesz_y = ComputeRieszPyramid(current_frame_float, self.max_levels)
            
            number_of_levels = len(self.previous_laplacian_pyramid) - 1 # Should be consistent

            motion_magnified_laplacian_pyramid = [None] * (number_of_levels + 1)

            # Prepare args for parallel processing of pyramid levels
            args_list = []
            for k in range(number_of_levels):
                args_list.append((
                    k,
                    current_laplacian_pyramid[k],
                    current_riesz_x[k],
                    current_riesz_y[k],
                    self.previous_laplacian_pyramid[k],
                    self.previous_riesz_x[k],
                    self.previous_riesz_y[k],
                    self.phase_cos[k],
                    self.phase_sin[k],
                    self.register0_cos[k],
                    self.register1_cos[k],
                    self.register0_sin[k],
                    self.register1_sin[k],
                    self.B, self.A,
                    self.amplification_factor,
                    self.gaussian_kernel_2d
                ))

            # Process pyramid levels in parallel
            try:
                results = list(self.thread_pool_executor.map(process_pyramid_level, args_list))
            except Exception as e:
                self.update_status(f"Error during parallel processing of pyramid levels: {e}")
                self.stop_live_amplification()
                break

            # Collect results and update state
            for res in results:
                k = res['level']
                motion_magnified_laplacian_pyramid[k] = res['motion_magnified_coeff']
                self.phase_cos[k] = res['phase_cos']
                self.phase_sin[k] = res['phase_sin']
                self.register0_cos[k] = res['register0_cos']
                self.register1_cos[k] = res['register1_cos']
                self.register0_sin[k] = res['register0_sin']
                self.register1_sin[k] = res['register1_sin']

            # The coarsest level (base of the Laplacian pyramid) is not motion magnified
            motion_magnified_laplacian_pyramid[number_of_levels] = current_laplacian_pyramid[number_of_levels]
            
            # Collapse the modified Laplacian pyramid to get the amplified frame
            amplified_frame_float = CollapseLaplacianPyramid(motion_magnified_laplacian_pyramid)
            amplified_frame_float = np.clip(amplified_frame_float, 0, 1) # Clip values to [0, 1]
            amplified_frame_uint8 = (amplified_frame_float * 255).astype(np.uint8)

            # Update previous states for next frame
            self.previous_laplacian_pyramid = current_laplacian_pyramid
            self.previous_riesz_x = current_riesz_x
            self.previous_riesz_y = current_riesz_y

            # Display frames (schedule on main Tkinter thread)
            self.master.after(1, lambda: self._update_display(frame_raw, amplified_frame_uint8))

            # Introduce a small sleep to prevent 100% CPU usage if processing is very fast
            # and to allow the GUI to update smoothly. Adjust as needed.
            # This sleep helps ensure the GUI thread gets enough time to process events.
            time.sleep(0.001)

        self.update_status("Live capture and process loop finished.")

    def _update_display(self, raw_frame_bgr, amplified_frame_bgr):
        # Resize frames to fit the labels if necessary
        # We target a fixed height, and calculate width to maintain aspect ratio
        target_height = 300 # Max height for display frames
        
        if raw_frame_bgr.shape[0] > 0: # Ensure frame is not empty
            aspect_ratio = raw_frame_bgr.shape[1] / raw_frame_bgr.shape[0]
            display_height = target_height
            display_width = int(target_height * aspect_ratio)

            # Convert OpenCV BGR image to RGB for PIL
            raw_frame_rgb = cv2.cvtColor(raw_frame_bgr, cv2.COLOR_BGR2RGB)
            amplified_frame_rgb = cv2.cvtColor(amplified_frame_bgr, cv2.COLOR_BGR2RGB)

            # Convert NumPy arrays to PIL Image
            img_raw = Image.fromarray(raw_frame_rgb)
            img_amplified = Image.fromarray(amplified_frame_rgb)

            # Resize for display using LANCZOS for quality
            img_raw = img_raw.resize((display_width, display_height), Image.LANCZOS)
            img_amplified = img_amplified.resize((display_width, display_height), Image.LANCZOS)

            # Convert PIL Image to ImageTk.PhotoImage
            photo_raw = ImageTk.PhotoImage(image=img_raw)
            photo_amplified = ImageTk.PhotoImage(image=img_amplified)

            # Update labels
            self.raw_feed_label.config(image=photo_raw)
            self.raw_feed_label.image = photo_raw # Keep a reference!
            self.amplified_feed_label.config(image=photo_amplified)
            self.amplified_feed_label.image = photo_amplified # Keep a reference!

    def stop_live_amplification(self):
        if not self.live_running:
            return

        self.live_running = False
        self.update_status("Stopping live amplification...")

        # Wait for the live thread to finish
        if self.live_thread and self.live_thread.is_alive():
            self.live_thread.join(timeout=5) # Give it a few seconds to finish
            if self.live_thread.is_alive():
                self.update_status("Warning: Live processing thread did not terminate gracefully.")

        if self.cap_live:
            self.cap_live.release()
            self.cap_live = None
        
        # Clear displayed images
        self.raw_feed_label.config(image='')
        self.amplified_feed_label.config(image='')
        self.raw_feed_label.image = None
        self.amplified_feed_label.image = None

        # Reset state variables
        self.previous_laplacian_pyramid = None
        self.previous_riesz_x = None
        self.previous_riesz_y = None
        self.phase_cos = None
        self.phase_sin = None
        self.register0_cos = None
        self.register1_cos = None
        self.register0_sin = None
        self.register1_sin = None

        self.start_live_button.config(state=tk.NORMAL)
        self.stop_live_button.config(state=tk.DISABLED)
        self.update_status("Live amplification stopped.")

    # Override destroy to ensure thread pool is shut down
    def on_closing(self):
        self.stop_live_amplification()
        self.thread_pool_executor.shutdown(wait=True)
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveVideoAmplifierApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing) # Handle window close event
    root.mainloop()
