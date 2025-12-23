import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from datetime import datetime
from typing import Optional

from src.camera import Camera
from src.model import PhoneDetectionModel, LocalPhoneDetectionModel
from config.config import Config


class PhoneDetectionApp:
    """Main GUI application for phone detection"""

    def __init__(self, root: tk.Tk):
        """
        Initialize the application

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title(Config.WINDOW_TITLE)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Set window size
        self.root.geometry("800x900")
        self.root.resizable(True, True)

        # Initialize camera
        self.camera = Camera(
            camera_source=Config.CAMERA_INDEX,
            width=Config.CAMERA_WIDTH,
            height=Config.CAMERA_HEIGHT
        )

        # Initialize model (API or Local)
        if Config.USE_LOCAL_MODEL:
            print("Using LOCAL YOLOv8 model")
            self.model = LocalPhoneDetectionModel(
                model_path=Config.LOCAL_MODEL_PATH,
                confidence=Config.CONFIDENCE_THRESHOLD
            )
            self.model_type = "Local"
        else:
            print("Using Roboflow API model")
            self.model = PhoneDetectionModel(
                api_key=Config.ROBOFLOW_API_KEY,
                workspace=Config.ROBOFLOW_WORKSPACE,
                project=Config.ROBOFLOW_PROJECT,
                version=Config.ROBOFLOW_VERSION,
                confidence=Config.CONFIDENCE_THRESHOLD,
                overlap=Config.OVERLAP_THRESHOLD
            )
            self.model_type = "API"

        # Application state
        self.is_running = False
        self.is_processing = False
        self.current_frame = None
        self.current_result = None
        self.last_inference_time = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Create screenshots directory
        self.screenshots_dir = "screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Max video display size
        self.max_video_width = 640
        self.max_video_height = 480

        # Setup UI
        self.setup_ui()

        # Load model in background
        self.load_model_async()

    def setup_ui(self):
        """Setup the user interface"""
        # Create main canvas with scrollbar
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Main container
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Video frame with fixed size
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        video_frame.pack(pady=5, fill="x")

        # Video display with fixed size
        self.video_label = ttk.Label(video_frame, text="Camera not started", anchor="center")
        self.video_label.pack(pady=5)
        # Set minimum size for video label
        self.video_label.config(width=80, height=30)

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Detection Status", padding="10")
        status_frame.pack(pady=5, fill="x")

        # Create grid for status items
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill="x")

        # Detection result
        ttk.Label(status_grid, text="Result:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.result_label = ttk.Label(status_grid, text="Waiting...", font=('Arial', 11, 'bold'))
        self.result_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Confidence
        ttk.Label(status_grid, text="Confidence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.confidence_label = ttk.Label(status_grid, text="0.00%")
        self.confidence_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Inference time
        ttk.Label(status_grid, text="Inference Time:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.time_label = ttk.Label(status_grid, text="0.00s")
        self.time_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # FPS
        ttk.Label(status_grid, text="Video FPS:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.fps_label = ttk.Label(status_grid, text="0")
        self.fps_label.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # Processing status
        ttk.Label(status_grid, text="Status:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.processing_label = ttk.Label(status_grid, text="Idle")
        self.processing_label.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        # Control buttons frame
        button_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        button_frame.pack(pady=5, fill="x")

        # First row of buttons
        button_row1 = ttk.Frame(button_frame)
        button_row1.pack(fill="x", pady=2)

        self.start_button = ttk.Button(
            button_row1,
            text="▶ Start Camera",
            command=self.start_detection,
            width=20
        )
        self.start_button.pack(side="left", padx=5, pady=2)

        self.stop_button = ttk.Button(
            button_row1,
            text="⏹ Stop Camera",
            command=self.stop_detection,
            state=tk.DISABLED,
            width=20
        )
        self.stop_button.pack(side="left", padx=5, pady=2)

        self.snapshot_button = ttk.Button(
            button_row1,
            text="📷 Take Snapshot & Detect",
            command=self.take_snapshot,
            state=tk.DISABLED,
            width=25
        )
        self.snapshot_button.pack(side="left", padx=5, pady=2)

        # Second row of buttons
        button_row2 = ttk.Frame(button_frame)
        button_row2.pack(fill="x", pady=2)

        self.save_screenshot_button = ttk.Button(
            button_row2,
            text="💾 Save Screenshot",
            command=self.save_screenshot,
            state=tk.DISABLED,
            width=20
        )
        self.save_screenshot_button.pack(side="left", padx=5, pady=2)

        self.clear_results_button = ttk.Button(
            button_row2,
            text="🗑 Clear Results",
            command=self.clear_results,
            state=tk.DISABLED,
            width=20
        )
        self.clear_results_button.pack(side="left", padx=5, pady=2)

        # Info label
        info_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        info_frame.pack(pady=5, fill="x")
        
        info_text = (
            "📋 How to use:\n"
            "1. Click 'Start Camera' to begin video stream\n"
            "2. Click 'Take Snapshot & Detect' to run phone detection\n"
            "3. Click 'Save Screenshot' to save current frame with detections\n"
            "4. Click 'Clear Results' to remove detections and try again"
        )
        info_label = ttk.Label(
            info_frame,
            text=info_text,
            font=('Arial', 9),
            foreground="blue",
            justify=tk.LEFT
        )
        info_label.pack()

        # Model status
        status_bar = ttk.Frame(main_frame)
        status_bar.pack(pady=5, fill="x")
        
        self.model_status_label = ttk.Label(
            status_bar,
            text="⏳ Loading model...",
            foreground="orange",
            font=('Arial', 9, 'bold')
        )
        self.model_status_label.pack()

        # Configure grid weights
        status_grid.columnconfigure(1, weight=1)

    def load_model_async(self):
        """Load model in background thread"""
        def load():
            success = self.model.load_model()
            if success:
                mode_text = f"✅ {self.model_type} model loaded successfully - Ready to use"
                self.model_status_label.config(
                    text=mode_text,
                    foreground="green"
                )
            else:
                self.model_status_label.config(
                    text="❌ Failed to load model",
                    foreground="red"
                )
                messagebox.showerror(
                    "Error",
                    "Failed to load model. Please check your API key and internet connection."
                )

        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def start_detection(self):
        """Start camera and continuous detection"""
        if not self.model.is_loaded:
            messagebox.showwarning(
                "Warning",
                "Model is not loaded yet. Please wait..."
            )
            return

        if self.camera.start():
            self.is_running = True
            self.last_inference_time = 0
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.snapshot_button.config(state=tk.NORMAL)
            self.save_screenshot_button.config(state=tk.NORMAL)

            # Keep window on top and focused
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after(100, lambda: self.root.attributes('-topmost', False))

            self.update_video()
        else:
            messagebox.showerror(
                "Error",
                "Failed to start camera. Please check if camera is connected."
            )

    def stop_detection(self):
        """Stop camera and detection"""
        self.is_running = False
        self.camera.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.snapshot_button.config(state=tk.DISABLED)
        self.save_screenshot_button.config(state=tk.DISABLED)
        
        # Reset video display
        self.video_label.config(image='', text="Camera stopped")

    def take_snapshot(self):
        """Take a single snapshot and run detection"""
        if not self.model.is_loaded:
            messagebox.showwarning(
                "Warning",
                "Model is not loaded yet. Please wait..."
            )
            return

        if self.current_frame is None:
            messagebox.showwarning(
                "Warning",
                "No frame available. Please start camera first."
            )
            return

        if self.is_processing:
            messagebox.showinfo(
                "Info",
                "Already processing a frame. Please wait..."
            )
            return

        # Disable snapshot button while processing to prevent multiple requests
        self.snapshot_button.config(state=tk.DISABLED)
        
        # Run inference in background thread
        self.run_inference_async(self.current_frame.copy())

    def save_screenshot(self):
        """Save current frame with detections to file"""
        if self.current_frame is None:
            messagebox.showwarning(
                "Warning",
                "No frame available to save."
            )
            return

        # Get frame with or without detections
        if self.current_result is not None:
            frame_to_save = self.model.draw_predictions(self.current_frame, self.current_result)
        else:
            frame_to_save = self.current_frame

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add detection info to filename
        if self.current_result and self.current_result['has_phone']:
            label = self.current_result['label'].replace(' ', '_')
            conf = int(self.current_result['confidence'] * 100)
            filename = f"screenshot_{timestamp}_{label}_{conf}pct.jpg"
        else:
            filename = f"screenshot_{timestamp}_no_phone.jpg"
        
        filepath = os.path.join(self.screenshots_dir, filename)

        # Save image
        success = cv2.imwrite(filepath, frame_to_save)
        
        if success:
            messagebox.showinfo(
                "Success",
                f"Screenshot saved to:\n{filepath}"
            )
        else:
            messagebox.showerror(
                "Error",
                "Failed to save screenshot."
            )

    def clear_results(self):
        """Clear detection results"""
        self.current_result = None
        self.result_label.config(text="Waiting...", foreground="black")
        self.confidence_label.config(text="0.00%")
        self.time_label.config(text="0.00s")
        self.clear_results_button.config(state=tk.DISABLED)

    def resize_frame_for_display(self, frame):
        """
        Resize frame to fit in display area while maintaining aspect ratio
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        
        # Calculate scaling factor
        width_scale = self.max_video_width / width
        height_scale = self.max_video_height / height
        scale = min(width_scale, height_scale, 1.0)  # Don't upscale
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize if needed
        if scale < 1.0:
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return frame

    def update_video(self):
        """Update video frame (called frequently for smooth video)"""
        if not self.is_running:
            return

        # Read frame from camera
        ret, frame = self.camera.read_frame()

        if ret:
            self.current_frame = frame

            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed > 1.0:
                self.current_fps = self.frame_count / elapsed
                self.fps_label.config(text=f"{self.current_fps:.1f}")
                self.frame_count = 0
                self.last_fps_time = current_time

            # Display frame (with or without detections)
            display_frame = frame
            if self.current_result is not None:
                display_frame = self.model.draw_predictions(frame, self.current_result)

            # Resize for display
            display_frame = self.resize_frame_for_display(display_frame)
            
            self.display_frame(display_frame)

        # Schedule next video update (high frequency for smooth video)
        self.root.after(Config.VIDEO_UPDATE_INTERVAL, self.update_video)

    def run_inference_async(self, frame):
        """Run inference in background thread to avoid blocking UI"""
        if self.is_processing:
            return

        self.is_processing = True
        self.processing_label.config(text="🔄 Processing...", foreground="orange")

        def inference_thread():
            try:
                # Run inference
                result = self.model.predict(frame)

                # Update UI in main thread
                self.root.after(0, lambda: self.update_inference_result(result))
            except Exception as e:
                print(f"Error in inference thread: {e}")
                self.root.after(0, lambda: self.inference_error(str(e)))

        thread = threading.Thread(target=inference_thread, daemon=True)
        thread.start()

    def update_inference_result(self, result):
        """Update UI with inference results (called from main thread)"""
        self.current_result = result
        self.last_inference_time = time.time()
        self.is_processing = False

        # Re-enable snapshot button after processing
        if self.is_running:
            self.snapshot_button.config(state=tk.NORMAL)

        # Enable clear results button
        self.clear_results_button.config(state=tk.NORMAL)

        # Update status labels
        if result['has_phone']:
            self.result_label.config(
                text=f"✅ Phone Detected: {result['label']}",
                foreground="green"
            )
        else:
            self.result_label.config(
                text="❌ No Phone Detected",
                foreground="red"
            )

        confidence_pct = result['confidence'] * 100
        self.confidence_label.config(text=f"{confidence_pct:.2f}%")
        self.time_label.config(text=f"{result['inference_time']:.3f}s")
        self.processing_label.config(text="✅ Ready", foreground="green")

    def inference_error(self, error_msg):
        """Handle inference error"""
        self.is_processing = False
        
        # Re-enable snapshot button even after error
        if self.is_running:
            self.snapshot_button.config(state=tk.NORMAL)
        
        self.processing_label.config(text="❌ Error", foreground="red")
        print(f"Inference error: {error_msg}")
        
        messagebox.showerror(
            "Detection Error",
            f"Failed to run detection:\n{error_msg}"
        )

    def display_frame(self, frame):
        """
        Display a frame in the video label

        Args:
            frame: Frame to display (BGR format)
        """
        # Convert frame to PhotoImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image=image)

        # Update video display
        self.video_label.config(image=photo, text='')
        self.video_label.image = photo  # Keep a reference

    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        self.camera.release()
        self.root.destroy()


def main():
    """Main entry point"""
    try:
        # Validate configuration
        Config.validate()

        # Create and run application
        root = tk.Tk()
        app = PhoneDetectionApp(root)
        root.mainloop()

    except ValueError as e:
        # Show error dialog if configuration is invalid
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Configuration Error", str(e))
        root.destroy()
    except Exception as e:
        # Show error dialog for unexpected errors
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
        root.destroy()


if __name__ == "__main__":
    main()
