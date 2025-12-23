import cv2
import numpy as np
from typing import Optional, Tuple, Union


class Camera:
    """Class to handle webcam capture (local or IP camera)"""

    def __init__(self, camera_source: Union[int, str] = 0, width: int = 640, height: int = 480):
        """
        Initialize camera

        Args:
            camera_source: Camera device index (0, 1, 2...) or IP camera URL
                          Examples:
                          - 0 (default webcam)
                          - "http://192.168.1.100:8080/video" (IP Webcam)
                          - "http://192.168.1.100:4747/video" (DroidCam)
            width: Frame width (may not apply to IP cameras)
            height: Frame height (may not apply to IP cameras)
        """
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_ip_camera = isinstance(camera_source, str)

    def start(self) -> bool:
        """
        Start camera capture

        Returns:
            True if camera started successfully, False otherwise
        """
        print(f"Starting camera: {self.camera_source}")
        self.cap = cv2.VideoCapture(self.camera_source)

        if not self.cap.isOpened():
            print(f"Failed to open camera: {self.camera_source}")
            return False

        # Set camera properties (only for local cameras, may not work for IP cameras)
        if not self.is_ip_camera:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Test reading a frame
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read initial frame from camera")
            self.release()
            return False

        print(f"Camera started successfully. Frame size: {frame.shape if frame is not None else 'Unknown'}")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera

        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Release camera resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.release()
        except:
            pass  # Ignore errors during cleanup
