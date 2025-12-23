import cv2
import numpy as np
from roboflow import Roboflow
from typing import Dict, Optional, Any
import time
import os


class PhoneDetectionModel:
    """Class to handle phone detection model inference"""

    def __init__(self, api_key: str, workspace: str, project: str, version: int,
                 confidence: float = 0.4, overlap: float = 0.3):
        """
        Initialize Roboflow model

        Args:
            api_key: Roboflow API key
            workspace: Roboflow workspace name
            project: Roboflow project name
            version: Model version number
            confidence: Confidence threshold for detections
            overlap: Overlap threshold for NMS
        """
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = version
        self.confidence = confidence
        self.overlap = overlap
        self.model = None
        self.is_loaded = False

    def load_model(self) -> bool:
        """
        Load the Roboflow model

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(self.workspace).project(self.project)
            self.model = project.version(self.version).model
            self.is_loaded = True
            print(f"Model loaded successfully: {self.workspace}/{self.project}/v{self.version}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            return False

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a frame

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Dictionary with classification results:
            {
                'has_phone': bool,
                'confidence': float,
                'label': str,
                'detections': list,
                'inference_time': float
            }
        """
        if not self.is_loaded or self.model is None:
            return {
                'has_phone': False,
                'confidence': 0.0,
                'label': 'Model not loaded',
                'detections': [],
                'inference_time': 0.0
            }

        try:
            start_time = time.time()

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save frame temporarily for Roboflow API
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)

            # Run inference
            predictions = self.model.predict(
                temp_path,
                confidence=self.confidence,
                overlap=self.overlap
            ).json()

            inference_time = time.time() - start_time

            # Process predictions
            detections = predictions.get('predictions', [])

            # Determine if phone is detected
            has_phone = len(detections) > 0

            # Get highest confidence detection
            max_confidence = 0.0
            detected_class = "No phone"

            if has_phone:
                max_detection = max(detections, key=lambda x: x.get('confidence', 0))
                max_confidence = max_detection.get('confidence', 0.0)
                detected_class = max_detection.get('class', 'phone')

            return {
                'has_phone': has_phone,
                'confidence': max_confidence,
                'label': detected_class,
                'detections': detections,
                'inference_time': inference_time
            }

        except Exception as e:
            print(f"Error during inference: {e}")
            return {
                'has_phone': False,
                'confidence': 0.0,
                'label': f'Error: {str(e)}',
                'detections': [],
                'inference_time': 0.0
            }

    def draw_predictions(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Draw prediction results on frame

        Args:
            frame: Input frame
            result: Prediction result from predict()

        Returns:
            Frame with drawn predictions
        """
        output = frame.copy()

        # Draw bounding boxes for all detections
        for detection in result.get('detections', []):
            x = int(detection.get('x', 0))
            y = int(detection.get('y', 0))
            width = int(detection.get('width', 0))
            height = int(detection.get('height', 0))
            confidence = detection.get('confidence', 0)
            class_name = detection.get('class', 'unknown')

            # Calculate bounding box coordinates
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            # Draw rectangle
            color = (0, 255, 0) if result['has_phone'] else (0, 0, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output


class LocalPhoneDetectionModel:
    """Class to handle local YOLOv8 phone detection model"""

    def __init__(self, model_path: str = "models/best.pt", confidence: float = 0.4):
        """
        Initialize local YOLO model

        Args:
            model_path: Path to YOLOv8 model weights (.pt file)
            confidence: Confidence threshold for detections
        """
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self.is_loaded = False

    def load_model(self) -> bool:
        """
        Load the local YOLO model

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from ultralytics import YOLO

            if not os.path.exists(self.model_path):
                print(f"Error: Model file not found at {self.model_path}")
                print("Please download the model first using: python scripts/download_model.py")
                return False

            self.model = YOLO(self.model_path)
            self.is_loaded = True
            print(f"Local model loaded successfully from {self.model_path}")
            return True
        except ImportError:
            print("Error: ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            print(f"Error loading local model: {e}")
            self.is_loaded = False
            return False

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a frame

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Dictionary with classification results:
            {
                'has_phone': bool,
                'confidence': float,
                'label': str,
                'detections': list,
                'inference_time': float
            }
        """
        if not self.is_loaded or self.model is None:
            return {
                'has_phone': False,
                'confidence': 0.0,
                'label': 'Model not loaded',
                'detections': [],
                'inference_time': 0.0
            }

        try:
            start_time = time.time()

            # Run inference
            results = self.model(frame, conf=self.confidence, verbose=False)

            inference_time = time.time() - start_time

            # Process results
            detections = []
            has_phone = False
            max_confidence = 0.0
            detected_class = "No phone"

            if len(results) > 0:
                result = results[0]
                boxes = result.boxes

                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    # Get class name
                    class_name = result.names[cls] if cls < len(result.names) else f"class_{cls}"

                    # Convert to Roboflow-like format for compatibility
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2

                    detection = {
                        'x': x_center,
                        'y': y_center,
                        'width': width,
                        'height': height,
                        'confidence': conf,
                        'class': class_name,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    }
                    detections.append(detection)

                    # Update max confidence
                    if conf > max_confidence:
                        max_confidence = conf
                        detected_class = class_name
                        has_phone = True

            return {
                'has_phone': has_phone,
                'confidence': max_confidence,
                'label': detected_class if has_phone else "No phone",
                'detections': detections,
                'inference_time': inference_time
            }

        except Exception as e:
            print(f"Error during local inference: {e}")
            return {
                'has_phone': False,
                'confidence': 0.0,
                'label': f'Error: {str(e)}',
                'detections': [],
                'inference_time': 0.0
            }

    def draw_predictions(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Draw prediction results on frame

        Args:
            frame: Input frame
            result: Prediction result from predict()

        Returns:
            Frame with drawn predictions
        """
        output = frame.copy()

        # Draw bounding boxes for all detections
        for detection in result.get('detections', []):
            x = int(detection.get('x', 0))
            y = int(detection.get('y', 0))
            width = int(detection.get('width', 0))
            height = int(detection.get('height', 0))
            confidence = detection.get('confidence', 0)
            class_name = detection.get('class', 'unknown')

            # Calculate bounding box coordinates
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            # Draw rectangle
            color = (0, 255, 0) if result['has_phone'] else (0, 0, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output
