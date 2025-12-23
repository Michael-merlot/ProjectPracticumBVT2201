import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the phone detection application"""

    # Roboflow API Configuration
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', '')
    ROBOFLOW_WORKSPACE = os.getenv('ROBOFLOW_WORKSPACE', 'phone-in-hand-detection')
    ROBOFLOW_PROJECT = os.getenv('ROBOFLOW_PROJECT', 'phone-in-hand-detection')
    ROBOFLOW_VERSION = int(os.getenv('ROBOFLOW_VERSION', '19'))

    # Camera Configuration
    # Can be:
    # - Integer (0, 1, 2...) for local webcam
    # - String URL for IP camera (e.g., "http://192.168.1.100:8080/video")
    CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', '0')

    # Parse CAMERA_SOURCE to int if it's a number, otherwise keep as string
    try:
        CAMERA_INDEX = int(CAMERA_SOURCE)
    except ValueError:
        CAMERA_INDEX = CAMERA_SOURCE  # It's a URL string

    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30

    # Model Configuration
    # "api" - Use Roboflow API (requires internet)
    # "local" - Use local YOLOv8 model (no internet required)
    USE_LOCAL_MODEL = os.getenv('USE_LOCAL_MODEL', 'false').lower() == 'true'
    LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', 'models/best.pt')

    # Inference Configuration
    CONFIDENCE_THRESHOLD = 0.4
    OVERLAP_THRESHOLD = 0.3
    INFERENCE_INTERVAL = 2000  # milliseconds between API calls (2 seconds)

    # UI Configuration
    WINDOW_TITLE = "Phone Detection App"
    VIDEO_UPDATE_INTERVAL = 30  # milliseconds (for smooth video, ~30 FPS)

    # Detection Mode
    DETECTION_MODE = "continuous"  # "continuous" or "snapshot"

    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.USE_LOCAL_MODEL and not cls.ROBOFLOW_API_KEY:
            raise ValueError(
                "ROBOFLOW_API_KEY not set and USE_LOCAL_MODEL is false. "
                "Please either:\n"
                "1. Set ROBOFLOW_API_KEY in .env file for API mode, OR\n"
                "2. Set USE_LOCAL_MODEL=true and download the model for local mode.\n"
                "See .env.example for template."
            )
        if cls.USE_LOCAL_MODEL and not os.path.exists(cls.LOCAL_MODEL_PATH):
            raise ValueError(
                f"USE_LOCAL_MODEL is true but model not found at {cls.LOCAL_MODEL_PATH}\n"
                "Please download the model using: python scripts/download_model.py"
            )
        return True
