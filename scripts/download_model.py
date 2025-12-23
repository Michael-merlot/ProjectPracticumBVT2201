"""
Script to download phone detection model from Roboflow

This script downloads the YOLOv8 model for phone detection from Roboflow
and saves it locally for offline use.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roboflow import Roboflow
from config.config import Config


def download_model():
    """Download the model from Roboflow"""

    print("=" * 60)
    print("Phone Detection Model Downloader")
    print("=" * 60)
    print()

    # Check if API key is set
    if not Config.ROBOFLOW_API_KEY:
        print("ERROR: ROBOFLOW_API_KEY not set in .env file")
        print()
        print("Please:")
        print("1. Create a .env file (copy from .env.example)")
        print("2. Add your Roboflow API key")
        print("3. Run this script again")
        return False

    print(f"API Key: {Config.ROBOFLOW_API_KEY[:10]}...")
    print(f"Workspace: {Config.ROBOFLOW_WORKSPACE}")
    print(f"Project: {Config.ROBOFLOW_PROJECT}")
    print(f"Version: {Config.ROBOFLOW_VERSION}")
    print()

    try:
        # Initialize Roboflow
        print("Connecting to Roboflow...")
        rf = Roboflow(api_key=Config.ROBOFLOW_API_KEY)

        # Get project
        print("Loading project...")
        project = rf.workspace(Config.ROBOFLOW_WORKSPACE).project(Config.ROBOFLOW_PROJECT)

        # Get version
        print("Getting model version...")
        version = project.version(Config.ROBOFLOW_VERSION)

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Download in YOLOv8 format
        print()
        print("Downloading model (this may take a few minutes)...")
        print("Format: YOLOv8 PyTorch")
        print()

        dataset = version.download("yolov8", location="models/dataset")

        print()
        print("=" * 60)
        print("Download complete!")
        print("=" * 60)
        print()
        print("Model dataset downloaded to: models/dataset/")
        print()
        print("Next steps:")
        print("1. Train the model locally, OR")
        print("2. Download pre-trained weights from Roboflow")
        print()
        print("To train locally:")
        print("  pip install ultralytics")
        print("  yolo detect train data=models/dataset/data.yaml model=yolov8n.pt epochs=100")
        print()
        print("After training, the model will be at:")
        print("  runs/detect/train/weights/best.pt")
        print()
        print("Then update .env:")
        print("  USE_LOCAL_MODEL=true")
        print("  LOCAL_MODEL_PATH=runs/detect/train/weights/best.pt")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR downloading model")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Possible issues:")
        print("1. Invalid API key")
        print("2. No internet connection")
        print("3. Model not accessible with this API key")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
