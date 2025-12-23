"""
Simple script to download pre-trained YOLOv8 model for phone detection

This script downloads a standard YOLOv8 model that can detect phones (cell phone class).
For better accuracy with "phone in hand", you should train your own model or
download from Roboflow manually.
"""

import os
import sys

def download_yolo_model():
    """Download standard YOLOv8 model"""

    print("=" * 60)
    print("YOLOv8 Model Downloader")
    print("=" * 60)
    print()
    print("This script will download a standard YOLOv8 model.")
    print("It can detect 'cell phone' among other objects.")
    print()
    print("Note: For better 'phone in hand' detection, download")
    print("the trained model manually from Roboflow.")
    print()

    try:
        from ultralytics import YOLO

        # Create models directory
        os.makedirs("models", exist_ok=True)

        print("Downloading YOLOv8 nano model...")
        print("This may take a few minutes...")
        print()

        # Download and save model
        model = YOLO('yolov8n.pt')

        # Move to models folder
        import shutil
        if os.path.exists('yolov8n.pt'):
            shutil.move('yolov8n.pt', 'models/best.pt')
            print("✓ Model downloaded successfully!")

        print()
        print("=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print()
        print("Model saved to: models/best.pt")
        print()
        print("Update your .env file:")
        print("  USE_LOCAL_MODEL=true")
        print("  LOCAL_MODEL_PATH=models/best.pt")
        print()
        print("Then run:")
        print("  python desktop_app/main.py")
        print()
        print("⚠️  Note: This is a general YOLOv8 model.")
        print("   It detects 'cell phone' but may not be as accurate")
        print("   for 'phone in hand' as the Roboflow model.")
        print()
        print("For better accuracy:")
        print("1. Go to: https://universe.roboflow.com/phone-in-hand-detection/phone-in-hand-detection/model/19")
        print("2. Click 'Export' → 'YOLOv8' → Download weights")
        print("3. Save as models/best.pt")
        print()

        return True

    except ImportError:
        print()
        print("ERROR: ultralytics not installed")
        print()
        print("Please install it first:")
        print("  pip install ultralytics")
        print()
        return False

    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR downloading model")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        return False

if __name__ == "__main__":
    success = download_yolo_model()
    sys.exit(0 if success else 1)
