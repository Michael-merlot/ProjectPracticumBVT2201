"""
QUICK VERSION: Just download dataset, skip training

Use this if you want to train manually or use pre-trained weights.
"""

import os
import sys

def download_dataset_only():
    """Download dataset without training"""

    print("=" * 60)
    print("Download Dataset Only (v32)")
    print("=" * 60)
    print()

    try:
        from roboflow import Roboflow
        print("✓ Roboflow installed")
    except ImportError:
        print("✗ Roboflow not installed")
        print()
        print("Install it with:")
        print("  pip install roboflow")
        return False

    print()
    print("Downloading dataset from Roboflow...")
    print("Version: 32")
    print()

    try:
        rf = Roboflow(api_key="35PrGrbBxIAYgoqBPcKh")
        project = rf.workspace("phone-in-hand-detection").project("phone-in-hand-detection")
        version = project.version(32)

        # Download to simple path
        dataset = version.download("yolov8", location="dataset_v32")

        print()
        print("=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print()
        print(f"Dataset location: dataset_v32/")
        print()
        print("Next steps:")
        print()
        print("Option 1: Train yourself")
        print("  pip install ultralytics")
        print("  yolo detect train data=dataset_v32/data.yaml model=yolov8n.pt epochs=50")
        print("  # Model will be in: runs/detect/train/weights/best.pt")
        print()
        print("Option 2: Use download_and_train_v32.py (automatic training)")
        print("  python scripts/download_and_train_v32.py")
        print()

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("Try moving project to shorter path:")
        print("  C:\\Phone\\ProjectPracticum")
        return False

if __name__ == "__main__":
    success = download_dataset_only()
    sys.exit(0 if success else 1)
