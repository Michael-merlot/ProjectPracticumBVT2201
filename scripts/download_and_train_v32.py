"""
Download and train YOLOv8 model from Roboflow (Version 32)

This script:
1. Downloads phone-in-hand dataset (version 32) from Roboflow
2. Trains YOLOv8 model on this dataset
3. Saves trained model to models/best.pt
"""

import os
import sys

def download_and_train():
    """Download dataset and train model"""

    print("=" * 60)
    print("Phone Detection Model - Download & Train (v32)")
    print("=" * 60)
    print()

    # Step 1: Install dependencies
    print("Step 1: Checking dependencies...")
    try:
        from roboflow import Roboflow
        from ultralytics import YOLO
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print()
        print("Please install:")
        print("  pip install roboflow ultralytics")
        return False

    print()

    # Step 2: Download dataset
    print("Step 2: Downloading dataset from Roboflow...")
    print("This may take a few minutes...")
    print()

    try:
        rf = Roboflow(api_key="35PrGrbBxIAYgoqBPcKh")
        project = rf.workspace("phone-in-hand-detection").project("phone-in-hand-detection")
        version = project.version(32)

        # Download to a simple path to avoid Windows issues
        dataset = version.download("yolov8", location="dataset_v32")

        print()
        print("✓ Dataset downloaded successfully!")
        print(f"Location: dataset_v32/")

    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print()
        print("Possible issues:")
        print("1. No internet connection")
        print("2. API key expired")
        print("3. Path too long (try moving project to C:\\Phone)")
        return False

    print()

    # Step 3: Train model
    print("Step 3: Training YOLOv8 model...")
    print("This will take some time (depends on your hardware)")
    print()
    print("Parameters:")
    print("  - Model: YOLOv8 nano (yolov8n.pt)")
    print("  - Epochs: 50")
    print("  - Image size: 640x640")
    print("  - Device: GPU if available, else CPU")
    print()

    try:
        # Initialize model
        model = YOLO('yolov8n.pt')

        # Train model
        results = model.train(
            data=os.path.join('dataset_v32', 'data.yaml'),
            epochs=50,
            imgsz=640,
            batch=16,
            name='phone_detection_v32',
            patience=10,
            save=True,
            device='0' if os.system('nvidia-smi') == 0 else 'cpu'
        )

        print()
        print("✓ Training complete!")
        print()

    except Exception as e:
        print(f"✗ Error training model: {e}")
        return False

    # Step 4: Copy trained model to models/
    print("Step 4: Saving trained model...")

    try:
        import shutil

        # Create models directory
        os.makedirs("models", exist_ok=True)

        # Find the best model weights
        trained_model_path = os.path.join('runs', 'detect', 'phone_detection_v32', 'weights', 'best.pt')

        if os.path.exists(trained_model_path):
            shutil.copy(trained_model_path, 'models/best.pt')
            print(f"✓ Model saved to: models/best.pt")
        else:
            print(f"✗ Trained model not found at: {trained_model_path}")
            print("Check runs/detect/ folder for your trained model")
            return False

    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False

    print()
    print("=" * 60)
    print("SUCCESS! Setup Complete!")
    print("=" * 60)
    print()
    print("Your trained model is ready at: models/best.pt")
    print()
    print("Next steps:")
    print("1. Update .env file:")
    print("     USE_LOCAL_MODEL=true")
    print("     LOCAL_MODEL_PATH=models/best.pt")
    print()
    print("2. Run the app:")
    print("     python desktop_app/main.py")
    print()
    print("This model is specifically trained for 'phone in hand' detection!")
    print()

    return True

if __name__ == "__main__":
    print("WARNING: Training will take 30-60 minutes depending on hardware.")
    print("Make sure you have:")
    print("  - Good internet connection (for download)")
    print("  - 5+ GB free disk space")
    print("  - GPU recommended (but works on CPU)")
    print()

    response = input("Continue? (yes/no): ").lower().strip()

    if response in ['yes', 'y']:
        success = download_and_train()
        sys.exit(0 if success else 1)
    else:
        print("Cancelled.")
        sys.exit(0)
