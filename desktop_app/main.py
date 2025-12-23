"""
Phone Detection Desktop Application

This is the main entry point for the phone detection desktop application.
It uses a webcam to capture images and runs inference using a Roboflow model
to detect if a phone is present in the frame.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gui import main

if __name__ == "__main__":
    main()
