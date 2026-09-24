"""
Comprehensive camera & registration test script.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import cv2
from src.vision.camera import Camera
from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer
from src.vision.face_database import FaceDatabase
from src.vision.vision_manager import VisionManager

print("Initializing VisionManager...")
vm = VisionManager()

print("\nCalling register_person for 'Test Prateek'...")
t0 = time.time()
success, msg = vm.register_person("Test Prateek", num_samples=3)
t1 = time.time()

print(f"\nResult: success={success}, msg={msg}, took={t1-t0:.2f}s")
