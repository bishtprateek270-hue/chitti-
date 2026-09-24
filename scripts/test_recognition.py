"""
Test recognition against registered database.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from src.vision.camera import Camera
from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer
from src.vision.face_database import FaceDatabase
from src.vision.vision_manager import VisionManager

print("Initializing VisionManager...")
vm = VisionManager()

print("\n--- Listing Registered Identities in faces.db ---")
names = vm.list_registered_people()
print(f"Registered names: {names}")

print("\n--- Testing Live Face Recognition ---")
result = vm.analyze_frame()
print(f"Summary: {result.summary_text}")
for face in result.faces:
    print(f"  Face: is_known={face.is_known}, name={face.name}, similarity={face.similarity:.2f}")

for obj, cnt in result.object_counts.items():
    print(f"  Object: {obj} x {cnt}")
