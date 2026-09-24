"""
Step by step trace of register_person.
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

print("1. Initializing components...")
cam = Camera(0, 640, 480)
detector = FaceDetector()
recognizer = FaceRecognizer()
db = FaceDatabase("data/vision/faces.db")

print("2. Opening camera...")
cam.open()

print("3. Capturing frame 1...")
frame = cam.capture_frame()
print(f"Captured frame shape: {frame.shape}")

print("4. Detecting faces...")
dets = detector.detect_faces(frame)
print(f"Detections: {len(dets)}")
for d in dets:
    print(f"  Det: bbox={d.bbox}, conf={d.confidence:.2f}")
    print("5. Extracting embedding...")
    emb = recognizer.extract_embedding(frame, d)
    print(f"  Embedding generated? {emb is not None}, len={len(emb) if emb else 0}")
    
print("6. Releasing camera...")
cam.release()
print("7. Done!")
