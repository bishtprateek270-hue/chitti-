"""
Test script to diagnose camera and face detection on local laptop.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import cv2
from src.vision.camera import Camera
from src.vision.face_detector import FaceDetector

print("Listing available camera indices...")
cams = Camera.list_available_cameras(5)
print(f"Available cameras: {cams}")

for idx in cams if cams else [0]:
    print(f"\nTesting camera index {idx}...")
    try:
        cam = Camera(device_index=idx, width=640, height=480)
        cam.open()
        print("Camera opened successfully!")
        
        # Read a few frames to let auto-exposure adjust
        for _ in range(5):
            frame = cam.capture_frame()
            time.sleep(0.05)
            
        print(f"Frame captured! Shape: {frame.shape}, Dtype: {frame.dtype}")
        
        detector = FaceDetector()
        faces = detector.detect_faces(frame)
        print(f"Detected {len(faces)} face(s).")
        for i, f in enumerate(faces):
            print(f"  Face {i+1}: BBox={f.bbox}, Confidence={f.confidence:.2f}")
            
        cam.release()
    except Exception as e:
        print(f"Error testing camera {idx}: {e}")
