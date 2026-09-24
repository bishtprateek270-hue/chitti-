"""
Test face analysis alone vs YOLO on live frame.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import cv2
from src.vision.camera import Camera
from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer
from src.vision.face_database import FaceDatabase

print("1. Capturing live frame from webcam...")
cam = Camera(0, 640, 480)
frame = cam.capture_frame()
print(f"   Frame captured! shape={frame.shape}")

print("2. Running Face Detection (YuNet)...")
t0 = time.time()
detector = FaceDetector()
dets = detector.detect_faces(frame)
print(f"   Face detection took {time.time()-t0:.3f}s. Found {len(dets)} faces.")

print("3. Running Face Recognition (SFace)...")
t0 = time.time()
recognizer = FaceRecognizer()
db = FaceDatabase("data/vision/faces.db")
reg_faces = db.get_all_registered_faces()
for d in dets:
    emb = recognizer.extract_embedding(frame, d)
    name, sim, is_known, pid = recognizer.match_identity(emb, reg_faces)
    print(f"   Match result: Name='{name}', Similarity={sim:.2f}, IsKnown={is_known}")
print(f"   Face recognition took {time.time()-t0:.3f}s.")

print("4. Releasing camera...")
cam.release()
print("5. Completed Face Pipeline successfully!")
