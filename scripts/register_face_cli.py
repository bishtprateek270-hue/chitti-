"""
Fast face registration tool.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import cv2
from src.vision.camera import Camera
from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer
from src.vision.face_database import FaceDatabase

def register_face_cli(name="Prateek Singh Bisht", num_samples=3):
    print(f"\n==========================================")
    print(f"Registering Face for: {name}")
    print(f"==========================================")
    
    cam = Camera(0, 640, 480)
    detector = FaceDetector(score_threshold=0.45)
    recognizer = FaceRecognizer()
    db = FaceDatabase("data/vision/faces.db")
    
    print("Opening webcam...")
    cam.open()
    
    collected_embeddings = []
    print("Capturing samples... Please look at the camera.")
    
    start_time = time.time()
    while len(collected_embeddings) < num_samples and (time.time() - start_time) < 15.0:
        try:
            frame = cam.capture_frame()
            dets = detector.detect_faces(frame)
            if len(dets) >= 1:
                # Take largest/highest confidence face
                best_det = max(dets, key=lambda d: d.confidence)
                emb = recognizer.extract_embedding(frame, best_det)
                if emb is not None:
                    collected_embeddings.append(emb)
                    print(f"  -> [Sample {len(collected_embeddings)}/{num_samples}] Captured! Confidence: {best_det.confidence:.2f}")
                    time.sleep(0.2)
            else:
                print("  [Looking for face... Please face camera]", end="\r")
                time.sleep(0.05)
        except Exception as e:
            print(f"Error capturing: {e}")
            time.sleep(0.1)
            
    cam.release()
    
    if len(collected_embeddings) > 0:
        arr = np.array(collected_embeddings, dtype=np.float32)
        mean_vec = np.mean(arr, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        final_embedding = mean_vec.tolist()
        
        person_id = db.register_or_update_face(name, final_embedding, sample_count=len(collected_embeddings))
        print(f"\nSUCCESS: Registered '{name}' (ID: {person_id}) with {len(collected_embeddings)} samples!")
        return True
    else:
        print("\nFAILED: No face samples captured.")
        return False

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Prateek Singh Bisht"
    register_face_cli(name)
