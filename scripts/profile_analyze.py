"""
Profile analyze_frame step by step.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from src.vision.vision_manager import VisionManager

print("1. Creating VisionManager...")
t0 = time.time()
vm = VisionManager()
print(f"   VisionManager created in {time.time()-t0:.2f}s")

print("2. Calling vm.analyze_frame()...")
t_start = time.time()

t_cam = time.time()
frame = vm.camera.capture_frame()
print(f"   Camera capture_frame() took {time.time()-t_cam:.2f}s, frame shape={frame.shape}")

t_face = time.time()
dets = vm.face_detector.detect_faces(frame)
print(f"   Face detection took {time.time()-t_face:.2f}s, found {len(dets)} faces")

t_rec = time.time()
registered_faces = vm.face_db.get_all_registered_faces()
for d in dets:
    emb = vm.face_recognizer.extract_embedding(frame, d)
    name, sim, is_known, pid = vm.face_recognizer.match_identity(emb, registered_faces)
    print(f"   Recognized face: {name}, sim={sim:.2f}, is_known={is_known}")
print(f"   Face recognition took {time.time()-t_rec:.2f}s")

t_obj = time.time()
objs = vm.object_detector.detect_objects(frame)
print(f"   Object detection took {time.time()-t_obj:.2f}s, found {len(objs)} objects: {[o.class_name for o in objs]}")

res = vm.analyze_frame(frame)
print(f"\nFinal Summary: {res.summary_text}")
print(f"Total time: {time.time()-t_start:.2f}s")
