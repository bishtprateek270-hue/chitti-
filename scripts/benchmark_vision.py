"""
Benchmark script for Chitti Phase 3 Computer Vision module.
Measures:
- Face detection latency (YuNet)
- Face recognition latency (SFace embedding & cosine matching)
- Object detection latency (YOLOv8n)
- Vision Manager combined pipeline latency
- GPU memory and CPU utilization
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import psutil

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import AppConfig
from src.vision.models import FaceDetection
from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer
from src.vision.face_database import FaceDatabase
from src.vision.object_detector import ObjectDetector
from src.vision.vision_manager import VisionManager


def run_benchmarks():
    print("=" * 60)
    print("CHITTI PHASE 3 COMPUTER VISION BENCHMARK")
    print("=" * 60)
    
    config = AppConfig()
    
    # 1. System Info
    cpu_threads = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    
    print(f"CPU Logical Threads: {cpu_threads}")
    print(f"Total RAM: {ram_gb:.1f} GB")
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {gpu_name}")
        print(f"Initial VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
    
    print("-" * 60)
    
    # 2. Benchmark Face Detection (YuNet)
    print("[1/4] Benchmarking Face Detection (YuNet)...")
    detector = FaceDetector(
        model_path=config.vision.face_detection_model,
        score_threshold=0.6,
    )
    
    # Synthetic frame (640x480) with a drawn face-like circle
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(test_frame, (320, 240), 80, (200, 200, 200), -1)
    
    # Warmup
    for _ in range(5):
        detector.detect_faces(test_frame)
        
    runs = 30
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = detector.detect_faces(test_frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    avg_det_ms = np.mean(latencies)
    std_det_ms = np.std(latencies)
    print(f"  -> Face Detection Latency (YuNet 640x480): {avg_det_ms:.2f} ms (+/- {std_det_ms:.2f} ms)")
    
    # 3. Benchmark Face Recognition (SFace)
    print("[2/4] Benchmarking Face Recognition (SFace)...")
    recognizer = FaceRecognizer(
        model_path=config.vision.face_recognition_model,
        cosine_threshold=config.vision.face_recognition_threshold,
    )
    
    # Mock YuNet raw face detection row (15 elements)
    mock_raw_face = np.array([240, 160, 160, 160, 280, 200, 360, 200, 320, 240, 290, 280, 350, 280, 0.95], dtype=np.float32)
    mock_face_obj = FaceDetection(bbox=(240, 160, 160, 160), confidence=0.95, raw_detection=mock_raw_face)
    mock_registered_vector = np.random.randn(128).astype(np.float32)
    mock_registered_vector = (mock_registered_vector / np.linalg.norm(mock_registered_vector)).tolist()
    
    # Warmup
    for _ in range(5):
        emb = recognizer.extract_embedding(test_frame, mock_face_obj)
        if emb is not None:
            _ = recognizer.match_identity(emb, [("p1", "Prateek", mock_registered_vector)])
        
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        emb = recognizer.extract_embedding(test_frame, mock_face_obj)
        if emb is not None:
            _ = recognizer.match_identity(emb, [("p1", "Prateek", mock_registered_vector)])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    avg_rec_ms = np.mean(latencies)
    std_rec_ms = np.std(latencies)
    print(f"  -> Face Recognition Latency (SFace Crop + Align + 128d Match): {avg_rec_ms:.2f} ms (+/- {std_rec_ms:.2f} ms)")
    
    # 4. Benchmark Object Detection (YOLOv8n)
    print("[3/4] Benchmarking Object Detection (YOLOv8n)...")
    yolo = ObjectDetector(
        model_path=config.vision.object_detection_model,
        confidence_threshold=config.vision.object_confidence_threshold,
        device=config.vision.device
    )
    
    # Warmup
    for _ in range(5):
        _ = yolo.detect_objects(test_frame)
        
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = yolo.detect_objects(test_frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    avg_yolo_ms = np.mean(latencies)
    std_yolo_ms = np.std(latencies)
    print(f"  -> Object Detection Latency (YOLOv8n 640x480): {avg_yolo_ms:.2f} ms (+/- {std_yolo_ms:.2f} ms)")
    
    # 5. Combined Vision Pipeline
    print("[4/4] Benchmarking Combined Vision Manager Pipeline...")
    vision_mgr = VisionManager()
    
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = vision_mgr.analyze_frame(test_frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    avg_total_ms = np.mean(latencies)
    std_total_ms = np.std(latencies)
    print(f"  -> Full Vision Analysis Pipeline: {avg_total_ms:.2f} ms (+/- {std_total_ms:.2f} ms)")
    print(f"  -> Achievable Vision Inference Rate: {1000.0 / avg_total_ms:.1f} FPS")
    
    if cuda_available:
        print(f"  -> Peak VRAM Allocated: {torch.cuda.max_memory_allocated(0)/(1024**2):.1f} MB")
        
    print("=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()
