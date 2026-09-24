"""
Test instant on-demand snapshot capture.
"""
import time
import cv2

for i in range(3):
    t0 = time.time()
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        cap.release()
        print(f"Snapshot {i+1}: ret={ret}, shape={frame.shape if ret else None}, took={time.time()-t0:.3f}s", flush=True)
    else:
        print(f"Snapshot {i+1}: Failed to open", flush=True)
    time.sleep(0.5)
