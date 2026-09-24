"""
Test OpenCV capture backends on Windows.
"""
import time
import cv2

for name, backend in [("CAP_ANY (Default / MSMF)", cv2.CAP_ANY), ("CAP_DSHOW (DirectShow)", cv2.CAP_DSHOW)]:
    print(f"\n--- Testing Backend: {name} ---")
    t0 = time.time()
    cap = cv2.VideoCapture(0, backend)
    print(f"VideoCapture(0) created in {time.time()-t0:.3f}s. isOpened={cap.isOpened()}")
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        for i in range(5):
            t_read = time.time()
            ret, frame = cap.read()
            print(f"  Frame {i+1} read: ret={ret}, shape={frame.shape if ret else None}, took={time.time()-t_read:.3f}s")
            time.sleep(0.05)
            
        cap.release()
        print(f"Released {name}.")
    else:
        print(f"Failed to open with {name}.")
