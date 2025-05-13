from ultralytics import YOLO
import cv2
import math
import time
from datetime import datetime
import winsound  # Use only on Windows. Use `playsound` on Linux/macOS if needed.

camera = cv2.VideoCapture(0)
camera.set(3, 720)
camera.set(4, 720)

detector = YOLO('yolov8n.pt')

labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

prev_time = time.time()

while True:
    ret, frame = camera.read()
    detections = detector(frame, stream=True)
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    for detection in detections:
        box_list = detection.boxes
        for box in box_list:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = labels[cls_id]
            conf = math.ceil(box.conf[0] * 100)

            # Color coding by object type
            if label == 'knife':
                color = (0, 0, 255)
                thickness = 4
                winsound.Beep(1000, 150)  # Beep on knife detection
            elif label == 'person':
                color = (0, 255, 255)
                thickness = 3
            elif label == 'cat':
                color = (128, 0, 128)
                thickness = 2
            else:
                color = (0, 255, 0)
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"{label} {conf:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display FPS and timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
