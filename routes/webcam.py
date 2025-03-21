from ultralytics import YOLO
import cv2
import base64
import numpy as np
from flask_socketio import emit

# Load YOLO model (trained on COCO dataset)
yolo_model = YOLO("yolov8n.pt")  

def stream_webcam():
    cap = cv2.VideoCapture(r"F:\new_project\v1\vid30.mp4")  # Change index based on webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = yolo_model(frame)[0]  

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)

            if class_id == 0:  # Detect persons only (class ID = 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)  # Add label

        # Encode frame to send over WebSocket
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        emit("video_frame", {"frame": frame_base64})  # Send frame to frontend
