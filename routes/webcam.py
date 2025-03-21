from ultralytics import YOLO
import cv2
import base64
import numpy as np
from flask_socketio import emit
from insightface.app import FaceAnalysis  # Face detection using SCRFD

# Load YOLO model (trained on COCO dataset) for person detection
yolo_model = YOLO("yolov8n.pt")

# Load SCRFD face detection model
face_detector = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_detector.prepare(ctx_id=0, det_size=(640, 640))  # Resize for performance

def stream_webcam():
    cap = cv2.VideoCapture(r"F:\new_project\v1\vid30.mp4")  # Change index based on webcam or video file

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = yolo_model(frame)[0]  

        person_boxes = []  # Store person bounding boxes

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)

            if class_id == 0:  # Detect persons only (class ID = 0)
                person_boxes.append((x1, y1, x2, y2))  # Store person box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)  # Add label

        # Run face detection only inside detected person regions
        for (px1, py1, px2, py2) in person_boxes:
            person_roi = frame[py1:py2, px1:px2]  # Crop person region

            if person_roi.size == 0:
                continue  # Skip empty regions

            faces = face_detector.get(person_roi)  # Detect faces in person region

            for face in faces:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                fx1, fx2 = fx1 + px1, fx2 + px1  # Adjust face coordinates relative to the full frame
                fy1, fy2 = fy1 + py1, fy2 + py1

                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)  # Draw face bounding box
                cv2.putText(frame, "Face", (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 0), 2, cv2.LINE_AA)  # Add label

        # Encode frame to send over WebSocket
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        emit("video_frame", {"frame": frame_base64})  # Send frame to frontend
