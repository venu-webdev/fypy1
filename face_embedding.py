import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pickle  # For saving embeddings

# Path to dataset where each folder is a student name
DATASET_PATH = r"F:\new_project\v1\face_attendece\face_dataset"
OUTPUT_EMBEDDINGS = r"F:\new_project\v1\face_attendece\face_embeddings\face_embeddings.pkl"

# Initialize InsightFace model
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use GPU if available
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Store embeddings
embeddings_dict = {}

for student_name in os.listdir(DATASET_PATH):
    student_path = os.path.join(DATASET_PATH, student_name)
    
    if os.path.isdir(student_path):
        embeddings_list = []

        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            faces = face_app.get(img)  # Detect faces
            if faces:
                face_embedding = faces[0].normed_embedding  # Get normalized embedding
                embeddings_list.append(face_embedding)

        if embeddings_list:
            embeddings_dict[student_name] = np.mean(embeddings_list, axis=0)  # Average embeddings

# Save embeddings
with open(OUTPUT_EMBEDDINGS, "wb") as f:
    pickle.dump(embeddings_dict, f)

print(f"Saved face embeddings for {len(embeddings_dict)} students.")
