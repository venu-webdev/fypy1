from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from routes.webcam import stream_webcam
import subprocess  # To call face_embedding.py

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

socketio.on_event("start_webcam", stream_webcam)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate-embeddings", methods=["POST"])
def generate_embeddings():
    try:
        subprocess.run(["python", "face_embedding.py"], check=True)  # Run script
        return jsonify({"status": "success", "message": "Embeddings generated!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
