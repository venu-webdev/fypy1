from flask import Flask, render_template
from flask_socketio import SocketIO
from routes.webcam import stream_webcam

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSockets

# Start webcam stream
socketio.on_event("start_webcam", stream_webcam)

@app.route("/")
def index():
    return render_template("index.html")  # Load frontend

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)