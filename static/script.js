const socket = io.connect("http://localhost:5000");

socket.on("video_frame", (data) => {
    document.getElementById("videoStream").src = "data:image/jpeg;base64," + data.frame;
});

socket.emit("start_webcam");
