"""
Flask web application for real-time hand sign recognition.

Usage:
    python app.py
"""

import base64
import numpy as np
import cv2
from flask import Flask, render_template_string, Response

from model import load_trained_model
from utils import init_hands_detector, extract_landmarks, draw_landmarks, get_label_map


app = Flask(__name__)

# Global state
model = None
hands = None
label_map = get_label_map()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hand Sign Recognition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #0f0f23;
            color: #e0e0e0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 2rem;
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .video-container {
            border: 2px solid #333;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 114, 255, 0.2);
        }
        img { display: block; max-width: 100%; }
        p { margin-top: 1rem; color: #888; }
    </style>
</head>
<body>
    <h1>🖐️ Hand Sign Recognition</h1>
    <div class="video-container">
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
    </div>
    <p>Show a hand sign (A–Z) to the camera. Press <b>Ctrl+C</b> in the terminal to stop.</p>
</body>
</html>
"""


def generate_frames():
    """Generate video frames with hand sign predictions."""
    global model, hands

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                draw_landmarks(frame, hand_lm)

                landmarks = extract_landmarks(hand_lm).reshape(1, -1)
                pred = model.predict(landmarks, verbose=0)[0]
                idx = np.argmax(pred)
                conf = pred[idx]

                if conf > 0.6:
                    letter = label_map[idx]
                    cv2.putText(
                        frame, f"{letter} ({conf:.0%})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 255, 0), 3,
                    )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    cap.release()


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print("[*] Loading model...")
    model = load_trained_model()
    hands = init_hands_detector(static_mode=False, max_hands=1)
    print("[*] Starting web app at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
