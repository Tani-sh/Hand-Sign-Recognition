# 🖐️ Hand Sign Recognition

Real-time ASL hand sign recognition app built for **Smart India Hackathon 2024**. Uses a webcam feed to detect hand landmarks via MediaPipe and classifies signs (A–Z) with a lightweight CNN — all served through a Flask web interface.

## ⚙️ How it works

1. **OpenCV** captures frames from the webcam
2. **MediaPipe Hands** extracts 21 skeletal landmarks per hand (x, y coordinates)
3. A **CNN** trained on the landmark vectors classifies the gesture into one of 26 letters
4. The predicted letter is overlaid on the live video feed in the browser

The model takes landmark coordinates as input (not raw images), which keeps it fast and orientation-invariant.

## 📁 Project structure

```
├── app.py              # Flask web app with live video feed
├── model.py            # CNN architecture (Dense layers on landmark features)
├── train.py            # Training script with early stopping + LR scheduling
├── predict.py          # CLI tool for single-frame prediction
├── utils.py            # MediaPipe helpers, landmark extraction, label mapping
├── requirements.txt
└── .gitignore
```

## 🚀 Running it

```bash
pip install -r requirements.txt

# Train the model (generates synthetic data for demo, swap in real data for production)
python train.py

# Launch the web app
python app.py
# Open http://localhost:5000 in your browser
```

📷 You'll need a webcam. The app runs at ~30 FPS on a decent laptop.

## 🔧 Tech stack

- **OpenCV** — frame capture and image processing
- **MediaPipe** — hand landmark detection (21 keypoints)
- **TensorFlow/Keras** — CNN model
- **Flask** — web interface with MJPEG streaming
- **scikit-learn** — train/test split, metrics
