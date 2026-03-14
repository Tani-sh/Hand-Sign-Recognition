"""
Real-time hand sign prediction using webcam, OpenCV, and MediaPipe.

Usage:
    python predict.py
"""

import cv2
import numpy as np
from model import load_trained_model
from utils import init_hands_detector, extract_landmarks, draw_landmarks, get_label_map


def predict_realtime():
    """Run real-time hand sign classification from the webcam."""

    # Load model and label map
    print("[*] Loading trained model...")
    model = load_trained_model()
    label_map = get_label_map()

    # Initialise MediaPipe Hands
    hands = init_hands_detector(static_mode=False, max_hands=1, min_confidence=0.7)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Cannot open webcam.")
        return

    print("[*] Webcam opened — press 'q' to quit.")
    text_buffer = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        predicted_letter = ""
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                draw_landmarks(frame, hand_landmarks)

                # Extract and predict
                landmarks = extract_landmarks(hand_landmarks)
                landmarks = landmarks.reshape(1, -1)
                prediction = model.predict(landmarks, verbose=0)[0]
                class_idx = np.argmax(prediction)
                confidence = prediction[class_idx]
                predicted_letter = label_map[class_idx]

        # Draw prediction on frame
        if predicted_letter and confidence > 0.6:
            cv2.putText(
                frame,
                f"{predicted_letter} ({confidence:.0%})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )
            # Display accumulated text
            cv2.putText(
                frame,
                f"Text: {text_buffer}",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Show a hand sign...",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (128, 128, 128),
                2,
            )

        # Controls info
        cv2.putText(
            frame,
            "q: quit | r: reset | s: save letter",
            (10, h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

        cv2.imshow("Hand Sign Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            text_buffer = ""
        elif key == ord("s") and predicted_letter:
            text_buffer += predicted_letter

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"[✓] Final text: {text_buffer}")


if __name__ == "__main__":
    predict_realtime()
