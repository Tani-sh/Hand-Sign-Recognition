"""
Utility functions for hand landmark extraction and data preprocessing.
"""

import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def init_hands_detector(static_mode=False, max_hands=1, min_confidence=0.7):
    """Initialise the MediaPipe Hands detector."""
    return mp_hands.Hands(
        static_image_mode=static_mode,
        max_num_hands=max_hands,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """
    Extract normalised (x, y) coordinates from MediaPipe hand landmarks.

    Returns:
        np.ndarray of shape (42,) — 21 landmarks × 2 coordinates, normalised
        relative to the wrist position.
    """
    coords = np.array(
        [(lm.x, lm.y) for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    # Normalise: shift origin to wrist (landmark 0) and scale
    wrist = coords[0]
    coords = coords - wrist

    # Scale by the max absolute value to keep values in [-1, 1]
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val

    return coords.flatten()


def draw_landmarks(image, hand_landmarks):
    """Draw hand landmarks and connections on the image."""
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
    )


def get_label_map():
    """Return a mapping from class index to letter (A–Z)."""
    return {i: chr(65 + i) for i in range(26)}


def create_dataset_from_landmarks(landmarks_list, labels_list):
    """
    Convert lists of landmark arrays and labels into numpy arrays.

    Args:
        landmarks_list: list of np.ndarray, each of shape (42,)
        labels_list: list of int labels (0–25)

    Returns:
        X: np.ndarray of shape (N, 42)
        y: np.ndarray of shape (N,)
    """
    X = np.array(landmarks_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    return X, y
