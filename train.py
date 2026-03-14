"""
Train the hand sign classification model on extracted landmarks.

Usage:
    python train.py
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras

from model import build_model, NUM_LANDMARKS, NUM_COORDS
from utils import get_label_map

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = "./data"
SAVE_DIR = "./saved_model"
MODEL_PATH = os.path.join(SAVE_DIR, "hand_sign_model.keras")
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42


def generate_synthetic_data(n_samples_per_class=200):
    """
    Generate synthetic landmark data for demonstration.

    In a real project, you would collect data from the ASL dataset
    or capture it via webcam using utils.extract_landmarks().
    """
    print("[*] Generating synthetic training data for demonstration...")
    np.random.seed(RANDOM_STATE)

    X_all, y_all = [], []
    n_features = NUM_LANDMARKS * NUM_COORDS  # 42

    for label in range(26):
        # Create class-specific patterns with some noise
        base_pattern = np.random.randn(n_features).astype(np.float32) * 0.5
        samples = np.tile(base_pattern, (n_samples_per_class, 1))
        noise = np.random.randn(n_samples_per_class, n_features).astype(np.float32) * 0.15
        samples += noise

        X_all.append(samples)
        y_all.extend([label] * n_samples_per_class)

    X = np.vstack(X_all)
    y = np.array(y_all)

    return X, y


def train():
    """Train the model and save the best weights."""
    # Load or generate data
    X, y = generate_synthetic_data()
    print(f"[*] Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[*] Train: {len(X_train)} | Test: {len(X_test)}")

    # Build model
    model = build_model()
    model.summary()

    # Callbacks
    os.makedirs(SAVE_DIR, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6,
        ),
    ]

    # Train
    print("[*] Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n[*] Evaluation on test set:")
    y_pred = model.predict(X_test).argmax(axis=1)
    label_map = get_label_map()
    target_names = [label_map[i] for i in range(26)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[✓] Test accuracy: {test_acc:.4f}")
    print(f"[✓] Model saved to {MODEL_PATH}")

    return history


if __name__ == "__main__":
    train()
