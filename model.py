"""
CNN model architecture for hand sign classification.
"""

from tensorflow import keras
from tensorflow.keras import layers


NUM_CLASSES = 26  # A-Z
NUM_LANDMARKS = 21
NUM_COORDS = 2   # x, y (normalised)


def build_model(input_shape=(NUM_LANDMARKS * NUM_COORDS,), num_classes=NUM_CLASSES):
    """
    Build a fully-connected classifier for hand landmark features.

    Architecture:
        Input (42) → Dense(512) → Dropout → Dense(256) → Dropout
        → Dense(128) → Dense(26, softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_trained_model(path="./saved_model/hand_sign_model.keras"):
    """Load a previously trained model from disk."""
    return keras.models.load_model(path)


if __name__ == "__main__":
    model = build_model()
    model.summary()
