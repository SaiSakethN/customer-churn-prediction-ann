from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

from src.preprocess import load_and_preprocess_data

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "churn_ann_model.keras"


def build_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_save_model():
    _, _, y_train, y_test, X_train_scaled, X_test_scaled = load_and_preprocess_data()

    model = build_model(X_train_scaled.shape[1])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Loss: {loss:.4f}")

    return history, model


if __name__ == "__main__":
    train_and_save_model()