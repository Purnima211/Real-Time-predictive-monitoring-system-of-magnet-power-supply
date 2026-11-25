"""
model.py
--------
Loads and wraps the pre-trained LSTM temperature model
(lstm_temperature_model.keras). Keeps TensorFlow import isolated to this
module so the rest of the application can be exercised without a working
TensorFlow installation (helpful for unit-testing detector/predictor
logic).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from utils import CONFIG, LOGGER


class ModelLoadError(Exception):
    """Raised when the LSTM model cannot be loaded or is corrupted."""


class TemperatureModel:
    """
    Thin wrapper around a Keras LSTM model used to predict the next
    temperature sample from a rolling window of past samples.
    """

    def __init__(self) -> None:
        self._model = None
        self._input_shape: Optional[Tuple[int, ...]] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def sequence_length(self) -> int:
        """
        Returns the sequence length (time-steps) the model expects,
        inferred from the model's input shape when available, otherwise
        falling back to CONFIG.SEQUENCE_LENGTH.
        """
        if self._input_shape and len(self._input_shape) >= 2 and self._input_shape[1]:
            return int(self._input_shape[1])
        return CONFIG.SEQUENCE_LENGTH

    @property
    def n_features(self) -> int:
        """
        Returns the number of input features per time-step expected by
        the model, inferred from the model's input shape when available.
        Falls back to 1 (univariate: temperature only).
        """
        if self._input_shape and len(self._input_shape) >= 3 and self._input_shape[2]:
            return int(self._input_shape[2])
        return 1

    def load(self, path: str = CONFIG.MODEL_PATH) -> None:
        """
        Loads the Keras model from disk.

        Raises ModelLoadError with a human-readable message on any
        failure (missing file, corrupted file, incompatible format).
        Never raises a raw TensorFlow/Keras exception to the caller.
        """
        if not os.path.isfile(path):
            raise ModelLoadError(f"Model file not found: {path}")

        try:
            # Imported lazily so the rest of the application can be
            # imported/tested without requiring TensorFlow to be
            # installed and initialized.
            import tensorflow as tf
        except ImportError as exc:
            raise ModelLoadError(
                "TensorFlow is not installed. Install the dependencies "
                "in requirements.txt before running the application."
            ) from exc

        try:
            self._model = tf.keras.models.load_model(path, compile=False)
        except Exception as exc:  # noqa: BLE001 - Keras raises many exception types
            self._model = None
            raise ModelLoadError(
                f"Model could not be loaded (file may be corrupted or "
                f"incompatible with the installed TensorFlow version): {exc}"
            ) from exc

        try:
            self._input_shape = self._model.input_shape
        except Exception:  # noqa: BLE001 - input_shape is best-effort metadata
            self._input_shape = None

        LOGGER.info(
            "Model loaded from %s (sequence_length=%s, n_features=%s)",
            path,
            self.sequence_length,
            self.n_features,
        )

    def predict_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Runs inference on a batch of input sequences.

        sequences: array of shape (n_samples, sequence_length, n_features)

        Returns a 1-D array of shape (n_samples,) with the predicted
        (scaled or unscaled, depending on how the model was trained)
        temperature for the sample following each sequence.
        """
        if not self.is_loaded:
            raise ModelLoadError("Model is not loaded.")

        try:
            raw_predictions = self._model.predict(sequences, verbose=0)
        except Exception as exc:  # noqa: BLE001 - defensive boundary for GUI safety
            raise ModelLoadError(f"Model inference failed: {exc}") from exc

        predictions = np.asarray(raw_predictions).reshape(-1)
        return predictions
