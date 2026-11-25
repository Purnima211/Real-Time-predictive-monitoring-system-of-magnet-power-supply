"""
predictor.py
------------
Bridges the raw CSV data and the LSTM model: builds rolling input
sequences, scales them consistently with how the model was trained,
runs inference, and hands the aligned actual/predicted arrays to the
AnomalyDetector.

Assumption (documented, since no training notebook was supplied):
the model was trained on a sliding window of CONFIG.SEQUENCE_LENGTH
past samples of the numeric feature columns to predict the next
temperature value. Features are standardized (zero mean, unit
variance) with scikit-learn's StandardScaler fitted on the window
itself, which is a common, robust choice for this kind of offline
monitoring tool when the original scaler artifact is not shipped
alongside the model. If the model was trained with a saved scaler,
swap the fit_transform calls below for a loaded scaler's transform.

Two entry points are exposed:

  * run(df, threshold)
        Original batch pipeline: loads a full CSV, fits the scaler on
        it, builds every sliding window at once, and predicts in one
        batched call. Unchanged from the original implementation.

  * initialize_stream(df) + predict_window(window)
        Real-time pipeline used by the SCADA dashboard's monitoring
        worker. initialize_stream() fits the scaler ONCE (over the
        dataset that is about to be streamed sample-by-sample) and
        calibrates the raw-vs-scaled output heuristic once, so units
        stay consistent for the lifetime of the run. predict_window()
        then takes a single rolling window of the latest
        model.sequence_length raw samples and returns one predicted
        temperature, with no per-sample refitting (which would be
        both non-causal and far too slow to run every sample).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from detector import AnomalyDetector, DetectionResult
from model import ModelLoadError, TemperatureModel
from utils import CONFIG, LOGGER


class PredictionError(Exception):
    """Raised when the CSV cannot be turned into valid model input."""


class TemperaturePredictor:
    """
    Orchestrates the full offline pipeline:
        CSV -> feature matrix -> sequences -> LSTM prediction
             -> absolute error -> anomaly detection
    """

    def __init__(self, model: TemperatureModel):
        self._model = model
        # Streaming state, populated by initialize_stream().
        self._stream_ready: bool = False
        self._stream_feature_columns: List[str] = []
        self._stream_scaler: Optional[StandardScaler] = None
        self._stream_target_col_idx: int = 0
        self._stream_looks_scaled: bool = False

    @staticmethod
    def _select_feature_columns(
        df: pd.DataFrame, n_features_expected: Optional[int] = None
    ) -> List[str]:
        """
        Chooses the numeric columns used as model input features.
        The target column (temperature) is always included since the
        model predicts the next temperature from past temperature (and
        any other numeric process variables present, e.g. current,
        voltage, coolant flow).

        If n_features_expected == 1 (a univariate model, e.g. input
        shape (sequence_length, 1) such as this project's LSTM), only
        the target column is used, regardless of how many other numeric
        columns happen to be present in the CSV. This keeps the feature
        set consistent with what the model was actually trained on.
        """
        if CONFIG.TARGET_COLUMN not in df.columns or not pd.api.types.is_numeric_dtype(
            df[CONFIG.TARGET_COLUMN]
        ):
            raise PredictionError(
                f"Target column '{CONFIG.TARGET_COLUMN}' is missing or non-numeric."
            )

        if n_features_expected == 1:
            return [CONFIG.TARGET_COLUMN]

        numeric_columns = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col != CONFIG.TIME_COLUMN
        ]
        if CONFIG.TARGET_COLUMN not in numeric_columns:
            raise PredictionError(
                f"Target column '{CONFIG.TARGET_COLUMN}' is missing or non-numeric."
            )
        return numeric_columns

    def _build_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Builds (sequences, targets, sample_index, feature_columns) from
        the raw dataframe using a sliding window of length
        self._model.sequence_length.
        """
        feature_columns = self._select_feature_columns(df, self._model.n_features)
        seq_len = self._model.sequence_length

        if len(df) <= seq_len:
            raise PredictionError(
                f"CSV must contain more than {seq_len} rows "
                f"(the model's required sequence length); got {len(df)}."
            )

        feature_matrix = df[feature_columns].to_numpy(dtype=float)

        # Guard against NaNs/Infs which would silently corrupt predictions.
        if not np.isfinite(feature_matrix).all():
            raise PredictionError(
                "CSV contains missing or non-numeric values in one of the "
                "required columns. Please clean the data and try again."
            )

        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)

        target_col_idx = feature_columns.index(CONFIG.TARGET_COLUMN)

        n_windows = len(df) - seq_len
        sequences = np.zeros((n_windows, seq_len, len(feature_columns)), dtype=float)
        for i in range(n_windows):
            sequences[i] = scaled[i:i + seq_len]

        # The "actual" value being predicted is the unscaled temperature
        # at the sample immediately following each window.
        actual_targets = feature_matrix[seq_len:seq_len + n_windows, target_col_idx]

        if CONFIG.TIME_COLUMN in df.columns:
            sample_index = df[CONFIG.TIME_COLUMN].to_numpy()[seq_len:seq_len + n_windows]
        else:
            sample_index = np.arange(seq_len, seq_len + n_windows)

        self._scaler = scaler
        self._target_col_idx = target_col_idx
        return sequences, actual_targets, sample_index, feature_columns

    def _inverse_scale_predictions(self, scaled_predictions: np.ndarray) -> np.ndarray:
        """
        The model outputs a prediction in the same scaled feature space
        the sequences were built in (see module docstring). This inverts
        just the target-column scaling so predicted values are directly
        comparable (same units) with the actual temperature column.
        """
        n_features = self._scaler.mean_.shape[0]
        dummy = np.zeros((len(scaled_predictions), n_features))
        dummy[:, self._target_col_idx] = scaled_predictions
        inverted = self._scaler.inverse_transform(dummy)
        return inverted[:, self._target_col_idx]

    def run(self, df: pd.DataFrame, threshold: float) -> DetectionResult:
        """
        Full pipeline entry point. Raises PredictionError or
        ModelLoadError on failure; both carry human-readable messages
        suitable for display in a dialog box.
        """
        if not self._model.is_loaded:
            raise ModelLoadError("Model is not loaded.")

        sequences, actual_targets, sample_index, feature_columns = self._build_sequences(df)

        LOGGER.info(
            "Running inference on %d sequences (%d features: %s)",
            len(sequences), len(feature_columns), feature_columns,
        )

        raw_predictions = self._model.predict_sequences(sequences)

        # If the model's output already appears to be in raw temperature
        # units (i.e. its scale roughly matches the actual data), skip
        # the inverse-scaling step. Otherwise treat it as scaled output.
        predicted_targets = self._reconcile_prediction_scale(raw_predictions, actual_targets)

        result = AnomalyDetector.build_result(
            sample_index=sample_index,
            actual=actual_targets,
            predicted=predicted_targets,
            threshold=threshold,
        )
        return result

    def _reconcile_prediction_scale(
        self, raw_predictions: np.ndarray, actual_targets: np.ndarray
    ) -> np.ndarray:
        """
        Heuristic to keep the tool usable regardless of whether the
        supplied .keras model was trained to output scaled or raw
        temperature values: compares the raw prediction range to the
        actual value range and only applies inverse scaling if the
        predictions look like standardized (roughly zero-mean, small
        magnitude) values rather than physical temperatures.
        """
        pred_std = float(np.std(raw_predictions))
        pred_mean = float(np.mean(raw_predictions))
        actual_mean = float(np.mean(actual_targets))
        actual_std = float(np.std(actual_targets)) or 1.0

        looks_scaled = abs(pred_mean) < 5 and pred_std < 5 and abs(pred_mean - actual_mean) > actual_std

        if looks_scaled:
            try:
                return self._inverse_scale_predictions(raw_predictions)
            except Exception as exc:  # noqa: BLE001 - fall back to raw predictions
                LOGGER.warning("Inverse scaling failed, using raw model output: %s", exc)
                return raw_predictions

        return raw_predictions

    # ------------------------------------------------------------------
    # Real-time / streaming API
    #
    # Used by the SCADA dashboard's monitoring worker to process one
    # sample at a time instead of a whole CSV at once. The scaler and
    # the raw-vs-scaled output decision are both computed ONCE here and
    # then frozen, so predict_window() is cheap and its output units
    # stay consistent for the whole monitoring session.
    # ------------------------------------------------------------------
    def initialize_stream(self, df: pd.DataFrame) -> None:
        """
        Prepares the predictor for sample-by-sample streaming.

        df: the dataset that is about to be streamed (used only to fit
            the feature scaler and to calibrate the output-scale
            heuristic -- it is NOT loaded into memory for later use).

        Raises PredictionError / ModelLoadError on failure.
        """
        if not self._model.is_loaded:
            raise ModelLoadError("Model is not loaded.")

        self._stream_feature_columns = self._select_feature_columns(df, self._model.n_features)
        seq_len = self._model.sequence_length

        if len(df) <= seq_len:
            raise PredictionError(
                f"Dataset must contain more than {seq_len} rows "
                f"(the model's required sequence length); got {len(df)}."
            )

        feature_matrix = df[self._stream_feature_columns].to_numpy(dtype=float)
        if not np.isfinite(feature_matrix).all():
            raise PredictionError(
                "Dataset contains missing or non-numeric values in one of "
                "the required columns. Please clean the data and try again."
            )

        self._stream_scaler = StandardScaler()
        self._stream_scaler.fit(feature_matrix)
        self._stream_target_col_idx = self._stream_feature_columns.index(CONFIG.TARGET_COLUMN)

        # Calibrate the raw-vs-scaled output heuristic once, using the
        # same logic as the batch pipeline, over a small leading slice
        # of the dataset. This avoids re-deciding (and potentially
        # flip-flopping) the output unit on every single incoming
        # sample during live monitoring.
        calib_n = min(len(df) - seq_len, 200)
        scaled_full = self._stream_scaler.transform(feature_matrix)
        calib_windows = np.zeros((calib_n, seq_len, len(self._stream_feature_columns)), dtype=float)
        for i in range(calib_n):
            calib_windows[i] = scaled_full[i:i + seq_len]
        calib_actual = feature_matrix[seq_len:seq_len + calib_n, self._stream_target_col_idx]

        calib_raw_predictions = self._model.predict_sequences(calib_windows)
        pred_std = float(np.std(calib_raw_predictions))
        pred_mean = float(np.mean(calib_raw_predictions))
        actual_mean = float(np.mean(calib_actual))
        actual_std = float(np.std(calib_actual)) or 1.0

        self._stream_looks_scaled = (
            abs(pred_mean) < 5 and pred_std < 5 and abs(pred_mean - actual_mean) > actual_std
        )
        self._stream_ready = True

        LOGGER.info(
            "Streaming predictor initialized (features=%s, sequence_length=%d, "
            "output_looks_scaled=%s)",
            self._stream_feature_columns, seq_len, self._stream_looks_scaled,
        )

    @property
    def feature_columns(self) -> List[str]:
        """Feature columns the currently-initialized stream expects, in order."""
        if not getattr(self, "_stream_ready", False):
            raise PredictionError("Predictor stream not initialized. Call initialize_stream() first.")
        return self._stream_feature_columns

    def predict_window(self, window: np.ndarray) -> float:
        """
        Runs inference on a single rolling window of raw (unscaled)
        samples and returns one predicted temperature value.

        window: array of shape (sequence_length, n_features), with
        columns in the same order as self.feature_columns, containing
        the most recent raw sensor readings (oldest first).
        """
        if not getattr(self, "_stream_ready", False):
            raise PredictionError("Predictor stream not initialized. Call initialize_stream() first.")

        window = np.asarray(window, dtype=float)
        if not np.isfinite(window).all():
            raise PredictionError("Rolling window contains missing or non-numeric values.")

        scaled_window = self._stream_scaler.transform(window)
        sequence = scaled_window.reshape(1, scaled_window.shape[0], scaled_window.shape[1])
        raw_prediction = self._model.predict_sequences(sequence)

        if self._stream_looks_scaled:
            n_features = self._stream_scaler.mean_.shape[0]
            dummy = np.zeros((1, n_features))
            dummy[:, self._stream_target_col_idx] = raw_prediction
            inverted = self._stream_scaler.inverse_transform(dummy)
            predicted = inverted[:, self._stream_target_col_idx]
        else:
            predicted = raw_prediction

        return float(predicted[0])
