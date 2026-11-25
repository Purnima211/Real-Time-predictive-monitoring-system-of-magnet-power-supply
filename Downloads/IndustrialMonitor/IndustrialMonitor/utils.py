"""
utils.py
--------
Shared configuration, dark industrial stylesheet, logging setup and
small helper functions (CSV reading, threshold-json loading, dialog boxes)
used across the Industrial Magnet Power Supply Monitoring System.

All functions here are defensive: they never raise uncaught exceptions
into the GUI thread. They return (result, error_message) style tuples
or None on failure so the caller can decide how to present the problem
to the operator.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple  # noqa: F401 (Tuple used by AppConfig)

import pandas as pd
from PySide6.QtWidgets import QMessageBox, QWidget


# ----------------------------------------------------------------------
# Application-wide configuration
#
# These constants describe the shape of data the trained LSTM model
# expects. They are intentionally centralized here so they can be
# adjusted in one place if the model was trained with a different
# sequence length or a different set of input features.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    MODEL_PATH: str = os.path.join("models", "lstm_temperature_model.keras")
    THRESHOLD_PATH: str = os.path.join("models", "anomaly_threshold.json")

    # Column in the CSV that holds the actual/measured temperature.
    TARGET_COLUMN: str = "TrueTemperature_C"

    # Column used for the X axis / sample index. If it is not present
    # in the CSV, a synthetic integer sample index is generated.
    TIME_COLUMN: str = "timestamp"

    # Number of past samples the LSTM consumes to predict the next
    # temperature value. Must match the window size used at training
    # time for lstm_temperature_model.keras. (Overridden at runtime by
    # TemperatureModel.sequence_length, which is inferred from the
    # loaded model's real input shape whenever possible.)
    SEQUENCE_LENGTH: int = 50

    # Fallback anomaly threshold (absolute prediction error) used only
    # if anomaly_threshold.json cannot be read. This keeps the
    # application usable (in a clearly-flagged degraded mode) instead
    # of refusing to run detection at all.
    DEFAULT_THRESHOLD: float = 2.5

    APP_TITLE: str = "Magnet Power Supply — Real-Time Predictive Monitoring"
    ORG_NAME: str = "Accelerator Controls Group"

    # ------------------------------------------------------------------
    # Real-time monitoring dashboard settings
    # ------------------------------------------------------------------
    # Maximum number of most-recent samples kept on the live graph.
    GRAPH_MAX_POINTS: int = 400

    # Base time (ms) between processed samples at 1x simulation speed.
    BASE_INTERVAL_MS: int = 200

    # Available simulation speed multipliers, shown in the speed selector.
    SIMULATION_SPEEDS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0)

    # Anomaly-percentage breakpoints for the gauge / status bands.
    WARNING_PCT: float = 40.0
    CRITICAL_PCT: float = 70.0

    # Interval (ms) between repeated beeps while in CRITICAL state.
    CRITICAL_ALARM_INTERVAL_MS: int = 700

    # Fractional margin added above/below the visible data range when
    # auto-scaling the live graph's Y-axis (so small fluctuations are
    # still clearly visible instead of being squashed to a flat line).
    GRAPH_Y_MARGIN_RATIO: float = 0.12
    # Minimum absolute margin (°C) used when the visible range is
    # extremely flat (near-zero span), to avoid a divide-by-zero-like
    # razor-thin Y range.
    GRAPH_Y_MIN_MARGIN: float = 0.3

    # Blink interval (ms) for the status / alarm LED indicators.
    LED_BLINK_INTERVAL_MS: int = 500


CONFIG = AppConfig()


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("IndustrialMonitor")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        file_handler = logging.FileHandler("monitor.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Non-fatal: continue with console logging only if the log
        # file cannot be created (e.g. read-only working directory).
        pass

    return logger


LOGGER = setup_logging()


# ----------------------------------------------------------------------
# Dark industrial theme (SCADA / control-room styling)
# ----------------------------------------------------------------------
DARK_STYLESHEET = """
QWidget {
    background-color: #1b1f24;
    color: #d7dde3;
    font-family: "Segoe UI", "Consolas", "Arial";
    font-size: 11pt;
}

QMainWindow {
    background-color: #14171b;
}

QToolButton, QPushButton {
    background-color: #262c33;
    border: 1px solid #3a4149;
    border-radius: 3px;
    padding: 8px;
    color: #e5e9ed;
    text-align: left;
}

QToolButton:hover, QPushButton:hover {
    background-color: #313945;
    border: 1px solid #4f8fce;
}

QToolButton:pressed, QPushButton:pressed {
    background-color: #1f242b;
}

QToolButton:disabled, QPushButton:disabled {
    color: #5a6169;
    background-color: #20242a;
    border: 1px solid #2c323a;
}

QLabel {
    color: #c7cdd3;
}

QStatusBar {
    background-color: #14171b;
    border-top: 1px solid #2c323a;
}

QStatusBar QLabel {
    padding: 2px 10px;
    border-right: 1px solid #2c323a;
}

QMenuBar {
    background-color: #14171b;
    color: #d7dde3;
}

QMenuBar::item:selected {
    background-color: #313945;
}

QMenu {
    background-color: #20242a;
    border: 1px solid #3a4149;
}

QMenu::item:selected {
    background-color: #313945;
}

QSplitter::handle {
    background-color: #2c323a;
}

QFrame#LeftPanel {
    background-color: #191d22;
    border-right: 1px solid #2c323a;
}

QLabel#PanelTitle {
    font-weight: bold;
    color: #7fb2e5;
    padding: 6px 4px;
    border-bottom: 1px solid #2c323a;
}

QLabel#StatusOK {
    color: #4caf50;
    font-weight: bold;
}

QLabel#StatusAlarm {
    color: #e53935;
    font-weight: bold;
}

QFrame#StatusCard {
    background-color: #20242a;
    border: 1px solid #2c323a;
    border-radius: 4px;
}

QLabel#CardTitle {
    color: #7d8894;
    font-size: 9pt;
    font-weight: bold;
    letter-spacing: 1px;
}

QLabel#CardValue {
    color: #e5e9ed;
    font-size: 15pt;
    font-weight: bold;
}

QGroupBox {
    border: 1px solid #2c323a;
    border-radius: 4px;
    margin-top: 14px;
    padding-top: 6px;
    font-weight: bold;
    color: #7fb2e5;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}

QComboBox {
    background-color: #262c33;
    border: 1px solid #3a4149;
    border-radius: 3px;
    padding: 5px;
    color: #e5e9ed;
}

QComboBox:hover {
    border: 1px solid #4f8fce;
}

QComboBox QAbstractItemView {
    background-color: #20242a;
    color: #e5e9ed;
    selection-background-color: #313945;
}

QTableWidget {
    background-color: #14171b;
    alternate-background-color: #191d22;
    gridline-color: #2c323a;
    border: 1px solid #2c323a;
}

QHeaderView::section {
    background-color: #20242a;
    color: #7fb2e5;
    padding: 5px;
    border: none;
    border-bottom: 1px solid #2c323a;
    font-weight: bold;
}

QTableWidget::item {
    padding: 3px;
}
"""


# ----------------------------------------------------------------------
# Dialog helpers
# ----------------------------------------------------------------------
def show_error_dialog(parent: Optional[QWidget], title: str, message: str) -> None:
    LOGGER.error("%s: %s", title, message)
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    box.exec()


def show_warning_dialog(parent: Optional[QWidget], title: str, message: str) -> None:
    LOGGER.warning("%s: %s", title, message)
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(message)
    box.exec()


def show_info_dialog(parent: Optional[QWidget], title: str, message: str) -> None:
    LOGGER.info("%s: %s", title, message)
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle(title)
    box.setText(message)
    box.exec()


# ----------------------------------------------------------------------
# Threshold / CSV loading helpers
# ----------------------------------------------------------------------
def load_threshold(path: str = CONFIG.THRESHOLD_PATH) -> Tuple[Optional[float], Optional[str]]:
    """
    Reads anomaly_threshold.json.

    Accepts either a bare numeric value, or an object containing one of
    the keys: "threshold", "anomaly_threshold", "value".

    Returns (threshold, error_message). On success error_message is None.
    On failure threshold is None and error_message describes the problem.
    """
    if not os.path.isfile(path):
        return None, f"Threshold file not found: {path}"

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        return None, f"Threshold file is not valid JSON: {exc}"
    except OSError as exc:
        return None, f"Could not read threshold file: {exc}"

    threshold_value = None
    if isinstance(data, (int, float)):
        threshold_value = float(data)
    elif isinstance(data, dict):
        for key in ("threshold", "anomaly_threshold", "value"):
            if key in data:
                try:
                    threshold_value = float(data[key])
                    break
                except (TypeError, ValueError):
                    return None, f"Threshold field '{key}' is not numeric."
    if threshold_value is None:
        return None, "Threshold JSON did not contain a recognizable numeric value."

    return threshold_value, None


def safe_read_csv(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Reads a CSV file defensively.

    Returns (dataframe, error_message). On success error_message is None.
    """
    if not os.path.isfile(path):
        return None, f"CSV file not found: {path}"

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None, "The selected CSV file is empty."
    except pd.errors.ParserError as exc:
        return None, f"The CSV file could not be parsed: {exc}"
    except OSError as exc:
        return None, f"The CSV file could not be opened: {exc}"
    except Exception as exc:  # noqa: BLE001 - defensive boundary for GUI safety
        return None, f"Unexpected error reading CSV file: {exc}"

    if df.empty:
        return None, "The selected CSV file contains no rows."

    if CONFIG.TARGET_COLUMN not in df.columns:
        return None, (
            f"CSV file is missing the required column "
            f"'{CONFIG.TARGET_COLUMN}'."
        )

    return df, None
