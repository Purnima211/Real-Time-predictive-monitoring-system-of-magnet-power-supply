"""
ui.py
-----
Main application window for the Industrial Magnet Power Supply
Real-Time Predictive Monitoring Dashboard. PySide6 + PyQtGraph, dark
industrial / SCADA theme.

Layout:
    - Left panel:   controls (Load / Start / Pause / Resume / Stop /
                     Reset / Export Logs), simulation speed selector,
                     system status box.
    - Main area:
        * Live actual-vs-predicted temperature graph (auto-scrolling,
          latest GRAPH_MAX_POINTS samples only).
        * Live status cards + circular anomaly gauge.
        * Event log table (state transitions only).

Real-time pipeline (see MonitoringWorker):
    read one sample -> push into rolling buffer of the latest
    model.sequence_length samples -> once full, run LSTM prediction via
    TemperaturePredictor.predict_window() -> compute error -> anomaly
    detection via AnomalyDetector -> emit result to the GUI thread.

The data source is deliberately abstracted (data_source.DataSource) so
a future PySerial-based PT100 sensor feed can replace CSVDataSource
without any change to this file.
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QMutex, QMutexLocker, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from data_source import CSVDataSource, DataSourceError
from detector import AnomalyDetector
from model import ModelLoadError, TemperatureModel
from predictor import PredictionError, TemperaturePredictor
from utils import (
    CONFIG,
    LOGGER,
    load_threshold,
    safe_read_csv,
    show_error_dialog,
    show_info_dialog,
    show_warning_dialog,
)
from widgets import AnomalyGauge, LabeledLed, StatusCard

# ----------------------------------------------------------------------
# Live-graph palette (matches the dark stylesheet in utils.py).
# ----------------------------------------------------------------------
COLOR_ACTUAL = "#4fc3f7"
COLOR_PREDICTED = "#ffb74d"
COLOR_PLOT_BG = "#14171b"
COLOR_GRID = "#2c323a"

STATUS_COLORS = {
    "NORMAL": "#4caf50",
    "WARNING": "#ffb300",
    "CRITICAL": "#e53935",
    "BUFFERING": "#7d8894",
    "STOPPED": "#7d8894",
}


@dataclass
class SampleResult:
    """One processed sample, emitted from the worker thread to the GUI thread."""

    index: int
    actual: float
    predicted: Optional[float]
    error: Optional[float]
    anomaly_pct: float
    status: str  # "BUFFERING" | "NORMAL" | "WARNING" | "CRITICAL"
    timestamp: str
    total_samples: Optional[int]


class MonitoringWorker(QThread):
    """
    Streams samples one at a time from a DataSource, maintains a
    rolling buffer of the latest model.sequence_length samples, and
    runs prediction + anomaly detection once the buffer is full.

    Runs entirely on a background thread; the GUI is only ever updated
    through the sample_ready / stream_error / stream_finished signals,
    so the UI never blocks while streaming or during inference.
    """

    sample_ready = Signal(object)   # SampleResult
    stream_error = Signal(str)
    stream_finished = Signal()

    def __init__(
        self,
        data_source: CSVDataSource,
        predictor: TemperaturePredictor,
        feature_columns: List[str],
        sequence_length: int,
        threshold: float,
        base_interval_ms: int = CONFIG.BASE_INTERVAL_MS,
        parent=None,
    ):
        super().__init__(parent)
        self._data_source = data_source
        self._predictor = predictor
        self._feature_columns = feature_columns
        self._sequence_length = sequence_length
        self._threshold = threshold
        self._base_interval_ms = base_interval_ms

        self._mutex = QMutex()
        self._speed = 1.0
        self._pause_flag = False
        self._stop_flag = False

    # -- thread-safe controls, callable from the GUI thread ---------
    def set_speed(self, speed: float) -> None:
        with QMutexLocker(self._mutex):
            self._speed = speed

    def pause(self) -> None:
        with QMutexLocker(self._mutex):
            self._pause_flag = True

    def resume(self) -> None:
        with QMutexLocker(self._mutex):
            self._pause_flag = False

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._stop_flag = True

    def _is_paused(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._pause_flag

    def _should_stop(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._stop_flag

    def _current_speed(self) -> float:
        with QMutexLocker(self._mutex):
            return self._speed

    # -- worker thread body -------------------------------------------
    def run(self) -> None:
        try:
            self._data_source.open()
        except DataSourceError as exc:
            self.stream_error.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never crash the app
            LOGGER.exception("Unexpected error opening data source")
            self.stream_error.emit(f"Unexpected error opening data source: {exc}")
            return

        total = self._data_source.total_samples()
        buffer: deque = deque(maxlen=self._sequence_length)
        idx = 0

        while not self._should_stop():
            if self._is_paused():
                self.msleep(100)
                continue

            try:
                sample = self._data_source.read_sample()
            except DataSourceError as exc:
                self.stream_error.emit(str(exc))
                break
            except Exception as exc:  # noqa: BLE001 - defensive boundary
                LOGGER.exception("Unexpected error reading sample")
                self.stream_error.emit(f"Unexpected error reading sample: {exc}")
                break

            if sample is None:
                break  # data source exhausted

            idx += 1
            try:
                feature_vec = np.array([sample[col] for col in self._feature_columns], dtype=float)
            except KeyError as exc:
                self.stream_error.emit(f"Sample is missing expected column: {exc}")
                break
            buffer.append(feature_vec)

            actual = float(sample[CONFIG.TARGET_COLUMN])
            predicted: Optional[float] = None
            error: Optional[float] = None
            anomaly_pct = 0.0
            status = "BUFFERING"

            if len(buffer) == self._sequence_length:
                window = np.array(buffer, dtype=float)
                try:
                    predicted = self._predictor.predict_window(window)
                except (PredictionError, ModelLoadError) as exc:
                    self.stream_error.emit(str(exc))
                    break
                except Exception as exc:  # noqa: BLE001 - defensive boundary
                    LOGGER.exception("Unexpected error during streaming prediction")
                    self.stream_error.emit(f"Unexpected error during prediction: {exc}")
                    break

                error = float(AnomalyDetector.compute_absolute_error(
                    np.array([actual]), np.array([predicted])
                )[0])
                anomaly_pct = self._anomaly_percentage(error)
                status = self._status_from_pct(anomaly_pct)

            self.sample_ready.emit(SampleResult(
                index=idx, actual=actual, predicted=predicted, error=error,
                anomaly_pct=anomaly_pct, status=status,
                timestamp=time.strftime("%H:%M:%S"), total_samples=total,
            ))

            speed = self._current_speed() or 1.0
            interval_ms = max(1, int(self._base_interval_ms / speed))
            self.msleep(interval_ms)

        self._data_source.close()
        self.stream_finished.emit()

    def _anomaly_percentage(self, error: float) -> float:
        if self._threshold <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * error / self._threshold))

    @staticmethod
    def _status_from_pct(pct: float) -> str:
        if pct <= CONFIG.WARNING_PCT:
            return "NORMAL"
        if pct <= CONFIG.CRITICAL_PCT:
            return "WARNING"
        return "CRITICAL"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(CONFIG.APP_TITLE)
        self.resize(1500, 950)

        self._model = TemperatureModel()
        self._predictor: Optional[TemperaturePredictor] = None
        self._threshold: float = CONFIG.DEFAULT_THRESHOLD
        self._threshold_is_default = True

        self._current_csv_path: Optional[str] = None
        self._current_df_len: int = 0
        self._worker: Optional[MonitoringWorker] = None

        self._last_status: Optional[str] = None
        self._samples_processed = 0
        self._monitor_start_time: Optional[float] = None
        self._prediction_active = False
        self._monitoring_running = False
        self._monitoring_paused = False

        # Live statistics (reset in on_reset()).
        self._stat_min: Optional[float] = None
        self._stat_max: Optional[float] = None
        self._stat_sum: float = 0.0
        self._stat_count: int = 0
        self._count_warnings: int = 0
        self._count_criticals: int = 0

        # Rolling arrays backing the live graph.
        self._graph_x: deque = deque(maxlen=CONFIG.GRAPH_MAX_POINTS)
        self._graph_actual: deque = deque(maxlen=CONFIG.GRAPH_MAX_POINTS)
        self._graph_predicted: deque = deque(maxlen=CONFIG.GRAPH_MAX_POINTS)

        self._build_ui()
        self._load_backend()

        self._critical_alarm_timer = QTimer(self)
        self._critical_alarm_timer.timeout.connect(QApplication.beep)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_monitoring_time)
        self._clock_timer.start(1000)

        self._update_readiness_checklist()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        main_area = self._build_main_area()  # builds table_log, needed by the left panel's button-state logic
        root_layout.addWidget(self._build_left_panel())
        root_layout.addWidget(main_area, stretch=1)

        self.setCentralWidget(central)
        self._build_status_bar()

    def _build_left_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("LeftPanel")
        panel.setFixedWidth(250)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("CONTROLS")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.btn_load = QPushButton("  Load Dataset")
        self.btn_start = QPushButton("  Start Monitoring")
        self.btn_pause = QPushButton("  Pause")
        self.btn_resume = QPushButton("  Resume")
        self.btn_stop = QPushButton("  Stop")
        self.btn_reset = QPushButton("  Reset")
        self.btn_export = QPushButton("  Export Logs")

        for btn in (
            self.btn_load, self.btn_start, self.btn_pause,
            self.btn_resume, self.btn_stop, self.btn_reset, self.btn_export,
        ):
            btn.setMinimumHeight(36)
            layout.addWidget(btn)

        speed_group = QGroupBox("SIMULATION SPEED")
        speed_layout = QVBoxLayout(speed_group)
        self.combo_speed = QComboBox()
        for speed in CONFIG.SIMULATION_SPEEDS:
            self.combo_speed.addItem(f"{speed:g}x", speed)
        self.combo_speed.setCurrentIndex(list(CONFIG.SIMULATION_SPEEDS).index(1.0))
        speed_layout.addWidget(self.combo_speed)
        layout.addWidget(speed_group)

        layout.addWidget(self._build_system_status_box())
        layout.addWidget(self._build_readiness_checklist_box())
        layout.addStretch(1)

        self.btn_load.clicked.connect(self.on_load_dataset)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_resume.clicked.connect(self.on_resume)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_export.clicked.connect(self.on_export_logs)
        self.combo_speed.currentIndexChanged.connect(self.on_speed_changed)

        self._set_monitoring_controls_enabled(has_dataset=False, running=False, paused=False)
        return panel

    def _build_system_status_box(self) -> QGroupBox:
        box = QGroupBox("SYSTEM STATUS")
        layout = QVBoxLayout(box)
        layout.setSpacing(3)

        self.lbl_model_status = QLabel("Model: not loaded")
        self.lbl_threshold_status = QLabel("Threshold: not loaded")
        self.lbl_dataset_status = QLabel("Dataset: none")
        self.lbl_dataset_status.setToolTip("")
        self.lbl_samples_status = QLabel("Samples: 0 / 0")
        self.lbl_time_status = QLabel("Monitoring time: 00:00:00")
        self.lbl_fps_status = QLabel("Rate: -- samples/s")

        for lbl in (
            self.lbl_model_status, self.lbl_threshold_status, self.lbl_dataset_status,
            self.lbl_samples_status, self.lbl_time_status, self.lbl_fps_status,
        ):
            # Dataset path can be long; elide it instead of letting it
            # overflow the fixed-width left panel. Other labels still
            # wrap normally since they're short.
            lbl.setWordWrap(lbl is not self.lbl_dataset_status)
            layout.addWidget(lbl)

        return box

    def _build_readiness_checklist_box(self) -> QGroupBox:
        box = QGroupBox("SYSTEM READINESS")
        layout = QVBoxLayout(box)
        layout.setSpacing(3)

        self.lbl_check_model = QLabel()
        self.lbl_check_threshold = QLabel()
        self.lbl_check_dataset = QLabel()
        self.lbl_check_monitoring = QLabel()
        self.lbl_check_prediction = QLabel()
        self.lbl_check_alarm = QLabel()
        self.lbl_check_logging = QLabel()

        self._checklist_labels = {
            "Model Loaded": self.lbl_check_model,
            "Threshold Loaded": self.lbl_check_threshold,
            "Dataset Loaded": self.lbl_check_dataset,
            "Monitoring Running": self.lbl_check_monitoring,
            "Prediction Active": self.lbl_check_prediction,
            "Alarm Ready": self.lbl_check_alarm,
            "Logging Enabled": self.lbl_check_logging,
        }
        for lbl in self._checklist_labels.values():
            layout.addWidget(lbl)

        return box

    def _build_main_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_graph_panel())
        splitter.addWidget(self._build_middle_row())
        splitter.addWidget(self._build_statistics_panel())
        splitter.addWidget(self._build_event_log_panel())
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setStretchFactor(3, 3)

        layout.addWidget(splitter)
        return container

    def _build_statistics_panel(self) -> QWidget:
        box = QGroupBox("LIVE STATISTICS")
        layout = QGridLayout(box)
        layout.setSpacing(8)

        self.card_stat_max = StatusCard("Max Temperature")
        self.card_stat_min = StatusCard("Min Temperature")
        self.card_stat_avg = StatusCard("Avg Temperature")
        self.card_stat_duration = StatusCard("Monitoring Duration")
        self.card_stat_warnings = StatusCard("Total Warnings")
        self.card_stat_criticals = StatusCard("Total Critical Events")

        cards = [
            self.card_stat_max, self.card_stat_min, self.card_stat_avg,
            self.card_stat_duration, self.card_stat_warnings, self.card_stat_criticals,
        ]
        for i, card in enumerate(cards):
            layout.addWidget(card, i // 6, i % 6)

        return box

    def _build_graph_panel(self) -> QWidget:
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget(background=COLOR_PLOT_BG)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("left", "Temperature (\u00b0C)")
        self.plot_widget.setLabel("bottom", "Sample")
        self.plot_widget.setTitle("Live Temperature — Actual vs Predicted", color="#c7cdd3", size="11pt")
        self.plot_widget.addLegend(offset=(10, 10))

        # We manage the visible range ourselves (sliding X window + a
        # margin-padded Y range around only the currently-visible data)
        # instead of relying on pyqtgraph's automatic range detection,
        # which recomputes on every point and produces visible jitter.
        self.plot_widget.enableAutoRange(x=False, y=False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)

        pen_actual = pg.mkPen(color=COLOR_ACTUAL, width=2)
        pen_predicted = pg.mkPen(color=COLOR_PREDICTED, width=2, style=Qt.PenStyle.DashLine)

        # No downsampling/interpolation: every incoming sample is plotted
        # exactly as received, preserving real sensor fluctuations.
        self.curve_actual = self.plot_widget.plot(
            [], [], pen=pen_actual, name="Actual Temperature", skipFiniteCheck=False,
        )
        self.curve_predicted = self.plot_widget.plot(
            [], [], pen=pen_predicted, name="Predicted Temperature", connect="finite",
        )

        return self.plot_widget

    def _update_graph_ranges(self) -> None:
        """
        Slides the X-axis to always show the latest window of samples,
        and auto-scales the Y-axis to the visible data's own min/max
        (with a small margin) so gentle real fluctuations are still
        clearly visible instead of looking like a flat line.
        """
        if not self._graph_x:
            return

        x_min, x_max = self._graph_x[0], self._graph_x[-1]
        if x_max == x_min:
            x_max = x_min + 1
        self.plot_widget.setXRange(x_min, x_max, padding=0.02)

        visible_values = [v for v in self._graph_actual if np.isfinite(v)]
        visible_values += [v for v in self._graph_predicted if np.isfinite(v)]
        if not visible_values:
            return

        y_min, y_max = min(visible_values), max(visible_values)
        span = y_max - y_min
        margin = max(span * CONFIG.GRAPH_Y_MARGIN_RATIO, CONFIG.GRAPH_Y_MIN_MARGIN)
        self.plot_widget.setYRange(y_min - margin, y_max + margin, padding=0)

    def _build_middle_row(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        cards_frame = QFrame()
        grid = QGridLayout(cards_frame)
        grid.setSpacing(8)

        self.card_current_temp = StatusCard("Current Temperature")
        self.card_predicted_temp = StatusCard("Predicted Temperature")
        self.card_error = StatusCard("Prediction Error")
        self.card_threshold = StatusCard("Threshold")
        self.card_status = StatusCard("System Status")
        self.card_last_update = StatusCard("Last Update")

        cards = [
            self.card_current_temp, self.card_predicted_temp, self.card_error,
            self.card_threshold, self.card_status, self.card_last_update,
        ]
        for i, card in enumerate(cards):
            grid.addWidget(card, i // 3, i % 3)

        layout.addWidget(cards_frame, stretch=3)

        # -- indicators column: blinking status LED + alarm LED --
        indicators_frame = QGroupBox("INDICATORS")
        indicators_layout = QVBoxLayout(indicators_frame)
        indicators_layout.setSpacing(8)

        self.led_status = LabeledLed("Status")
        self.led_alarm = LabeledLed("Alarm")
        indicators_layout.addWidget(self.led_status)
        indicators_layout.addWidget(self.led_alarm)
        indicators_layout.addStretch(1)

        layout.addWidget(indicators_frame, stretch=2)

        # -- gauge column: gauge + explicit percentage label below it --
        gauge_frame = QFrame()
        gauge_layout = QVBoxLayout(gauge_frame)
        gauge_layout.setContentsMargins(0, 0, 0, 0)
        gauge_layout.setSpacing(4)

        self.gauge = AnomalyGauge()
        self.lbl_gauge_pct = QLabel("Anomaly Level: 0.0%")
        self.lbl_gauge_pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_gauge_pct.setStyleSheet("color: #c7cdd3; font-weight: bold; font-size: 10pt;")

        gauge_layout.addWidget(self.gauge, stretch=1)
        gauge_layout.addWidget(self.lbl_gauge_pct)

        layout.addWidget(gauge_frame, stretch=3)

        return container

    def _build_event_log_panel(self) -> QWidget:
        container = QGroupBox("EVENT LOG — STATE TRANSITIONS")
        layout = QVBoxLayout(container)

        header_row = QHBoxLayout()
        header_row.addStretch(1)
        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.setMaximumWidth(120)
        self.btn_clear_log.clicked.connect(self.on_clear_log)
        header_row.addWidget(self.btn_clear_log)
        layout.addLayout(header_row)

        self.table_log = QTableWidget(0, 6)
        self.table_log.setHorizontalHeaderLabels([
            "Timestamp", "Actual Temp (\u00b0C)", "Predicted Temp (\u00b0C)",
            "Prediction Error", "Anomaly %", "Status",
        ])
        self.table_log.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_log.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_log.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_log.setAlternatingRowColors(True)

        layout.addWidget(self.table_log)
        return container

    def _build_status_bar(self):
        bar = QStatusBar()
        mono = QFont("Consolas")

        self.lbl_bar_sample = QLabel("Sample: --")
        self.lbl_bar_state = QLabel("NO DATA")
        self.lbl_bar_state.setObjectName("StatusOK")

        self.lbl_bar_sample.setFont(mono)
        bar.addWidget(self.lbl_bar_sample)
        bar.addPermanentWidget(self.lbl_bar_state)
        self.setStatusBar(bar)

    # ------------------------------------------------------------------
    # Backend (model + threshold) loading — unchanged from the original
    # batch application, reused as-is.
    # ------------------------------------------------------------------
    def _load_backend(self):
        try:
            self._model.load(CONFIG.MODEL_PATH)
            self.lbl_model_status.setText(f"Model: loaded ({os.path.basename(CONFIG.MODEL_PATH)})")
        except ModelLoadError as exc:
            self.lbl_model_status.setText("Model: FAILED TO LOAD")
            show_error_dialog(
                self, "Model Load Error",
                f"{exc}\n\nThe application will remain open, but monitoring "
                f"cannot run until a valid model file is available at:\n"
                f"{CONFIG.MODEL_PATH}",
            )

        threshold, error = load_threshold(CONFIG.THRESHOLD_PATH)
        if error is not None:
            self._threshold = CONFIG.DEFAULT_THRESHOLD
            self._threshold_is_default = True
            self.lbl_threshold_status.setText(f"Threshold: using default ({CONFIG.DEFAULT_THRESHOLD})")
            show_warning_dialog(
                self, "Threshold Load Error",
                f"{error}\n\nFalling back to the default threshold "
                f"({CONFIG.DEFAULT_THRESHOLD}). Anomaly percentages may not "
                f"reflect the calibrated boundary.",
            )
        else:
            self._threshold = threshold
            self._threshold_is_default = False
            self.lbl_threshold_status.setText(f"Threshold: {threshold}")

        self.card_threshold.set_value(f"{self._threshold:.4f}")
        self._predictor = TemperaturePredictor(self._model)
        self._update_readiness_checklist()

    # ------------------------------------------------------------------
    # Control state helpers
    # ------------------------------------------------------------------
    def _set_monitoring_controls_enabled(self, has_dataset: bool, running: bool, paused: bool):
        self.btn_load.setEnabled(not running)
        self.btn_start.setEnabled(has_dataset and not running and self._model.is_loaded)
        self.btn_pause.setEnabled(running and not paused)
        self.btn_resume.setEnabled(running and paused)
        self.btn_stop.setEnabled(running)
        self.btn_reset.setEnabled(not running)
        self.btn_export.setEnabled(self.table_log.rowCount() > 0)
        self.btn_clear_log.setEnabled(self.table_log.rowCount() > 0)
        self._monitoring_running = running
        self._monitoring_paused = paused
        self._update_readiness_checklist()

    def _set_dataset_label(self, path: str, row_count: int) -> None:
        """Elides long dataset paths so they never overflow the fixed-width left panel."""
        metrics = QFontMetrics(self.lbl_dataset_status.font())
        available_width = self.lbl_dataset_status.width() or 210
        elided = metrics.elidedText(os.path.basename(path), Qt.TextElideMode.ElideMiddle, available_width)
        self.lbl_dataset_status.setText(f"Dataset: {elided} ({row_count} rows)")
        self.lbl_dataset_status.setToolTip(path)

    def _update_readiness_checklist(self) -> None:
        """
        Single source of truth for the SYSTEM READINESS checklist.
        Called whenever any of the underlying conditions change, so it
        always reflects the live state of the same data stream driving
        the rest of the dashboard.
        """
        model_loaded = self._model.is_loaded
        threshold_loaded = not self._threshold_is_default
        dataset_loaded = self._current_csv_path is not None
        monitoring_running = getattr(self, "_monitoring_running", False)
        prediction_active = self._prediction_active
        alarm_ready = model_loaded and threshold_loaded
        logging_enabled = model_loaded  # event log is always active once the app is usable

        states = {
            "Model Loaded": model_loaded,
            "Threshold Loaded": threshold_loaded,
            "Dataset Loaded": dataset_loaded,
            "Monitoring Running": monitoring_running,
            "Prediction Active": prediction_active,
            "Alarm Ready": alarm_ready,
            "Logging Enabled": logging_enabled,
        }
        for name, ok in states.items():
            label = self._checklist_labels[name]
            if ok:
                label.setText(f"\u2713  {name}")
                label.setStyleSheet("color: #4caf50; font-weight: bold;")
            else:
                label.setText(f"\u2717  {name}")
                label.setStyleSheet("color: #6b7280;")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def on_load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Magnet Power Supply Dataset", "", "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        df, error = safe_read_csv(path)
        if error is not None:
            show_error_dialog(self, "Invalid CSV", error)
            return

        if not self._model.is_loaded or self._predictor is None:
            show_error_dialog(self, "Model Not Available", "The LSTM model is not loaded.")
            return

        try:
            self._predictor.initialize_stream(df)
        except (PredictionError, ModelLoadError) as exc:
            show_error_dialog(self, "Dataset Not Compatible", str(exc))
            return

        self._current_csv_path = path
        self._current_df_len = len(df)
        self._set_dataset_label(path, len(df))

        self.on_reset()  # clear any previous run's graph/log/state
        self._set_monitoring_controls_enabled(has_dataset=True, running=False, paused=False)

    def on_start(self):
        if self._current_csv_path is None or self._predictor is None:
            show_warning_dialog(self, "No Dataset", "Load a dataset before starting monitoring.")
            return

        self._clear_live_state()  # every Start begins a clean, non-overlapping session

        data_source = CSVDataSource(self._current_csv_path, self._predictor.feature_columns)
        self._worker = MonitoringWorker(
            data_source=data_source,
            predictor=self._predictor,
            feature_columns=self._predictor.feature_columns,
            sequence_length=self._model.sequence_length,
            threshold=self._threshold,
            base_interval_ms=CONFIG.BASE_INTERVAL_MS,
        )
        self._worker.set_speed(self.combo_speed.currentData())
        self._worker.sample_ready.connect(self._on_sample_ready)
        self._worker.stream_error.connect(self._on_stream_error)
        self._worker.stream_finished.connect(self._on_stream_finished)

        self._samples_processed = 0
        self._monitor_start_time = time.time()
        self.card_status.set_value("RUNNING", STATUS_COLORS["BUFFERING"])
        self.led_status.set_state(STATUS_COLORS["BUFFERING"], blinking=True, text="RUNNING")
        self.lbl_bar_state.setText("MONITORING")
        self.lbl_bar_state.setStyleSheet("color: #4caf50; font-weight: bold;")

        self._worker.start()
        self._set_monitoring_controls_enabled(has_dataset=True, running=True, paused=False)

    def on_pause(self):
        if self._worker is not None:
            self._worker.pause()
            self.card_status.set_value("PAUSED", STATUS_COLORS["BUFFERING"])
            self.led_status.set_state(STATUS_COLORS["BUFFERING"], blinking=False, text="PAUSED")
            self._set_monitoring_controls_enabled(has_dataset=True, running=True, paused=True)

    def on_resume(self):
        if self._worker is not None:
            self._worker.resume()
            self.card_status.set_value("RUNNING", STATUS_COLORS["BUFFERING"])
            self.led_status.set_state(STATUS_COLORS["BUFFERING"], blinking=True, text="RUNNING")
            self._set_monitoring_controls_enabled(has_dataset=True, running=True, paused=False)

    def on_stop(self):
        if self._worker is not None:
            self._worker.stop()
            self.btn_stop.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_resume.setEnabled(False)

    def on_reset(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        self._worker = None
        self._clear_live_state()
        self._set_monitoring_controls_enabled(
            has_dataset=self._current_csv_path is not None, running=False, paused=False,
        )

    def _clear_live_state(self) -> None:
        """
        Clears everything the live stream drives -- graph, gauge, LEDs,
        status cards, live statistics, and the event log -- without
        touching the loaded dataset/model/predictor. Shared by Start
        (so each run begins clean) and Reset.
        """
        self._graph_x.clear()
        self._graph_actual.clear()
        self._graph_predicted.clear()
        self.curve_actual.setData([], [])
        self.curve_predicted.setData([], [])

        self.table_log.setRowCount(0)
        self._last_status = None
        self._samples_processed = 0
        self._monitor_start_time = None
        self._prediction_active = False

        self._stat_min = None
        self._stat_max = None
        self._stat_sum = 0.0
        self._stat_count = 0
        self._count_warnings = 0
        self._count_criticals = 0

        self.gauge.set_value(0)
        self.lbl_gauge_pct.setText("Anomaly Level: 0.0%")
        for card in (
            self.card_current_temp, self.card_predicted_temp,
            self.card_error, self.card_last_update,
        ):
            card.set_value("--")
        for card in (
            self.card_stat_max, self.card_stat_min, self.card_stat_avg,
            self.card_stat_duration, self.card_stat_warnings, self.card_stat_criticals,
        ):
            card.set_value("--")
        self.card_status.set_value("READY")
        self.led_status.set_state("#4a5158", blinking=False, text="IDLE")
        self.led_alarm.set_state("#4a5158", blinking=False, text="IDLE")

        self.lbl_samples_status.setText(f"Samples: 0 / {self._current_df_len}")
        self.lbl_time_status.setText("Monitoring time: 00:00:00")
        self.lbl_fps_status.setText("Rate: -- samples/s")
        self.lbl_bar_sample.setText("Sample: --")
        self.lbl_bar_state.setText("READY" if self._current_csv_path else "NO DATA")
        self.lbl_bar_state.setStyleSheet("")

        if self._critical_alarm_timer.isActive():
            self._critical_alarm_timer.stop()

        self._update_readiness_checklist()

    def on_export_logs(self):
        if self.table_log.rowCount() == 0:
            show_warning_dialog(self, "Nothing to Export", "The event log is empty.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Event Log as CSV", "event_log.csv", "CSV Files (*.csv)",
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                headers = [
                    self.table_log.horizontalHeaderItem(c).text()
                    for c in range(self.table_log.columnCount())
                ]
                writer.writerow(headers)
                for row in range(self.table_log.rowCount()):
                    writer.writerow([
                        self.table_log.item(row, col).text() if self.table_log.item(row, col) else ""
                        for col in range(self.table_log.columnCount())
                    ])
            show_info_dialog(self, "Log Exported", f"Event log exported to:\n{path}")
        except OSError as exc:
            show_error_dialog(self, "Export Failed", f"Could not export event log: {exc}")

    def on_clear_log(self):
        if self.table_log.rowCount() == 0:
            return
        reply = QMessageBox.question(
            self, "Clear Event Log", "Clear all event log entries? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.table_log.setRowCount(0)
            self.btn_export.setEnabled(False)
            self.btn_clear_log.setEnabled(False)

    def on_speed_changed(self, _index: int):
        if self._worker is not None:
            self._worker.set_speed(self.combo_speed.currentData())

    # ------------------------------------------------------------------
    # Worker signal handlers (run on the GUI thread)
    # ------------------------------------------------------------------
    def _on_sample_ready(self, result: SampleResult):
        self._samples_processed = result.index

        # -- live graph: append the raw sample as-is (no smoothing/averaging) --
        self._graph_x.append(result.index)
        self._graph_actual.append(result.actual)
        self._graph_predicted.append(result.predicted if result.predicted is not None else float("nan"))

        x = np.fromiter(self._graph_x, dtype=float)
        self.curve_actual.setData(x, np.fromiter(self._graph_actual, dtype=float))
        self.curve_predicted.setData(x, np.fromiter(self._graph_predicted, dtype=float))
        self._update_graph_ranges()

        # -- live statistics (actual temperature is available from sample 1) --
        self._stat_count += 1
        self._stat_sum += result.actual
        self._stat_min = result.actual if self._stat_min is None else min(self._stat_min, result.actual)
        self._stat_max = result.actual if self._stat_max is None else max(self._stat_max, result.actual)
        self.card_stat_max.set_value(f"{self._stat_max:.2f} \u00b0C")
        self.card_stat_min.set_value(f"{self._stat_min:.2f} \u00b0C")
        self.card_stat_avg.set_value(f"{self._stat_sum / self._stat_count:.2f} \u00b0C")
        self.card_stat_warnings.set_value(str(self._count_warnings))
        self.card_stat_criticals.set_value(str(self._count_criticals))

        # -- status cards --
        self.card_current_temp.set_value(f"{result.actual:.2f} \u00b0C")
        self.card_last_update.set_value(result.timestamp)

        if result.predicted is not None:
            if not self._prediction_active:
                self._prediction_active = True
                self._update_readiness_checklist()

            self.card_predicted_temp.set_value(f"{result.predicted:.2f} \u00b0C")
            self.card_error.set_value(f"{result.error:.4f}")
            self.gauge.set_value(result.anomaly_pct)
            self.lbl_gauge_pct.setText(f"Anomaly Level: {result.anomaly_pct:.1f}%")
            status_color = STATUS_COLORS.get(result.status, "#c7cdd3")
            self.card_status.set_value(result.status, status_color)
            self.led_status.set_state(status_color, blinking=True, text=result.status)
        else:
            self.card_predicted_temp.set_value("buffering...")
            self.card_error.set_value("--")
            self.card_status.set_value("BUFFERING", STATUS_COLORS["BUFFERING"])
            self.led_status.set_state(STATUS_COLORS["BUFFERING"], blinking=True, text="BUFFERING")

        # -- system status box --
        total_text = str(result.total_samples) if result.total_samples is not None else "?"
        self.lbl_samples_status.setText(f"Samples: {result.index} / {total_text}")
        self.lbl_bar_sample.setText(f"Sample: {result.index}")

        if self._monitor_start_time is not None:
            elapsed = max(time.time() - self._monitor_start_time, 1e-6)
            fps = result.index / elapsed
            self.lbl_fps_status.setText(f"Rate: {fps:.1f} samples/s")

        # -- state-transition event log + alarms (never repeated for the same state) --
        if result.status != "BUFFERING" and result.status != self._last_status:
            if result.status == "WARNING":
                self._count_warnings += 1
                self.card_stat_warnings.set_value(str(self._count_warnings))
            elif result.status == "CRITICAL":
                self._count_criticals += 1
                self.card_stat_criticals.set_value(str(self._count_criticals))

            self._append_event_log(result)
            self._handle_alarm_transition(result.status)
            self._last_status = result.status
            self.lbl_bar_state.setText(result.status)
            self.lbl_bar_state.setStyleSheet(
                f"color: {STATUS_COLORS.get(result.status, '#c7cdd3')}; font-weight: bold;"
            )

    def _append_event_log(self, result: SampleResult):
        row = self.table_log.rowCount()
        self.table_log.insertRow(row)
        values = [
            result.timestamp,
            f"{result.actual:.3f}",
            f"{result.predicted:.3f}" if result.predicted is not None else "--",
            f"{result.error:.4f}" if result.error is not None else "--",
            f"{result.anomaly_pct:.1f}",
            result.status,
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_log.setItem(row, col, item)
        self.table_log.scrollToBottom()
        self.btn_export.setEnabled(True)
        self.btn_clear_log.setEnabled(True)

    def _handle_alarm_transition(self, new_status: str):
        """
        Drives both the audible alarm and the Alarm LED. Only ever
        called on an actual state transition (see _on_sample_ready), so
        beeps/LED changes never repeat while the system stays in the
        same state.
        """
        if new_status == "CRITICAL":
            if not self._critical_alarm_timer.isActive():
                QApplication.beep()
                self._critical_alarm_timer.start(CONFIG.CRITICAL_ALARM_INTERVAL_MS)
            self.led_alarm.set_state(STATUS_COLORS["CRITICAL"], blinking=True, text="CRITICAL ALARM")
        elif new_status == "WARNING":
            if self._critical_alarm_timer.isActive():
                self._critical_alarm_timer.stop()
            QApplication.beep()
            self.led_alarm.set_state(STATUS_COLORS["WARNING"], blinking=True, text="WARNING")
        else:  # NORMAL
            if self._critical_alarm_timer.isActive():
                self._critical_alarm_timer.stop()
            self.led_alarm.set_state("#4a5158", blinking=False, text="IDLE")

    def _on_stream_error(self, message: str):
        show_error_dialog(self, "Monitoring Error", message)
        self._set_monitoring_controls_enabled(
            has_dataset=self._current_csv_path is not None, running=False, paused=False,
        )
        if self._critical_alarm_timer.isActive():
            self._critical_alarm_timer.stop()
        self.led_status.set_state(STATUS_COLORS["CRITICAL"], blinking=False, text="ERROR")
        self.led_alarm.set_state("#4a5158", blinking=False, text="IDLE")

    def _on_stream_finished(self):
        was_running = self._worker is not None
        self._worker = None
        self._set_monitoring_controls_enabled(
            has_dataset=self._current_csv_path is not None, running=False, paused=False,
        )
        if was_running:
            self.card_status.set_value("STOPPED", STATUS_COLORS["STOPPED"])
            self.led_status.set_state("#4a5158", blinking=False, text="STOPPED")
            self.lbl_bar_state.setText("STOPPED")
            self.lbl_bar_state.setStyleSheet("")
        if self._critical_alarm_timer.isActive():
            self._critical_alarm_timer.stop()
        self.led_alarm.set_state("#4a5158", blinking=False, text="IDLE")

    def _update_monitoring_time(self):
        if self._monitor_start_time is None:
            return
        elapsed = int(time.time() - self._monitor_start_time)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        duration_text = f"{h:02d}:{m:02d}:{s:02d}"
        self.lbl_time_status.setText(f"Monitoring time: {duration_text}")
        self.card_stat_duration.set_value(duration_text)

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            reply = QMessageBox.question(
                self, "Monitoring Running",
                "Monitoring is still running. Exit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.stop()
            self._worker.wait(2000)
        event.accept()
