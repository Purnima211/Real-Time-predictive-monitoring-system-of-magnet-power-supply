"""
plotting.py
-----------
Matplotlib canvases embedded in the PySide6 GUI:

  * MainGraphCanvas  - actual vs predicted temperature + anomaly markers
  * ErrorGraphCanvas - prediction error vs threshold with shaded
                        anomaly regions

Both support the standard Matplotlib navigation toolbar (zoom/pan/save)
and a synchronized crosshair readout.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np

from detector import DetectionResult

# Industrial / control-room palette.
COLOR_BG = "#1b1f24"
COLOR_AXES_BG = "#14171b"
COLOR_GRID = "#2c323a"
COLOR_TEXT = "#c7cdd3"
COLOR_ACTUAL = "#4fc3f7"
COLOR_PREDICTED = "#ffb74d"
COLOR_ANOMALY = "#ef5350"
COLOR_ERROR = "#81c784"
COLOR_THRESHOLD = "#e53935"
COLOR_ANOMALY_BAND = "#ef535055"  # translucent red

plt.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_AXES_BG,
    "axes.edgecolor": COLOR_GRID,
    "axes.labelcolor": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": COLOR_GRID,
    "text.color": COLOR_TEXT,
    "font.size": 9,
})


class _BaseCanvas(FigureCanvasQTAgg):
    """Common setup shared by both plot canvases, including crosshair."""

    def __init__(self, title: str, ylabel: str, parent=None):
        self._figure = Figure(figsize=(5, 3), dpi=100, facecolor=COLOR_BG)
        super().__init__(self._figure)
        self.setParent(parent)

        self.axes = self._figure.add_subplot(111)
        self.axes.set_facecolor(COLOR_AXES_BG)
        self.axes.set_title(title, color=COLOR_TEXT, fontsize=10, fontweight="bold")
        self.axes.set_xlabel("Sample")
        self.axes.set_ylabel(ylabel)
        self.axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        self._crosshair_v = self.axes.axvline(color="#8899a6", linewidth=0.8, alpha=0.0)
        self._crosshair_h = self.axes.axhline(color="#8899a6", linewidth=0.8, alpha=0.0)
        self._crosshair_text = self.axes.text(
            0.02, 0.95, "", transform=self.axes.transAxes,
            fontsize=8, color=COLOR_TEXT, va="top",
            bbox=dict(boxstyle="round", facecolor=COLOR_AXES_BG, edgecolor=COLOR_GRID, alpha=0.85),
        )

        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("axes_leave_event", self._on_leave)

        self._figure.tight_layout()

    def _on_move(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            if self._crosshair_v.get_alpha():
                self._hide_crosshair()
            return

        self._crosshair_v.set_xdata([event.xdata, event.xdata])
        self._crosshair_v.set_alpha(0.7)
        self._crosshair_h.set_ydata([event.ydata, event.ydata])
        self._crosshair_h.set_alpha(0.7)
        self._crosshair_text.set_text(self._crosshair_label(event.xdata, event.ydata))
        self._crosshair_text.set_alpha(1.0)
        self.draw_idle()

    def _on_leave(self, _event):
        self._hide_crosshair()

    def _hide_crosshair(self):
        self._crosshair_v.set_alpha(0.0)
        self._crosshair_h.set_alpha(0.0)
        self._crosshair_text.set_alpha(0.0)
        self.draw_idle()

    def _crosshair_label(self, x: float, y: float) -> str:
        return f"x = {x:.2f}\ny = {y:.2f}"

    def clear(self):
        self.axes.clear()
        self.axes.set_facecolor(COLOR_AXES_BG)
        self.axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        self._crosshair_v = self.axes.axvline(color="#8899a6", linewidth=0.8, alpha=0.0)
        self._crosshair_h = self.axes.axhline(color="#8899a6", linewidth=0.8, alpha=0.0)
        self._crosshair_text = self.axes.text(
            0.02, 0.95, "", transform=self.axes.transAxes,
            fontsize=8, color=COLOR_TEXT, va="top",
            bbox=dict(boxstyle="round", facecolor=COLOR_AXES_BG, edgecolor=COLOR_GRID, alpha=0.85),
        )


class MainGraphCanvas(_BaseCanvas):
    """Actual vs predicted temperature, with anomaly points highlighted."""

    def __init__(self, parent=None):
        super().__init__("Magnet Power Supply Temperature", "Temperature (\u00b0C)", parent)

    def plot(self, result: DetectionResult):
        self.clear()
        x = np.arange(result.n_samples)

        self.axes.plot(
            x, result.actual_temperature,
            color=COLOR_ACTUAL, linewidth=1.2, label="Actual Temperature",
        )
        self.axes.plot(
            x, result.predicted_temperature,
            color=COLOR_PREDICTED, linewidth=1.2, linestyle="--", label="Predicted Temperature",
        )

        if result.n_anomalies:
            anomaly_x = x[result.anomaly_mask]
            anomaly_y = result.actual_temperature[result.anomaly_mask]
            self.axes.scatter(
                anomaly_x, anomaly_y,
                color=COLOR_ANOMALY, s=28, zorder=5,
                edgecolors="#ffffff", linewidths=0.4,
                label="Detected Anomaly",
            )

        self.axes.set_title("Magnet Power Supply Temperature", color=COLOR_TEXT, fontsize=10, fontweight="bold")
        self.axes.set_xlabel("Sample")
        self.axes.set_ylabel("Temperature (\u00b0C)")
        legend = self.axes.legend(loc="upper right", framealpha=0.85, facecolor=COLOR_AXES_BG)
        for text in legend.get_texts():
            text.set_color(COLOR_TEXT)

        self._figure.tight_layout()
        self.draw_idle()

    def _crosshair_label(self, x: float, y: float) -> str:
        return f"Sample = {x:.0f}\nTemperature = {y:.2f} \u00b0C"


class ErrorGraphCanvas(_BaseCanvas):
    """Prediction error vs threshold, with anomaly regions shaded."""

    def __init__(self, parent=None):
        super().__init__("Prediction Error", "Absolute Error", parent)

    def plot(self, result: DetectionResult):
        self.clear()
        x = np.arange(result.n_samples)

        self.axes.plot(
            x, result.error,
            color=COLOR_ERROR, linewidth=1.0, label="Prediction Error",
        )
        self.axes.axhline(
            result.threshold, color=COLOR_THRESHOLD, linewidth=1.0,
            linestyle="--", label=f"Threshold ({result.threshold:.3f})",
        )

        for start, end in result.anomaly_regions:
            self.axes.axvspan(start, end, color=COLOR_ANOMALY_BAND, zorder=0)

        handles, labels = self.axes.get_legend_handles_labels()
        if result.anomaly_regions:
            handles.append(Patch(facecolor=COLOR_ANOMALY, alpha=0.35))
            labels.append("Anomaly Region")

        self.axes.set_title("Prediction Error", color=COLOR_TEXT, fontsize=10, fontweight="bold")
        self.axes.set_xlabel("Sample")
        self.axes.set_ylabel("Absolute Error")
        legend = self.axes.legend(handles, labels, loc="upper right", framealpha=0.85, facecolor=COLOR_AXES_BG)
        for text in legend.get_texts():
            text.set_color(COLOR_TEXT)

        self._figure.tight_layout()
        self.draw_idle()

    def _crosshair_label(self, x: float, y: float) -> str:
        return f"Sample = {x:.0f}\nError = {y:.3f}"


def make_toolbar(canvas: FigureCanvasQTAgg, parent=None) -> NavigationToolbar2QT:
    """
    Builds the standard Matplotlib navigation toolbar (pan, zoom,
    home/reset, save) styled to match the dark theme.
    """
    toolbar = NavigationToolbar2QT(canvas, parent)
    toolbar.setStyleSheet(
        f"QToolBar {{ background-color: {COLOR_AXES_BG}; border: none; }}"
        f"QToolButton {{ background-color: {COLOR_AXES_BG}; }}"
    )
    return toolbar
