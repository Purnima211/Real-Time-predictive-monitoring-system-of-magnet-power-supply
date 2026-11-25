"""
widgets.py
----------
Custom SCADA-style widgets used by the real-time monitoring dashboard:

  * AnomalyGauge - circular 0-100% needle gauge with green/yellow/red
                    color bands and smooth needle animation.
  * StatusCard    - small live-value tile ("Current Temperature: 42.3 C")
                    used for the row of live status readouts.

Kept separate from ui.py so the main window file stays focused on
layout and control flow.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, QTimer, Property
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

COLOR_NORMAL = QColor("#4caf50")
COLOR_WARNING = QColor("#ffb300")
COLOR_CRITICAL = QColor("#e53935")
COLOR_NEEDLE = QColor("#e5e9ed")
COLOR_TRACK_BG = QColor("#2c323a")


class LedIndicator(QWidget):
    """
    A small circular industrial LED that can either stay solidly lit or
    blink at a fixed interval. Used for:
      * the main status LED (always "alive"-blinking; green/yellow/red
        for NORMAL/WARNING/CRITICAL)
      * the alarm LED (solid grey when idle, blinking yellow/red for
        WARNING/CRITICAL alarm states)
    """

    def __init__(self, diameter: int = 16, blink_interval_ms: int = 500, parent=None):
        super().__init__(parent)
        self._diameter = diameter
        self.setFixedSize(diameter + 8, diameter + 8)

        self._color = QColor("#4a5158")  # idle/grey by default
        self._blinking = False
        self._lit = True

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._toggle)
        self._timer.start(blink_interval_ms)

    def _toggle(self) -> None:
        self._lit = (not self._lit) if self._blinking else True
        self.update()

    def set_state(self, color: str, blinking: bool) -> None:
        """color: any Qt color string, e.g. '#4caf50'. blinking: whether the LED flashes."""
        self._color = QColor(color)
        self._blinking = blinking
        if not blinking:
            self._lit = True
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color = self._color if self._lit else self._color.darker(320)
        painter.setPen(QPen(QColor("#0d0f12"), 2))
        painter.setBrush(color)
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        painter.drawEllipse(center, self._diameter / 2.0, self._diameter / 2.0)

        if self._lit:
            # subtle glow highlight to read as "lit" rather than a flat disc
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 60))
            painter.drawEllipse(
                QPointF(center.x() - self._diameter * 0.18, center.y() - self._diameter * 0.18),
                self._diameter * 0.22, self._diameter * 0.22,
            )
        painter.end()


class LabeledLed(QFrame):
    """LED + text label side by side, e.g. "● STATUS" or "● ALARM"."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self.led = LedIndicator()
        self._label = QLabel(label.upper())
        self._label.setStyleSheet("color: #c7cdd3; font-weight: bold; font-size: 9pt;")

        self._state_label = QLabel("--")
        self._state_label.setStyleSheet("color: #9aa4ae; font-size: 9pt;")

        layout.addWidget(self.led)
        layout.addWidget(self._label)
        layout.addStretch(1)
        layout.addWidget(self._state_label)

    def set_state(self, color: str, blinking: bool, text: str) -> None:
        self.led.set_state(color, blinking)
        self._state_label.setText(text)
        self._state_label.setStyleSheet(f"color: {color}; font-size: 9pt; font-weight: bold;")


class AnomalyGauge(QWidget):
    """
    Circular needle gauge showing the current anomaly percentage
    (0-100%), with three color bands:
        0-40%   green   NORMAL
        41-70%  yellow  WARNING
        71-100% red     CRITICAL
    The needle animates smoothly toward each new value instead of
    jumping, so rapid updates read as a live instrument rather than a
    flickering number.
    """

    #: sweep of the gauge arc, in degrees (a speedometer-style 270 deg arc)
    _SWEEP_DEG = 270.0
    #: angle (Qt convention: 0 deg = 3 o'clock, positive = counter-clockwise)
    #: where the 0% position of the needle starts
    _START_DEG = 225.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._animation = QPropertyAnimation(self, b"needleValue", self)
        self._animation.setDuration(350)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

    # -- animated Qt property -----------------------------------------
    def _get_needle_value(self) -> float:
        return self._value

    def _set_needle_value(self, value: float) -> None:
        self._value = value
        self.update()

    needleValue = Property(float, _get_needle_value, _set_needle_value)

    # -- public API ------------------------------------------------------
    def set_value(self, value: float) -> None:
        """Animates the needle to a new anomaly percentage (0-100)."""
        value = max(0.0, min(100.0, float(value)))
        self._animation.stop()
        self._animation.setStartValue(self._value)
        self._animation.setEndValue(value)
        self._animation.start()

    @staticmethod
    def status_for(value: float) -> tuple[str, QColor]:
        if value <= 40.0:
            return "NORMAL", COLOR_NORMAL
        if value <= 70.0:
            return "WARNING", COLOR_WARNING
        return "CRITICAL", COLOR_CRITICAL

    # -- painting ----------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height())
        painter.translate(self.width() / 2.0, self.height() / 2.0)
        scale = side / 220.0
        painter.scale(scale, scale)

        radius = 78.0
        arc_rect = QRectF(-radius, -radius, radius * 2, radius * 2)

        # Background track.
        pen = QPen(COLOR_TRACK_BG, 15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawArc(arc_rect, int(self._START_DEG * 16), int(-self._SWEEP_DEG * 16))

        # Colored bands: green / yellow / red.
        self._draw_band(painter, arc_rect, 0.0, 40.0, COLOR_NORMAL)
        self._draw_band(painter, arc_rect, 40.0, 70.0, COLOR_WARNING)
        self._draw_band(painter, arc_rect, 70.0, 100.0, COLOR_CRITICAL)

        # Tick marks + numeric labels at 0, 20, 40, 60, 80, 100.
        self._draw_ticks(painter, radius)

        # Needle.
        angle_deg = self._START_DEG - (self._value / 100.0) * self._SWEEP_DEG
        angle_rad = math.radians(angle_deg)
        needle_len = radius - 20.0
        tip = QPointF(needle_len * math.cos(angle_rad), -needle_len * math.sin(angle_rad))

        painter.setPen(QPen(COLOR_NEEDLE, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(0, 0), tip)
        painter.setBrush(COLOR_NEEDLE)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0, 0), 7, 7)

        # Percentage + status text, centered in the lower half of the dial.
        status_text, status_color = self.status_for(self._value)
        painter.setPen(status_color)
        painter.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        painter.drawText(QRectF(-radius, 14, radius * 2, 30), Qt.AlignmentFlag.AlignCenter, f"{self._value:.0f}%")

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(QRectF(-radius, 42, radius * 2, 20), Qt.AlignmentFlag.AlignCenter, status_text)

        painter.end()

    def _draw_band(self, painter: QPainter, rect: QRectF, pct_start: float, pct_end: float, color: QColor) -> None:
        band_start_deg = self._START_DEG - (pct_start / 100.0) * self._SWEEP_DEG
        band_span_deg = -((pct_end - pct_start) / 100.0) * self._SWEEP_DEG
        pen = QPen(color, 16, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawArc(rect, int(band_start_deg * 16), int(band_span_deg * 16))

    def _draw_ticks(self, painter: QPainter, radius: float) -> None:
        """Draws tick marks + numeric labels at 0, 20, 40, 60, 80, 100."""
        tick_font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        outer_r = radius + 12.0
        inner_r = radius + 4.0
        label_r = radius + 20.0

        for pct in range(0, 101, 20):
            angle_deg = self._START_DEG - (pct / 100.0) * self._SWEEP_DEG
            angle_rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(angle_rad), -math.sin(angle_rad)

            p_outer = QPointF(outer_r * cos_a, outer_r * sin_a)
            p_inner = QPointF(inner_r * cos_a, inner_r * sin_a)
            painter.setPen(QPen(QColor("#c7cdd3"), 1.6))
            painter.drawLine(p_outer, p_inner)

            painter.setPen(QColor("#9aa4ae"))
            painter.setFont(tick_font)
            lx, ly = label_r * cos_a, label_r * sin_a
            painter.drawText(QRectF(lx - 16, ly - 8, 32, 16), Qt.AlignmentFlag.AlignCenter, str(pct))


class StatusCard(QFrame):
    """Small tile displaying a live label + value, e.g. "PREDICTED TEMP: 41.8 C"."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusCard")
        self.setFrameShape(QFrame.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        self._title_label = QLabel(title.upper())
        self._title_label.setObjectName("CardTitle")

        self._value_label = QLabel("--")
        self._value_label.setObjectName("CardValue")

        layout.addWidget(self._title_label)
        layout.addWidget(self._value_label)

    def set_value(self, text: str, color: str | None = None) -> None:
        self._value_label.setText(text)
        if color:
            self._value_label.setStyleSheet(f"color: {color}; font-size: 15pt; font-weight: bold;")
        else:
            self._value_label.setStyleSheet("")
