"""
data_source.py
---------------
Modular streaming data source abstraction for the real-time monitoring
pipeline.

Every data source (CSV replay today, a live sensor tomorrow) implements
the same small DataSource interface: open() / read_sample() /
total_samples() / close(). The monitoring worker in ui.py only ever
talks to that interface, so swapping the acquisition backend never
requires touching the GUI or the prediction pipeline.

Today:
    CSVDataSource   - replays a CSV file row-by-row to simulate a live
                       sensor feed (used for the current dataset-driven
                       monitoring mode).

Future hardware path (not implemented yet, stubbed below so the shape
of the integration is obvious):
    PT100  ->  ICP DAS M7017  ->  RS485  ->  PySerial
    SerialDataSource will read one live temperature value per
    read_sample() call, exactly like CSVDataSource reads one row.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional

import pandas as pd


class DataSourceError(Exception):
    """Raised when a data source cannot be opened or read from."""


class DataSource(abc.ABC):
    """
    Common interface every data source must implement.

    A "sample" is a dict mapping each required feature column name to
    its numeric value for that reading, e.g. {"TrueTemperature_C": 42.7}.
    """

    @abc.abstractmethod
    def open(self) -> None:
        """Opens/connects the source. Raises DataSourceError on failure."""

    @abc.abstractmethod
    def read_sample(self) -> Optional[Dict[str, float]]:
        """
        Returns the next sample as a dict, or None when the source is
        exhausted (end of CSV). Live sources (e.g. serial) should block
        briefly for the next reading rather than returning None.
        """

    @abc.abstractmethod
    def total_samples(self) -> Optional[int]:
        """Total known sample count, or None if unbounded (e.g. a live sensor)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Releases any resources (file handles, serial ports, ...)."""


class CSVDataSource(DataSource):
    """
    Replays a CSV file one row at a time, in order, simulating a
    continuous live feed. This is today's stand-in for the future
    PT100 / RS485 hardware source.
    """

    def __init__(self, path: str, feature_columns: List[str]):
        self._path = path
        self._feature_columns = feature_columns
        self._df: Optional[pd.DataFrame] = None
        self._cursor = 0

    def open(self) -> None:
        try:
            self._df = pd.read_csv(self._path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataSourceError(f"Could not open CSV '{self._path}': {exc}") from exc

        missing = [c for c in self._feature_columns if c not in self._df.columns]
        if missing:
            raise DataSourceError(
                f"CSV '{self._path}' is missing required column(s): {missing}"
            )
        self._cursor = 0

    def read_sample(self) -> Optional[Dict[str, float]]:
        if self._df is None or self._cursor >= len(self._df):
            return None
        row = self._df.iloc[self._cursor]
        self._cursor += 1
        try:
            return {col: float(row[col]) for col in self._feature_columns}
        except (TypeError, ValueError) as exc:
            raise DataSourceError(f"Non-numeric value at row {self._cursor - 1}: {exc}") from exc

    def total_samples(self) -> Optional[int]:
        return None if self._df is None else len(self._df)

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """Full dataframe, used once up-front to calibrate the predictor's scaler."""
        return self._df

    def close(self) -> None:
        self._df = None
        self._cursor = 0


class SerialDataSource(DataSource):
    """
    Future live-hardware data source:
        PT100 -> ICP DAS M7017 (RTD-to-digital module) -> RS485 -> PySerial

    Implementing open()/read_sample()/total_samples()/close() here with
    the exact same signatures as CSVDataSource is the only change needed
    to move this application from CSV playback to live sensor readings
    -- ui.py and the monitoring worker do not need to change.

    Intentionally left unimplemented (raises NotImplementedError) since
    no hardware/protocol details were provided. A typical implementation
    would:
        1. Open the serial port with `pyserial` (port, baudrate, parity
           etc. matching the M7017's Modbus-RTU/RS485 configuration).
        2. In read_sample(), issue a Modbus read-holding-registers
           request for the PT100 channel, convert the returned raw
           value to degrees Celsius, and return
           {CONFIG.TARGET_COLUMN: value}.
        3. Return None from total_samples() (a live sensor has no
           known end).
    """

    def __init__(self, port: str, baudrate: int = 9600, feature_columns: Optional[List[str]] = None):
        self._port = port
        self._baudrate = baudrate
        self._feature_columns = feature_columns or []

    def open(self) -> None:
        raise NotImplementedError(
            "Live PT100/RS485 sensor support is not implemented yet. "
            "Implement SerialDataSource using pyserial to enable it."
        )

    def read_sample(self) -> Optional[Dict[str, float]]:
        raise NotImplementedError

    def total_samples(self) -> Optional[int]:
        return None

    def close(self) -> None:
        pass
