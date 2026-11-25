"""
detector.py
-----------
Anomaly detection logic: compares the absolute prediction error against
the configured threshold and produces contiguous anomaly regions for
plotting/highlighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class DetectionResult:
    """
    Container for everything the GUI needs to render results and
    populate the status bar / results export.
    """

    sample_index: np.ndarray          # X axis (sample number)
    actual_temperature: np.ndarray     # Measured temperature
    predicted_temperature: np.ndarray  # Model prediction
    error: np.ndarray                  # Absolute prediction error
    threshold: float
    anomaly_mask: np.ndarray           # Boolean array, True = anomaly
    anomaly_regions: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def n_anomalies(self) -> int:
        return int(np.count_nonzero(self.anomaly_mask))

    @property
    def n_samples(self) -> int:
        return int(self.sample_index.shape[0])

    @property
    def anomaly_percentage(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return 100.0 * self.n_anomalies / self.n_samples


class AnomalyDetector:
    """
    Stateless detector: given prediction errors and a threshold,
    determines which samples are anomalous and groups them into
    contiguous regions for highlighting on the error plot.
    """

    @staticmethod
    def compute_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return np.abs(actual.astype(float) - predicted.astype(float))

    @staticmethod
    def detect(error: np.ndarray, threshold: float) -> np.ndarray:
        return error > float(threshold)

    @staticmethod
    def find_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Converts a boolean anomaly mask into a list of (start, end)
        index pairs (inclusive) describing contiguous anomaly regions.
        Used to shade anomaly bands on the error plot.
        """
        regions: List[Tuple[int, int]] = []
        in_region = False
        start = 0

        for i, flagged in enumerate(mask):
            if flagged and not in_region:
                in_region = True
                start = i
            elif not flagged and in_region:
                in_region = False
                regions.append((start, i - 1))

        if in_region:
            regions.append((start, len(mask) - 1))

        return regions

    @classmethod
    def build_result(
        cls,
        sample_index: np.ndarray,
        actual: np.ndarray,
        predicted: np.ndarray,
        threshold: float,
    ) -> DetectionResult:
        error = cls.compute_absolute_error(actual, predicted)
        mask = cls.detect(error, threshold)
        regions = cls.find_regions(mask)

        return DetectionResult(
            sample_index=sample_index,
            actual_temperature=actual,
            predicted_temperature=predicted,
            error=error,
            threshold=threshold,
            anomaly_mask=mask,
            anomaly_regions=regions,
        )
