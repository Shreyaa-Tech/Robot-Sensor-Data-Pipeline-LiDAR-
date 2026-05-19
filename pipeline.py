"""
pipeline.py - Data cleaning pipeline for LiDAR sensor readings

"""

import numpy as np
import pandas as pd
from sensor import Sensor


class DataPipeline:
    def __init__(self, sensor: Sensor) -> None:
        if sensor.raw_data is None:
            raise ValueError("Sensor has no data. Did you call sensor.load()?")
        self.sensor = sensor
        self._processed: pd.DataFrame | None = None


    # Public API
    def run(self) -> "DataPipeline":
        """Execute the full cleaning pipeline. Returns self for chaining."""
        df = self.sensor.raw_data.copy()

        df = self._add_outlier_flags(df)
        df = self._clip_physical_bounds(df)

        self._processed = df
        return self

    @property
    def cleaned_data(self) -> pd.DataFrame:
        """Readings with outliers removed."""
        self._check_run()
        return self._processed[~self._processed["is_outlier"]].copy()

    @property
    def outliers(self) -> pd.DataFrame:
        """Readings that were flagged as outliers."""
        self._check_run()
        return self._processed[self._processed["is_outlier"]].copy()

    @property
    def full_data(self) -> pd.DataFrame:
        """All readings with outlier flag column attached."""
        self._check_run()
        return self._processed.copy()

    def stats(self) -> dict:
        """Return cleaning summary statistics."""
        self._check_run()
        total = len(self._processed)
        n_outliers = int(self._processed["is_outlier"].sum())
        return {
            "total_readings": total,
            "outliers_detected": n_outliers,
            "outlier_pct": round(100 * n_outliers / total, 2),
            "clean_readings": total - n_outliers,
            "iqr_multiplier_k": self.sensor.config.noise_threshold,
        }


    # Private helpers
    def _add_outlier_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        dist = df["distance_m"]
        q1, q3 = dist.quantile(0.25), dist.quantile(0.75)
        iqr = q3 - q1
        k = self.sensor.config.noise_threshold

        lower_fence = q1 - k * iqr
        upper_fence = q3 + k * iqr

        df["is_outlier"] = (dist < lower_fence) | (dist > upper_fence)
        df["iqr_lower"] = lower_fence
        df["iqr_upper"] = upper_fence
        return df

    def _clip_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.sensor.config
        out_of_bounds = (
            (df["distance_m"] < cfg.min_distance_m) |
            (df["distance_m"] > cfg.max_distance_m)
        )
        # Physical bound violations are also treated as outliers
        df["is_outlier"] = df["is_outlier"] | out_of_bounds
        return df

    def _check_run(self) -> None:
        if self._processed is None:
            raise RuntimeError("Pipeline not yet run. Call .run() first.")
