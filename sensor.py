"""
sensor.py - LiDAR Sensor abstraction

"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class SensorConfig:
    """Holds sensor-level configuration parameters."""
    sensor_id: str = "LIDAR_01"
    noise_threshold: float = 3.0   # IQR multiplier; higher = more permissive
    min_distance_m: float = 0.1    # Physical minimum range (metres)
    max_distance_m: float = 15.0   # Physical maximum range (metres)


class Sensor:
    REQUIRED_COLUMNS = {"timestamp", "angle_deg", "distance_m", "intensity", "sensor_id"}

    def __init__(self, config: SensorConfig | None = None) -> None:
        self.config = config or SensorConfig()
        self._raw_data: pd.DataFrame | None = None


    # Public API
    def load(self, filepath: str | Path) -> "Sensor":
        """Load a CSV file of LiDAR readings. Returns self for chaining."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path, parse_dates=["timestamp"])
        self._validate_schema(df)

        # Filter to this sensor's readings only
        if "sensor_id" in df.columns:
            df = df[df["sensor_id"] == self.config.sensor_id].copy()

        self._raw_data = df.reset_index(drop=True)
        return self

    @property
    def raw_data(self) -> pd.DataFrame:
        """Return the raw (unprocessed) readings."""
        if self._raw_data is None:
            raise RuntimeError("No data loaded. Call .load() first.")
        return self._raw_data.copy()

    @property
    def reading_count(self) -> int:
        return len(self._raw_data) if self._raw_data is not None else 0

    def summary(self) -> dict:
        """Basic statistics on the raw readings."""
        if self._raw_data is None:
            return {}
        d = self._raw_data["distance_m"]
        return {
            "sensor_id": self.config.sensor_id,
            "total_readings": len(self._raw_data),
            "distance_mean_m": round(float(d.mean()), 4),
            "distance_std_m": round(float(d.std()), 4),
            "distance_min_m": round(float(d.min()), 4),
            "distance_max_m": round(float(d.max()), 4),
        }


    # Private helpers
    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
