# LiDAR Sensor Data Pipeline

A Python pipeline that ingests simulated LiDAR sensor readings, cleans outliers using statistical methods and produces three visualisations.

---

## Project Structure

```
sensor_pipeline/
├── sensor.py           # Sensor class: loads & validates raw CSV readings
├── pipeline.py         # DataPipeline class: outlier detection & cleaning
├── report.py           # Report class: generates 3 Matplotlib visualisations
├── main.py             # CLI entry point
├── lidar_data.csv      # Simulated LiDAR dataset (500 readings, ~5% outliers)
├── outputs/            # Generated PNG figures land here
│   ├── fig1_raw_data.png
│   ├── fig2_cleaned_data.png
│   └── fig3_anomaly_heatmap.png
└── tests/
    └── test_pipeline.py   # pytest unit tests (≥80% coverage)
```

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib pytest pytest-cov
```

---

## Running the Pipeline

```bash
python main.py
```

With optional arguments:

```bash
python main.py --data lidar_data.csv --threshold 2.5 --output outputs/
```

| Argument      | Default          | Description                              |
|---------------|------------------|------------------------------------------|
| `--data`      | `lidar_data.csv` | Path to the LiDAR CSV input file         |
| `--threshold` | `3.0`            | IQR multiplier k for outlier detection   |
| `--output`    | `outputs/`       | Directory to save visualisation PNGs     |

---

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Design Overview

### `Sensor` class (`sensor.py`)
- Loads a CSV of LiDAR readings and validates required columns
- Filters readings by `sensor_id`
- Exposes raw data and summary statistics
- Configurable via `SensorConfig` dataclass (noise threshold, physical bounds)

### `DataPipeline` class (`pipeline.py`)
Outlier detection uses a two-stage approach:

1. **IQR Fence Method**  computes Q1, Q3, IQR and flags readings outside `[Q1 - k·IQR, Q3 + k·IQR]` where `k` is the configurable noise threshold
2. **Physical Bounds Clipping**  readings outside the sensor's min/max range are flagged regardless of IQR result

### `Report` class (`report.py`)
Generates three PNG figures:
- **Figure 1**  Raw data in polar + cartesian views
- **Figure 2**  Cleaned data overlaid with outlier positions, plus before/after histogram
- **Figure 3**  Anomaly heatmap: angle × intensity bins coloured by anomaly rate

---

## Dataset

`lidar_data.csv` contains 500 simulated readings from sensor `LIDAR_01`:

| Column       | Description                    |
|--------------|--------------------------------|
| `timestamp`  | ISO datetime of the reading    |
| `angle_deg`  | Scan angle in degrees (0-360°) |
| `distance_m` | Measured distance in metres    |
| `intensity`  | Return intensity (50-255)      |
| `sensor_id`  | Sensor identifier              |

~5% of readings are injected outliers (distances 20–50m) to simulate sensor noise.

---


