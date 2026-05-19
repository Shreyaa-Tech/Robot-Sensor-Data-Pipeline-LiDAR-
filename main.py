"""
main.py

Usage:
    python main.py [--data PATH] [--threshold FLOAT] [--output DIR]


"""

import argparse
import sys
from pathlib import Path

from sensor import Sensor, SensorConfig
from pipeline import DataPipeline
from report import Report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiDAR Sensor Data Cleaning Pipeline"
    )
    parser.add_argument(
        "--data", default="lidar_data.csv",
        help="Path to the LiDAR CSV file (default: lidar_data.csv)"
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0,
        help="IQR multiplier for outlier detection (default: 3.0)"
    )
    parser.add_argument(
        "--output", default="outputs",
        help="Directory to save visualisation PNGs (default: outputs/)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n LiDAR Sensor Data Pipeline \n")

    # 1. Configure and load sensor
    config = SensorConfig(noise_threshold=args.threshold)
    sensor = Sensor(config).load(args.data)
    print(f"[Sensor]   Loaded {sensor.reading_count} readings")
    print(f"           Summary: {sensor.summary()}\n")

    # 2. Run cleaning pipeline
    pipeline = DataPipeline(sensor).run()
    stats = pipeline.stats()
    print(f"[Pipeline] Outliers detected: {stats['outliers_detected']} "
          f"({stats['outlier_pct']}%)")
    print(f"           Clean readings retained: {stats['clean_readings']}\n")

    # 3. Generate report visualisations
    print("[Report]   Generating visualisations...")
    report = Report(pipeline, output_dir=args.output)
    saved = report.generate_all()
    print(f"\n  All done. {len(saved)} figures saved to '{args.output}/'.\n")


if __name__ == "__main__":
    main()
