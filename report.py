"""
report.py - Visualisation and report for the LiDAR pipeline

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering - no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from pipeline import DataPipeline


class Report:
    def __init__(self, pipeline: DataPipeline, output_dir: str | Path = ".") -> None:
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._saved_files: list[Path] = []


    # Public API
    def generate_all(self) -> list[Path]:
        self._saved_files = []
        self._plot_raw_data()
        self._plot_cleaned_data()
        self._plot_anomaly_heatmap()
        return self._saved_files


    # Figure 1. Raw data
    def _plot_raw_data(self) -> None:
        raw = self.pipeline.sensor.raw_data
        angles_rad = np.radians(raw["angle_deg"])
        distances = raw["distance_m"]

        fig = plt.figure(figsize=(14, 6), facecolor="#0f1117")
        fig.suptitle("Figure 1. Raw LiDAR Distance Readings", color="white",
                     fontsize=14, fontweight="bold", y=1.01)

        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        # --- Polar plot ---
        ax_polar = fig.add_subplot(gs[0], projection="polar")
        ax_polar.set_facecolor("#1a1d27")
        scatter = ax_polar.scatter(
            angles_rad, distances,
            c=distances, cmap="plasma",
            s=8, alpha=0.8
        )
        ax_polar.set_title("Polar View", color="white", pad=12)
        ax_polar.tick_params(colors="gray")
        ax_polar.spines["polar"].set_color("#444")
        cbar = fig.colorbar(scatter, ax=ax_polar, pad=0.12, shrink=0.7)
        cbar.set_label("Distance (m)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # --- Cartesian plot ---
        ax_cart = fig.add_subplot(gs[1])
        ax_cart.set_facecolor("#1a1d27")
        ax_cart.plot(raw["angle_deg"], distances, color="#7b8cde", linewidth=0.9,
                     alpha=0.7, label="Distance (m)")
        ax_cart.set_xlabel("Angle (°)", color="white")
        ax_cart.set_ylabel("Distance (m)", color="white")
        ax_cart.set_title("Cartesian View", color="white")
        ax_cart.tick_params(colors="white")
        ax_cart.spines[:].set_color("#444")
        ax_cart.legend(facecolor="#2a2d3a", labelcolor="white", fontsize=9)

        self._save(fig, "fig1_raw_data.png")


    # Figure 2. Cleaned data with outlier overlay
    def _plot_cleaned_data(self) -> None:
        full = self.pipeline.full_data
        clean = self.pipeline.cleaned_data
        outliers = self.pipeline.outliers

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")
        fig.suptitle("Figure 2. Cleaned Data vs Outliers", color="white",
                     fontsize=14, fontweight="bold")

        for ax in axes:
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")

        # Left: overlay cleaned + outliers
        ax = axes[0]
        ax.plot(clean["angle_deg"], clean["distance_m"],
                color="#4caf82", linewidth=1.0, label="Clean readings", alpha=0.85)
        ax.scatter(outliers["angle_deg"], outliers["distance_m"],
                   color="#ff6b6b", s=30, zorder=5, label=f"Outliers (n={len(outliers)})")
        # IQR fence lines
        if len(full) > 0:
            ax.axhline(full["iqr_upper"].iloc[0], color="#ffd166", linewidth=1,
                       linestyle="--", alpha=0.6, label="IQR upper fence")
            ax.axhline(full["iqr_lower"].iloc[0], color="#ffd166", linewidth=1,
                       linestyle="--", alpha=0.6, label="IQR lower fence")
        ax.set_xlabel("Angle (°)", color="white")
        ax.set_ylabel("Distance (m)", color="white")
        ax.set_title("Clean + Outlier Overlay", color="white")
        ax.legend(facecolor="#2a2d3a", labelcolor="white", fontsize=8)

        # Right: before/after comparison (histograms)
        ax2 = axes[1]
        ax2.hist(self.pipeline.sensor.raw_data["distance_m"],
                 bins=40, alpha=0.55, color="#7b8cde", label="Raw")
        ax2.hist(clean["distance_m"],
                 bins=40, alpha=0.75, color="#4caf82", label="Cleaned")
        ax2.set_xlabel("Distance (m)", color="white")
        ax2.set_ylabel("Count", color="white")
        ax2.set_title("Distance Distribution: Before vs After", color="white")
        ax2.legend(facecolor="#2a2d3a", labelcolor="white", fontsize=9)

        self._save(fig, "fig2_cleaned_data.png")

    # Figure 3. Anomaly heatmap
    def _plot_anomaly_heatmap(self) -> None:
        full = self.pipeline.full_data.copy()

        # Bin angle (36 bins of 10°) and intensity (20 bins)
        angle_bins = np.arange(0, 361, 10)
        intensity_bins = np.linspace(
            full["intensity"].min(), full["intensity"].max(), 21
        )

        full["angle_bin"] = pd.cut(full["angle_deg"], bins=angle_bins,
                                   labels=angle_bins[:-1], include_lowest=True)
        full["intensity_bin"] = pd.cut(full["intensity"], bins=intensity_bins,
                                       labels=np.round(intensity_bins[:-1], 0),
                                       include_lowest=True)

        # Pivot: fraction of anomalies per (angle_bin, intensity_bin) cell
        pivot = (
            full.groupby(["angle_bin", "intensity_bin"], observed=True)["is_outlier"]
            .mean()
            .unstack("intensity_bin")
            .fillna(0)
        )

        # Custom red-highlight colourmap
        cmap = LinearSegmentedColormap.from_list(
            "anomaly", ["#1a1d27", "#ffd166", "#ff6b6b"]
        )

        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0f1117")
        ax.set_facecolor("#1a1d27")
        fig.suptitle("Figure 3. Anomaly Heatmap (Angle × Intensity)",
                     color="white", fontsize=14, fontweight="bold")

        im = ax.imshow(
            pivot.T.values,
            aspect="auto", cmap=cmap,
            extent=[0, 360, float(intensity_bins[0]), float(intensity_bins[-1])],
            origin="lower", vmin=0, vmax=1
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Anomaly Rate", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_xlabel("Angle (°)", color="white")
        ax.set_ylabel("Intensity", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

        # Mark outlier angles with a rug
        outlier_angles = self.pipeline.outliers["angle_deg"].values
        for a in outlier_angles:
            ax.axvline(a, color="#ff6b6b", alpha=0.25, linewidth=0.7)

        self._save(fig, "fig3_anomaly_heatmap.png")


    # Utility
    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        self._saved_files.append(path)
        print(f"  Saved → {path}")
