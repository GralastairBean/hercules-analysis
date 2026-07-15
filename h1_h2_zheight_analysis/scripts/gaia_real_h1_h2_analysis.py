from pathlib import Path
from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def log(message: str, started_at: float) -> None:
    elapsed_seconds = perf_counter() - started_at
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp} | +{elapsed_seconds:7.2f}s] {message}", flush=True)


def main() -> None:
    started_at = perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    classified_csv = data_dir / "gaia_real_h1_h2_classified.csv"
    output_plot = data_dir / "gaia_real_h1_h2_comparison.png"

    log("Starting real Gaia Hercules plotting script.", started_at)
    log(f"Loading classified Hercules sample from {classified_csv}...", started_at)
    classified_df = pd.read_csv(classified_csv)
    log(f"Loaded {len(classified_df):,} classified stars.", started_at)

    if "h_group" not in classified_df.columns or "z_height_pc" not in classified_df.columns:
        raise ValueError(
            "The classified CSV does not contain the expected columns. Run the data-pull script first."
        )

    h1_df = classified_df[classified_df["h_group"] == "H1"].copy()
    h2_df = classified_df[classified_df["h_group"] == "H2"].copy()

    log(f"Isolated {len(h1_df):,} Hercules 1 stars from CSV.", started_at)
    log(f"Isolated {len(h2_df):,} Hercules 2 stars from CSV.", started_at)

    plot_series = pd.concat([h1_df["z_height_pc"], h2_df["z_height_pc"]], ignore_index=True).dropna()
    if plot_series.empty:
        raise ValueError("No H1/H2 stars with z-height values were found in the classified CSV.")

    z_limit = np.percentile(np.abs(plot_series), 99)
    z_limit = float(np.clip(z_limit, 50.0, 300.0))
    z_range = (-z_limit, z_limit)
    log(f"Using plot range {z_range[0]:.1f} to {z_range[1]:.1f} pc based on the data.", started_at)

    log("Creating comparison plot for the vertical z distribution...", started_at)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(
        h1_df["z_height_pc"],
        bins=100,
        range=z_range,
        density=True,
        histtype="step",
        linewidth=2.5,
        color="darkgreen",
        label="Hercules 1 (H1)",
    )
    ax1.hist(
        h2_df["z_height_pc"],
        bins=100,
        range=z_range,
        density=True,
        histtype="step",
        linewidth=2.5,
        color="darkorange",
        label="Hercules 2 (H2)",
    )
    ax1.hist(h1_df["z_height_pc"], bins=100, range=z_range, density=True, alpha=0.1, color="darkgreen")
    ax1.hist(h2_df["z_height_pc"], bins=100, range=z_range, density=True, alpha=0.1, color="darkorange")
    ax1.set_title("Stellar Probability Density Profile", fontsize=12, pad=10)
    ax1.set_xlabel("Galactic Height z (pc)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.set_xlim(z_range)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10, loc="upper right")

    ax2.hist(
        h1_df["z_height_pc"],
        bins=500,
        range=z_range,
        density=True,
        cumulative=True,
        histtype="step",
        linewidth=2.5,
        color="darkgreen",
        label="H1 Cumulative",
    )
    ax2.hist(
        h2_df["z_height_pc"],
        bins=500,
        range=z_range,
        density=True,
        cumulative=True,
        histtype="step",
        linewidth=2.5,
        color="darkorange",
        label="H2 Cumulative",
    )
    ax2.set_title("Cumulative Spatial Distribution (CDF)", fontsize=12, pad=10)
    ax2.set_xlabel("Galactic Height z (pc)", fontsize=11)
    ax2.set_ylabel("Fraction of Total Population", fontsize=11)
    ax2.set_xlim(z_range)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10, loc="lower right")

    plt.suptitle("Vertical Structure Comparison: Hercules 1 vs Hercules 2", fontsize=14, y=0.98)
    plt.tight_layout()

    log(f"Saving plot to {output_plot}...", started_at)
    plt.savefig(output_plot, dpi=300)
    plt.show()

    log("Finished real Gaia Hercules plotting.", started_at)


if __name__ == "__main__":
    main()