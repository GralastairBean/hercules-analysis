from pathlib import Path
from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSIFIED_CSV_NAME = "gaia_data_processed.csv"
OUTPUT_PNG_NAME = "processed_gross_error_check_histograms.png"


def log(message: str, started_at: float) -> None:
    elapsed_seconds = perf_counter() - started_at
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp} | +{elapsed_seconds:7.2f}s] {message}", flush=True)


def main() -> None:
    started_at = perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    classified_csv = data_dir / CLASSIFIED_CSV_NAME
    output_png = data_dir / OUTPUT_PNG_NAME

    log(f"Loading classified sample from {classified_csv}...", started_at)
    df = pd.read_csv(classified_csv)
    log(f"Loaded {len(df):,} rows.", started_at)

    required_columns = {"z_height_pc", "vr_kms", "lz_kpc_kms"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"The classified CSV is missing required columns: {missing_text}. Run the processing script first."
        )

    if "distance_pc" not in df.columns:
        if "parallax" not in df.columns:
            raise ValueError(
                "The classified CSV is missing both distance_pc and parallax columns. "
                "Run the processing script first."
            )
        df["distance_pc"] = 1000.0 / df["parallax"]

    plot_df = df.dropna(subset=["distance_pc", "z_height_pc", "vr_kms", "lz_kpc_kms"]).copy()
    if plot_df.empty:
        raise ValueError("No rows with distance_pc, z_height_pc, vr_kms, and lz_kpc_kms were found.")

    log("Creating gross error check histograms (distance, z height, Vr, and Lz)...", started_at)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    ax1.hist(
        plot_df["distance_pc"],
        bins=180,
        color="black",
        alpha=0.85,
    )
    ax1.set_title("Distance Distribution", fontsize=12, pad=10)
    ax1.set_xlabel("Distance (pc)", fontsize=11)
    ax1.set_ylabel("Star count", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.25)

    z_limit = float(plot_df["distance_pc"].max())
    ax2.hist(
        plot_df["z_height_pc"],
        bins=180,
        range=(-z_limit, z_limit),
        color="black",
        alpha=0.85,
    )
    ax2.set_title("Z Height Distribution", fontsize=12, pad=10)
    ax2.set_xlabel("Z Height Above Galactic Plane (pc)", fontsize=11)
    ax2.set_ylabel("Star count", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.25)

    vr_low = float(np.percentile(plot_df["vr_kms"], 0.5))
    vr_high = float(np.percentile(plot_df["vr_kms"], 99.5))
    ax3.hist(
        plot_df["vr_kms"],
        bins=180,
        range=(vr_low, vr_high),
        color="black",
        alpha=0.85,
    )
    ax3.set_xlim(vr_low, vr_high)
    ax3.set_title("Vr Distribution", fontsize=12, pad=10)
    ax3.set_xlabel("Vr (km/s)", fontsize=11)
    ax3.set_ylabel("Star count", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.25)

    lz_low = float(np.percentile(plot_df["lz_kpc_kms"], 0.5))
    lz_high = float(np.percentile(plot_df["lz_kpc_kms"], 99.5))
    ax4.hist(
        plot_df["lz_kpc_kms"],
        bins=180,
        range=(lz_low, lz_high),
        color="black",
        alpha=0.85,
    )
    ax4.set_xlim(lz_low, lz_high)
    ax4.set_title("Lz Distribution", fontsize=12, pad=10)
    ax4.set_xlabel("Lz (kpc km/s)", fontsize=11)
    ax4.set_ylabel("Star count", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.25)

    fig.text(
        0.98,
        0.965,
        f"n = {len(plot_df):,}",
        ha="right",
        va="top",
        fontsize=10,
        color="black",
    )

    plt.suptitle("Gross Error Check: Distance, Z Height, Vr, and Lz Histograms", fontsize=14, y=0.995)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    log(f"Saving gross error check plot to {output_png}...", started_at)
    plt.savefig(output_png, dpi=300)
    plt.show()

    log("Finished gross error check plot.", started_at)


if __name__ == "__main__":
    main()