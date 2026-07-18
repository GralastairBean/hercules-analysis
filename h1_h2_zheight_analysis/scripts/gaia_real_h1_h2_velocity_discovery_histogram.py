from pathlib import Path
from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd


CLASSIFIED_CSV_NAME = "gaia_real_h1_h2_classified.csv"
OUTPUT_PNG_NAME = "gaia_real_h1_h2_velocity_discovery_histogram.png"


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

    required_columns = {"lz_kpc_kms"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"The classified CSV is missing required columns: {missing_text}. Run the processing script first."
        )

    plot_df = df.dropna(subset=["lz_kpc_kms"]).copy()
    if plot_df.empty:
        raise ValueError("No rows with lz_kpc_kms were found in the classified CSV.")

    log("Creating clean Lz histogram...", started_at)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))

    lz_low = float(plot_df["lz_kpc_kms"].quantile(0.005))
    lz_high = float(plot_df["lz_kpc_kms"].quantile(0.995))
    ax.hist(
        plot_df["lz_kpc_kms"],
        bins=250,
        range=(lz_low, lz_high),
        color="black",
        alpha=0.85,
    )

    ax.set_title("Velocity Discovery: Lz Histogram", fontsize=13, pad=12)
    ax.set_xlabel("Lz (kpc km/s)", fontsize=11)
    ax.set_ylabel("Star count", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.text(
        0.98,
        0.95,
        f"n = {len(plot_df):,}",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        ha="right",
        va="top",
        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 4},
    )

    plt.tight_layout()

    log(f"Saving velocity discovery histogram to {output_png}...", started_at)
    plt.savefig(output_png, dpi=300)
    plt.show()

    log("Finished velocity discovery histogram.", started_at)


if __name__ == "__main__":
    main()