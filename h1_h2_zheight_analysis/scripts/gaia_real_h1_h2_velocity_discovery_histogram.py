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

    required_columns = {"lz_kpc_kms", "h_group"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"The classified CSV is missing required columns: {missing_text}. Run the processing script first."
        )

    plot_df = df.dropna(subset=["lz_kpc_kms", "h_group"]).copy()
    if plot_df.empty:
        raise ValueError("No rows with lz_kpc_kms and h_group were found in the classified CSV.")

    h1_df = plot_df[plot_df["h_group"] == "H1"]
    h2_df = plot_df[plot_df["h_group"] == "H2"]
    if h1_df.empty or h2_df.empty:
        raise ValueError("H1 and H2 groups were not both found in the classified CSV.")

    h1_lz = float(h1_df["lz_kpc_kms"].median())
    h2_lz = float(h2_df["lz_kpc_kms"].median())

    log("Creating Lz histogram with H1/H2 delineation lines...", started_at)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))

    lz_low = float(plot_df["lz_kpc_kms"].quantile(0.005))
    lz_high = float(plot_df["lz_kpc_kms"].quantile(0.995))
    ax.hist(
        plot_df["lz_kpc_kms"],
        bins=180,
        range=(lz_low, lz_high),
        color="black",
        alpha=0.85,
    )

    ax.axvline(h1_lz, color="darkgreen", linestyle="--", linewidth=2.0, label=f"H1 median Lz = {h1_lz:.1f}")
    ax.axvline(h2_lz, color="darkorange", linestyle="--", linewidth=2.0, label=f"H2 median Lz = {h2_lz:.1f}")

    ax.set_title("Velocity Discovery: Lz Histogram", fontsize=13, pad=12)
    ax.set_xlabel("Lz (kpc km/s)", fontsize=11)
    ax.set_ylabel("Star count", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(fontsize=10, loc="upper right")
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