from pathlib import Path
from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd


CLASSIFIED_CSV_NAME = "gaia_real_h1_h2_classified.csv"
OUTPUT_PNG_NAME = "gaia_real_h1_h2_gross_error_check_histograms.png"


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

    required_columns = {"z_height_pc"}
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

    plot_df = df.dropna(subset=["distance_pc", "z_height_pc"]).copy()
    if plot_df.empty:
        raise ValueError("No rows with both distance_pc and z_height_pc were found.")

    log("Creating gross error check histograms (distance and z height)...", started_at)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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

    fig.text(
        0.98,
        0.965,
        f"n = {len(plot_df):,}",
        ha="right",
        va="top",
        fontsize=10,
        color="black",
    )

    plt.suptitle("Gross Error Check: Distance and Z Height Histograms", fontsize=14, y=0.995)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    log(f"Saving gross error check plot to {output_png}...", started_at)
    plt.savefig(output_png, dpi=300)
    plt.show()

    log("Finished gross error check plot.", started_at)


if __name__ == "__main__":
    main()