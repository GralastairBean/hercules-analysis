from pathlib import Path
from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSIFIED_CSV_NAME = "gaia_real_h1_h2_classified.csv"
OUTPUT_PLOT_NAME = "gaia_real_h1_h2_hr_diagrams.png"

PARALLAX_MIN_MAS = 0.1


def log(message: str, started_at: float) -> None:
    elapsed_seconds = perf_counter() - started_at
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp} | +{elapsed_seconds:7.2f}s] {message}", flush=True)


def prepare_hr_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "parallax",
        "h_group",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"The classified CSV is missing required HR columns: {missing_text}."
        )

    hr_df = df.copy()
    hr_df = hr_df[hr_df["parallax"] > PARALLAX_MIN_MAS].copy()
    hr_df["bp_rp_color"] = hr_df["phot_bp_mean_mag"] - hr_df["phot_rp_mean_mag"]
    hr_df["abs_g_mag"] = hr_df["phot_g_mean_mag"] + 5.0 * np.log10(hr_df["parallax"]) - 10.0
    hr_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    hr_df.dropna(subset=["bp_rp_color", "abs_g_mag"], inplace=True)
    return hr_df


def plot_group_hr(
    ax: plt.Axes,
    group_df: pd.DataFrame,
    group_name: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
):
    if group_df.empty:
        ax.set_title(f"{group_name}: no valid stars", fontsize=12, pad=10)
        ax.set_xlabel("Gaia BP-RP colour", fontsize=11)
        ax.set_ylabel("Gaia G absolute magnitude", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)
        return None

    ax.scatter(
        group_df["bp_rp_color"],
        group_df["abs_g_mag"],
        s=1,
        c="black",
        alpha=0.04,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    hb = ax.hexbin(
        group_df["bp_rp_color"],
        group_df["abs_g_mag"],
        gridsize=110,
        mincnt=1,
        bins="log",
        cmap="afmhot",
        linewidths=0,
        alpha=0.95,
        zorder=2,
    )

    ax.set_title(f"{group_name}", fontsize=12, pad=8, fontweight="bold")
    ax.set_xlabel("Gaia BP-RP colour", fontsize=11)
    ax.set_ylabel("Gaia G absolute magnitude", fontsize=11)
    ax.grid(False)
    ax.invert_yaxis()
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    ax.text(
        0.98,
        0.04,
        f"n = {len(group_df):,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 3},
    )
    return hb

def main() -> None:
    started_at = perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    classified_csv = data_dir / CLASSIFIED_CSV_NAME
    output_plot = data_dir / OUTPUT_PLOT_NAME

    log("Starting Gaia Hercules HR analysis.", started_at)
    log(f"Loading classified sample from {classified_csv}...", started_at)
    df = pd.read_csv(classified_csv)
    log(f"Loaded {len(df):,} classified stars.", started_at)

    hr_df = prepare_hr_columns(df)
    log(f"Prepared {len(hr_df):,} stars with valid HR quantities.", started_at)

    h1_df = hr_df[hr_df["h_group"] == "H1"].copy()
    h2_df = hr_df[hr_df["h_group"] == "H2"].copy()
    log(f"H1 stars used in HR diagram: {len(h1_df):,}", started_at)
    log(f"H2 stars used in HR diagram: {len(h2_df):,}", started_at)

    if h1_df.empty and h2_df.empty:
        raise ValueError("No H1/H2 stars with valid HR values were found.")

    log("Creating side-by-side HR diagrams...", started_at)
    x_lo = float(hr_df["bp_rp_color"].min())
    x_hi = float(hr_df["bp_rp_color"].max())
    y_lo = float(hr_df["abs_g_mag"].min())
    y_hi = float(hr_df["abs_g_mag"].max())
    x_span = max(x_hi - x_lo, 1e-6)
    y_span = max(y_hi - y_lo, 1e-6)
    x_limits = (x_lo - 0.03 * x_span, x_hi + 0.03 * x_span)
    y_limits = (y_hi + 0.03 * y_span, y_lo - 0.03 * y_span)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.5), sharex=True, sharey=True)

    h1_hb = plot_group_hr(ax1, h1_df, "H1", x_limits, y_limits)
    h2_hb = plot_group_hr(ax2, h2_df, "H2", x_limits, y_limits)

    if h1_hb is not None:
        cbar1 = fig.colorbar(h1_hb, ax=ax1, pad=0.01)
        cbar1.set_label("Hexbin density (log10 count)", fontsize=9)
    if h2_hb is not None:
        cbar2 = fig.colorbar(h2_hb, ax=ax2, pad=0.01)
        cbar2.set_label("Hexbin density (log10 count)", fontsize=9)

    plt.suptitle("Hercules H1/H2 Hertzsprung-Russell Diagrams", fontsize=15, y=0.99, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    log(f"Saving HR diagram figure to {output_plot}...", started_at)
    plt.savefig(output_plot, dpi=300)
    plt.show()

    log("Finished Gaia Hercules HR analysis.", started_at)


if __name__ == "__main__":
    main()