from pathlib import Path
from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSIFIED_CSV_NAME = "gaia_real_h1_h2_classified.csv"
OUTPUT_PLOT_NAME = "gaia_real_h1_h2_velocity_discovery.png"
OUTPUT_PDF_NAME = "gaia_real_h1_h2_velocity_discovery.pdf"

# Reference kinematics used for marker overlay (literature-style conventions).
R0_KPC = 8.2
VCIRC_LSR_KMS = 232.0
U_SUN_PEC_KMS = 11.1
V_SUN_PEC_KMS = 12.24


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
    output_plot = data_dir / OUTPUT_PLOT_NAME
    output_pdf = data_dir / OUTPUT_PDF_NAME

    log(f"Loading classified sample from {classified_csv}...", started_at)
    df = pd.read_csv(classified_csv)
    log(f"Loaded {len(df):,} classified stars.", started_at)

    required_columns = {"lz_kpc_kms", "vr_kms"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"The classified CSV is missing required columns: {missing_text}. Run the processing script first.")

    velocity_df = df.dropna(subset=["lz_kpc_kms", "vr_kms"]).copy()
    if velocity_df.empty:
        raise ValueError("No stars with both Lz and Vr values were found in the classified CSV.")
    n_total = len(velocity_df)

    log("Creating Hercules velocity discovery plot...", started_at)
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    lz_full_low = float(np.percentile(velocity_df["lz_kpc_kms"], 0.5))
    lz_full_high = float(np.percentile(velocity_df["lz_kpc_kms"], 99.5))
    vr_full_low = float(np.percentile(velocity_df["vr_kms"], 0.5))
    vr_full_high = float(np.percentile(velocity_df["vr_kms"], 99.5))

    full_lz_range = (lz_full_low, lz_full_high)
    full_vr_range = (vr_full_low, vr_full_high)

    lsr_vr_kms = 0.0
    lsr_lz_kpc_kms = R0_KPC * VCIRC_LSR_KMS

    # U_sun is toward Galactic center, while positive Vr here is outward.
    sun_vr_kms = -U_SUN_PEC_KMS
    sun_lz_kpc_kms = R0_KPC * (VCIRC_LSR_KMS + V_SUN_PEC_KMS)

    mesh = ax.hist2d(
        velocity_df["vr_kms"],
        velocity_df["lz_kpc_kms"],
        bins=150,
        range=[full_vr_range, full_lz_range],
        cmap="cividis",
    )
    ax.set_title("Parent Sample Lz-Vr Map", fontsize=12, pad=10)
    ax.set_xlabel("Vr (km/s)", fontsize=11)
    ax.set_ylabel("Lz (kpc km/s)", fontsize=11)
    ax.set_xlim(full_vr_range)
    ax.set_ylim(full_lz_range)
    ax.scatter(
        lsr_vr_kms,
        lsr_lz_kpc_kms,
        marker="o",
        s=70,
        facecolor="none",
        edgecolor="white",
        linewidths=1.8,
        zorder=5,
        label="LSR",
    )
    ax.scatter(
        sun_vr_kms,
        sun_lz_kpc_kms,
        marker="x",
        s=90,
        c="yellow",
        linewidths=2.0,
        zorder=5,
        label="Sun",
    )
    ax.grid(True, linestyle="--", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(fontsize=10, loc="upper left", labelcolor="white")
        frame = legend.get_frame()
        frame.set_facecolor("black")
        frame.set_alpha(0.35)
        frame.set_edgecolor("none")
    fig.colorbar(mesh[3], ax=ax, label="Stars per bin")

    ax.text(
        0.02,
        0.02,
        (
            f"Refs: R0={R0_KPC:.1f} kpc, Vc={VCIRC_LSR_KMS:.0f} km/s, "
            f"(U,V)_sun=({U_SUN_PEC_KMS:.1f}, {V_SUN_PEC_KMS:.2f}) km/s"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 4},
    )
    ax.text(
        0.98,
        0.95,
        f"n = {n_total:,}",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        ha="right",
        va="top",
        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 4},
    )

    plt.suptitle("Hercules Discovery Step: Parent Lz-Vr Structure", fontsize=14, y=0.98)
    plt.tight_layout()

    log(f"Saving velocity discovery plot to {output_plot}...", started_at)
    plt.savefig(output_plot, dpi=300)
    log(f"Saving velocity discovery plot to {output_pdf}...", started_at)
    plt.savefig(output_pdf)
    plt.show()

    log("Finished Hercules velocity discovery plot.", started_at)


if __name__ == "__main__":
    main()