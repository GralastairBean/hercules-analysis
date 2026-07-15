from pathlib import Path
from datetime import datetime
from time import perf_counter

import astropy.coordinates as coord
import astropy.units as u
import pandas as pd


RAW_CSV_NAME = "gaia_real_h1_h2_raw.csv"
CLASSIFIED_CSV_NAME = "gaia_real_h1_h2_classified.csv"

MAX_DISTANCE_PC = 1000.0
RUWE_MAX = 1.4
H1_V_MIN = -55.0
H1_V_MAX = -43.0
H2_V_MIN = -43.0
H2_V_MAX = -30.0


def log(message: str, started_at: float) -> None:
    elapsed_seconds = perf_counter() - started_at
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp} | +{elapsed_seconds:7.2f}s] {message}", flush=True)


def add_galactocentric_columns(df: pd.DataFrame) -> pd.DataFrame:
    galactocentric_frame = coord.Galactocentric()
    skycoord = coord.SkyCoord(
        ra=df["ra"].to_numpy() * u.deg,
        dec=df["dec"].to_numpy() * u.deg,
        distance=(1000.0 / df["parallax"].to_numpy()) * u.pc,
        pm_ra_cosdec=df["pmra"].to_numpy() * u.mas / u.yr,
        pm_dec=df["pmdec"].to_numpy() * u.mas / u.yr,
        radial_velocity=df["radial_velocity"].to_numpy() * u.km / u.s,
        frame="icrs",
    )

    galactocentric = skycoord.transform_to(galactocentric_frame)
    enriched = df.copy()
    x_kpc = galactocentric.x.to_value(u.kpc)
    y_kpc = galactocentric.y.to_value(u.kpc)
    cylindrical_radius_kpc = (x_kpc**2 + y_kpc**2) ** 0.5

    enriched["distance_pc"] = 1000.0 / enriched["parallax"]
    enriched["x_kpc"] = x_kpc
    enriched["y_kpc"] = y_kpc
    enriched["u_velocity_kms"] = galactocentric.v_x.to_value(u.km / u.s)
    enriched["v_velocity_kms"] = galactocentric.v_y.to_value(u.km / u.s)
    enriched["w_velocity_kms"] = galactocentric.v_z.to_value(u.km / u.s)
    enriched["local_v_kms"] = enriched["v_velocity_kms"] - galactocentric_frame.galcen_v_sun.y.to_value(u.km / u.s)
    enriched["vr_kms"] = (
        x_kpc * enriched["u_velocity_kms"] + y_kpc * enriched["v_velocity_kms"]
    ) / cylindrical_radius_kpc
    # Use the literature sign convention: prograde disk stars have positive Lz.
    enriched["lz_kpc_kms"] = -(x_kpc * enriched["v_velocity_kms"] - y_kpc * enriched["u_velocity_kms"])
    enriched["z_height_pc"] = galactocentric.z.to_value(u.pc)
    return enriched


def main() -> None:
    started_at = perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    raw_csv = data_dir / RAW_CSV_NAME
    classified_csv = data_dir / CLASSIFIED_CSV_NAME

    log("Starting Gaia Hercules processing step.", started_at)
    log(f"Loading raw Gaia sample from {raw_csv}...", started_at)
    df = pd.read_csv(raw_csv)
    log(f"Loaded {len(df):,} raw rows.", started_at)

    log("Computing Galactocentric velocities and z heights...", started_at)
    df = add_galactocentric_columns(df)

    log(f"Applying quality cuts: distance <= {MAX_DISTANCE_PC:.0f} pc and ruwe < {RUWE_MAX:.1f}...", started_at)
    quality_filter = (df["distance_pc"] <= MAX_DISTANCE_PC) & (df["ruwe"] < RUWE_MAX)
    clean_df = df[quality_filter].copy()
    log(f"Quality-cut sample contains {len(clean_df):,} stars.", started_at)

    log("Splitting cleaned sample into Hercules 1 and Hercules 2 groups using local V velocity...", started_at)
    h1_filter = (clean_df["local_v_kms"] >= H1_V_MIN) & (clean_df["local_v_kms"] < H1_V_MAX)
    h2_filter = (clean_df["local_v_kms"] >= H2_V_MIN) & (clean_df["local_v_kms"] <= H2_V_MAX)

    h1_df = clean_df[h1_filter].copy()
    h2_df = clean_df[h2_filter].copy()

    log(f"Isolated {len(h1_df):,} Hercules 1 stars.", started_at)
    log(f"Isolated {len(h2_df):,} Hercules 2 stars.", started_at)
    log(f"Removed {len(clean_df) - len(h1_df) - len(h2_df):,} non-member stars.", started_at)

    classified_df = clean_df.copy()
    classified_df["h_group"] = "Other"
    classified_df.loc[h1_filter, "h_group"] = "H1"
    classified_df.loc[h2_filter, "h_group"] = "H2"

    log(f"Saving classified sample to {classified_csv}...", started_at)
    classified_df.to_csv(classified_csv, index=False)

    log("Finished Gaia Hercules processing step.", started_at)


if __name__ == "__main__":
    main()
