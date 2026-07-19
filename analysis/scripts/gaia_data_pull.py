import os
from pathlib import Path
from datetime import datetime
from time import perf_counter, sleep

import pandas as pd
from astroquery.gaia import Gaia


# QUERY FILTERS
PARALLAX_MIN_MAS = 1.0  # Parallax = 1.0 sets search radius to d = 1 kpc
PARALLAX_OVER_ERROR_MIN = 10  # Keep only stars with <= 10% parallax error
RADIAL_VELOCITY_ERROR_MAX = 5.0  # Keep only stars with absolute radial-velocity error <= 5 km/s
MAX_RETRIES = 8
RETRY_BASE_SECONDS = 60
GAIA_USERNAME_ENV = "GAIA_USERNAME"
GAIA_PASSWORD_ENV = "GAIA_PASSWORD"

QUERY = """
    SELECT
        source_id,
        ra,
        dec,
        parallax,
        pmra,
        pmdec,
        radial_velocity,
        radial_velocity_error,
        ruwe,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag
    FROM gaiadr3.gaia_source
    WHERE parallax IS NOT NULL
        AND parallax_over_error >= {parallax_over_error_min}
        AND pmra IS NOT NULL
        AND pmdec IS NOT NULL
        AND radial_velocity_error IS NOT NULL
        AND radial_velocity_error <= {radial_velocity_error_max}
        AND phot_g_mean_mag IS NOT NULL
        AND phot_bp_mean_mag IS NOT NULL
        AND phot_rp_mean_mag IS NOT NULL
        AND parallax >= {parallax_min_mas}
"""


def log(message: str, started_at: float) -> None:
    elapsed_seconds = perf_counter() - started_at
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp} | +{elapsed_seconds:7.2f}s] {message}", flush=True)


def run_query_with_retries(query: str, started_at: float):
    """Run the Gaia async job with retry/backoff for transient network drops."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"Submitting Gaia query (attempt {attempt}/{MAX_RETRIES})...", started_at)
            job = Gaia.launch_job_async(query=query)
            log("Gaia query returned. Loading results into pandas...", started_at)
            return job.get_results()
        except (ConnectionResetError, TimeoutError, OSError) as exc:
            if attempt == MAX_RETRIES:
                raise
            backoff_seconds = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            log(
                (
                    f"Transient Gaia/network error: {exc.__class__.__name__}. "
                    f"Retrying in {backoff_seconds}s..."
                ),
                started_at,
            )
            sleep(backoff_seconds)


def login_to_gaia(started_at: float) -> None:
    username = os.getenv(GAIA_USERNAME_ENV)
    password = os.getenv(GAIA_PASSWORD_ENV)
    if not username or not password:
        raise RuntimeError(
            f"Missing Gaia credentials. Set {GAIA_USERNAME_ENV} and {GAIA_PASSWORD_ENV} environment variables."
        )

    log("Logging in to Gaia TAP using configured account...", started_at)
    Gaia.login(user=username, password=password)
    log("Gaia login successful.", started_at)


def main() -> None:
    started_at = perf_counter()
    login_to_gaia(started_at)
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    raw_csv = data_dir / "gaia_data_pull_raw.csv"

    log("Starting data pull.", started_at)
    log(
        (
            "Submitting query for "
            f"all sources with parallax >= {PARALLAX_MIN_MAS:.3f} mas..."
        ),
        started_at,
    )
    query = QUERY.format(
        parallax_min_mas=PARALLAX_MIN_MAS,
        parallax_over_error_min=PARALLAX_OVER_ERROR_MIN,
        radial_velocity_error_max=RADIAL_VELOCITY_ERROR_MAX,
    )
    table = run_query_with_retries(query=query, started_at=started_at)
    df = table.to_pandas()
    log(f"Loaded {len(df):,} rows.", started_at)

    log(f"Saving query results to {raw_csv}...", started_at)
    df.to_csv(raw_csv, index=False)

    log("Finished data pull.", started_at)


if __name__ == "__main__":
    main()