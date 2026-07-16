from pathlib import Path
from datetime import datetime
from time import perf_counter, sleep

import pandas as pd
from astroquery.gaia import Gaia


# Proof-of-concept pull settings. Keep this intentionally small so the
# end-to-end pipeline can be validated quickly before expanding the sample.
# timing: 100k=40 min, 200k=2hrs...
PARALLAX_MIN_MAS = 1.0  # Parallax = 1.0 sets search radius to d = 1 kpc
MAX_RETRIES = 8
RETRY_BASE_SECONDS = 60

QUERY = """
    SELECT
        source_id,
        ra,
        dec,
        parallax,
        pmra,
        pmdec,
        radial_velocity,
        ruwe,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag
    FROM gaiadr3.gaia_source
    WHERE parallax IS NOT NULL
        AND parallax > 0
        AND pmra IS NOT NULL
        AND pmdec IS NOT NULL
        AND radial_velocity IS NOT NULL
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


def main() -> None:
    started_at = perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    raw_csv = data_dir / "gaia_real_h1_h2_raw.csv"

    log("Starting proof-of-concept Gaia Hercules data pull.", started_at)
    log(
        (
            "Submitting Gaia DR3 proof-of-concept query for "
            f"all sources with parallax >= {PARALLAX_MIN_MAS:.3f} mas..."
        ),
        started_at,
    )
    query = QUERY.format(parallax_min_mas=PARALLAX_MIN_MAS)
    table = run_query_with_retries(query=query, started_at=started_at)
    df = table.to_pandas()
    log(f"Loaded {len(df):,} rows.", started_at)

    log(f"Saving raw query results to {raw_csv}...", started_at)
    df.to_csv(raw_csv, index=False)

    log("Finished proof-of-concept Gaia Hercules data pull.", started_at)


if __name__ == "__main__":
    main()