from datetime import datetime
from time import perf_counter, sleep

from astroquery.gaia import Gaia


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
        AND parallax_over_error >= 10
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
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"Submitting Gaia count query (attempt {attempt}/{MAX_RETRIES})...", started_at)
            job = Gaia.launch_job_async(query=query)
            log("Gaia count query returned.", started_at)
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

    count_query = f"""
    SELECT COUNT(*) AS n_matches
    FROM (
{QUERY.format(parallax_min_mas=PARALLAX_MIN_MAS).strip()}
    ) AS q
    """

    log("Starting Gaia count check.", started_at)
    results = run_query_with_retries(query=count_query, started_at=started_at)
    count_value = int(results[0][0])
    log(f"Matching rows: {count_value:,}", started_at)
    log("Finished Gaia count check.", started_at)


if __name__ == "__main__":
    main()