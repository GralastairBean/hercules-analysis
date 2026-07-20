# Hercules Analysis Repo for Dissertation
## Overview
An analysis of the Hercules Moving Group using *Gaia* data. The project is open-ended, currently in the brain-storm phase.
The idea at the moment is...
1.  First find the various HMG in the raw data (2-3 groups?).
2. Apply an appropriate filtering mechanism to accurately catagorise stars as H1, H2 or other.
3. Make various comparisons between the HMG and the generic population to confirm their status as peturbed disc stars.
3.1 Compare z-height
3.2 Compare HR diagrams (shows different ages as y not c turn-off)
3.3 Compare metallicity

## Pipeline
### Initial Setup: must be run in this order for pipeline to work...
1. Log into your Gaia access account by setting username & password in a new python terminal
2. Run scripts\gaia_count_query.py to see how many rows the data pull will get and to check log in and connection
3. Run analysis\scripts\gaia_data_pull.py to pull the raw data into a csv analysis\data\gaia_data_pull_raw.csv
4. Run analysis\scripts\data_processing_classify.py to transform motions/positions and catagorise raw data into H1, H2, other
5. Run analysis\scripts\gross_error_check_plots.py and check processed data appears as expected.

### You can now run any of the following in any order as they all point back to the processed csv...
- analysis\scripts\hercules_discovery_plot.py = shows the H1,2 and 3 regions on a Lz vs Vr plot.
- analysis\scripts\hercules_discovery_plot_contours.py =  as above but with percentile contours.
- analysis\scripts\hercules_discovery_histogram.py shows Lz bumps for H1 and H2.
- analysis\scripts\analysis_zheight_comparison.py = plot H1, H2 and other z-heights.
- analysis\scripts\analysis_HR_comparison.py = plot H1, H2 and other HR diagrams. 