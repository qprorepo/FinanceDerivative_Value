# Data

This project uses two real, publicly-accessible datasets. **Neither is
committed to this repository** — the Fama-French files carry their own
redistribution terms, and the NOAA file is ~67 MB, too large for a git
repository without LFS. Instead, this file documents exactly where to get
them and how to verify you have the right files.

## 1. Fama-French factor panels (Kenneth R. French Data Library)

| File (place in `data/raw/`)                       | Description                          |
|-----------------------------------------------------|---------------------------------------|
| `F-F_Research_Data_5_Factors_2x3.csv`                | 5-factor model, **monthly**           |
| `F-F_Research_Data_5_Factors_2x3_daily.csv`          | 5-factor model, **daily**             |
| `F-F_Research_Data_Factors_daily.csv`                | 3-factor model, **daily**             |
| `F-F_Research_Data_Factors_weekly.csv`               | 3-factor model, **weekly**            |

**Source:** [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
(Dartmouth Tuck School of Business). Download the "CSV" links for
"Fama/French 5 Factors (2x3)" and "Fama/French 3 Factors" (both daily and
weekly variants), unzip, and place the four `.csv` files directly under
`data/raw/`.

**License / terms of use:** the data are provided free for academic
research by Prof. French; see the "Data Library" page's terms. This
project does not redistribute the files — you must download them yourself.

**Format note:** these files have a free-text header, a blank line, a
comma-led column-header row, the numeric data block, and (for the monthly
5-factor file) a second "Annual Factors" block followed by a copyright
footer. `qcmc.data_loading.load_fama_french` parses this robustly via
regex date-matching rather than fixed line numbers — see that module's
docstring and `tests/test_data_loading.py` for the exact parsing contract.

## 2. NOAA Storm Events Database

| File (place in `data/raw/`)                                        | Description                    |
|------------------------------------------------------------------------|---------------------------------|
| `StormEvents_details-ftp_v1_0_d2024_c20260728.csv`                      | 2024 annual "details" file      |

**Source:** [NOAA National Centers for Environmental Information — Storm Events Database](https://www.ncdc.noaa.gov/stormevents/)
(bulk CSV files are served from
`https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/`). Download
the 2024 "details" file (the exact `cYYYYMMDD` creation-date suffix in the
filename will differ depending on when NOAA last reprocessed the year —
the `qcmc.cat_pricing.load_noaa_storm_losses` loader only needs the
`DAMAGE_PROPERTY`, `DAMAGE_CROPS`, `EVENT_TYPE`, `STATE`,
`BEGIN_DATE_TIME`, and `MAGNITUDE` columns, so any recent reprocessing of
the 2024 file is compatible).

**License:** NOAA Storm Events data is a **U.S. Government work** and is
in the public domain (no restrictions on use).

**Scale:** ~69,800 total 2024 event records, of which ~14,800 report
non-zero property or crop damage (a "$0.00K" damage figure — the
overwhelming majority of records — means no economic loss was reported for
that event, not that the field is missing).
