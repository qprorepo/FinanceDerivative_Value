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

## Verifying your download

After placing all five files in `data/raw/`, run:

```bash
python -c "
from qcmc.data_loading import load_fama_french
from qcmc.cat_pricing import load_noaa_storm_losses

ff = load_fama_french('data/raw/F-F_Research_Data_5_Factors_2x3_daily.csv', 'daily')
print('FF 5-factor daily:', ff.shape, ff.index.min().date(), '->', ff.index.max().date())

noaa = load_noaa_storm_losses('data/raw/StormEvents_details-ftp_v1_0_d2024_c20260728.csv')
print('NOAA events:', noaa['n_total_events'], 'total,', noaa['n_loss_events'], 'with reported loss')
"
```

Expected order of magnitude: several thousand daily FF rows spanning
decades; ~69,000-70,000 total NOAA 2024 records with ~14,000-15,000
reporting damage (exact counts vary slightly with NOAA's periodic
reprocessing of past-year data, hence the `c20260728` creation-date suffix
in the filename above — an event whose damage estimate is revised between
reprocessing runs will shift these counts by a handful of records; this
does not materially change any reported figure or conclusion).

## Processed / cached artefacts

`data/raw/` is git-ignored (see `.gitignore`). If you want a fast-loading
cached version for repeated notebook runs, `notebooks/QCMC_Quantum_Finance_Analysis.ipynb`
will happily read Parquet caches if you add that convenience yourself
(not included by default, to keep the reference pipeline's data path fully
transparent and auditable from the raw CSVs).
