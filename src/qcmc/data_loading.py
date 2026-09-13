"""
Robust Fama-French factor CSV parser and market calibration
(manuscript Sec. "Data Ingestion: Fama-French Factors & NOAA Storm Events").

Calibrates the GBM drift/volatility used in the option-pricing experiments
and the empirical :math:`5\\times5` factor covariance matrix
:math:`\\bm\\Sigma` used throughout every multi-asset, systemic-risk, and
QUBO-portfolio experiment.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = [
    "load_fama_french",
    "calibrate_market_parameters",
    "FACTOR_NAMES",
    "TRADING_DAYS_PER_YEAR",
]

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
TRADING_DAYS_PER_YEAR = 252


def load_fama_french(path: str, date_kind: str) -> pd.DataFrame:
    """Parse a Fama-French factor CSV (monthly / daily / weekly variants
    as distributed by the Kenneth R. French Data Library).

    The raw files carry a free-text header block, a blank line, the column
    header row, the numeric data block, then (for monthly files) a second
    "Annual Factors" block, and finally a copyright footer. The first
    contiguous numeric block is isolated robustly by regex-matching the
    date token at the start of each line, rather than relying on fixed line
    numbers, because the header-block length varies across files.

    Parameters
    ----------
    date_kind : {'monthly', 'daily', 'weekly'}
        Governs the ``strptime`` format applied to the leading date column
        (``%Y%m`` for monthly, ``%Y%m%d`` otherwise).

    Returns
    -------
    pd.DataFrame
        Indexed by ``Date``, with all factor columns converted from the
        file's native percentage units to decimal fractions.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw_lines = fh.readlines()

    if date_kind == "monthly":
        date_re = re.compile(r"^\s*(\d{6})\s*,")
    else:
        date_re = re.compile(r"^\s*(\d{8})\s*,")

    header_idx = None
    for i, line in enumerate(raw_lines):
        if line.lstrip().startswith(",") and "Mkt-RF" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not locate header row in {path}")

    columns = [c.strip() for c in raw_lines[header_idx].strip().split(",")]
    columns[0] = "Date"

    data_rows = []
    for line in raw_lines[header_idx + 1 :]:
        m = date_re.match(line)
        if not m:
            if data_rows:
                break
            else:
                continue
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != len(columns):
            continue
        data_rows.append(parts)

    df = pd.DataFrame(data_rows, columns=columns)
    for c in columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    if date_kind == "monthly":
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    else:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

    for c in columns[1:]:
        df[c] = df[c] / 100.0

    df = df.set_index("Date").sort_index()
    return df


def calibrate_market_parameters(
    ff_daily5: pd.DataFrame,
    calibration_start: str = "2020-08-01",
    factor_window_start: str = "2015-01-01",
) -> dict:
    """Calibrates GBM parameters and the 5-factor covariance/correlation
    matrix from a loaded 5-factor daily Fama-French panel.

    Returns a dict with keys ``mu_annual``, ``sigma_annual``, ``rf_annual``
    (scalars), ``sigma_factors_annual``, ``corr_factors``,
    ``mu_factors_annual``, ``vol_factors_annual`` (5-vectors/matrices, in
    the order given by :data:`FACTOR_NAMES`), and ``window`` (the
    calibration sub-DataFrame).
    """
    window = ff_daily5.loc[calibration_start:]
    mkt_total_return = window["Mkt-RF"] + window["RF"]

    mu_daily = mkt_total_return.mean()
    sigma_daily = mkt_total_return.std(ddof=1)
    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR
    sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    rf_annual = window["RF"].mean() * TRADING_DAYS_PER_YEAR

    factor_window = ff_daily5.loc[factor_window_start:, FACTOR_NAMES]
    sigma_factors_daily = factor_window.cov().values
    sigma_factors_annual = sigma_factors_daily * TRADING_DAYS_PER_YEAR
    corr_factors = factor_window.corr().values
    mu_factors_annual = factor_window.mean().values * TRADING_DAYS_PER_YEAR
    vol_factors_annual = np.sqrt(np.diag(sigma_factors_annual))

    return dict(
        window=window,
        mu_annual=float(mu_annual),
        sigma_annual=float(sigma_annual),
        rf_annual=float(rf_annual),
        sigma_factors_annual=sigma_factors_annual,
        corr_factors=corr_factors,
        mu_factors_annual=mu_factors_annual,
        vol_factors_annual=vol_factors_annual,
    )
