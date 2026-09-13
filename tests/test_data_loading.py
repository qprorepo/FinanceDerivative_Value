import numpy as np
import pandas as pd
import pytest

from qcmc.data_loading import FACTOR_NAMES, calibrate_market_parameters, load_fama_french


def _write_synthetic_ff_daily(tmp_path, n_days=400):
    """Writes a small synthetic Fama-French-5-factor-style daily CSV
    matching the real file's header/footer structure closely enough for
    the parser to exercise its header-detection and blank-line-termination
    logic."""
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(0)
    rows = []
    for d in dates:
        vals = rng.normal(0.02, 0.5, size=6)  # in percent, matching real file units
        rows.append([d.strftime("%Y%m%d")] + [f"{v:.2f}" for v in vals])

    header = "This file contains synthetic factors for testing purposes only\n\n"
    col_header = ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
    body = "\n".join(",".join(r) for r in rows)
    footer = "\n\nCopyright 2026 Synthetic Data Generator. All rights reserved.\n"

    path = tmp_path / "F-F_Research_Data_5_Factors_2x3_daily.csv"
    path.write_text(header + col_header + body + footer)
    return str(path), dates, rows


def test_load_fama_french_daily_parses_header_and_dates(tmp_path):
    path, dates, rows = _write_synthetic_ff_daily(tmp_path)
    df = load_fama_french(path, date_kind="daily")

    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == len(rows)
    assert df.index[0] == dates[0]
    assert df.index[-1] == dates[-1]
    assert df.index.is_monotonic_increasing


def test_load_fama_french_converts_percent_to_decimal(tmp_path):
    path, _, rows = _write_synthetic_ff_daily(tmp_path)
    df = load_fama_french(path, date_kind="daily")
    first_row_pct = [float(x) for x in rows[0][1:]]
    first_row_decimal = df.iloc[0].values
    np.testing.assert_allclose(first_row_decimal, np.array(first_row_pct) / 100.0, atol=1e-12)


def test_load_fama_french_monthly_date_format(tmp_path):
    header = "Synthetic monthly test file\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n"
    rows = ["202001,1.00,0.50,-0.20,0.10,0.05,0.01", "202002,-2.00,0.30,0.40,-0.10,0.02,0.01"]
    footer = "\n\n Annual Factors: January-December \n\n,Mkt-RF,SMB\n2020,5.0,1.0\n"
    path = tmp_path / "F-F_Research_Data_5_Factors_2x3.csv"
    path.write_text(header + "\n".join(rows) + footer)

    df = load_fama_french(str(path), date_kind="monthly")
    assert len(df) == 2  # the 'Annual Factors' block must NOT be included
    assert df.index[0] == pd.Timestamp("2020-01-01")
    assert df.index[1] == pd.Timestamp("2020-02-01")


def test_load_fama_french_raises_on_missing_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("this file has no Mkt-RF header at all\n1,2,3\n")
    with pytest.raises(ValueError):
        load_fama_french(str(path), date_kind="daily")


def test_calibrate_market_parameters_returns_sane_shapes(tmp_path):
    path, _, _ = _write_synthetic_ff_daily(tmp_path, n_days=1600)
    df = load_fama_french(path, date_kind="daily")
    result = calibrate_market_parameters(
        df,
        calibration_start=df.index[500].strftime("%Y-%m-%d"),
        factor_window_start=df.index[0].strftime("%Y-%m-%d"),
    )
    assert isinstance(result["mu_annual"], float)
    assert isinstance(result["sigma_annual"], float)
    assert result["sigma_annual"] > 0
    assert result["sigma_factors_annual"].shape == (5, 5)
    assert result["corr_factors"].shape == (5, 5)
    np.testing.assert_allclose(np.diag(result["corr_factors"]), 1.0, atol=1e-9)
    assert result["mu_factors_annual"].shape == (5,)
    assert result["vol_factors_annual"].shape == (5,)
    assert np.all(result["vol_factors_annual"] > 0)


def test_factor_names_constant_matches_expected_order():
    assert FACTOR_NAMES == ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
