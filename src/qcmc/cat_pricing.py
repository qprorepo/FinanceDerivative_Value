"""
Catastrophe (excess-of-loss) tail-risk pricing from real NOAA Storm Events
data (manuscript Sec. "Catastrophe tail-risk pricing", Eq. ``cat_payoff``).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from qcmc.quantum_ae import DiscretisedDistribution

__all__ = [
    "parse_damage",
    "load_noaa_storm_losses",
    "discretise_empirical_log_spaced",
    "discretise_empirical_linear",
    "cat_excess_payoff",
]

_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "": 1.0}


def parse_damage(x) -> float:
    """Parse a NOAA ``DAMAGE_PROPERTY`` / ``DAMAGE_CROPS`` string
    (e.g. ``'1.50K'``, ``'2.00M'``, ``'0.00'``) into US dollars.

    Returns ``np.nan`` for missing or unparseable values.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s == "":
        return np.nan
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([KMB]?)$", s)
    if not m:
        return np.nan
    value, suffix = m.groups()
    return float(value) * _SUFFIX.get(suffix, 1.0)


def load_noaa_storm_losses(csv_path: str) -> dict:
    """Load and clean a NOAA Storm Events "details" CSV, returning the
    strictly-positive-loss severity sample and summary statistics.

    Returns a dict with keys ``cat_losses_usd`` (np.ndarray),
    ``n_total_events``, ``n_loss_events``, ``top_event_types``
    (a pandas Series of aggregate damage by ``EVENT_TYPE``, descending).
    """
    storm_cols = [
        "DAMAGE_PROPERTY",
        "DAMAGE_CROPS",
        "EVENT_TYPE",
        "STATE",
        "BEGIN_DATE_TIME",
        "MAGNITUDE",
    ]
    storm_raw = pd.read_csv(csv_path, usecols=storm_cols, low_memory=False)
    storm_raw["property_damage_usd"] = storm_raw["DAMAGE_PROPERTY"].apply(parse_damage)
    storm_raw["crop_damage_usd"] = storm_raw["DAMAGE_CROPS"].apply(parse_damage)
    storm_raw["total_damage_usd"] = storm_raw[["property_damage_usd", "crop_damage_usd"]].sum(
        axis=1, skipna=True
    )

    cat_losses_usd = storm_raw.loc[storm_raw["total_damage_usd"] > 0, "total_damage_usd"].to_numpy()

    top_event_types = (
        storm_raw.loc[storm_raw["total_damage_usd"] > 0]
        .groupby("EVENT_TYPE")["total_damage_usd"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    return dict(
        cat_losses_usd=cat_losses_usd,
        n_total_events=len(storm_raw),
        n_loss_events=len(cat_losses_usd),
        top_event_types=top_event_types,
    )


def discretise_empirical_log_spaced(
    samples: np.ndarray, n_qubits: int, clip_hi_pct: float = 99.9
) -> DiscretisedDistribution:
    """Log-spaced discretisation of an *empirical* (non-parametric) sample,
    generalising :func:`qcmc.quantum_ae.discretise_lognormal` to real,
    possibly heavy-tailed data (manuscript Sec. "Log-spaced binning and
    discretisation error"). Events above the ``clip_hi_pct`` percentile are
    folded into the top bin to preserve the tail probability mass.
    """
    n_bins = 2**n_qubits
    lo = max(samples.min(), 1e-6)
    hi = np.percentile(samples, clip_hi_pct)
    edges = np.exp(np.linspace(np.log(lo), np.log(hi), n_bins + 1))
    counts, _ = np.histogram(samples, bins=edges)
    counts[-1] += int(np.sum(samples > hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    probs = counts / counts.sum()
    probs = np.clip(probs, 1e-16, None)
    probs = probs / probs.sum()
    return DiscretisedDistribution(n_qubits, centres, probs, log_spaced=True)


def discretise_empirical_linear(
    samples: np.ndarray, n_qubits: int, clip_hi_pct: float = 99.9
) -> DiscretisedDistribution:
    """Linear-spaced counterpart of :func:`discretise_empirical_log_spaced`,
    used only as a discretisation-error comparison baseline."""
    n_bins = 2**n_qubits
    lo, hi = samples.min(), np.percentile(samples, clip_hi_pct)
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(samples, bins=edges)
    counts[-1] += int(np.sum(samples > hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    probs = counts / counts.sum()
    probs = np.clip(probs, 1e-16, None)
    probs = probs / probs.sum()
    return DiscretisedDistribution(n_qubits, centres, probs, log_spaced=False)


def cat_excess_payoff(x: np.ndarray, L0: float, Lmax: float) -> np.ndarray:
    """Manuscript Eq. ``cat_payoff``, the capped excess-of-loss payoff:

    .. math::
        f_{\\mathrm{CAT}}(x) = \\frac{\\min(\\max(x-L_0,0),\\,L_{\\max}-L_0)}{L_{\\max}-L_0}.
    """
    return np.clip(np.maximum(x - L0, 0.0), 0.0, Lmax - L0) / (Lmax - L0)
