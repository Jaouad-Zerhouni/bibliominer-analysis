"""Growth indicators and sample size.

The formulas are those of the literature, written out explicitly so that
they can be checked:

  - **CAGR**, compound annual growth
        ``((N_end / N_start)^(1/n) - 1) × 100``   (n = number of intervals)

  - **AGR**, simple annual growth, from one year to the next
        ``(N_t - N_{t-1}) / N_{t-1} × 100``

  - **RGR**, relative growth rate (Mahapatra, 1985)
        ``R = (ln W₂ - ln W₁) / (T₂ - T₁)``
    where W is the **cumulative** number of publications. RGR mechanically
    decreases over time: a mature corpus grows more slowly in relative
    terms.

  - **Doubling time**, the time needed to double the stock
        ``Dt = ln(2) / RGR = 0.693 / RGR``

  - **Cochran**, representative sample size
        ``n₀ = Z²·p·q / e²``  then the finite population correction
        ``n = n₀ / (1 + (n₀ - 1)/N)``

CAGR and AGR answer two different questions: the first smooths the whole
period, the second shows the jolts. Giving them together avoids
concluding to steady growth where there is only one good year.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _docs_per_year(corpus) -> pd.Series:
    """Documents per year, with no gap in the series."""
    years = pd.to_numeric(corpus.documents["year"], errors="coerce").dropna()
    if years.empty:
        return pd.Series(dtype=int)
    counts = years.astype(int).value_counts().sort_index()
    full = range(int(counts.index.min()), int(counts.index.max()) + 1)
    return counts.reindex(full, fill_value=0)


def cagr(corpus) -> Optional[float]:
    """Compound annual growth, as a percentage.

    Returns ``None`` if the corpus covers less than two years or if the first
    year is empty: the rate would then be infinite, which makes no sense.
    """
    per_year = _docs_per_year(corpus)
    if len(per_year) < 2:
        return None
    first, last = int(per_year.iloc[0]), int(per_year.iloc[-1])
    n = len(per_year) - 1
    if first <= 0 or n <= 0:
        return None
    return round(((last / first) ** (1 / n) - 1) * 100, 2)


def agr(corpus) -> pd.DataFrame:
    """Simple annual growth, year by year.

    The first year has no rate (no previous year), nor does a year following
    an empty year: dividing by zero would give an infinity, which is better
    left empty than shown as a wrong number.
    """
    per_year = _docs_per_year(corpus)
    empty = pd.DataFrame(columns=["year", "documents", "agr", "cumulative"])
    if per_year.empty:
        return empty

    df = pd.DataFrame({"year": per_year.index.astype(int),
                       "documents": per_year.to_numpy(dtype=int)})
    prev = df["documents"].shift(1)
    df["agr"] = np.where(prev > 0,
                         ((df["documents"] - prev) / prev * 100).round(2),
                         np.nan)
    df["cumulative"] = df["documents"].cumsum()
    return df


def rgr_doubling_time(corpus) -> pd.DataFrame:
    """RGR and doubling time, year by year.

    Columns: ``year``, ``documents``, ``cumulative``, ``ln_cumulative``,
    ``rgr``, ``doubling_time``.

    The RGR of a year is computed between the cumulative count of the
    previous year and that of the current year. The first year therefore has
    none.
    """
    per_year = _docs_per_year(corpus)
    empty = pd.DataFrame(columns=["year", "documents", "cumulative",
                                  "ln_cumulative", "rgr", "doubling_time"])
    if per_year.empty:
        return empty

    df = pd.DataFrame({"year": per_year.index.astype(int),
                       "documents": per_year.to_numpy(dtype=int)})
    df["cumulative"] = df["documents"].cumsum()
    # ln(0) does not exist: an empty first year stays without a value.
    df["ln_cumulative"] = np.where(df["cumulative"] > 0,
                                   np.log(df["cumulative"].replace(0, np.nan)),
                                   np.nan)
    prev_ln = df["ln_cumulative"].shift(1)
    # The doubling time is computed on the EXACT RGR. Rounded first to 4
    # decimals, an RGR of 0.00004 became 0 and the doubling time disappeared;
    # an RGR of 0.00015 rounded to 0.0001 gave 6,931 years instead of 4,621.
    rgr = df["ln_cumulative"] - prev_ln
    df["rgr"] = rgr
    df["doubling_time"] = np.where(rgr > 0, math.log(2) / rgr, np.nan)
    df["ln_cumulative"] = df["ln_cumulative"].round(4)
    df["rgr"] = df["rgr"].round(4)
    df["doubling_time"] = df["doubling_time"].round(2)
    return df


def cochran_sample_size(corpus, confidence: float = 0.95,
                        margin: float = 0.05,
                        proportion: float = 0.5) -> Dict[str, Any]:
    """Representative sample size of the corpus (Cochran, 1977).

    How many documents must be read for the conclusions to hold for the whole
    corpus? ``proportion = 0.5`` is the most cautious choice: it is the value
    that maximises the variance, hence the required size.

    The usual confidence levels are tabulated; outside them, the quantile of
    the normal distribution is approximated without depending on SciPy.
    """
    n_pop = int(len(corpus.documents))
    z = _z_score(confidence)
    p = float(proportion)
    q = 1.0 - p

    n0 = (z ** 2) * p * q / (margin ** 2)
    # Finite population correction: without it, we would sometimes ask for more
    # documents than the corpus contains.
    n = n0 / (1 + (n0 - 1) / n_pop) if n_pop > 0 else n0

    required = int(math.ceil(min(n, n_pop))) if n_pop else 0
    return {
        "population": n_pop,
        "confidence": confidence,
        "margin_of_error": margin,
        "proportion": p,
        "z_score": round(z, 4),
        "sample_size_infinite": int(math.ceil(n0)),
        "sample_size": required,
        "sampling_fraction": round(100 * required / n_pop, 1) if n_pop else 0.0,
    }


#: Usual quantiles of the standard normal distribution (two-sided).
_Z = {0.80: 1.2816, 0.85: 1.4395, 0.90: 1.6449, 0.95: 1.9600,
      0.98: 2.3263, 0.99: 2.5758}


def _z_score(confidence: float) -> float:
    confidence = float(confidence)
    # Table only for an EXACTLY tabulated value. Rounding first sent 0.975 to
    # 0.97 (Python's floating-point rounding), and gave z = 2.1705 instead of
    # 2.2414.
    for key, z in _Z.items():
        if abs(confidence - key) < 1e-9:
            return z
    # Rational approximation from Abramowitz & Stegun (26.2.23), error
    # < 4.5·10⁻⁴, sufficient here and without SciPy.
    p = 1 - (1 - confidence) / 2
    t = math.sqrt(-2.0 * math.log(1 - p)) if p > 0.5 else math.sqrt(-2.0 * math.log(p))
    num = 2.515517 + 0.802853 * t + 0.010328 * t * t
    den = 1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3
    z = t - num / den
    return abs(z)


def trend_forecast(corpus, horizon: int = 5,
                   model: str = "linear") -> Dict[str, Any]:
    """Least-squares trend analysis, and projection.

    Two models:

      - **linear**: ``N = a + b·t``, fitted directly on the counts. ``b``
        reads as "additional publications per year".
      - **exponential**: ``ln N = a + b·t``, i.e. ``N = e^a · e^(b·t)``.
        Fitted on the logarithms, it suits accelerating growth. Empty years
        are excluded (ln 0 does not exist).

    **R² is computed on the original scale in both cases**, otherwise an R²
    of logarithms would be compared with an R² of counts, and the exponential
    would always look better.

    The last year is often **incomplete** (indexing in progress): it pulls the
    trend down. `last_year_partial` flags it, but it is not removed
    automatically: the user decides.

    Returns ``{"table": DataFrame, "fit": {...}}`` where the table carries the
    observed **and** projected years, with ``kind`` set to "observed" or
    "forecast".
    """
    per_year = _docs_per_year(corpus)
    empty = pd.DataFrame(columns=["year", "documents", "fitted", "kind"])
    if len(per_year) < 3:
        # Two points always give an R² of 1: a trend only makes sense from three
        # observations on.
        return {"table": empty, "fit": None,
                "message": "At least three years are needed."}

    years = per_year.index.to_numpy(dtype=float)
    counts = per_year.to_numpy(dtype=float)
    t = years - years[0]

    if model == "exponential":
        mask = counts > 0
        if mask.sum() < 3:
            return {"table": empty, "fit": None,
                    "message": "Too many empty years for an exponential fit."}
        b, a = np.polyfit(t[mask], np.log(counts[mask]), 1)
        predict = lambda x: np.exp(a) * np.exp(b * x)  # noqa: E731
        slope_label = "annual growth factor"
        slope_value = round(float(np.exp(b)), 4)
    else:
        b, a = np.polyfit(t, counts, 1)
        predict = lambda x: a + b * x                   # noqa: E731
        slope_label = "additional documents per year"
        slope_value = round(float(b), 3)

    fitted = predict(t)
    ss_res = float(np.sum((counts - fitted) ** 2))
    ss_tot = float(np.sum((counts - counts.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Standard error of the estimate: 2 parameters consumed (slope, intercept).
    dof = max(len(counts) - 2, 1)
    rmse = float(np.sqrt(ss_res / dof))

    rows = [{"year": int(y), "documents": int(n), "fitted": round(float(f), 2),
             "kind": "observed"}
            for y, n, f in zip(years, counts, fitted)]

    last = int(years[-1])
    for k in range(1, horizon + 1):
        x = float(last + k - years[0])
        # A negative predicted count makes no sense: it is floored at zero.
        rows.append({"year": last + k, "documents": None,
                     "fitted": round(max(float(predict(x)), 0.0), 2),
                     "kind": "forecast"})

    return {
        "table": pd.DataFrame(rows),
        "fit": {
            "model": model,
            "slope": slope_value,
            "slope_label": slope_label,
            "intercept": round(float(a), 3),
            "r2": round(float(r2), 4),
            "rmse": round(rmse, 2),
            "years_used": int(len(counts)),
            "last_year": last,
            # True when the last year is the current (or a future) year: its indexing
            # is not finished. It was always None, while the docstring promised to flag
            # it.
            "last_year_partial": bool(last >= date.today().year),
        },
    }


def growth_summary(corpus) -> Dict[str, Any]:
    """Summary of the growth indicators, for the interface tiles."""
    a = agr(corpus)
    r = rgr_doubling_time(corpus)
    mean_agr = float(a["agr"].dropna().mean()) if not a.empty else float("nan")
    mean_rgr = float(r["rgr"].dropna().mean()) if not r.empty else float("nan")
    mean_dt = float(r["doubling_time"].dropna().mean()) if not r.empty else float("nan")
    return {
        "cagr": cagr(corpus),
        "agr_mean": round(mean_agr, 2) if not math.isnan(mean_agr) else None,
        "rgr_mean": round(mean_rgr, 4) if not math.isnan(mean_rgr) else None,
        "doubling_time_mean": round(mean_dt, 2) if not math.isnan(mean_dt) else None,
    }
