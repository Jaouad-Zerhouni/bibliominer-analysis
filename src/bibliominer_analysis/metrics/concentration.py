"""Concentration of production: Gini, Lorenz curve, ratios.

A corpus of a hundred articles can be the work of a hundred people who
each published once, or of three people who wrote thirty each. Rankings do
not distinguish these two situations: they show the top of the table,
never the shape of the distribution.

  - **Gini**: 0 = everyone produces the same, 1 = a single entity produces
    everything. On authors, a Gini above ~0.6 signals a field carried by a
    few people.
  - **Lorenz curve**: the representation of which the Gini is only the
    summary. It shows *where* the inequality lies, which a single number
    does not.
  - **CR4 / CR10**: share of the top 4 and top 10. Easier to read than a
    Gini for a non-specialist, and enough for most purposes.

Lotka (`laws.py`) answers a neighbouring but different question: it tests
whether the distribution follows a power law. Here nothing is tested; it
is measured.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

#: Dimensions on which concentration makes sense.
UNITS = ("authors", "institutions", "countries", "sources", "cities")


def gini(values) -> Optional[float]:
    """Gini coefficient of a series of positive values.

    Direct formula on the sorted series. Zero values count: an author with
    zero documents is part of the population, and excluding them would
    artificially lower the measured inequality.
    """
    x = np.sort(np.asarray([v for v in values if v is not None and v >= 0],
                           dtype=float))
    n = x.size
    if n == 0:
        return None
    total = x.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return round(float((2.0 * (index * x).sum()) / (n * total) - (n + 1.0) / n), 4)


def lorenz(values, points: int = 40) -> pd.DataFrame:
    """Lorenz curve: cumulative share of the population x cumulative share of
    the total.

    Columns: ``population_share``, ``value_share``.

    The diagonal represents perfect equality; the area between the curve and
    the diagonal is exactly what the Gini summarises.
    """
    x = np.sort(np.asarray([v for v in values if v is not None and v >= 0],
                           dtype=float))
    if x.size == 0 or x.sum() <= 0:
        return pd.DataFrame({"population_share": [0.0, 100.0],
                             "value_share": [0.0, 100.0]})

    cum = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
    pop = np.linspace(0.0, 1.0, x.size + 1)
    # The curve is sampled: beyond a few dozen points it gains nothing in
    # readability and makes the answer heavier.
    if pop.size > points:
        keep = np.unique(np.linspace(0, pop.size - 1, points).astype(int))
        pop, cum = pop[keep], cum[keep]
    return pd.DataFrame({
        "population_share": np.round(100.0 * pop, 2),
        "value_share": np.round(100.0 * cum, 2),
    })


def _series(corpus, unit: str) -> pd.Series:
    """Documents per entity, for the requested unit."""
    if unit == "authors":
        a = corpus.authors
        a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
        if a.empty:
            return pd.Series(dtype=float)
        key = a["scopus_id"].fillna("name:" + a["name"].map(str))
        return a.assign(key=key).drop_duplicates(["key", "eid"]).groupby("key").size()

    if unit == "sources":
        d = corpus.documents[["eid", "source"]].dropna()
        d = d[d["source"].map(str).str.strip() != ""]
        return d.groupby("source")["eid"].nunique()

    frame, column = None, None
    if unit == "institutions":
        from .production import _org_frame
        frame, column = _org_frame(corpus, "parent"), "org"
    elif unit == "countries":
        frame, column = corpus.affiliations, "country"
    elif unit == "cities":
        frame, column = corpus.affiliations, "city"
    if frame is None or column not in getattr(frame, "columns", []):
        return pd.Series(dtype=float)

    f = frame[["eid", column]].dropna()
    f = f[f[column].map(str).str.strip() != ""]
    if f.empty:
        return pd.Series(dtype=float)
    return f.drop_duplicates().groupby(column)["eid"].nunique()


def concentration(corpus, unit: str = "authors") -> Dict[str, Any]:
    """Concentration measures for one dimension.

    Returns: ``unit``, ``entities``, ``documents``, ``gini``, ``cr4``,
    ``cr10``, ``top_share_10pct``, ``lorenz`` (list of points).

    ``top_share_10pct`` is the share produced by the 10 % most active, the
    wording any reader understands, unlike the Gini.
    """
    s = _series(corpus, unit)
    empty = {"unit": unit, "entities": 0, "documents": 0, "gini": None,
             "cr4": None, "cr10": None, "top_share_10pct": None,
             "lorenz": lorenz([]).to_dict("records")}
    if s.empty:
        return empty

    values = np.sort(s.to_numpy(dtype=float))[::-1]
    total = values.sum()
    if total <= 0:
        return empty

    n = values.size
    top10pct = max(1, int(round(0.1 * n)))
    return {
        "unit": unit,
        "entities": int(n),
        "documents": int(total),
        "gini": gini(values),
        "cr4": round(100.0 * values[:4].sum() / total, 1),
        "cr10": round(100.0 * values[:10].sum() / total, 1),
        "top_share_10pct": round(100.0 * values[:top10pct].sum() / total, 1),
        "lorenz": lorenz(values).to_dict("records"),
    }


def concentration_summary(corpus) -> pd.DataFrame:
    """One row per dimension: the comparison says more than the detail.

    Columns: ``unit``, ``entities``, ``documents``, ``gini``, ``cr4``,
    ``cr10``, ``top_share_10pct``.
    """
    cols = ["unit", "entities", "documents", "gini", "cr4", "cr10",
            "top_share_10pct"]
    rows = []
    for unit in UNITS:
        r = concentration(corpus, unit)
        rows.append({k: r[k] for k in cols})
    return pd.DataFrame(rows, columns=cols)
