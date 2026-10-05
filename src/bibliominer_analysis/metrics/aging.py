"""Ageing of the literature: Price index, age of references.

Two corpora of the same size and impact can relate to time in opposite
ways. A field that mostly cites work less than five years old renews
itself quickly; a field that cites founders from 1970 is settled. No
production or citation indicator shows this.

  - **Price index** (1970): share of references **less than five years**
    old at the time of publication. Above ~50 %, one speaks of a research
    front; below ~30 %, of an archival field.
  - **Median age of references**: half of the references are older.
  - **Cited half-life**: the age below which half of the references lie.
    On an age distribution it is the median; the name is a bibliometric
    convention.

Age is computed document by document (citing year minus reference year),
never globally: a corpus spanning ten years would otherwise mix
incompatible time perspectives.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

#: Price threshold, in years.
PRICE_WINDOW = 5

#: Beyond this, the reference year is considered wrong.
MAX_AGE = 200


def _reference_ages(corpus) -> pd.DataFrame:
    """One row per dated reference: ``eid``, ``year``, ``ref_year``, ``age``."""
    refs = corpus.references
    if refs.empty or "ref_year" not in refs.columns:
        return pd.DataFrame(columns=["eid", "year", "ref_year", "age"])

    r = refs[["eid", "ref_year"]].copy()
    r["ref_year"] = pd.to_numeric(r["ref_year"], errors="coerce")
    docs = corpus.documents[["eid"]].copy()
    docs["year"] = pd.to_numeric(corpus.documents["year"], errors="coerce")
    r = r.merge(docs, on="eid", how="left").dropna(subset=["ref_year", "year"])
    if r.empty:
        return pd.DataFrame(columns=["eid", "year", "ref_year", "age"])

    r["age"] = r["year"] - r["ref_year"]
    # A negative age is possible and legitimate (an "in press" reference that
    # appeared the following year); it is brought back to zero. An absurd age
    # comes from a misread year and must go.
    r = r[(r["age"] >= -2) & (r["age"] <= MAX_AGE)]
    r["age"] = r["age"].clip(lower=0)
    return r


def price_index(corpus, window: int = PRICE_WINDOW) -> Dict[str, Any]:
    """Price index of the corpus, and the age measures that go with it.

    Returns: ``price_index`` (%), ``references``, ``median_age``,
    ``mean_age``, ``half_life``, ``documents_with_references``.

    ``half_life`` is the median age, named after the bibliometric convention:
    half of the cited references are younger than this value.
    """
    r = _reference_ages(corpus)
    if r.empty:
        return {"price_index": None, "references": 0, "median_age": None,
                "mean_age": None, "half_life": None,
                "documents_with_references": 0, "window": window}

    ages = r["age"].to_numpy(dtype=float)
    recent = int((ages < window).sum())
    return {
        "price_index": round(100.0 * recent / ages.size, 1),
        "references": int(ages.size),
        "median_age": round(float(np.median(ages)), 1),
        "mean_age": round(float(ages.mean()), 1),
        "half_life": round(float(np.median(ages)), 1),
        "documents_with_references": int(r["eid"].nunique()),
        "window": window,
    }


def price_index_by_year(corpus, window: int = PRICE_WINDOW) -> pd.DataFrame:
    """Price index year by year.

    Columns: ``year``, ``documents``, ``references``, ``price_index``,
    ``median_age``.

    This is the useful reading: a rising index signals a field that is
    speeding up, a falling one a field that is settling.
    """
    cols = ["year", "documents", "references", "price_index", "median_age"]
    r = _reference_ages(corpus)
    if r.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for year, g in r.groupby("year", sort=True):
        ages = g["age"].to_numpy(dtype=float)
        rows.append({
            "year": int(year),
            "documents": int(g["eid"].nunique()),
            "references": int(ages.size),
            "price_index": round(100.0 * float((ages < window).sum()) / ages.size, 1),
            "median_age": round(float(np.median(ages)), 1),
        })
    return pd.DataFrame(rows, columns=cols)


def reference_age_distribution(corpus, max_age: int = 40) -> pd.DataFrame:
    """Distribution of reference ages.

    Columns: ``age``, ``references``, ``share``, ``cumulative_share``.

    The cumulative share is what is read to locate the half-life: the age at
    which it crosses 50 %.
    """
    cols = ["age", "references", "share", "cumulative_share"]
    r = _reference_ages(corpus)
    if r.empty:
        return pd.DataFrame(columns=cols)

    ages = r["age"].astype(int)
    total = int(ages.size)
    counts = ages.value_counts().sort_index()
    grid = np.arange(0, int(min(max_age, counts.index.max())) + 1)
    counts = counts.reindex(grid, fill_value=0)

    out = pd.DataFrame({
        "age": grid.astype(int),
        "references": counts.to_numpy().astype(int),
    })
    out["share"] = (100.0 * out["references"] / total).round(2)
    # The cumulative share is computed on the REAL total, not on the displayed
    # part: otherwise it would reach 100 % at the cut-off, hiding the tail.
    out["cumulative_share"] = out["share"].cumsum().round(2)
    return out
