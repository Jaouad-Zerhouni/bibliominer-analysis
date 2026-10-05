"""Reference Publication Year Spectroscopy (RPYS).

Marx et al. (2014). What is counted is not the publications of the corpus
but the **publication years of the works it cites**. The raw curve is of
little interest: it always grows, because there is mechanically more
recent literature. What matters is the **deviation from the rolling
median**:

    deviation(t) = citations(t) - median(citations over t-2 ... t+2)

A positive peak signals a year that the corpus cites far more than the
trend: almost always a founding work published that year. It is the only
method that brings up the historical roots of a field without having to
know them in advance.

The **median** is used, not the mean: a single peak would crush its own
reference if it were averaged, and would therefore erase itself.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

#: Half-width of the rolling window (5 years in total: t-2 ... t+2).
HALF_WINDOW = 2


def reference_spectroscopy(corpus, year_min: Optional[int] = None,
                           year_max: Optional[int] = None) -> pd.DataFrame:
    """Distribution of reference years and deviation from the rolling median.

    Columns: ``year``, ``references``, ``median_5``, ``deviation``,
    ``is_peak``, ``top_reference``.

    ``is_peak`` marks the years whose deviation exceeds the median of the
    positive deviations: a reading aid, not a statistical test.
    """
    cols = ["year", "references", "median_5", "deviation", "is_peak", "top_reference"]
    refs = corpus.references
    if refs.empty or "ref_year" not in refs.columns:
        return pd.DataFrame(columns=cols)

    r = refs.copy()
    r["ref_year"] = pd.to_numeric(r["ref_year"], errors="coerce")
    r = r.dropna(subset=["ref_year"])
    r["ref_year"] = r["ref_year"].astype(int)
    # Absurd years exist in exports (0, 2098). They are bounded to a reasonable
    # range rather than letting an isolated point flatten everything else.
    last = int(pd.to_numeric(corpus.documents["year"],
                             errors="coerce").max() or r["ref_year"].max())
    r = r[(r["ref_year"] >= 1800) & (r["ref_year"] <= last)]
    if year_min is not None:
        r = r[r["ref_year"] >= int(year_min)]
    if year_max is not None:
        r = r[r["ref_year"] <= int(year_max)]
    if r.empty:
        return pd.DataFrame(columns=cols)

    counts = r.groupby("ref_year").size().rename("references")
    years = np.arange(int(counts.index.min()), int(counts.index.max()) + 1)
    counts = counts.reindex(years, fill_value=0)

    values = counts.to_numpy(dtype=float)
    medians = np.empty_like(values)
    for i in range(values.size):
        lo = max(0, i - HALF_WINDOW)
        hi = min(values.size, i + HALF_WINDOW + 1)
        medians[i] = np.median(values[lo:hi])

    out = pd.DataFrame({
        "year": years.astype(int),
        "references": counts.to_numpy().astype(int),
        "median_5": np.round(medians, 1),
        "deviation": np.round(values - medians, 1),
    })

    # NON-strict comparison: a corpus with a single peak sees the median of the
    # positive deviations equal exactly that peak. With ">" it would never be
    # flagged, precisely in the case where the marker is most useful.
    positive = out.loc[out["deviation"] > 0, "deviation"]
    threshold = float(positive.median()) if not positive.empty else 0.0
    out["is_peak"] = (out["deviation"] > 0) & (out["deviation"] >= threshold)

    # For each peak year, the most cited reference of that year: it is the one
    # to name when commenting on the chart.
    top = _top_reference_per_year(r)
    out["top_reference"] = out["year"].map(top)
    out.loc[~out["is_peak"], "top_reference"] = None
    return out.reset_index(drop=True)


def _top_reference_per_year(refs: pd.DataFrame) -> dict:
    """Most frequently cited reference, year by year."""
    r = refs.copy()
    key = r["ref_doi"].map(str).str.strip().str.lower()
    fallback = r["ref_title"].map(str).str.strip().str.lower()
    r["key"] = np.where(key.isin(["", "nan", "none"]), fallback, key)
    r = r[r["key"].notna() & (r["key"] != "") & (r["key"] != "nan")]
    if r.empty:
        return {}

    # `first()` skips missing values: empty strings are turned into missing
    # values first, which gives the first non-empty value without a Python
    # function per group. The `lambda` version called one per (year, reference)
    # pair, 9 s on 10,000 references.
    for column in ("ref_title", "ref_authors"):
        text = r[column].astype("string").str.strip()
        r[column] = text.mask(text == "")
    grouped = r.groupby(["ref_year", "key"])
    counts = pd.DataFrame({
        "n": grouped["eid"].nunique(),
        "label": grouped["ref_title"].first(),
        "authors": grouped["ref_authors"].first(),
    }).reset_index()
    counts["label"] = counts["label"].astype(object).where(counts["label"].notna(), None)
    counts["authors"] = counts["authors"].astype(object).where(counts["authors"].notna(), None)
    counts = counts.sort_values(["ref_year", "n"], ascending=[True, False], kind="stable")
    best = counts.drop_duplicates("ref_year")

    out = {}
    for row in best.itertuples():
        label = row.label or "?"
        if row.authors:
            label = f"{row.authors}: {label}"
        out[int(row.ref_year)] = f"{label[:110]} ({row.n}×)"
    return out
