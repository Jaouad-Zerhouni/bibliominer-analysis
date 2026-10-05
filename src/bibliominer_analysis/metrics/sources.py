"""Analysis of **journals**: impact, production over time.

The counterpart, for publication venues, of what `impact.py` does for
authors and organisations. Bradford (in `laws.py`) says *how many*
journals concentrate the field; here we say *which ones*, and with what
weight.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .impact import g_index, h_index, m_index


def _docs_with_citations(corpus) -> pd.DataFrame:
    d = corpus.documents[["eid", "source", "year"]].copy()
    d["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                   errors="coerce").fillna(0).astype(int)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d[d["source"].notna() & (d["source"].map(str).str.strip() != "")]
    return d


def sources_impact(corpus, n: Optional[int] = 20,
                   min_documents: int = 1) -> pd.DataFrame:
    """Ranking of journals with h, g, m, citations and first year.

    Columns: ``source``, ``documents``, ``citations``, ``h_index``,
    ``g_index``, ``m_index``, ``first_year``, ``last_year``.
    """
    d = _docs_with_citations(corpus)
    empty = pd.DataFrame(columns=["source", "documents", "citations", "h_index",
                                  "g_index", "m_index", "first_year", "last_year"])
    if d.empty:
        return empty

    corpus_last = d["year"].max()
    rows = []
    for source, g in d.groupby("source", sort=False):
        cites = g["citations"].tolist()
        years = g["year"].dropna()
        first = int(years.min()) if not years.empty else None
        h = h_index(cites)
        rows.append({
            "source": source,
            "documents": int(g["eid"].nunique()),
            "citations": int(sum(cites)),
            "h_index": h,
            "g_index": g_index(cites),
            "m_index": m_index(h, first, corpus_last),
            "first_year": first,
            "last_year": int(years.max()) if not years.empty else None,
        })

    out = pd.DataFrame(rows)
    out = out[out["documents"] >= min_documents]
    out = out.sort_values(["h_index", "citations", "documents"],
                          ascending=False, kind="stable").reset_index(drop=True)
    return out.head(n) if n else out


def top_sources_ranked(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """Journals by number of documents, the simplest ranking."""
    d = _docs_with_citations(corpus)
    if d.empty:
        return pd.DataFrame(columns=["source", "documents", "citations"])
    out = (d.groupby("source")
             .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
             .reset_index()
             .sort_values(["documents", "citations"], ascending=False, kind="stable")
             .reset_index(drop=True))
    return out.head(n) if n else out


def sources_over_time(corpus, n: int = 8, cumulative: bool = True) -> pd.DataFrame:
    """Yearly production of the top `n` journals.

    Columns: ``year``, ``source``, ``documents``, ``cumulative``.

    Every journal is present for **every** year of the range, at zero if
    needed: an interrupted series would read as missing data, and the
    cumulative curve would be wrong.
    """
    d = _docs_with_citations(corpus).dropna(subset=["year"])
    if d.empty:
        return pd.DataFrame(columns=["year", "source", "documents", "cumulative"])

    top = (d.groupby("source")["eid"].nunique()
             .sort_values(ascending=False, kind="stable").head(n).index)
    d = d[d["source"].isin(top)]
    counts = (d.groupby(["source", "year"])["eid"].nunique()
                .rename("documents").reset_index())

    years = np.arange(int(d["year"].min()), int(d["year"].max()) + 1)
    grid = pd.MultiIndex.from_product([top, years], names=["source", "year"])
    counts = (counts.set_index(["source", "year"]).reindex(grid, fill_value=0)
                    .reset_index())
    counts["cumulative"] = counts.groupby("source")["documents"].cumsum()
    counts["year"] = counts["year"].astype(int)
    order = {s: i for i, s in enumerate(top)}
    counts = counts.sort_values(
        ["source", "year"], key=lambda s: s.map(order) if s.name == "source" else s, kind="stable")
    return counts.reset_index(drop=True)
