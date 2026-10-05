"""Timelines: authors, terms and mean citations year by year.

A ranking says *who* dominates; a timeline says *when*. The two often
contradict each other: a highly ranked author may have published nothing
for six years, and that gap is precisely what is informative.
"""

from __future__ import annotations


import numpy as np
import pandas as pd


def authors_over_time(corpus, n: int = 12) -> pd.DataFrame:
    """Yearly production of the top `n` authors.

    Columns: ``author``, ``year``, ``documents``, ``citations``,
    ``citations_per_year``, ``cumulative``.

    ``citations_per_year`` relates citations to the age of the document:
    without it, the early years always look better, since they have had more
    time to accumulate.

    Only the years in which the author actually published are returned: here
    the absence of a point IS the information (an interruption), unlike a
    cumulative series.
    """
    cols = ["author", "year", "documents", "citations", "citations_per_year",
            "cumulative"]
    a = corpus.authors[["eid", "name"]].dropna(subset=["name"])
    a = a[a["name"].map(str).str.strip() != ""].drop_duplicates(["eid", "name"])
    if a.empty:
        return pd.DataFrame(columns=cols)

    d = corpus.documents[["eid", "year"]].copy()
    d["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                   errors="coerce").fillna(0).astype(int)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    a = a.merge(d, on="eid", how="left").dropna(subset=["year"])
    if a.empty:
        return pd.DataFrame(columns=cols)

    top = (a.groupby("name")["eid"].nunique()
             .sort_values(ascending=False, kind="stable").head(n).index)
    a = a[a["name"].isin(top)]

    last_year = d["year"].max()
    out = (a.groupby(["name", "year"])
             .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
             .reset_index().rename(columns={"name": "author"}))
    age = (last_year - out["year"] + 1).clip(lower=1)
    out["citations_per_year"] = (out["citations"] / age).round(2)
    out = out.sort_values(["author", "year"], kind="stable")
    out["cumulative"] = out.groupby("author")["documents"].cumsum()
    out["year"] = out["year"].astype(int)

    order = {name: i for i, name in enumerate(top)}
    out = out.sort_values(["author", "year"],
                          key=lambda s: s.map(order) if s.name == "author" else s, kind="stable")
    return out[cols].reset_index(drop=True)


def word_dynamics(corpus, n: int = 10, kind: str = "author",
                  cumulative: bool = True) -> pd.DataFrame:
    """Occurrences of the top `n` terms, year by year.

    Columns: ``year``, ``keyword``, ``occurrences``, ``cumulative``.

    This is the reading that distinguishes an **emerging** term from an
    **established** one: two keywords with the same total can have opposite
    trajectories. Every year of the range is present, at zero if needed,
    otherwise the cumulative count would be wrong.
    """
    cols = ["year", "keyword", "occurrences", "cumulative"]
    k = corpus.keywords
    if kind in ("author", "index"):
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna() & (k["keyword"].map(str).str.strip() != "")]
    if k.empty:
        return pd.DataFrame(columns=cols)

    years = corpus.documents[["eid", "year"]].copy()
    years["year"] = pd.to_numeric(years["year"], errors="coerce")
    k = k.merge(years, on="eid", how="left").dropna(subset=["year"])
    k = k.drop_duplicates(["eid", "keyword"])
    if k.empty:
        return pd.DataFrame(columns=cols)

    top = (k.groupby("keyword")["eid"].nunique()
             .sort_values(ascending=False, kind="stable").head(n).index)
    k = k[k["keyword"].isin(top)]

    counts = (k.groupby(["keyword", "year"])["eid"].nunique()
                .rename("occurrences").reset_index())
    grid_years = np.arange(int(k["year"].min()), int(k["year"].max()) + 1)
    grid = pd.MultiIndex.from_product([top, grid_years], names=["keyword", "year"])
    counts = (counts.set_index(["keyword", "year"]).reindex(grid, fill_value=0)
                    .reset_index())
    counts["cumulative"] = counts.groupby("keyword")["occurrences"].cumsum()
    counts["year"] = counts["year"].astype(int)

    order = {w: i for i, w in enumerate(top)}
    counts = counts.sort_values(
        ["keyword", "year"],
        key=lambda s: s.map(order) if s.name == "keyword" else s, kind="stable")
    return counts[cols].reset_index(drop=True)


def average_citations_per_year(corpus) -> pd.DataFrame:
    """Mean citations per article, by publication year.

    Columns: ``year``, ``documents``, ``mean_citations``,
    ``mean_citations_per_year``, ``citable_years``.

    The second column is the first divided by the age. It is the one to read:
    the first always decreases towards recent years, and that is not a
    weakening of the corpus, only missing time.
    """
    d = corpus.documents[["eid", "year"]].copy()
    d["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                   errors="coerce").fillna(0).astype(int)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d.dropna(subset=["year"])
    if d.empty:
        return pd.DataFrame(columns=["year", "documents", "mean_citations",
                                     "mean_citations_per_year", "citable_years"])

    last_year = int(d["year"].max())
    out = (d.groupby("year")
             .agg(documents=("eid", "nunique"), mean_citations=("citations", "mean"))
             .reset_index())
    out["year"] = out["year"].astype(int)
    out["citable_years"] = (last_year - out["year"] + 1).clip(lower=1)
    out["mean_citations_per_year"] = (out["mean_citations"]
                                      / out["citable_years"]).round(2)
    out["mean_citations"] = out["mean_citations"].round(2)
    return out.sort_values("year", kind="stable").reset_index(drop=True)
