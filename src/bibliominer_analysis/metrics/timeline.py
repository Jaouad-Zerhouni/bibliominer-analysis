"""Chronologies : auteurs, termes et citations moyennes année par année.

Un classement dit *qui* domine ; une chronologie dit *quand*. Les deux se
contredisent souvent — un auteur très bien classé peut n'avoir rien publié
depuis six ans — et c'est précisément l'écart qui est informatif.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def authors_over_time(corpus, n: int = 12) -> pd.DataFrame:
    """Production annuelle des `n` auteurs principaux.

    Colonnes : ``author``, ``year``, ``documents``, ``citations``,
    ``citations_per_year``, ``cumulative``.

    ``citations_per_year`` rapporte les citations à l'ancienneté du document :
    sans cela, les premières années paraissent toujours meilleures, puisqu'elles
    ont eu plus de temps pour accumuler.

    Seules les années où l'auteur a effectivement publié sont renvoyées : ici
    l'absence de point EST l'information (une interruption), contrairement à une
    série cumulée.
    """
    cols = ["author", "year", "documents", "citations", "citations_per_year",
            "cumulative"]
    a = corpus.authors[["eid", "name"]].dropna(subset=["name"])
    a = a[a["name"].astype(str).str.strip() != ""].drop_duplicates(["eid", "name"])
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
             .sort_values(ascending=False).head(n).index)
    a = a[a["name"].isin(top)]

    last_year = d["year"].max()
    out = (a.groupby(["name", "year"])
             .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
             .reset_index().rename(columns={"name": "author"}))
    age = (last_year - out["year"] + 1).clip(lower=1)
    out["citations_per_year"] = (out["citations"] / age).round(2)
    out = out.sort_values(["author", "year"])
    out["cumulative"] = out.groupby("author")["documents"].cumsum()
    out["year"] = out["year"].astype(int)

    order = {name: i for i, name in enumerate(top)}
    out = out.sort_values(["author", "year"],
                          key=lambda s: s.map(order) if s.name == "author" else s)
    return out[cols].reset_index(drop=True)


def word_dynamics(corpus, n: int = 10, kind: str = "author",
                  cumulative: bool = True) -> pd.DataFrame:
    """Occurrences des `n` termes principaux, année par année.

    Colonnes : ``year``, ``keyword``, ``occurrences``, ``cumulative``.

    C'est la lecture qui distingue un terme **émergent** d'un terme
    **installé** : deux mots-clés de même total peuvent avoir des trajectoires
    opposées. Toutes les années de l'intervalle sont présentes, à zéro si
    besoin, sinon le cumul serait faux.
    """
    cols = ["year", "keyword", "occurrences", "cumulative"]
    k = corpus.keywords
    if kind in ("author", "index"):
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna() & (k["keyword"].astype(str).str.strip() != "")]
    if k.empty:
        return pd.DataFrame(columns=cols)

    years = corpus.documents[["eid", "year"]].copy()
    years["year"] = pd.to_numeric(years["year"], errors="coerce")
    k = k.merge(years, on="eid", how="left").dropna(subset=["year"])
    k = k.drop_duplicates(["eid", "keyword"])
    if k.empty:
        return pd.DataFrame(columns=cols)

    top = (k.groupby("keyword")["eid"].nunique()
             .sort_values(ascending=False).head(n).index)
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
        key=lambda s: s.map(order) if s.name == "keyword" else s)
    return counts[cols].reset_index(drop=True)


def average_citations_per_year(corpus) -> pd.DataFrame:
    """Citations moyennes par article, selon l'année de publication.

    Colonnes : ``year``, ``documents``, ``mean_citations``,
    ``mean_citations_per_year``, ``citable_years``.

    La seconde colonne est la première divisée par l'ancienneté. C'est celle
    qu'il faut lire : la première décroît toujours vers les années récentes, et
    ce n'est pas un affaiblissement du corpus, seulement le temps qui manque.
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
    return out.sort_values("year").reset_index(drop=True)
