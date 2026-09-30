"""Vieillissement de la littérature : indice de Price, âge des références.

Deux corpus de même taille et de même impact peuvent avoir des rapports au
temps opposés. Un domaine qui cite surtout des travaux de moins de cinq ans se
renouvelle vite ; un domaine qui cite des fondateurs de 1970 est stabilisé.
Aucun indicateur de production ou de citation ne montre cela.

  - **Indice de Price** (1970) : part des références âgées de **moins de cinq
    ans** au moment de la publication. Au-dessus de ~50 %, on parle d'un front
    de recherche ; en dessous de ~30 %, d'un champ d'archive.
  - **Âge médian des références** : la moitié des références sont plus vieilles.
  - **Demi-vie citée** : l'âge en deçà duquel se trouve la moitié des
    références. Sur une distribution d'âges, c'est la médiane, on la nomme
    ainsi par convention bibliométrique.

L'âge se calcule document par document (année du citant moins année de la
référence), jamais globalement : un corpus qui s'étale sur dix ans mélangerait
sinon des points de vue temporels incompatibles.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

#: Seuil de Price, en années.
PRICE_WINDOW = 5

#: Au-delà, on considère l'année de référence comme erronée.
MAX_AGE = 200


def _reference_ages(corpus) -> pd.DataFrame:
    """Une ligne par référence datée : ``eid``, ``year``, ``ref_year``, ``age``."""
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
    # Un âge négatif est possible et légitime (référence « in press » parue
    # l'année suivante) ; on le ramène à zéro. Un âge délirant vient d'une
    # année mal lue et doit sortir.
    r = r[(r["age"] >= -2) & (r["age"] <= MAX_AGE)]
    r["age"] = r["age"].clip(lower=0)
    return r


def price_index(corpus, window: int = PRICE_WINDOW) -> Dict[str, Any]:
    """Indice de Price du corpus, et les mesures d'âge qui l'accompagnent.

    Retour : ``price_index`` (%), ``references``, ``median_age``,
    ``mean_age``, ``half_life``, ``documents_with_references``.

    ``half_life`` est l'âge médian, nommé selon la convention bibliométrique :
    la moitié des références citées sont plus jeunes que cette valeur.
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
    """Indice de Price année par année.

    Colonnes : ``year``, ``documents``, ``references``, ``price_index``,
    ``median_age``.

    C'est la lecture utile : un indice qui monte signale un domaine qui
    s'accélère, un indice qui baisse un domaine qui se sédimente.
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
    """Distribution des âges de référence.

    Colonnes : ``age``, ``references``, ``share``, ``cumulative_share``.

    La part cumulée est ce qu'on lit pour situer la demi-vie : l'âge où elle
    franchit 50 %.
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
    # Le cumul se calcule sur le total RÉEL, pas sur la partie affichée :
    # sinon il atteindrait 100 % à la borne, en cachant la queue.
    out["cumulative_share"] = out["share"].cumsum().round(2)
    return out
