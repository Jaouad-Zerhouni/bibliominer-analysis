"""Concentration de la production : Gini, courbe de Lorenz, ratios.

Un corpus de cent articles peut être l'œuvre de cent personnes ayant publié une
fois chacune, ou de trois personnes en ayant écrit trente. Les classements ne
distinguent pas ces deux situations : ils montrent le haut du tableau, jamais la
forme de la distribution.

  - **Gini**, 0 = tout le monde produit autant, 1 = une seule entité produit
    tout. Sur des auteurs, un Gini au-dessus de ~0,6 signale un domaine porté
    par quelques personnes.
  - **Courbe de Lorenz**, la représentation dont le Gini n'est que le résumé.
    Elle montre *où* se situe l'inégalité, ce qu'un seul nombre ne dit pas.
  - **CR4 / CR10**, part des 4 et des 10 premiers. Plus lisible qu'un Gini
    pour un lecteur non spécialiste, et suffisant pour la plupart des propos.

Lotka (`laws.py`) répond à une question voisine mais différente : il teste si la
distribution suit une loi de puissance. Ici on ne teste rien, on mesure.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

#: Dimensions sur lesquelles la concentration a un sens.
UNITS = ("authors", "institutions", "countries", "sources", "cities")


def gini(values) -> Optional[float]:
    """Coefficient de Gini d'une série de valeurs positives.

    Formule directe sur la série triée. Les valeurs nulles comptent : un auteur
    à zéro document fait partie de la population, et l'exclure ferait baisser
    artificiellement l'inégalité mesurée.
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
    """Courbe de Lorenz : part cumulée de la population × part cumulée du total.

    Colonnes : ``population_share``, ``value_share``.

    La diagonale représente l'égalité parfaite ; l'aire entre la courbe et la
    diagonale est exactement ce que résume le Gini.
    """
    x = np.sort(np.asarray([v for v in values if v is not None and v >= 0],
                           dtype=float))
    if x.size == 0 or x.sum() <= 0:
        return pd.DataFrame({"population_share": [0.0, 100.0],
                             "value_share": [0.0, 100.0]})

    cum = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
    pop = np.linspace(0.0, 1.0, x.size + 1)
    # On échantillonne : au-delà de quelques dizaines de points la courbe ne
    # gagne rien en lisibilité et alourdit la réponse.
    if pop.size > points:
        keep = np.unique(np.linspace(0, pop.size - 1, points).astype(int))
        pop, cum = pop[keep], cum[keep]
    return pd.DataFrame({
        "population_share": np.round(100.0 * pop, 2),
        "value_share": np.round(100.0 * cum, 2),
    })


def _series(corpus, unit: str) -> pd.Series:
    """Documents par entité, pour l'unité demandée."""
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
    """Mesures de concentration pour une dimension.

    Retour : ``unit``, ``entities``, ``documents``, ``gini``, ``cr4``,
    ``cr10``, ``top_share_10pct``, ``lorenz`` (liste de points).

    ``top_share_10pct`` est la part produite par les 10 % les plus actifs,
    la formulation que comprend n'importe quel lecteur, contrairement au Gini.
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
    """Une ligne par dimension, la comparaison est plus parlante que le détail.

    Colonnes : ``unit``, ``entities``, ``documents``, ``gini``, ``cr4``,
    ``cr10``, ``top_share_10pct``.
    """
    cols = ["unit", "entities", "documents", "gini", "cr4", "cr10",
            "top_share_10pct"]
    rows = []
    for unit in UNITS:
        r = concentration(corpus, unit)
        rows.append({k: r[k] for k in cols})
    return pd.DataFrame(rows, columns=cols)
