"""Spectroscopie des années de référence (RPYS).

Marx & al. (2014). On ne compte pas les publications du corpus mais les
**années de publication des travaux qu'il cite**. La courbe brute est
inintéressante — elle croît toujours, parce qu'il y a mécaniquement plus de
littérature récente. Ce qui compte est l'**écart à la médiane glissante** :

    écart(t) = citations(t) − médiane(citations sur t−2 … t+2)

Un pic positif signale une année où le corpus cite bien plus que la tendance :
presque toujours un travail fondateur publié cette année-là. C'est la seule
méthode qui fait remonter les racines historiques d'un domaine sans qu'on ait
à les connaître d'avance.

On prend la **médiane** et non la moyenne : un unique pic écraserait sa propre
référence si on moyennait, et s'effacerait donc lui-même.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

#: Demi-largeur de la fenêtre glissante (5 ans au total : t−2 … t+2).
HALF_WINDOW = 2


def reference_spectroscopy(corpus, year_min: Optional[int] = None,
                           year_max: Optional[int] = None) -> pd.DataFrame:
    """Distribution des années de référence et écart à la médiane glissante.

    Colonnes : ``year``, ``references``, ``median_5``, ``deviation``,
    ``is_peak``, ``top_reference``.

    ``is_peak`` marque les années dont l'écart dépasse la médiane des écarts
    positifs — un repère de lecture, pas un test statistique.
    """
    cols = ["year", "references", "median_5", "deviation", "is_peak", "top_reference"]
    refs = corpus.references
    if refs.empty or "ref_year" not in refs.columns:
        return pd.DataFrame(columns=cols)

    r = refs.copy()
    r["ref_year"] = pd.to_numeric(r["ref_year"], errors="coerce")
    r = r.dropna(subset=["ref_year"])
    r["ref_year"] = r["ref_year"].astype(int)
    # Des années aberrantes existent dans les exports (0, 2098). On borne au
    # raisonnable plutôt que de laisser un point isolé aplatir tout le reste.
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

    # Comparaison LARGE, et non stricte : un corpus qui n'a qu'un seul pic voit
    # la médiane des écarts positifs valoir exactement ce pic. Avec « > » il ne
    # serait jamais signalé — c'est-à-dire précisément dans le cas où le repère
    # est le plus utile.
    positive = out.loc[out["deviation"] > 0, "deviation"]
    threshold = float(positive.median()) if not positive.empty else 0.0
    out["is_peak"] = (out["deviation"] > 0) & (out["deviation"] >= threshold)

    # Pour chaque année de pic, la référence la plus citée de cette année-là :
    # c'est elle qu'on veut nommer quand on commente le graphique.
    top = _top_reference_per_year(r)
    out["top_reference"] = out["year"].map(top)
    out.loc[~out["is_peak"], "top_reference"] = None
    return out.reset_index(drop=True)


def _top_reference_per_year(refs: pd.DataFrame) -> dict:
    """Référence la plus fréquemment citée, année par année."""
    r = refs.copy()
    key = r["ref_doi"].astype(str).str.strip().str.lower()
    fallback = r["ref_title"].astype(str).str.strip().str.lower()
    r["key"] = np.where(key.isin(["", "nan", "none"]), fallback, key)
    r = r[r["key"].notna() & (r["key"] != "") & (r["key"] != "nan")]
    if r.empty:
        return {}

    # `first()` saute les valeurs manquantes : on transforme d'abord les
    # chaînes vides en manquantes, et on obtient la première valeur non vide
    # sans fonction Python par groupe. La version à `lambda` en appelait une
    # par couple (année, référence) — 9 s sur 10 000 références.
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
    counts = counts.sort_values(["ref_year", "n"], ascending=[True, False])
    best = counts.drop_duplicates("ref_year")

    out = {}
    for row in best.itertuples():
        label = row.label or "?"
        if row.authors:
            label = f"{row.authors} — {label}"
        out[int(row.ref_year)] = f"{label[:110]} ({row.n}×)"
    return out
