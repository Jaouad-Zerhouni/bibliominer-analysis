"""Indicateurs de croissance et taille d'échantillon.

Les formules sont celles de la littérature, écrites explicitement pour qu'on
puisse les vérifier :

  - **CAGR** — croissance annuelle composée
        ``((N_fin / N_début)^(1/n) − 1) × 100``   (n = nombre d'intervalles)

  - **AGR** — croissance annuelle simple, d'une année sur l'autre
        ``(N_t − N_{t−1}) / N_{t−1} × 100``

  - **RGR** — taux de croissance relatif (Mahapatra, 1985)
        ``R = (ln W₂ − ln W₁) / (T₂ − T₁)``
    où W est le nombre **cumulé** de publications. Le RGR décroît
    mécaniquement avec le temps : un corpus mûr croît moins vite en relatif.

  - **Temps de doublement** — durée nécessaire pour doubler le stock
        ``Dt = ln(2) / RGR = 0,693 / RGR``

  - **Cochran** — taille d'échantillon représentatif
        ``n₀ = Z²·p·q / e²``  puis correction pour population finie
        ``n = n₀ / (1 + (n₀ − 1)/N)``

CAGR et AGR répondent à deux questions différentes : le premier lisse toute la
période, le second montre les à-coups. Les donner ensemble évite de conclure à
une croissance régulière là où il n'y a qu'une bonne année.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _docs_per_year(corpus) -> pd.Series:
    """Documents par année, sans trou dans la série."""
    years = pd.to_numeric(corpus.documents["year"], errors="coerce").dropna()
    if years.empty:
        return pd.Series(dtype=int)
    counts = years.astype(int).value_counts().sort_index()
    full = range(int(counts.index.min()), int(counts.index.max()) + 1)
    return counts.reindex(full, fill_value=0)


def cagr(corpus) -> Optional[float]:
    """Croissance annuelle composée, en pourcentage.

    Renvoie ``None`` si le corpus couvre moins de deux ans ou si la première
    année est vide — le taux serait alors infini, ce qui n'a aucun sens.
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
    """Croissance annuelle simple, année par année.

    La première année n'a pas de taux (pas d'année précédente), et une année
    qui suit une année vide non plus — diviser par zéro donnerait un infini
    qu'on préfère laisser vide plutôt que d'afficher un nombre faux.
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
    """RGR et temps de doublement, année par année.

    Colonnes : ``year``, ``documents``, ``cumulative``, ``ln_cumulative``,
    ``rgr``, ``doubling_time``.

    Le RGR d'une année se calcule entre le cumul de l'année précédente et
    celui de l'année courante. La première année n'en a donc pas.
    """
    per_year = _docs_per_year(corpus)
    empty = pd.DataFrame(columns=["year", "documents", "cumulative",
                                  "ln_cumulative", "rgr", "doubling_time"])
    if per_year.empty:
        return empty

    df = pd.DataFrame({"year": per_year.index.astype(int),
                       "documents": per_year.to_numpy(dtype=int)})
    df["cumulative"] = df["documents"].cumsum()
    # ln(0) n'existe pas : une année initiale vide reste sans valeur.
    df["ln_cumulative"] = np.where(df["cumulative"] > 0,
                                   np.log(df["cumulative"].replace(0, np.nan)),
                                   np.nan)
    prev_ln = df["ln_cumulative"].shift(1)
    # Le temps de doublement se calcule sur le RGR EXACT. Arrondi d'abord à
    # 4 décimales, un RGR de 0,00004 devenait 0 et le temps de doublement
    # disparaissait ; un RGR de 0,00015 arrondi à 0,0001 donnait 6 931 ans au
    # lieu de 4 621.
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
    """Taille d'échantillon représentatif du corpus (Cochran, 1977).

    Combien de documents faut-il lire pour que les conclusions valent pour
    tout le corpus ? ``proportion = 0,5`` est le choix le plus prudent : c'est
    la valeur qui maximise la variance, donc la taille requise.

    Les niveaux de confiance courants sont tabulés ; en dehors, on approche le
    quantile de la loi normale sans dépendre de SciPy.
    """
    n_pop = int(len(corpus.documents))
    z = _z_score(confidence)
    p = float(proportion)
    q = 1.0 - p

    n0 = (z ** 2) * p * q / (margin ** 2)
    # Correction pour population finie : sans elle, on demanderait parfois
    # plus de documents que le corpus n'en contient.
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


#: Quantiles usuels de la loi normale centrée réduite (bilatéral).
_Z = {0.80: 1.2816, 0.85: 1.4395, 0.90: 1.6449, 0.95: 1.9600,
      0.98: 2.3263, 0.99: 2.5758}


def _z_score(confidence: float) -> float:
    confidence = float(confidence)
    # Table seulement pour une valeur EXACTEMENT tabulée. Arrondir d'abord
    # envoyait 0,975 sur 0,97 (l'arrondi flottant de Python), et donnait
    # z = 2,1705 au lieu de 2,2414.
    for key, z in _Z.items():
        if abs(confidence - key) < 1e-9:
            return z
    # Approximation rationnelle d'Abramowitz & Stegun (26.2.23), erreur
    # < 4,5·10⁻⁴, suffisante ici et sans SciPy.
    p = 1 - (1 - confidence) / 2
    t = math.sqrt(-2.0 * math.log(1 - p)) if p > 0.5 else math.sqrt(-2.0 * math.log(p))
    num = 2.515517 + 0.802853 * t + 0.010328 * t * t
    den = 1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3
    z = t - num / den
    return abs(z)


def trend_forecast(corpus, horizon: int = 5,
                   model: str = "linear") -> Dict[str, Any]:
    """Analyse de tendance par moindres carrés, et projection.

    Deux modèles :

      - **linéaire** : ``N = a + b·t`` — ajusté directement sur les effectifs.
        ``b`` se lit comme « publications supplémentaires par an ».
      - **exponentiel** : ``ln N = a + b·t``, soit ``N = e^a · e^(b·t)``.
        Ajusté sur les logarithmes, il convient à une croissance qui
        s'accélère. Les années vides en sont exclues (ln 0 n'existe pas).

    Le **R² est calculé sur l'échelle d'origine dans les deux cas**, sinon on
    comparerait un R² de logarithmes à un R² d'effectifs — et l'exponentiel
    paraîtrait toujours meilleur.

    La dernière année est souvent **incomplète** (indexation en cours) : elle
    tire la tendance vers le bas. `last_year_partial` la signale, mais on ne
    l'écarte pas d'office — c'est à l'utilisateur de trancher.

    Renvoie ``{"table": DataFrame, "fit": {...}}`` où la table porte les années
    observées **et** projetées, avec ``kind`` valant « observed » ou « forecast ».
    """
    per_year = _docs_per_year(corpus)
    empty = pd.DataFrame(columns=["year", "documents", "fitted", "kind"])
    if len(per_year) < 3:
        # Deux points donnent toujours un R² de 1 : une tendance n'a de sens
        # qu'à partir de trois observations.
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
    # Erreur type de l'estimation : 2 paramètres consommés (pente, ordonnée).
    dof = max(len(counts) - 2, 1)
    rmse = float(np.sqrt(ss_res / dof))

    rows = [{"year": int(y), "documents": int(n), "fitted": round(float(f), 2),
             "kind": "observed"}
            for y, n, f in zip(years, counts, fitted)]

    last = int(years[-1])
    for k in range(1, horizon + 1):
        x = float(last + k - years[0])
        # Un effectif prédit négatif n'a pas de sens : on plancher à zéro.
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
            # Vrai quand la dernière année est l'année en cours (ou future) :
            # son indexation n'est pas terminée. Valait toujours None, alors
            # que la docstring promettait de le signaler.
            "last_year_partial": bool(last >= date.today().year),
        },
    }


def growth_summary(corpus) -> Dict[str, Any]:
    """Résumé des indicateurs de croissance, pour les tuiles de l'interface."""
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
