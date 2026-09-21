"""Lois bibliométriques : Lotka, Bradford, Zipf.

Ces trois lois décrivent des concentrations très inégales, chacune sur un
objet différent :

  - **Lotka (1926)** — la productivité des AUTEURS. Le nombre d'auteurs ayant
    publié *x* articles décroît en 1/x^n. Beaucoup publient une fois, très peu
    publient beaucoup.

  - **Bradford (1934)** — la dispersion des SOURCES. En classant les revues par
    productivité décroissante et en formant trois zones contenant chacune un
    tiers des articles, le nombre de revues croît géométriquement d'une zone à
    l'autre. La première zone est le « noyau » du domaine.

  - **Zipf (1949)** — la fréquence des MOTS. La fréquence d'un terme est
    inversement proportionnelle à son rang, élevé à une puissance *s*.

Chaque fonction renvoie à la fois les données OBSERVÉES et l'ajustement
THÉORIQUE : sans les deux, on ne peut pas juger si la loi s'applique au
corpus. Un exposant seul ne veut rien dire.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Ajuste y = C · x^(-b) par moindres carrés sur les logarithmes.

    Renvoie l'exposant, la constante et le R² — ce dernier est indispensable :
    il dit si la loi DÉCRIT vraiment le corpus. Un exposant sans R² est un
    chiffre sans garantie.
    """
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return {"exponent": float("nan"), "constant": float("nan"), "r2": float("nan")}

    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)

    predicted = slope * lx + intercept
    ss_res = float(np.sum((ly - predicted) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"exponent": float(-slope), "constant": float(10 ** intercept),
            "r2": float(r2)}


def _zeta(a: float, terms: int = 2000) -> float:
    """ζ(a) = Σ 1/n^a — la constante qui fait de C/n^a une VRAIE distribution.

    Sans elle, « C » n'est qu'une ordonnée à l'origine en unités de comptage :
    elle ne peut pas se comparer à une part observée, qui est une probabilité.
    Diviser par ζ(a) garantit que les parts théoriques somment à 1 sur tous les
    entiers, exactement comme les parts observées.

    La somme est tronquée puis complétée par le terme d'Euler-Maclaurin : la
    queue d'une série en 1/n^a décroît lentement quand `a` est proche de 1, et
    l'ignorer surestimerait C de plusieurs pour cent.
    """
    if not np.isfinite(a) or a <= 1.0:
        return float("nan")
    n = np.arange(1, terms + 1, dtype=float)
    head = float(np.sum(n ** (-a)))
    tail = terms ** (1.0 - a) / (a - 1.0) - 0.5 * terms ** (-a)
    return head + tail


#: Termes de la série pour l'équation de vraisemblance. Près de a = 1 la queue
#: de Σ ln n · n^-a décroît très lentement : 4 000 termes y biaisaient
#: l'espérance, et donc l'exposant, de façon mesurable.
_MLE_TERMS = 200_000
_MLE_GRID = np.arange(1, _MLE_TERMS + 1, dtype=float)
_MLE_LOG_GRID = np.log(_MLE_GRID)


def _zeta_log_ratio(a: float) -> float:
    """Σ(ln n · n^-a) / ζ(a) — l'espérance de ln(x) sous la loi de puissance.

    C'est la quantité qu'il faut égaler à la moyenne observée de ln(x) pour
    résoudre l'équation de vraisemblance.
    """
    weights = _MLE_GRID ** (-a)
    return float(np.sum(_MLE_LOG_GRID * weights) / np.sum(weights))


def _lotka_mle(values: np.ndarray, lo: float = 1.01, hi: float = 6.0,
               iterations: int = 60) -> float:
    """Exposant de Lotka par maximum de vraisemblance (loi zêta discrète).

    **Pourquoi pas les moindres carrés sur log-log.** C'est la méthode qu'on
    trouve partout, et elle est mauvaise ici — Clauset, Shalizi & Newman (2009)
    l'ont montré. Elle donne le même poids à chaque NIVEAU de productivité :
    sur un corpus réel, la queue compte beaucoup de niveaux ne contenant qu'un
    seul auteur, et ces points isolés dominent la régression au point d'aplatir
    la pente. Mesuré sur le corpus d'essai : 0,755 par moindres carrés contre
    1,754 par vraisemblance. Le premier est inférieur à 1, donc la loi n'y a
    même pas de constante de normalisation — l'ajustement était inutilisable.

    Le maximum de vraisemblance pondère par AUTEUR, ce qui est la bonne unité
    d'observation. On résout par dichotomie l'équation E[ln x] = moyenne
    observée de ln x ; la fonction est monotone décroissante en `a`, la
    dichotomie converge donc sûrement.
    """
    v = np.asarray([x for x in values if x is not None and x >= 1], dtype=float)
    if v.size == 0:
        return float("nan")
    target = float(np.mean(np.log(v)))
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _zeta_log_ratio(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Lotka — productivité des auteurs
# ---------------------------------------------------------------------------

def lotka(corpus) -> Dict[str, object]:
    """Distribution du nombre d'auteurs par nombre de publications.

    Renvoie ``{"table": DataFrame, "fit": {...}}``.

    Une ligne se lit : « ``n_authors`` AUTEURS ont publié ``documents_written``
    documents CHACUN ». Ce n'est pas un décompte d'articles à auteur unique —
    la formulation inverse, à laquelle le tableau se prête, est le contresens
    classique sur cette loi.

    La table donne, pour chaque nombre de documents *x*, trois colonnes :

      - ``share_observed`` — la part réelle des auteurs ;
      - ``share_fitted``   — la loi AJUSTÉE ``y = C/x^a``, l'exposant estimé
        par maximum de vraisemblance et ``C = 1/ζ(a)`` ;
      - ``share_lotka``    — Lotka STRICT, exposant 2, soit ``(1/x²)/ζ(2)``.

    Les deux dernières répondent à des questions distinctes : la production
    suit-elle *une* loi de puissance, et suit-elle *celle de Lotka* ? La
    seconde est bien plus exigeante, et la confondre avec la première fait
    conclure à un écart là où il n'y en a pas.
    """
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    empty = pd.DataFrame(columns=["documents_written", "n_authors",
                                  "share_observed", "share_lotka"])
    if a.empty:
        return {"table": empty, "fit": _fit_power_law(np.array([]), np.array([]))}

    key = a["scopus_id"].fillna("name:" + a["name"].map(str))
    per_author = a.assign(key=key).drop_duplicates(
        subset=["key", "eid"]).groupby("key").size()

    dist = per_author.value_counts().sort_index()
    total_authors = int(dist.sum())

    x = dist.index.to_numpy(dtype=float)
    y = dist.to_numpy(dtype=float)

    observed = y / total_authors

    # Lotka STRICT : l'exposant vaut 2 par hypothèse. C'est la référence
    # historique, celle à laquelle on compare — pas un ajustement.
    zeta2 = np.pi ** 2 / 6.0
    share_lotka = (1.0 / x ** 2) / zeta2

    # Lotka GÉNÉRALISÉ : y_n = C / n^a, l'exposant estimé sur le corpus et
    # C = 1/ζ(a) qui en fait une distribution de probabilité. C'est cette
    # courbe qui dit si la production suit UNE loi de puissance ; la version
    # stricte dit seulement si elle suit CELLE de Lotka — question différente
    # et bien plus exigeante.
    ols = _fit_power_law(x, y)
    exponent = _lotka_mle(per_author.to_numpy(dtype=float))
    zeta_a = _zeta(exponent)
    if np.isfinite(zeta_a) and zeta_a > 0:
        constant = 1.0 / zeta_a
        share_fitted = constant * x ** (-exponent)
    else:
        # Un exposant ≤ 1 rend la série divergente : aucune constante ne
        # normalise la loi. On le dit plutôt que de renvoyer un nombre qui
        # n'aurait pas de sens.
        constant = float("nan")
        share_fitted = np.full_like(x, np.nan, dtype=float)

    table = pd.DataFrame({
        "documents_written": dist.index.astype(int),
        "n_authors": dist.to_numpy(dtype=int),
        "share_observed": observed.round(4),
        "share_fitted": np.round(share_fitted, 4),
        "share_lotka": share_lotka.round(4),
    }).reset_index(drop=True)

    fit = {
        "exponent": float(exponent),
        "constant": float(constant),
        # R² de la régression log-log : une mesure DESCRIPTIVE de l'alignement
        # des points, pas la qualité de l'estimation. Il est conservé parce
        # qu'on le lit partout, et l'exposant des moindres carrés à côté pour
        # qu'on voie l'écart entre les deux méthodes.
        "r2": float(ols["r2"]),
        "exponent_ols": float(ols["exponent"]),
    }

    # Le test compare les répartitions sur TOUS les niveaux 1..x_max, pas
    # seulement ceux qu'on a observés. Un niveau sans aucun auteur (personne à
    # 4 documents, mais quelqu'un à 5) a une part observée nulle et une part
    # théorique non nulle : l'omettre faisait sauter cette masse de la
    # répartition théorique, et sous-estimait l'écart D.
    grid = np.arange(1, int(x.max()) + 1, dtype=float)
    observed_grid = pd.Series(observed, index=x).reindex(grid, fill_value=0.0).to_numpy()
    fitted_grid = (constant * grid ** (-exponent)) if np.isfinite(constant) \
        else np.full_like(grid, np.nan)
    lotka_grid = (1.0 / grid ** 2) / zeta2

    return {"table": table, "fit": fit,
            "total_authors": total_authors,
            # Deux tests, deux questions. Le premier : la production suit-elle
            # la loi AJUSTÉE ? Le second : suit-elle Lotka strict ?
            "ks_test": kolmogorov_smirnov(observed_grid, fitted_grid,
                                          sample_size=total_authors),
            "ks_test_strict": kolmogorov_smirnov(observed_grid, lotka_grid,
                                                 sample_size=total_authors)}


def kolmogorov_smirnov(observed: np.ndarray, theoretical: np.ndarray,
                       sample_size: Optional[int] = None) -> Dict[str, object]:
    """Test de Kolmogorov-Smirnov entre la distribution observée et Lotka.

    Un exposant ajusté et un R² disent que la loi *ressemble* aux données. Ils
    ne disent PAS si l'écart restant est compatible avec le hasard — c'est la
    question à laquelle ce test répond, et c'est celle que pose un relecteur.

    On compare les fonctions de répartition cumulées : ``D`` est leur écart
    maximal. La valeur critique à 5 % est ``1,36/√N`` (Pao, 1985), où **N est
    le nombre d'OBSERVATIONS — les auteurs**, pas le nombre de niveaux de
    productivité. `observed` et `theoretical` sont des parts par niveau ;
    `sample_size` porte N. Sans lui, on retombe sur le nombre de valeurs
    passées, ce qui n'a de sens que si chaque valeur est une observation.

    Défaut corrigé : N valait le nombre de niveaux (souvent moins de dix), la
    valeur critique dépassait alors 0,4 et presque tout corpus « suivait
    Lotka ».

    Réserve : l'exposant de la loi AJUSTÉE est estimé sur les mêmes données.
    Le test est alors conservateur (il rejette moins qu'il ne devrait) ; la
    version stricte, à exposant 2 fixé a priori, n'a pas ce biais.

    La p-valeur vient de la série de Kolmogorov, calculée ici directement :
    elle évite une dépendance à scipy pour une somme qui converge en quelques
    termes.
    """
    obs = np.asarray(observed, dtype=float)
    theo = np.asarray(theoretical, dtype=float)
    if obs.size == 0 or not np.isfinite(theo).all():
        return {"d": None, "p_value": None, "critical_5pct": None, "follows_lotka": None}
    n = int(sample_size) if sample_size else obs.size

    cdf_obs = np.cumsum(obs)
    cdf_theo = np.cumsum(theo)
    d = float(np.max(np.abs(cdf_obs - cdf_theo)))
    critical = 1.36 / np.sqrt(n)

    # Série de Kolmogorov : Q(t) = 2 Σ (−1)^{k−1} exp(−2k²t²).
    t = np.sqrt(n) * d
    if t <= 0:
        p = 1.0
    else:
        terms = [((-1) ** (k - 1)) * np.exp(-2.0 * (k ** 2) * (t ** 2))
                 for k in range(1, 101)]
        p = float(min(1.0, max(0.0, 2.0 * sum(terms))))

    return {
        "d": round(d, 4),
        "p_value": round(p, 4),
        "critical_5pct": round(float(critical), 4),
        # « Suit Lotka » = l'écart n'est PAS significatif. Un test qui ne
        # rejette pas n'est pas une preuve que la loi tient ; c'est seulement
        # l'absence de preuve du contraire, et il faut le dire ainsi.
        "follows_lotka": bool(d <= critical),
    }


# ---------------------------------------------------------------------------
# Bradford — dispersion des sources
# ---------------------------------------------------------------------------

def bradford(corpus, zones: int = 3) -> Dict[str, object]:
    """Sources classées par productivité, réparties en zones de Bradford.

    Renvoie ``{"table": DataFrame, "zones": DataFrame, "multiplier": float}``.

    Chaque zone contient approximativement le même NOMBRE D'ARTICLES ; c'est
    le nombre de SOURCES par zone qui croît, en principe géométriquement. Le
    « multiplicateur » est le rapport moyen entre le nombre de sources de deux
    zones successives : c'est lui qui dit si la loi tient.
    """
    d = corpus.documents
    d = d[d["source"].notna() & (d["source"].map(str).str.strip() != "")]
    empty = pd.DataFrame(columns=["rank", "source", "documents",
                                  "cumulative", "zone"])
    if d.empty:
        return {"table": empty,
                "zones": pd.DataFrame(columns=["zone", "sources", "documents"]),
                "multiplier": float("nan")}

    counts = (d.groupby("source")["eid"].nunique()
                .sort_values(ascending=False, kind="stable").reset_index(name="documents"))
    counts["rank"] = np.arange(1, len(counts) + 1)
    counts["cumulative"] = counts["documents"].cumsum()

    total = int(counts["documents"].sum())

    # Affectation des zones. Subtilité : une source appartient à la zone
    # qu'elle est en train de REMPLIR, pas à celle qu'elle fait déborder.
    # Sinon une revue très productive — dépassant à elle seule le premier
    # tiers — se retrouverait en zone 2, alors qu'elle EST le noyau.
    target = total / zones
    zone, acc, assigned = 1, 0, []
    for docs in counts["documents"].tolist():
        assigned.append(zone)
        acc += docs
        # `while`, pas `if` : une revue assez productive pour remplir DEUX
        # zones d'un coup doit faire avancer le compteur de deux. Avec `if`,
        # la revue suivante tombait dans une zone déjà pleine.
        while zone < zones and acc >= target * zone:
            zone += 1
    counts["zone"] = assigned

    summary = (counts.groupby("zone")
                     .agg(sources=("source", "count"),
                          documents=("documents", "sum"))
                     .reset_index())

    # Multiplicateur : rapport moyen du nombre de sources entre zones
    # successives. Théoriquement constant si la loi s'applique.
    ratios = []
    s = summary["sources"].tolist()
    for i in range(1, len(s)):
        if s[i - 1] > 0:
            ratios.append(s[i] / s[i - 1])
    multiplier = float(np.mean(ratios)) if ratios else float("nan")

    table = counts[["rank", "source", "documents", "cumulative", "zone"]]
    return {"table": table, "zones": summary, "multiplier": round(multiplier, 2),
            "total_documents": total, "total_sources": int(len(counts))}


# ---------------------------------------------------------------------------
# Zipf — fréquence des mots
# ---------------------------------------------------------------------------

def zipf(corpus, n: Optional[int] = 100, kind: str = "author") -> Dict[str, object]:
    """Fréquence des mots-clés en fonction de leur rang.

    Renvoie ``{"table": DataFrame, "fit": {...}}``. La table porte le rang, le
    terme, sa fréquence observée et la fréquence **prédite** par l'ajustement,
    pour qu'on puisse superposer les deux courbes et juger visuellement.
    """
    k = corpus.keywords
    if kind != "all":
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna()]
    empty = pd.DataFrame(columns=["rank", "keyword", "frequency", "predicted"])
    if k.empty:
        return {"table": empty, "fit": _fit_power_law(np.array([]), np.array([]))}

    k = k.copy()
    k["norm"] = k["keyword"].map(str).str.strip().str.lower()
    counts = (k.groupby("norm")
                .agg(keyword=("keyword", lambda s: s.mode().iat[0]),
                     frequency=("eid", "nunique"))
                .reset_index(drop=True)
                .sort_values("frequency", ascending=False, kind="stable")
                .reset_index(drop=True))
    counts["rank"] = np.arange(1, len(counts) + 1)

    # L'ajustement porte sur TOUTE la distribution ; la troncature à n ne
    # concerne que l'affichage. Ajuster sur un extrait fausserait l'exposant.
    fit = _fit_power_law(counts["rank"].to_numpy(dtype=float),
                         counts["frequency"].to_numpy(dtype=float))

    out = counts.head(n) if n else counts
    out = out.copy()
    if not np.isnan(fit["exponent"]):
        out["predicted"] = (fit["constant"]
                            * out["rank"] ** (-fit["exponent"])).round(2)
    else:
        out["predicted"] = np.nan

    return {"table": out[["rank", "keyword", "frequency", "predicted"]],
            "fit": fit, "total_keywords": int(len(counts))}
