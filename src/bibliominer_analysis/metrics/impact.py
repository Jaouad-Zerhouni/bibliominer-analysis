"""Indices d'impact : h, g, et dérivés.

Les définitions retenues sont les définitions canoniques, pas des variantes :

  - **h** (Hirsch, 2005) : le plus grand entier *h* tel que *h* documents
    aient chacun **au moins** *h* citations.

  - **g** (Egghe, 2006) : le plus grand entier *g* tel que les *g* documents
    les plus cités totalisent **au moins** *g²* citations. Le g-index est
    toujours ≥ h : il tient compte des articles très cités, que le h-index
    plafonne.

Les citations sont celles de Scopus (`Cited by`) — donc les citations
**mondiales** de l'article, pas seulement celles reçues à l'intérieur du
corpus. C'est la convention des classements bibliométriques usuels.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def h_index(citations: Iterable[int]) -> int:
    """h-index d'une série de comptes de citations."""
    c = np.sort(np.asarray([x for x in citations if x is not None and x >= 0],
                           dtype=float))[::-1]
    if c.size == 0:
        return 0
    ranks = np.arange(1, c.size + 1)
    return int(np.max(np.where(c >= ranks, ranks, 0)))


def g_index(citations: Iterable[int]) -> int:
    """g-index d'une série de comptes de citations."""
    c = np.sort(np.asarray([x for x in citations if x is not None and x >= 0],
                           dtype=float))[::-1]
    if c.size == 0:
        return 0
    ranks = np.arange(1, c.size + 1)
    cumulative = np.cumsum(c)
    ok = cumulative >= ranks ** 2
    return int(np.max(np.where(ok, ranks, 0)))


def i10_index(citations: Iterable[int]) -> int:
    """Nombre de documents ayant au moins 10 citations."""
    return int(sum(1 for x in citations if x is not None and x >= 10))


def e_index(citations: Iterable[int]) -> float:
    """e-index (Zhang, 2009) : les citations EXCÉDENTAIRES du noyau h.

    Le h-index ignore tout ce qui dépasse. Deux auteurs de h = 10 sont
    indiscernables, que leurs dix articles aient 10 citations chacun ou 500 :
    le premier a 100 citations dans son noyau, le second 5000, et h vaut 10
    dans les deux cas. Le e-index mesure exactement ce que h jette :

        e² = (somme des citations des h articles du noyau) − h²
        e  = √(e²)

    Il ne remplace pas h, il le **complète** — c'est le sens du mot chez Zhang.
    Lu avec h, il distingue une œuvre régulière d'une œuvre portée par quelques
    travaux très cités.

    e est toujours ≥ 0 : par définition du h-index, chacun des h articles du
    noyau a au moins h citations, donc leur somme atteint au moins h².
    """
    c = np.sort(np.asarray([x for x in citations if x is not None and x >= 0],
                           dtype=float))[::-1]
    if c.size == 0:
        return 0.0
    h = h_index(c)
    if h == 0:
        return 0.0
    excess = float(c[:h].sum()) - h ** 2
    return round(float(np.sqrt(max(excess, 0.0))), 2)


def m_index(h: int, first_year, last_year) -> Optional[float]:
    """h rapporté à l'ancienneté : m = h / nombre d'années d'activité.

    Le h-index ne peut que croître avec le temps : comparer un chercheur de
    trente ans de carrière à un chercheur de cinq ans par leur seul h n'a pas
    de sens. Le m corrige exactement ce biais.

    `last_year` est la dernière année **du corpus**, jamais l'année courante :
    sinon tous les m d'un corpus arrêté en 2020 baisseraient chaque 1ᵉʳ janvier
    sans qu'aucune donnée n'ait changé.
    """
    if first_year is None or last_year is None:
        return None
    if pd.isna(first_year) or pd.isna(last_year):
        return None
    span = int(last_year) - int(first_year) + 1
    if span <= 0:
        return None
    return round(h / span, 2)


# ---------------------------------------------------------------------------
# Au niveau du corpus
# ---------------------------------------------------------------------------

def corpus_impact(corpus) -> dict:
    """Indices d'impact du corpus entier."""
    cites = pd.to_numeric(corpus.documents["cited_by"],
                          errors="coerce").fillna(0).astype(int)
    total = int(cites.sum())
    n = int(len(cites))
    return {
        "documents": n,
        "citations": total,
        "citations_per_doc": round(total / n, 2) if n else 0.0,
        "h_index": h_index(cites),
        "g_index": g_index(cites),
        "i10_index": i10_index(cites),
        "e_index": e_index(cites),
        "uncited": int((cites == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Par auteur
# ---------------------------------------------------------------------------

def _rank_counts(ranks, total: int) -> dict:
    """Répartition des signatures d'un auteur par RANG : 1ᵉʳ, 2ᵉ, 3ᵉ, 4ᵉ et plus.

    Le rang de signature n'est pas un détail : dans la plupart des disciplines
    il encode le rôle. Premier auteur, c'est avoir porté le travail ; septième
    sur huit, c'est y avoir contribué. Deux auteurs à 50 documents peuvent
    avoir des carrières opposées, et seule cette ventilation le montre.

    Les quatre comptes somment TOUJOURS au nombre de documents : « 4ᵉ et plus »
    est calculé par différence. Un rang illisible y tombe donc aussi — c'est le
    seau fourre-tout, et il vaut mieux un tableau qui s'additionne qu'une
    colonne muette. En pratique les exports Scopus numérotent toujours les
    signataires, ce cas ne se produit pas.
    """
    r = pd.to_numeric(ranks, errors="coerce")
    first = int((r == 1).sum())
    second = int((r == 2).sum())
    third = int((r == 3).sum())
    return {
        "first_author": first,
        "second_author": second,
        "third_author": third,
        "later_author": int(total - first - second - third),
    }


def authors_impact(corpus, n: Optional[int] = 20,
                   min_documents: int = 1) -> pd.DataFrame:
    """Classement des auteurs avec leurs indices d'impact.

    Colonnes : ``author``, ``scopus_id``, ``documents``, ``citations``,
    ``h_index``, ``g_index``, ``first_author``, ``years``, ``first_year``,
    ``last_year``.

    Un auteur est identifié par son **identifiant Scopus** s'il existe ; sinon
    par son nom. Deux homonymes sans identifiant restent indiscernables — c'est
    une limite des données, pas du calcul, et mieux vaut le savoir que de le
    masquer.

    Le calcul se fait sur les documents DU CORPUS : un h-index de 6 ici
    signifie « 6 documents de ce corpus, cités au moins 6 fois chacun ». Ce
    n'est pas le h-index global de la personne, qui porterait sur toute son
    œuvre.
    """
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    empty = pd.DataFrame(columns=["author", "scopus_id", "documents", "citations",
                                  "h_index", "g_index", "first_author",
                                  "first_year", "last_year"])
    if a.empty:
        return empty

    docs = corpus.documents[["eid", "year"]].copy()
    docs["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                      errors="coerce").fillna(0).astype(int)
    a = a.merge(docs, on="eid", how="left")
    a["citations"] = a["citations"].fillna(0).astype(int)
    a["rank"] = pd.to_numeric(a["position"], errors="coerce")
    a["key"] = a["scopus_id"].fillna("name:" + a["name"].map(str))

    corpus_last = pd.to_numeric(corpus.documents["year"], errors="coerce").max()

    rows = []
    for key, g in a.groupby("key", sort=False):
        # Un auteur peut apparaître deux fois sur le même document (rare, mais
        # les exports le font) : on dédoublonne, sinon citations et indices
        # seraient gonflés.
        per_doc = g.drop_duplicates(subset=["eid"])
        cites = per_doc["citations"].tolist()
        years = pd.to_numeric(per_doc["year"], errors="coerce").dropna()
        h = h_index(cites)
        rows.append({
            "m_index": m_index(h, years.min() if not years.empty else None,
                               corpus_last),
            "author": g["name"].mode().iat[0] if not g["name"].empty else None,
            "scopus_id": g["scopus_id"].dropna().iat[0]
            if g["scopus_id"].notna().any() else None,
            "documents": int(len(per_doc)),
            "citations": int(sum(cites)),
            "h_index": h,
            "g_index": g_index(cites),
            "i10_index": i10_index(cites),
            "e_index": e_index(cites),
            **_rank_counts(per_doc["rank"], len(per_doc)),
            "first_year": int(years.min()) if not years.empty else None,
            "last_year": int(years.max()) if not years.empty else None,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return empty
    out = out[out["documents"] >= min_documents]
    out = out.sort_values(["h_index", "citations", "documents"],
                          ascending=False, kind="stable").reset_index(drop=True)
    return out.head(n) if n else out


def institutions_impact(corpus, n: Optional[int] = 20,
                        level: str = "parent") -> pd.DataFrame:
    """Mêmes indices, agrégés par organisation.

    `level` vaut « parent » (l'établissement) ou « subparent » (l'unité
    interne : laboratoire, école, département).

    Un document compte UNE fois par organisation, même si trois de ses auteurs
    y sont rattachés — sinon les citations seraient comptées trois fois.
    """
    from .production import _org_frame

    aff = _org_frame(corpus, level)
    empty = pd.DataFrame(columns=["institution", "documents", "citations",
                                  "h_index", "g_index", "country"])
    if aff.empty:
        return empty

    pairs = aff[["eid", "org", "country"]].drop_duplicates(subset=["eid", "org"])
    pairs = pairs.rename(columns={"org": "parent1"})
    docs = corpus.documents[["eid"]].copy()
    docs["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                      errors="coerce").fillna(0).astype(int)
    docs["year"] = pd.to_numeric(corpus.documents["year"], errors="coerce")
    pairs = pairs.merge(docs, on="eid", how="left")
    pairs["citations"] = pairs["citations"].fillna(0).astype(int)
    corpus_last = docs["year"].max()

    rows = []
    for inst, g in pairs.groupby("parent1", sort=False):
        cites = g["citations"].tolist()
        countries = g["country"].dropna()
        years = g["year"].dropna()
        h = h_index(cites)
        rows.append({
            "institution": inst,
            "documents": int(g["eid"].nunique()),
            "citations": int(sum(cites)),
            "h_index": h,
            "g_index": g_index(cites),
            "m_index": m_index(h, years.min() if not years.empty else None,
                               corpus_last),
            "first_year": int(years.min()) if not years.empty else None,
            "country": countries.mode().iat[0] if not countries.empty else None,
        })

    out = pd.DataFrame(rows).sort_values(
        ["h_index", "citations", "documents"], ascending=False, kind="stable").reset_index(drop=True)
    return out.head(n) if n else out


def normalized_citations(corpus) -> pd.DataFrame:
    """Citations rapportées à la moyenne de leur **année de publication**.

    Colonnes : ``eid``, ``citations``, ``year_mean_citations``,
    ``normalized_citations``.

    Comparer les citations brutes de deux articles publiés à huit ans d'écart
    n'a pas de sens : le plus ancien gagne presque toujours, et cela ne dit
    rien de sa qualité. Le score normalisé divise par la moyenne de la cohorte :

        1,0  = exactement la moyenne de son année
        3,0  = trois fois mieux que ses contemporains

    C'est la seule colonne qui permette de classer ensemble un article de 2016
    et un de 2024. Une année sans aucune citation donne 0 plutôt qu'une
    division par zéro.
    """
    d = corpus.documents[["eid"]].copy()
    d["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                   errors="coerce").fillna(0).astype(int)
    d["year"] = pd.to_numeric(corpus.documents["year"], errors="coerce")

    means = d.groupby("year")["citations"].transform("mean")
    d["year_mean_citations"] = means.round(2)
    d["normalized_citations"] = np.where(
        means > 0, (d["citations"] / means).round(3), 0.0)
    return d[["eid", "citations", "year_mean_citations", "normalized_citations"]]
