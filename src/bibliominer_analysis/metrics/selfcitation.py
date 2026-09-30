"""Auto-citations : quelle part de l'influence vient de soi-même.

Un compte de citations ne dit pas d'où elles viennent. Un auteur cité vingt
fois par ses propres articles et un auteur cité vingt fois par des tiers ont le
même total, et une réputation qui n'a rien à voir. C'est l'un des rares
indicateurs que les comités de lecture demandent explicitement.

Une citation est une **auto-citation** à un niveau donné quand le document
citant et le document cité partagent au moins une entité à ce niveau : un
auteur, une organisation, un pays. Les trois niveaux ne se recouvrent pas,
deux équipes distinctes d'un même pays produisent une auto-citation *nationale*
sans aucune auto-citation d'auteur.

**Limite à énoncer avant tout usage** : le calcul ne porte que sur les
citations INTERNES au corpus. Les auto-citations venues d'articles hors corpus
sont invisibles ici. Le taux mesuré est donc un plancher, jamais le taux réel,
et l'annoncer autrement serait trompeur.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

LEVELS = ("authors", "institutions", "countries")


def _members(corpus, level: str) -> Dict[str, set]:
    """eid → ensemble des entités du document, au niveau demandé."""
    out: Dict[str, set] = {}

    if level == "authors":
        a = corpus.authors
        a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
        if a.empty:
            return out
        key = a["scopus_id"].fillna("name:" + a["name"].map(str))
        frame = pd.DataFrame({"eid": a["eid"].to_numpy(), "key": key.to_numpy()})
    elif level == "institutions":
        from .production import _org_frame
        org = _org_frame(corpus, "parent")
        if org.empty:
            return out
        frame = pd.DataFrame({"eid": org["eid"].to_numpy(),
                              "key": org["org"].to_numpy()})
    elif level == "countries":
        aff = corpus.affiliations
        aff = aff[aff["country"].notna()
                  & (aff["country"].map(str).str.strip() != "")]
        if aff.empty:
            return out
        frame = pd.DataFrame({"eid": aff["eid"].to_numpy(),
                              "key": aff["country"].to_numpy()})
    else:
        return out

    for eid, key in zip(frame["eid"], frame["key"]):
        if pd.notna(key):
            out.setdefault(str(eid), set()).add(str(key))
    return out


def self_citation_rate(corpus, level: str = "authors") -> Dict[str, Any]:
    """Taux global d'auto-citation à un niveau donné.

    Retour : ``level``, ``citations``, ``self_citations``, ``external``,
    ``self_rate`` (%), et ``coverage_note``.
    """
    from .local import citation_pairs

    pairs = citation_pairs(corpus)
    base = {"level": level, "citations": 0, "self_citations": 0,
            "external": 0, "self_rate": None}
    if pairs.empty:
        return base

    members = _members(corpus, level)
    if not members:
        return base

    total = len(pairs)
    self_count = 0
    for citing, cited in zip(pairs["citing"], pairs["cited"]):
        a, b = members.get(str(citing)), members.get(str(cited))
        if a and b and (a & b):
            self_count += 1

    return {
        "level": level,
        "citations": int(total),
        "self_citations": int(self_count),
        "external": int(total - self_count),
        "self_rate": round(100.0 * self_count / total, 1) if total else None,
    }


def self_citation_summary(corpus) -> pd.DataFrame:
    """Les trois niveaux côte à côte.

    Colonnes : ``level``, ``citations``, ``self_citations``, ``external``,
    ``self_rate``.

    Les niveaux ne sont PAS strictement emboîtés : un auteur qui a changé
    d'établissement, ou de pays, entre l'article cité et l'article citant
    produit une auto-citation d'auteur sans auto-citation d'institution. En
    pratique le taux « pays » dépasse presque toujours le taux « auteur », et
    c'est l'écart entre les deux qui informe ; mais ce n'est pas une garantie,
    et un taux institutionnel plus bas que le taux auteur est possible.
    """
    cols = ["level", "citations", "self_citations", "external", "self_rate"]
    rows = [self_citation_rate(corpus, lv) for lv in LEVELS]
    return pd.DataFrame(rows, columns=cols)


def _received_citations(pairs: pd.DataFrame,
                        authors_of: Dict[str, set]) -> tuple:
    """Citations reçues par auteur, et combien venaient de ses propres documents.

    Un passage par citation, sur les SEULS auteurs du document cité : la
    version précédente parcourait tous les auteurs du corpus à chaque
    citation, et devenait inutilisable sur quelques milliers de documents.
    """
    received: Dict[str, int] = {}
    self_hits: Dict[str, int] = {}
    for citing, cited in zip(pairs["citing"].map(str), pairs["cited"].map(str)):
        citing_authors = authors_of.get(citing, set())
        for key in authors_of.get(cited, ()):
            received[key] = received.get(key, 0) + 1
            if key in citing_authors:
                self_hits[key] = self_hits.get(key, 0) + 1
    return received, self_hits


def authors_self_citation(corpus, n: Optional[int] = 20,
                          min_citations: int = 1) -> pd.DataFrame:
    """Auto-citation par auteur.

    Colonnes : ``author``, ``documents``, ``local_citations``,
    ``self_citations``, ``external_citations``, ``self_rate``.

    ``self_citations`` compte les citations reçues par un auteur depuis des
    documents qu'il a lui-même signés.

    Un auteur est identifié comme partout ailleurs dans le package : par son
    identifiant Scopus, sinon par son nom. Défaut corrigé : ce seul calcul
    passait par le NOM, et fusionnait deux homonymes, l'un « s'auto-citait »
    alors en citant l'autre.
    """
    from .local import citation_pairs

    cols = ["author", "documents", "local_citations", "self_citations",
            "external_citations", "self_rate"]
    pairs = citation_pairs(corpus)
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    if pairs.empty or a.empty:
        return pd.DataFrame(columns=cols)

    a = a.assign(key=a["scopus_id"].fillna("name:" + a["name"].map(str)))
    docs_of: Dict[str, set] = {}
    authors_of: Dict[str, set] = {}
    for eid, key in zip(a["eid"].map(str), a["key"]):
        docs_of.setdefault(key, set()).add(eid)
        authors_of.setdefault(eid, set()).add(key)
    name_of = a.groupby("key")["name"].agg(lambda s: s.mode().iat[0]).to_dict()

    received, self_hits = _received_citations(pairs, authors_of)

    rows = []
    for key, docs in docs_of.items():
        total = received.get(key, 0)
        if total < min_citations:
            continue
        own = self_hits.get(key, 0)
        rows.append({
            "author": name_of.get(key, key),
            "documents": len(docs),
            "local_citations": total,
            "self_citations": own,
            "external_citations": total - own,
            "self_rate": round(100.0 * own / total, 1) if total else 0.0,
        })

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols).sort_values(
        ["local_citations", "self_rate"], ascending=False, kind="stable").reset_index(drop=True)
    return out.head(n) if n else out
