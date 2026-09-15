"""Citations **locales** : les citations reçues à l'intérieur du corpus.

C'est la distinction la plus mal comprise de la bibliométrie, et la plus utile.

  - **GC** (*global citations*) : ce que Scopus compte, toutes sources
    confondues. Un article très cité par une communauté voisine y brille.
  - **LC** (*local citations*) : combien de documents **de ce corpus** le
    citent. C'est la mesure de l'influence *dans le domaine étudié*.

Un article peut avoir 800 citations mondiales et 0 locale : il est important
ailleurs, pas ici. L'inverse existe aussi, et signale un travail fondateur pour
la communauté précise qu'on analyse. Sans LC on ne peut construire ni
l'historiographe, ni les classements « local cited », ni le réseau de citation
directe — d'où ce module en socle.

Le rapprochement référence → document se fait en deux temps :

  1. **par DOI**, quand les deux côtés en ont un. C'est un identifiant, donc
     sans ambiguïté.
  2. **par titre normalisé** sinon (minuscules, sans ponctuation ni accents,
     espaces réduits). On exige un titre d'au moins 25 caractères : en dessous,
     des titres génériques (« Introduction », « Machine learning ») créeraient
     de faux rapprochements, et une fausse citation est pire qu'une manquante.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

#: En dessous de cette longueur, un titre n'est pas assez discriminant.
MIN_TITLE_LEN = 25


def _norm_doi(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None
    # Les exports mélangent « 10.1000/x », « https://doi.org/10.1000/x » et
    # « doi:10.1000/x » : on ne garde que la partie qui commence à « 10. ».
    m = re.search(r"10\.\d{4,9}/\S+", s)
    return m.group(0).rstrip(".,;)") if m else None


def _norm_title(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def citation_pairs(corpus) -> pd.DataFrame:
    """Les liens de citation **internes** au corpus.

    Colonnes : ``citing`` (eid du citant), ``cited`` (eid du cité), ``via``
    (« doi » ou « title », comment le rapprochement a été fait).

    Une paire n'apparaît qu'une fois même si le citant liste deux fois la même
    référence, et l'auto-citation d'un document par lui-même est écartée.
    """
    empty = pd.DataFrame(columns=["citing", "cited", "via"])
    docs, refs = corpus.documents, corpus.references
    if docs.empty or refs.empty:
        return empty

    # Index des documents du corpus, côté « cité ».
    by_doi, by_title = {}, {}
    for eid, doi, title in zip(docs["eid"], docs.get("doi"), docs.get("title")):
        d = _norm_doi(doi)
        if d and d not in by_doi:
            by_doi[d] = eid
        t = _norm_title(title)
        if t and len(t) >= MIN_TITLE_LEN and t not in by_title:
            by_title[t] = eid

    if not by_doi and not by_title:
        return empty

    rows = []
    for citing, rdoi, rtitle in zip(refs["eid"], refs.get("ref_doi"),
                                    refs.get("ref_title")):
        cited, via = None, None
        d = _norm_doi(rdoi)
        if d is not None:
            cited = by_doi.get(d)
            via = "doi"
        if cited is None:
            t = _norm_title(rtitle)
            if t is not None and len(t) >= MIN_TITLE_LEN:
                cited = by_title.get(t)
                via = "title"
        if cited is not None and cited != citing:
            rows.append((citing, cited, via))

    if not rows:
        return empty
    out = pd.DataFrame(rows, columns=["citing", "cited", "via"])
    # Le DOI prime sur le titre quand les deux existent pour la même paire.
    out["_rank"] = (out["via"] == "title").astype(int)
    out = (out.sort_values("_rank")
              .drop_duplicates(subset=["citing", "cited"])
              .drop(columns="_rank")
              .reset_index(drop=True))
    return out


def local_citations(corpus) -> pd.DataFrame:
    """Compte de citations locales par document. Colonnes ``eid``, ``local_citations``."""
    pairs = citation_pairs(corpus)
    base = pd.DataFrame({"eid": corpus.documents["eid"]})
    if pairs.empty:
        base["local_citations"] = 0
        return base
    counts = pairs.groupby("cited").size().rename("local_citations")
    base = base.merge(counts, left_on="eid", right_index=True, how="left")
    base["local_citations"] = base["local_citations"].fillna(0).astype(int)
    return base


def _doc_frame(corpus) -> pd.DataFrame:
    d = corpus.documents[["eid", "title", "year", "source", "doi"]].copy()
    d["global_citations"] = pd.to_numeric(
        corpus.documents["cited_by"], errors="coerce").fillna(0).astype(int)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    first = (corpus.authors[pd.to_numeric(corpus.authors["position"],
                                          errors="coerce").eq(1)]
             [["eid", "name"]].drop_duplicates("eid"))
    d = d.merge(first.rename(columns={"name": "first_author"}), on="eid", how="left")
    return d


def most_local_cited_documents(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """Les documents les plus cités **par les autres documents du corpus**.

    Colonnes : ``label``, ``title``, ``first_author``, ``year``, ``source``,
    ``local_citations``, ``global_citations``, ``lc_gc_ratio``,
    ``local_citations_per_year``.

    ``lc_gc_ratio`` est le pourcentage des citations mondiales qui viennent du
    corpus : élevé, le travail est spécifique au domaine ; proche de zéro, sa
    notoriété vient d'ailleurs.
    """
    lc = local_citations(corpus)
    d = _doc_frame(corpus).merge(lc, on="eid", how="left")
    d["local_citations"] = d["local_citations"].fillna(0).astype(int)
    d = d[d["local_citations"] > 0]
    if d.empty:
        return pd.DataFrame(columns=["label", "title", "first_author", "year",
                                     "source", "local_citations",
                                     "global_citations", "lc_gc_ratio",
                                     "local_citations_per_year"])

    last_year = pd.to_numeric(corpus.documents["year"], errors="coerce").max()
    age = (last_year - d["year"] + 1).clip(lower=1)
    d["local_citations_per_year"] = (d["local_citations"] / age).round(2)
    d["lc_gc_ratio"] = np.where(
        d["global_citations"] > 0,
        (100 * d["local_citations"] / d["global_citations"]).round(1), 0.0)
    d["label"] = _labels(d)

    out = d.sort_values(["local_citations", "global_citations"], ascending=False)
    out = out[["label", "title", "first_author", "year", "source",
               "local_citations", "global_citations", "lc_gc_ratio",
               "local_citations_per_year"]].reset_index(drop=True)
    out["year"] = out["year"].astype("Int64")
    return out.head(n) if n else out


def _labels(d: pd.DataFrame) -> pd.Series:
    """« HOSNI M., 2019 » — l'étiquette courte usuelle en bibliométrie."""
    author = d["first_author"].fillna("ANONYMOUS").astype(str).str.upper()
    year = d["year"].astype("Int64").astype(str).replace("<NA>", "n.d.")
    base = author + ", " + year
    # Deux documents du même auteur la même année : on suffixe a, b, c…
    dup = base.duplicated(keep=False)
    if dup.any():
        suffix = base.groupby(base).cumcount()
        letters = suffix.map(lambda i: "" if i == 0 else "-" + chr(97 + min(i, 25)))
        base = base + letters.where(dup, "")
    return base


def most_local_cited_authors(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """Auteurs classés par citations reçues **depuis l'intérieur du corpus**.

    Un document cité localement 5 fois apporte 5 à chacun de ses signataires :
    c'est le comptage entier (*full counting*), celui de bibliometrix.
    """
    lc = local_citations(corpus)
    a = corpus.authors[["eid", "name"]].dropna(subset=["name"])
    a = a[a["name"].astype(str).str.strip() != ""].drop_duplicates(["eid", "name"])
    if a.empty:
        return pd.DataFrame(columns=["author", "local_citations", "documents"])
    a = a.merge(lc, on="eid", how="left")
    a["local_citations"] = a["local_citations"].fillna(0).astype(int)
    out = (a.groupby("name")
             .agg(local_citations=("local_citations", "sum"),
                  documents=("eid", "nunique"))
             .reset_index().rename(columns={"name": "author"}))
    out = out[out["local_citations"] > 0]
    out = out.sort_values(["local_citations", "documents"],
                          ascending=False).reset_index(drop=True)
    return out.head(n) if n else out


def most_local_cited_sources(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """Revues classées par citations locales cumulées de leurs articles."""
    lc = local_citations(corpus)
    d = corpus.documents[["eid", "source"]].copy()
    d = d[d["source"].notna() & (d["source"].astype(str).str.strip() != "")]
    if d.empty:
        return pd.DataFrame(columns=["source", "local_citations", "documents"])
    d = d.merge(lc, on="eid", how="left")
    d["local_citations"] = d["local_citations"].fillna(0).astype(int)
    out = (d.groupby("source")
             .agg(local_citations=("local_citations", "sum"),
                  documents=("eid", "nunique"))
             .reset_index())
    out = out[out["local_citations"] > 0]
    out = out.sort_values(["local_citations", "documents"],
                          ascending=False).reset_index(drop=True)
    return out.head(n) if n else out


def historiograph(corpus, n: int = 25) -> dict:
    """Graphe de **citation directe** entre les documents les plus influents.

    L'historiographe de Garfield : on garde les `n` documents les plus cités
    localement et on ne trace que les citations qui les relient entre eux. Lu
    de gauche à droite (le temps), il montre la filiation des idées — quel
    travail s'appuie sur quel autre — ce qu'aucun classement ne donne.

    Retour : ``{"nodes": [...], "edges": [...], "n_nodes", "n_edges"}``.
    Chaque nœud porte ``id``, ``label``, ``year``, ``local_citations``,
    ``global_citations``, ``title``.
    """
    pairs = citation_pairs(corpus)
    lc = local_citations(corpus)
    d = _doc_frame(corpus).merge(lc, on="eid", how="left")
    d["local_citations"] = d["local_citations"].fillna(0).astype(int)

    keep = d[d["local_citations"] > 0].sort_values(
        ["local_citations", "global_citations"], ascending=False).head(n)
    if keep.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    keep = keep.copy()
    keep["label"] = _labels(keep)
    ids = set(keep["eid"])
    edges = pairs[pairs["citing"].isin(ids) & pairs["cited"].isin(ids)]

    nodes = [{
        "id": str(r.eid),
        "label": str(r.label),
        "year": int(r.year) if pd.notna(r.year) else None,
        "local_citations": int(r.local_citations),
        "global_citations": int(r.global_citations),
        "title": None if pd.isna(r.title) else str(r.title),
    } for r in keep.itertuples()]

    return {
        "nodes": nodes,
        "edges": [{"source": str(s), "target": str(t)}
                  for s, t in zip(edges["citing"], edges["cited"])],
        "n_nodes": len(nodes),
        "n_edges": int(len(edges)),
    }


#: Unités sur lesquelles on peut agréger un réseau de citation directe.
CITATION_UNITS = ("documents", "authors", "sources", "countries", "institutions")


def citation_network(corpus, unit: str = "sources", top_n: int = 40,
                     min_weight: int = 1, level: str = "parent") -> dict:
    """Réseau de **citation directe**, agrégé à l'unité demandée.

    La co-citation dit « ces deux travaux sont cités ensemble ». Le couplage dit
    « ces deux travaux citent les mêmes choses ». La citation directe dit tout
    autre chose, et c'est la seule des trois qui ait un **sens** : *qui cite
    qui*. C'est donc la seule à produire un graphe orienté.

    Agrégée aux auteurs, aux revues ou aux pays, elle montre les rapports de
    dépendance intellectuelle : une revue qui alimente tout le domaine sans
    jamais le citer en retour se voit immédiatement.

    Les auto-citations d'une entité vers elle-même sont écartées : au niveau
    d'un pays elles écraseraient tout le reste, et ne disent rien d'un échange.
    """
    from collections import Counter

    pairs = citation_pairs(corpus)
    empty = {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}
    if pairs.empty:
        return empty

    members = _unit_members(corpus, unit, level)
    if not members:
        return empty

    flows: Counter = Counter()
    for citing, cited in zip(pairs["citing"], pairs["cited"]):
        for a in members.get(citing, ()):
            for b in members.get(cited, ()):
                if a != b:
                    flows[(a, b)] += 1

    if not flows:
        return empty

    weight_of: Counter = Counter()
    for (a, b), w in flows.items():
        weight_of[a] += w
        weight_of[b] += w
    keep = {k for k, _ in weight_of.most_common(top_n)}

    edges = [{"source": a, "target": b, "weight": int(w)}
             for (a, b), w in flows.items()
             if w >= min_weight and a in keep and b in keep]
    if not edges:
        return empty

    linked = {e["source"] for e in edges} | {e["target"] for e in edges}
    # `occurrences` = citations échangées, c'est ce qui donne sa taille au nœud.
    nodes = [{"id": k, "label": k, "occurrences": int(weight_of[k]),
              "degree": int(sum(e["weight"] for e in edges
                                if e["source"] == k or e["target"] == k))}
             for k in sorted(linked, key=lambda x: -weight_of[x])]

    return {"nodes": nodes, "edges": sorted(edges, key=lambda e: -e["weight"]),
            "n_nodes": len(nodes), "n_edges": len(edges)}


def _unit_members(corpus, unit: str, level: str = "parent") -> dict:
    """eid → entités du document, selon l'unité d'agrégation."""
    out: dict = {}

    if unit == "documents":
        titles = corpus.documents.set_index("eid")["title"].to_dict()
        return {str(k): (str(v)[:70],) for k, v in titles.items() if pd.notna(v)}

    if unit == "authors":
        frame = corpus.authors[["eid", "name"]].dropna()
        frame = frame[frame["name"].astype(str).str.strip() != ""]
        column = "name"
    elif unit == "sources":
        frame = corpus.documents[["eid", "source"]].dropna()
        frame = frame[frame["source"].astype(str).str.strip() != ""]
        column = "source"
    elif unit == "countries":
        frame = corpus.affiliations[["eid", "country"]].dropna()
        frame = frame[frame["country"].astype(str).str.strip() != ""]
        column = "country"
    elif unit == "institutions":
        from .production import _org_frame
        frame = _org_frame(corpus, level)[["eid", "org"]].dropna()
        column = "org"
    else:
        return {}

    for eid, value in zip(frame["eid"], frame[column]):
        out.setdefault(str(eid), set()).add(str(value))
    return {k: tuple(v) for k, v in out.items()}
