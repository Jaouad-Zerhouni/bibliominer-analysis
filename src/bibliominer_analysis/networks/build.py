"""Réseaux bibliométriques : co-citation, co-mots, co-signature.

Le principe est le même dans les trois cas : deux entités sont **liées**
quand elles apparaissent ensemble dans le même document. Seule change
l'entité — une référence citée, un mot-clé, un auteur.

    document 1 : [A, B, C]  ->  liens A-B, A-C, B-C
    document 2 : [A, B]     ->  lien  A-B  (poids 2 au total)

Deux garde-fous, sans lesquels ces réseaux deviennent inutilisables :

  - **On limite d'abord les nœuds** aux entités les plus fréquentes. Un corpus
    de 3 000 documents produit des centaines de milliers de paires : le calcul
    devient lent et le graphe illisible.
  - **On filtre les liens faibles** (poids minimal), sinon le graphe est un
    nuage de liens uniques sans structure.

Les fonctions renvoient un dictionnaire `{"nodes": [...], "edges": [...]}`,
directement affichable — le package ne dessine pas, il fournit la structure.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional

import pandas as pd


def _pairs_from_groups(groups: Dict[Any, List[str]],
                       keep: Optional[set] = None) -> Counter:
    """Compte les co-occurrences. `keep` restreint aux entités retenues."""
    pairs: Counter = Counter()
    for items in groups.values():
        uniq = sorted({i for i in items if i and (keep is None or i in keep)})
        if len(uniq) < 2:
            continue
        pairs.update(combinations(uniq, 2))
    return pairs


def _to_graph(counts: Counter, pairs: Counter,
              labels: Dict[str, str], min_weight: int) -> Dict[str, Any]:
    """Assemble nœuds + liens, et retire les nœuds devenus isolés.

    Un nœud sans aucun lien après filtrage n'apporte rien à la lecture d'un
    réseau : on le retire plutôt que de laisser des points flottants.
    """
    edges = [{"source": a, "target": b, "weight": int(w)}
             for (a, b), w in pairs.items() if w >= min_weight]

    linked = {e["source"] for e in edges} | {e["target"] for e in edges}
    degree: Counter = Counter()
    for e in edges:
        degree[e["source"]] += e["weight"]
        degree[e["target"]] += e["weight"]

    nodes = [{"id": k,
              "label": labels.get(k, k),
              "occurrences": int(counts[k]),
              "degree": int(degree[k])}
             for k in counts if k in linked]
    nodes.sort(key=lambda n: -n["occurrences"])

    return {"nodes": nodes, "edges": sorted(edges, key=lambda e: -e["weight"]),
            "n_nodes": len(nodes), "n_edges": len(edges)}


# ---------------------------------------------------------------------------
# Co-citation
# ---------------------------------------------------------------------------

def _ref_key(row) -> Optional[str]:
    """Identité d'une référence citée : son DOI, sinon son titre normalisé.

    Le DOI est fiable ; le titre ne l'est qu'après normalisation (casse,
    ponctuation, espaces). Une référence sans ni l'un ni l'autre est écartée —
    on ne peut pas la rapprocher d'une autre sans risquer de fusionner des
    travaux différents.
    """
    doi = row.get("ref_doi")
    if isinstance(doi, str) and doi.strip():
        return "doi:" + doi.strip().lower()
    title = row.get("ref_title")
    if isinstance(title, str) and len(title.strip()) >= 12:
        norm = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        if norm:
            return "t:" + norm
    return None


def co_citation(corpus, top_n: int = 50, min_weight: int = 2) -> Dict[str, Any]:
    """Réseau de co-citation : deux références citées par les mêmes documents.

    Il révèle les **fondements intellectuels** du corpus : les travaux que les
    auteurs mobilisent ensemble forment des regroupements thématiques.
    """
    refs = corpus.references
    if refs.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    r = refs.copy()
    r["key"] = r.apply(_ref_key, axis=1)
    r = r[r["key"].notna()]
    if r.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    counts = Counter(r.drop_duplicates(subset=["eid", "key"])["key"])
    keep = {k for k, _ in counts.most_common(top_n)}

    labels: Dict[str, str] = {}
    for key, g in r[r["key"].isin(keep)].groupby("key"):
        title = g["ref_title"].dropna()
        year = g["ref_year"].dropna()
        authors = g["ref_authors"].dropna()
        name = title.iat[0] if not title.empty else key
        first = authors.iat[0].split(",")[0].strip() if not authors.empty else ""
        y = int(year.iat[0]) if not year.empty else None
        labels[key] = "%s%s — %s" % (first + " " if first else "",
                                     "(%d)" % y if y else "", name[:70])

    groups = r[r["key"].isin(keep)].groupby("eid")["key"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({k: counts[k] for k in keep}), pairs, labels, min_weight)


# ---------------------------------------------------------------------------
# Co-mots
# ---------------------------------------------------------------------------

def co_word(corpus, top_n: int = 50, min_weight: int = 2,
            kind: str = "author") -> Dict[str, Any]:
    """Réseau de co-occurrence des mots-clés — la **structure thématique**."""
    k = corpus.keywords
    if kind != "all":
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna()]
    if k.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    k = k.copy()
    k["norm"] = k["keyword"].astype(str).str.strip().str.lower()
    k = k.drop_duplicates(subset=["eid", "norm"])

    counts = Counter(k["norm"])
    keep = {w for w, _ in counts.most_common(top_n)}
    labels = (k[k["norm"].isin(keep)]
              .groupby("norm")["keyword"]
              .agg(lambda s: s.mode().iat[0]).to_dict())

    groups = k[k["norm"].isin(keep)].groupby("eid")["norm"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({w: counts[w] for w in keep}), pairs, labels, min_weight)


# ---------------------------------------------------------------------------
# Co-signature
# ---------------------------------------------------------------------------

def co_authorship(corpus, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
    """Réseau de collaboration : deux auteurs signant les mêmes documents.

    `min_weight` vaut 1 par défaut : une seule co-signature est déjà une
    collaboration réelle, contrairement à une co-citation isolée.
    """
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].astype(str).str.strip() != "")]
    if a.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    a = a.copy()
    a["key"] = a["scopus_id"].fillna("name:" + a["name"].astype(str))
    a = a.drop_duplicates(subset=["eid", "key"])

    counts = Counter(a["key"])
    keep = {k for k, _ in counts.most_common(top_n)}
    labels = (a[a["key"].isin(keep)]
              .groupby("key")["name"]
              .agg(lambda s: s.mode().iat[0]).to_dict())

    groups = a[a["key"].isin(keep)].groupby("eid")["key"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({k: counts[k] for k in keep}), pairs, labels, min_weight)


def co_institution(corpus, top_n: int = 50, min_weight: int = 1,
                   level: str = "parent") -> Dict[str, Any]:
    """Collaboration entre ORGANISATIONS présentes sur le même document.

    `level` vaut « parent » (établissements) ou « subparent » (unités
    internes). Au niveau unité, le réseau montre quels laboratoires
    travaillent ensemble — une information que le niveau établissement masque
    complètement quand deux équipes d'une même université collaborent.

    Les chercheurs sans rattachement sont écartés : « Independent researcher »
    n'est pas une organisation et polluerait le centre du réseau.
    """
    from ..metrics.production import _org_frame

    aff = _org_frame(corpus, level)
    if aff.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    aff = aff.drop_duplicates(subset=["eid", "org"])
    counts = Counter(aff["org"])
    keep = {k for k, _ in counts.most_common(top_n)}
    labels = {k: k for k in keep}

    groups = aff[aff["org"].isin(keep)].groupby("eid")["org"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({k: counts[k] for k in keep}), pairs, labels, min_weight)


def co_country(corpus, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
    """Collaboration entre PAYS : deux pays signataires du même document.

    C'est le réseau qui montre l'insertion internationale du corpus.
    """
    aff = corpus.affiliations
    aff = aff[aff["country"].notna() & (aff["country"].astype(str).str.strip() != "")]
    if aff.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    aff = aff.drop_duplicates(subset=["eid", "country"])
    counts = Counter(aff["country"])
    keep = {k for k, _ in counts.most_common(top_n)}
    labels = {k: k for k in keep}

    groups = aff[aff["country"].isin(keep)].groupby("eid")["country"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({k: counts[k] for k in keep}), pairs, labels, min_weight)


def bibliographic_coupling(corpus, top_n: int = 50,
                           min_weight: int = 2) -> Dict[str, Any]:
    """Couplage bibliographique : deux DOCUMENTS partageant des références.

    C'est le miroir de la co-citation. La co-citation regarde en arrière —
    quels travaux anciens sont cités ensemble — et évolue avec le temps. Le
    couplage regarde le présent : deux articles qui puisent aux mêmes sources
    traitent probablement du même sujet, et ce lien est **figé** dès leur
    publication.

    Le poids d'un lien est le nombre de références communes.
    """
    refs = corpus.references
    if refs.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    r = refs.copy()
    r["key"] = r.apply(_ref_key, axis=1)
    r = r[r["key"].notna()].drop_duplicates(subset=["eid", "key"])
    if r.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    # On garde les documents ayant le plus de références identifiables : ce
    # sont eux qui peuvent réellement se coupler. Un document à deux
    # références n'apporte que du bruit.
    per_doc = r.groupby("eid")["key"].apply(set)
    keep = set(per_doc.map(len).sort_values(ascending=False).head(top_n).index)

    pairs: Counter = Counter()
    docs = sorted(keep)
    for i, a in enumerate(docs):
        sa = per_doc[a]
        for b in docs[i + 1:]:
            shared = len(sa & per_doc[b])
            if shared >= min_weight:
                pairs[(a, b)] = shared

    titles = corpus.documents.set_index("eid")["title"].to_dict()
    years = corpus.documents.set_index("eid")["year"].to_dict()

    def label(eid: str) -> str:
        t = titles.get(eid) or eid
        y = years.get(eid)
        return "%s%s" % ("(%s) " % int(y) if pd.notna(y) else "", str(t)[:70])

    counts = Counter({eid: len(per_doc[eid]) for eid in keep})
    labels = {eid: label(eid) for eid in keep}
    return _to_graph(counts, pairs, labels, min_weight)


def _first_cited_author(value: Any) -> Optional[str]:
    """Premier auteur d'une référence citée, normalisé.

    L'analyse de co-citation d'auteurs (ACA) se fonde par convention sur le
    PREMIER auteur cité : les listes d'auteurs des références sont trop
    hétérogènes d'une base à l'autre pour être découpées entièrement de façon
    fiable. On assume cette convention plutôt que de produire du bruit.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    first = re.split(r"[;,]", value.strip())[0].strip()
    # « Chen T. » et « Chen, T. » doivent se rejoindre ; on garde le patronyme
    # et l'initiale quand elle existe.
    first = re.sub(r"\s+", " ", first)
    return first.lower() if len(first) >= 2 else None


def co_citation_authors(corpus, top_n: int = 50,
                        min_weight: int = 2) -> Dict[str, Any]:
    """Co-citation d'AUTEURS : deux auteurs cités par les mêmes documents.

    Complément du réseau de co-citation de références : là où celui-ci pointe
    des travaux précis, celui-ci fait apparaître les **écoles de pensée**.
    """
    refs = corpus.references
    if refs.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    r = refs.copy()
    r["key"] = r["ref_authors"].map(_first_cited_author)
    r = r[r["key"].notna()]
    if r.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    r = r.drop_duplicates(subset=["eid", "key"])
    counts = Counter(r["key"])
    keep = {k for k, _ in counts.most_common(top_n)}

    # Libellé : la graphie la plus fréquente parmi celles rencontrées.
    labels: Dict[str, str] = {}
    for key, g in r[r["key"].isin(keep)].groupby("key"):
        raw = g["ref_authors"].dropna().map(
            lambda s: re.split(r"[;,]", s.strip())[0].strip())
        labels[key] = raw.mode().iat[0] if not raw.empty else key

    groups = r[r["key"].isin(keep)].groupby("eid")["key"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({k: counts[k] for k in keep}), pairs, labels, min_weight)


# ---------------------------------------------------------------------------
# Pays — pour la carte
# ---------------------------------------------------------------------------

def country_map(corpus) -> pd.DataFrame:
    """Production par pays, avec le taux de collaboration internationale.

    ``sca`` (single country articles) : documents dont TOUTES les affiliations
    sont du même pays. ``mca`` (multiple country) : les autres. Le rapport
    mca/total est l'indicateur d'ouverture internationale usuel.
    """
    aff = corpus.affiliations
    aff = aff[aff["country"].notna() & (aff["country"].astype(str).str.strip() != "")]
    empty = pd.DataFrame(columns=["country", "documents", "citations",
                                  "sca", "mca", "mca_ratio"])
    if aff.empty:
        return empty

    per_doc = aff.groupby("eid")["country"].nunique()
    multi = set(per_doc[per_doc > 1].index)

    pairs = aff[["eid", "country"]].drop_duplicates()
    cites = corpus.documents[["eid"]].copy()
    cites["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                       errors="coerce").fillna(0).astype(int)
    pairs = pairs.merge(cites, on="eid", how="left")
    pairs["is_multi"] = pairs["eid"].isin(multi)

    g = (pairs.groupby("country")
              .agg(documents=("eid", "nunique"),
                   citations=("citations", "sum"),
                   mca=("is_multi", "sum"))
              .reset_index())
    g["mca"] = g["mca"].astype(int)
    g["sca"] = g["documents"] - g["mca"]
    g["mca_ratio"] = (100 * g["mca"] / g["documents"]).round(1)
    return g.sort_values("documents", ascending=False).reset_index(drop=True)


def co_city(corpus, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
    """Collaboration entre VILLES présentes sur le même document.

    Le réseau des pays montre l'ouverture internationale ; celui-ci montre
    quelque chose que le pays écrase complètement : la structure INTERNE d'un
    pays. Un corpus à 90 % marocain peut cacher un réseau Rabat–Meknès–Oujda
    dense, ou trois équipes qui s'ignorent — même pays, lecture opposée.

    Chaque lien porte un ``scope`` : « national » quand les deux villes
    partagent le pays, « international » sinon. Sans cette distinction, une
    collaboration de couloir et une collaboration transcontinentale auraient
    exactement le même aspect.
    """
    aff = corpus.affiliations
    if aff.empty or "city" not in aff.columns:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    a = aff[["eid", "city", "country"]].copy()
    a["city"] = a["city"].astype(str).str.strip()
    a = a[(a["city"] != "") & (~a["city"].str.lower().isin({"nan", "none"}))]
    if a.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    a = a.drop_duplicates(subset=["eid", "city"])
    country_of = (a.dropna(subset=["country"])
                   .groupby("city")["country"]
                   .agg(lambda s: s.mode().iat[0]).to_dict())

    counts = Counter(a["city"])
    keep = {c for c, _ in counts.most_common(top_n)}
    labels = {c: c for c in keep}

    groups = a[a["city"].isin(keep)].groupby("eid")["city"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    graph = _to_graph(Counter({c: counts[c] for c in keep}), pairs, labels, min_weight)

    for edge in graph["edges"]:
        ca, cb = country_of.get(edge["source"]), country_of.get(edge["target"])
        edge["scope"] = ("national" if ca is not None and ca == cb
                         else "international")
    for node in graph["nodes"]:
        node["country"] = country_of.get(node["id"])
    return graph
