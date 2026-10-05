"""Bibliometric networks: co-citation, co-word, co-authorship.

The principle is the same in all three cases: two entities are **linked**
when they appear together in the same document. Only the entity changes:
a cited reference, a keyword, an author.

    document 1: [A, B, C]  ->  links A-B, A-C, B-C
    document 2: [A, B]     ->  link  A-B  (weight 2 in total)

Two safeguards, without which these networks become unusable:

  - **Nodes are limited first** to the most frequent entities. A corpus
    of 3,000 documents produces hundreds of thousands of pairs: the
    computation becomes slow and the graph unreadable.
  - **Weak links are filtered** (minimum weight); otherwise the graph is
    a cloud of single links without structure.

The functions return a dictionary `{"nodes": [...], "edges": [...]}`,
directly displayable: this module does not draw, it provides the
structure.
"""

from __future__ import annotations

from .._stable import top_by_count

import re
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional

import pandas as pd


def _pairs_from_groups(groups: Dict[Any, List[str]],
                       keep: Optional[set] = None) -> Counter:
    """Counts co-occurrences. `keep` restricts to the retained entities."""
    pairs: Counter = Counter()
    for items in groups.values():
        uniq = sorted({i for i in items if i and (keep is None or i in keep)})
        if len(uniq) < 2:
            continue
        pairs.update(combinations(uniq, 2))
    return pairs


def _to_graph(counts: Counter, pairs: Counter,
              labels: Dict[str, str], min_weight: int) -> Dict[str, Any]:
    """Assembles nodes + links, and removes the nodes that became isolated.

    A node without any link after filtering adds nothing to the reading of a
    network: it is removed rather than leaving floating points.
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
    # COMPLETE order, ties broken by the identifier: `counts` is often filled
    # from a set, whose order Python changes at every run. The order of the
    # nodes decides the drawing order and which ones get a label: the figure
    # changed from one launch to the next.
    nodes.sort(key=lambda n: (-n["occurrences"], str(n["id"])))
    edges.sort(key=lambda e: (-e["weight"], str(e["source"]), str(e["target"])))

    return {"nodes": nodes, "edges": edges,
            "n_nodes": len(nodes), "n_edges": len(edges)}


# ---------------------------------------------------------------------------
# Co-citation
# ---------------------------------------------------------------------------

def _ref_key(row) -> Optional[str]:
    """Identity of a cited reference: its DOI, otherwise its normalised title.

    The DOI is reliable; the title only is after normalisation (case,
    punctuation, spaces). A reference with neither is left out: it cannot be
    matched with another without risking merging different works.
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
    """Co-citation network: two references cited by the same documents.

    It reveals the **intellectual foundations** of the corpus: the works that
    authors draw on together form thematic groupings.
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
    keep = {k for k, _ in top_by_count(counts, top_n)}

    # The same reference is written differently from one citing article to
    # another ("Minku L.L." here, "Mahmood Y." there). The label takes the MOST
    # FREQUENT spelling, on a tie the first in alphabetical order (`mode`
    # sorts), never the first one met, which depended on the order of the
    # export's rows.
    labels: Dict[str, str] = {}
    for key, g in r[r["key"].isin(keep)].groupby("key"):
        title = g["ref_title"].dropna().mode()
        year = g["ref_year"].dropna().mode()
        authors = g["ref_authors"].dropna().mode()
        name = title.iat[0] if not title.empty else key
        first = authors.iat[0].split(",")[0].strip() if not authors.empty else ""
        y = int(year.iat[0]) if not year.empty else None
        labels[key] = "%s%s · %s" % (first + " " if first else "",
                                     "(%d)" % y if y else "", name[:70])

    groups = r[r["key"].isin(keep)].groupby("eid")["key"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({k: counts[k] for k in keep}), pairs, labels, min_weight)


# ---------------------------------------------------------------------------
# Co-mots
# ---------------------------------------------------------------------------

def co_word(corpus, top_n: int = 50, min_weight: int = 2,
            kind: str = "author") -> Dict[str, Any]:
    """Keyword co-occurrence network, the **thematic structure**."""
    k = corpus.keywords
    if kind != "all":
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna()]
    if k.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    k = k.copy()
    k["norm"] = k["keyword"].map(str).str.strip().str.lower()
    k = k.drop_duplicates(subset=["eid", "norm"])

    counts = Counter(k["norm"])
    keep = {w for w, _ in top_by_count(counts, top_n)}
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
    """Collaboration network: two authors signing the same documents.

    `min_weight` is 1 by default: a single co-signature is already a real
    collaboration, unlike an isolated co-citation.
    """
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    if a.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    a = a.copy()
    a["key"] = a["scopus_id"].fillna("name:" + a["name"].map(str))
    a = a.drop_duplicates(subset=["eid", "key"])

    counts = Counter(a["key"])
    keep = {k for k, _ in top_by_count(counts, top_n)}
    labels = (a[a["key"].isin(keep)]
              .groupby("key")["name"]
              .agg(lambda s: s.mode().iat[0]).to_dict())

    groups = a[a["key"].isin(keep)].groupby("eid")["key"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({k: counts[k] for k in keep}), pairs, labels, min_weight)


def co_institution(corpus, top_n: int = 50, min_weight: int = 1,
                   level: str = "parent") -> Dict[str, Any]:
    """Collaboration between ORGANISATIONS present on the same document.

    `level` is "parent" (institutions) or "subparent" (internal units). At
    unit level, the network shows which laboratories work together,
    information that the institution level hides completely when two teams of
    the same university collaborate.

    Researchers without an affiliation are left out: "Independent researcher"
    is not an organisation and would pollute the centre of the network.
    """
    from ..metrics.production import _org_frame

    aff = _org_frame(corpus, level)
    if aff.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    aff = aff.drop_duplicates(subset=["eid", "org"])
    counts = Counter(aff["org"])
    keep = {k for k, _ in top_by_count(counts, top_n)}
    labels = {k: k for k in keep}

    groups = aff[aff["org"].isin(keep)].groupby("eid")["org"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({k: counts[k] for k in keep}), pairs, labels, min_weight)


def co_country(corpus, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
    """Collaboration between COUNTRIES: two countries signing the same document.

    It is the network that shows the international integration of the
    corpus.
    """
    aff = corpus.affiliations
    aff = aff[aff["country"].notna() & (aff["country"].map(str).str.strip() != "")]
    if aff.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    aff = aff.drop_duplicates(subset=["eid", "country"])
    counts = Counter(aff["country"])
    keep = {k for k, _ in top_by_count(counts, top_n)}
    labels = {k: k for k in keep}

    groups = aff[aff["country"].isin(keep)].groupby("eid")["country"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({k: counts[k] for k in keep}), pairs, labels, min_weight)


def bibliographic_coupling(corpus, top_n: int = 50,
                           min_weight: int = 2) -> Dict[str, Any]:
    """Bibliographic coupling: two DOCUMENTS sharing references.

    It is the mirror of co-citation. Co-citation looks backwards (which older
    works are cited together) and evolves over time. Coupling looks at the
    present: two articles drawing on the same sources probably deal with the
    same subject, and this link is **fixed** as soon as they are published.

    The weight of a link is the number of shared references.
    """
    refs = corpus.references
    if refs.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    r = refs.copy()
    r["key"] = r.apply(_ref_key, axis=1)
    r = r[r["key"].notna()].drop_duplicates(subset=["eid", "key"])
    if r.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    # The documents with the most identifiable references are kept: they are
    # the ones that can really couple. A document with two references only
    # brings noise.
    per_doc = r.groupby("eid")["key"].apply(set)
    keep = set(per_doc.map(len).sort_values(ascending=False, kind="stable").head(top_n).index)

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
    """First author of a cited reference, normalised.

    Author co-citation analysis (ACA) is based by convention on the FIRST
    cited author: the author lists of references are too heterogeneous from
    one database to another to be split entirely and reliably. This
    convention is assumed rather than producing noise.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    first = re.split(r"[;,]", value.strip())[0].strip()
    # "Chen T." and "Chen, T." must join; the surname is kept, with the initial
    # when there is one.
    first = re.sub(r"\s+", " ", first)
    return first.lower() if len(first) >= 2 else None


def co_citation_authors(corpus, top_n: int = 50,
                        min_weight: int = 2) -> Dict[str, Any]:
    """AUTHOR co-citation: two authors cited by the same documents.

    A complement to the reference co-citation network: where that one points
    to specific works, this one reveals **schools of thought**.
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
    keep = {k for k, _ in top_by_count(counts, top_n)}

    # Label: the most frequent spelling among those met.
    labels: Dict[str, str] = {}
    for key, g in r[r["key"].isin(keep)].groupby("key"):
        raw = g["ref_authors"].dropna().map(
            lambda s: re.split(r"[;,]", s.strip())[0].strip())
        labels[key] = raw.mode().iat[0] if not raw.empty else key

    groups = r[r["key"].isin(keep)].groupby("eid")["key"].apply(list).to_dict()
    pairs = _pairs_from_groups(groups, keep)
    return _to_graph(Counter({k: counts[k] for k in keep}), pairs, labels, min_weight)


# ---------------------------------------------------------------------------
# Countries, for the map
# ---------------------------------------------------------------------------

def country_map(corpus) -> pd.DataFrame:
    """Production per country, with the international collaboration rate.

    ``sca`` (single country articles): documents whose affiliations are ALL
    from the same country. ``mca`` (multiple country): the others. The
    mca/total ratio is the usual indicator of international openness.

    ``map_name``: the name of the country on the base map (world-atlas), empty
    if it is too small to appear on it; ``lon``/``lat``: where to place it
    (centre of the territory, or capital of a small country). Empty for a
    name that is not a recognised country. So the interface and the figures
    place countries the same way, without a table of names to maintain on
    each side.
    """
    aff = corpus.affiliations
    aff = aff[aff["country"].notna() & (aff["country"].map(str).str.strip() != "")]
    empty = pd.DataFrame(columns=["country", "documents", "citations",
                                  "sca", "mca", "mca_ratio", "map_name", "lon", "lat"])
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
    from ..io.countries import atlas_name, position
    g["map_name"] = g["country"].map(atlas_name)
    where = g["country"].map(position)
    g["lon"] = where.map(lambda p: round(p[0], 4) if p else None)
    g["lat"] = where.map(lambda p: round(p[1], 4) if p else None)
    return g.sort_values("documents", ascending=False, kind="stable").reset_index(drop=True)


def co_city(corpus, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
    """Collaboration between CITIES present on the same document.

    The country network shows international openness; this one shows
    something the country completely crushes: the INTERNAL structure of a
    country. A 90 % Moroccan corpus can hide a dense Rabat-Meknes-Oujda
    network, or three teams ignoring each other: same country, opposite
    readings.

    Each link carries a ``scope``: "national" when the two cities share the
    country, "international" otherwise. Without this distinction, a corridor
    collaboration and a transcontinental one would look exactly the same.
    """
    aff = corpus.affiliations
    if aff.empty or "city" not in aff.columns:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    a = aff[["eid", "city", "country"]].copy()
    a["city"] = a["city"].map(str).str.strip()
    a = a[(a["city"] != "") & (~a["city"].str.lower().isin({"nan", "none"}))]
    if a.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    a = a.drop_duplicates(subset=["eid", "city"])
    country_of = (a.dropna(subset=["country"])
                   .groupby("city")["country"]
                   .agg(lambda s: s.mode().iat[0]).to_dict())

    counts = Counter(a["city"])
    keep = {c for c, _ in top_by_count(counts, top_n)}
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
