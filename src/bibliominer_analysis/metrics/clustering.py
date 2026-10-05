"""Clustering documents by **bibliographic coupling**.

The thematic map of `themes.py` groups *words*. Here *documents* are
grouped: two articles that draw on the same references probably deal with
the same subject, even when they do not use the same vocabulary. That is
the difference that matters: coupling catches proximities that keywords
miss, especially between communities that name the same thing
differently.

Each cluster is placed on two axes, as in Callon's strategic diagram:

  - **centrality**: the strength of the cluster's links to the OTHER
    clusters;
  - **impact**: the mean citations of its documents.

The link is normalised by **association strength** before clustering.
Without it, an article with three hundred references would couple
strongly with everyone through size alone, and would form an artificial
cluster.
"""

from __future__ import annotations

from .._stable import ordered_communities, ordered_graph, top_by_count

from collections import Counter
from typing import Any, Dict

import pandas as pd


def clustering_by_coupling(corpus, top_n: int = 100, min_weight: int = 3,
                           min_cluster_size: int = 3,
                           impact: str = "local") -> Dict[str, Any]:
    """Clusters of coupled documents, with centrality and impact.

    `impact` is ``"local"`` (citations received within the corpus) or
    ``"global"`` (Scopus citations). Local is the most telling: it measures
    the cluster's influence **on the field under study**.

    Returns ``{"clusters": DataFrame, "medians": {...}, "n_documents": int}``.
    """
    from ..networks import build as nets
    from ..networks.analysis import normalize
    from .local import local_citations

    graph = normalize(nets.bibliographic_coupling(corpus, top_n=top_n,
                                                  min_weight=min_weight),
                      "association")
    cols = ["cluster", "label", "terms", "documents", "centrality", "impact",
            "mean_year", "top_document", "quadrant"]
    empty = pd.DataFrame(columns=cols)
    if graph["n_nodes"] == 0 or graph["n_edges"] == 0:
        return {"clusters": empty, "medians": {"centrality": 0.0, "impact": 0.0},
                "n_documents": 0}

    import networkx as nx

    G = ordered_graph(graph["nodes"], graph["edges"], node_attrs=False)

    try:
        groups = nx.community.louvain_communities(G, weight="weight", seed=20240101)
    except Exception:
        groups = nx.community.greedy_modularity_communities(G, weight="weight")

    # Per-document data, to describe the clusters afterwards.
    docs = corpus.documents[["eid", "title", "year"]].copy()
    docs["year"] = pd.to_numeric(docs["year"], errors="coerce")
    docs["global"] = pd.to_numeric(corpus.documents["cited_by"],
                                   errors="coerce").fillna(0).astype(int)
    docs = docs.merge(local_citations(corpus), on="eid", how="left")
    docs["local"] = docs["local_citations"].fillna(0).astype(int)
    docs = docs.set_index("eid")

    kw = corpus.keywords.copy()
    kw = kw[kw["keyword"].notna()]
    kw["norm"] = kw["keyword"].map(str).str.strip().str.lower()

    rows = []
    for i, members in enumerate(ordered_communities(groups), start=1):
        if len(members) < min_cluster_size:
            continue

        external = 0.0
        for u, v, data in G.edges(data=True):
            u_in, v_in = u in members, v in members
            if u_in != v_in:
                external += float(data.get("weight", 0.0))

        present = sorted(m for m in members if m in docs.index)
        sub = docs.loc[present] if present else docs.iloc[0:0]
        measure = sub[impact] if impact in ("local", "global") else sub["local"]

        terms = Counter(kw[kw["eid"].isin(members)]["norm"])
        top_terms = [t for t, _ in top_by_count(terms, 8)]
        # The most cited document of the cluster; on a tie, the smallest
        # identifier. `idxmax` took the first one met, whose order came from a set
        # (random hashing).
        best = (sub[impact].sort_index(kind="stable")
                .sort_values(ascending=False, kind="stable").index[0]
                if not sub.empty and sub[impact].max() > 0 else None)

        rows.append({
            "cluster": i,
            "label": top_terms[0] if top_terms else f"cluster {i}",
            "terms": ", ".join(top_terms),
            "documents": len(members),
            "centrality": round(100.0 * external, 3),
            "impact": round(float(measure.mean()), 2) if not sub.empty else 0.0,
            "mean_year": int(round(sub["year"].mean())) if sub["year"].notna().any() else None,
            "top_document": None if best is None else str(docs.loc[best, "title"])[:90],
        })

    if not rows:
        return {"clusters": empty, "medians": {"centrality": 0.0, "impact": 0.0},
                "n_documents": graph["n_nodes"]}

    df = pd.DataFrame(rows)
    med_c = float(df["centrality"].median())
    med_i = float(df["impact"].median())

    def quadrant(r) -> str:
        high_c, high_i = r["centrality"] >= med_c, r["impact"] >= med_i
        if high_c and high_i:
            return "motor"
        if high_c and not high_i:
            return "basic"
        if not high_c and high_i:
            return "niche"
        return "emerging_declining"

    df["quadrant"] = df.apply(quadrant, axis=1)
    df = df.sort_values("documents", ascending=False, kind="stable").reset_index(drop=True)
    return {
        "clusters": df[cols],
        "medians": {"centrality": round(med_c, 3), "impact": round(med_i, 2)},
        "n_documents": graph["n_nodes"],
    }
