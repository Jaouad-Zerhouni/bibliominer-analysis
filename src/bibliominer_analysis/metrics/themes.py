"""Thematic analysis: Callon's strategic map and emerging topics.

The **thematic map** (Callon, 1991) places each cluster of keywords on two
axes:

  - **centrality**: the intensity of the cluster's links with the OTHER
    clusters. It measures how connected the theme is to the rest of the
    field.
  - **density**: the intensity of the links INSIDE the cluster. It
    measures how developed, how structured the theme is.

Cutting at the medians gives four quadrants:

    density ↑
      niche themes      │   motor themes
      (developed,       │   (developed,
       weakly linked)   │    central)
    ──────────────────── ┼────────────────────► centrality
      emerging or       │   basic themes
      declining         │   (central,
                        │    little developed)

The bottom-left quadrant is ambiguous by construction: a theme that is
little developed and weakly linked may be appearing **or** disappearing.
The chronological reading decides, not the map.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def trend_topics(corpus, n: int = 25, min_documents: int = 2,
                 kind: str = "author") -> pd.DataFrame:
    """Topics and their position in time.

    For each term: the **median** of the years in which it appears, framed by
    the first and third quartiles. The median says when the term was used the
    most; the interquartile range says whether it is concentrated on one
    period or spread out.

    The median is preferred to the mean: a single old occurrence must not
    pull the whole term into the past.
    """
    k = corpus.keywords
    if kind != "all":
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna()]
    empty = pd.DataFrame(columns=["keyword", "documents", "year_q1",
                                  "year_median", "year_q3"])
    if k.empty:
        return empty

    years = corpus.documents[["eid", "year"]].copy()
    years["year"] = pd.to_numeric(years["year"], errors="coerce")
    k = k.merge(years, on="eid", how="left").dropna(subset=["year"])
    if k.empty:
        return empty

    k = k.copy()
    k["norm"] = k["keyword"].map(str).str.strip().str.lower()
    k = k.drop_duplicates(subset=["eid", "norm"])

    g = (k.groupby("norm")
           .agg(keyword=("keyword", lambda s: s.mode().iat[0]),
                documents=("eid", "nunique"),
                year_q1=("year", lambda s: float(np.percentile(s, 25))),
                year_median=("year", "median"),
                year_q3=("year", lambda s: float(np.percentile(s, 75))))
           .reset_index(drop=True))

    g = g[g["documents"] >= min_documents]
    for c in ("year_q1", "year_median", "year_q3"):
        g[c] = g[c].round().astype("Int64")

    return (g.sort_values(["year_median", "documents"], ascending=[True, False], kind="stable")
             .reset_index(drop=True)
             .head(n))


def thematic_map(corpus, top_n: int = 100, min_weight: int = 2,
                 min_cluster_size: int = 2,
                 kind: str = "author") -> Dict[str, Any]:
    """Callon's strategic map, computed on the co-word network.

    Returns ``{"clusters": DataFrame, "medians": {...}}``.

    Formulas of Callon et al. (1991):

        centrality = 10 × Σ e_kh   (links between a term of the cluster and a term outside it)
        density    = 100 × Σ e_ij / w   (internal links, w = number of terms)

    where **e_ij is the equivalence index** c_ij² / (c_i · c_j), between 0 and
    1, not the raw co-occurrence c_ij. The constants 10 and 100 are calibrated
    for this index. Fixed defect: the raw weights were used, which favoured
    frequent terms and could move a theme to another quadrant, not only change
    the scale of the axes.

    Clusters are detected by greedy modularity on the same normalised
    network: it is a deterministic algorithm, so two runs give the same map,
    which is essential for a published result.
    """
    from ..networks import build as nets
    from ..networks.analysis import normalize
    from .._stable import ordered_communities, ordered_graph

    graph = normalize(nets.co_word(corpus, top_n=top_n, min_weight=min_weight,
                                   kind=kind), "equivalence")
    empty = pd.DataFrame(columns=["cluster", "label", "terms", "n_terms",
                                  "occurrences", "centrality", "density",
                                  "quadrant"])
    if graph["n_nodes"] == 0:
        return {"clusters": empty, "medians": {"centrality": 0.0, "density": 0.0}}

    import networkx as nx

    # FIXED order of construction and numbering: without it, greedy modularity
    # broke its ties by the hashing of strings, different at every run (see
    # `_stable`).
    G = ordered_graph(graph["nodes"], graph["edges"])
    communities = ordered_communities(
        nx.community.greedy_modularity_communities(G, weight="weight"))

    # The `min_weight` threshold is used to DETECT the clusters, not to measure
    # them. Centrality and density are computed on all co-occurrences between
    # the retained terms (Callon: every external link counts). Measured on the
    # thresholded network, a small corpus broke into islands: 13 themes out of
    # 15 with zero centrality, a median of 0, and no theme left as "niche" or
    # "emerging", meaningless quadrants.
    kept = set(G.nodes)
    full = normalize(nets.co_word(corpus, top_n=top_n, min_weight=1, kind=kind),
                     "equivalence")
    measured = [(e["source"], e["target"], float(e.get("weight", 1)))
                for e in full["edges"]
                if e["source"] in kept and e["target"] in kept]

    rows = []
    for i, members in enumerate(communities, start=1):
        members = set(members)
        if len(members) < min_cluster_size:
            continue

        internal = external = 0.0
        for u, v, w in measured:
            u_in, v_in = u in members, v in members
            if u_in and v_in:
                internal += w
            elif u_in or v_in:
                external += w

        occ = {n: G.nodes[n]["occurrences"] for n in members}
        # At equal frequency, alphabetical order: it chooses the cluster's NAME
        # when two terms are tied.
        ordered = sorted(members, key=lambda n: (-occ[n], str(G.nodes[n]["label"]).lower(), str(n)))
        labels = [G.nodes[n]["label"] for n in ordered]

        rows.append({
            "cluster": i,
            # The cluster is named after its most frequent term: that is the reading
            # convention of thematic maps.
            "label": labels[0],
            "terms": ", ".join(labels[:8]),
            "n_terms": len(members),
            "occurrences": int(sum(occ.values())),
            "centrality": round(10.0 * external, 2),
            "density": round(100.0 * internal / len(members), 2),
        })

    if not rows:
        return {"clusters": empty, "medians": {"centrality": 0.0, "density": 0.0}}

    df = pd.DataFrame(rows)
    med_c = float(df["centrality"].median())
    med_d = float(df["density"].median())

    def quadrant(r) -> str:
        high_c, high_d = r["centrality"] >= med_c, r["density"] >= med_d
        if high_c and high_d:
            return "motor"
        if high_c and not high_d:
            return "basic"
        if not high_c and high_d:
            return "niche"
        return "emerging_declining"

    df["quadrant"] = df.apply(quadrant, axis=1)
    df = df.sort_values("occurrences", ascending=False, kind="stable").reset_index(drop=True)
    return {"clusters": df, "medians": {"centrality": round(med_c, 2),
                                        "density": round(med_d, 2)}}
