"""Thematic evolution: how themes carry over from one period to the next.

The thematic map (`themes.py`) is a snapshot. Here several are taken, over
successive periods, and the clusters are linked to each other: a theme
that splits in two, two themes that merge, a theme that disappears. It is
the only view that really tells an **emerging** subject from a
**declining** one; the bottom-left quadrant of Callon's map confuses the
two.

The link between a cluster of one period and a cluster of the next is the
**weighted inclusion index**:

    inclusion = Σ_{shared t} min(occ_A(t), occ_B(t))
                ────────────────────────────────────────
                min(Σ occurrences of A, Σ occurrences of B)

The numerator takes the MINIMUM of the occurrences of each term shared by
the two periods: a term cannot pass on more occurrences than the poorer
period carries, which keeps the index between 0 and 1.

The division is by the **smaller** of the two, not by the union: a small
cluster entirely absorbed by a large one must give 1, because it was
indeed absorbed. Dividing by the union would crush that case, which is
precisely the one to show.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


def _period_clusters(sub, top_n: int, min_weight: int,
                     min_cluster_size: int, kind: str) -> List[dict]:
    """Co-word clusters of a sub-corpus, with their complete terms."""
    from ..networks import build as nets

    graph = nets.co_word(sub, top_n=top_n, min_weight=min_weight, kind=kind)
    if graph["n_nodes"] == 0:
        return []

    import networkx as nx

    from .._stable import ordered_communities, ordered_graph

    G = ordered_graph(graph["nodes"], graph["edges"])

    out = []
    for members in ordered_communities(
            nx.community.greedy_modularity_communities(G, weight="weight")):
        members = set(members)
        if len(members) < min_cluster_size:
            continue
        occ = {G.nodes[m]["label"]: int(G.nodes[m]["occurrences"]) for m in members}
        ordered = sorted(occ, key=lambda w: (-occ[w], w.lower(), w))
        out.append({
            "label": ordered[0],
            "terms": occ,                       # terme -> occurrences
            "n_terms": len(occ),
            "occurrences": int(sum(occ.values())),
            "top_terms": ", ".join(ordered[:8]),
        })
    # The largest cluster first: the Sankey is read by this weight.
    return sorted(out, key=lambda c: (-c["occurrences"], c["label"]))


def _slice_bounds(corpus, cuts: Optional[Sequence[int]],
                  n_periods: int) -> List[tuple]:
    """Inclusive (start, end) bounds of each period."""
    years = pd.to_numeric(corpus.documents["year"], errors="coerce").dropna()
    if years.empty:
        return []
    lo, hi = int(years.min()), int(years.max())

    if cuts:
        pts = sorted({int(c) for c in cuts if lo <= int(c) < hi})
    else:
        # Without explicit cut points: slices of comparable size, not of equal
        # duration. A corpus whose production explodes would otherwise give a
        # meaningless first slice.
        if n_periods < 2:
            return [(lo, hi)]
        qs = [years.quantile(i / n_periods) for i in range(1, n_periods)]
        pts = sorted({int(round(q)) for q in qs if lo <= int(round(q)) < hi})

    bounds, start = [], lo
    for p in pts:
        bounds.append((start, p))
        start = p + 1
    bounds.append((start, hi))
    return [b for b in bounds if b[0] <= b[1]]


def thematic_evolution(corpus, cuts: Optional[Sequence[int]] = None,
                       n_periods: int = 3, top_n: int = 60,
                       min_weight: int = 2, min_cluster_size: int = 2,
                       kind: str = "author",
                       min_inclusion: float = 0.05) -> Dict[str, Any]:
    """Thematic flows between successive periods.

    Returns ::

        {"periods": [{"label", "year_min", "year_max", "documents", "clusters"}],
         "nodes":   [{"name", "period", "label", "occurrences", "terms"}],
         "flows":   [{"source", "target", "value", "inclusion", "shared_terms"}]}

    ``source`` and ``target`` reuse the nodes' ``name``, which prefixes the
    cluster with its period: two periods can have a cluster with the same
    name, and they must above all not be merged into one.
    """
    bounds = _slice_bounds(corpus, cuts, n_periods)
    if len(bounds) < 2:
        return {"periods": [], "nodes": [], "flows": []}

    periods, nodes, per_period = [], [], []
    for lo, hi in bounds:
        sub = corpus.filter(years=(lo, hi))
        clusters = _period_clusters(sub, top_n, min_weight, min_cluster_size, kind)
        label = f"{lo}-{hi}" if lo != hi else str(lo)
        periods.append({"label": label, "year_min": lo, "year_max": hi,
                        "documents": len(sub), "clusters": len(clusters)})
        for c in clusters:
            name = f"{label}: {c['label']}"
            nodes.append({"name": name, "period": label, "label": c["label"],
                          "occurrences": c["occurrences"],
                          "n_terms": c["n_terms"], "terms": c["top_terms"]})
            per_period.append((label, name, c["terms"]))

    flows = []
    for i in range(len(bounds) - 1):
        left = [x for x in per_period if x[0] == periods[i]["label"]]
        right = [x for x in per_period if x[0] == periods[i + 1]["label"]]
        for _, src, a in left:
            for _, dst, b in right:
                shared = set(a) & set(b)
                if not shared:
                    continue
                num = sum(min(a[w], b[w]) for w in shared)
                den = min(sum(a.values()), sum(b.values()))
                inc = num / den if den else 0.0
                if inc < min_inclusion:
                    continue
                flows.append({
                    "source": src, "target": dst,
                    "value": int(num),
                    "inclusion": round(inc, 3),
                    "shared_terms": ", ".join(sorted(shared)[:6]),
                })

    flows.sort(key=lambda f: -f["value"])
    # A node without any flow has nothing to show in a Sankey: it would make an
    # orphan column. Only the linked ones are kept.
    linked = {f["source"] for f in flows} | {f["target"] for f in flows}
    nodes = [n for n in nodes if n["name"] in linked]
    return {"periods": periods, "nodes": nodes, "flows": flows}
