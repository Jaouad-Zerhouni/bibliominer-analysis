"""Network measures: centralities, clustering, normalisations.

A raw bibliometric network reads badly: everyone sees the big node in the
centre, nobody sees the discreet node that links two communities. This
module computes what the eye does not see.

**The centralities do not say the same thing**, and confusing them is the
classic mistake:

  - **degree**: the number of links. Who is the most active.
  - **betweenness**: how often a node lies on the shortest path between
    two others. Who **bridges** groups that do not talk to each other. A
    node can have a mediocre degree and a huge betweenness: it is often
    the most interesting one of the corpus.
  - **closeness**: the mean distance to the rest. Who reaches everyone
    quickly.
  - **PageRank**: being cited by nodes that are themselves central counts
    more than being cited by isolated nodes.

**Normalisations**: a raw link mechanically favours frequent entities.
Relating the co-occurrence to what chance would produce (*association
strength*) corrects this bias, and really changes the map.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

#: Fixed seed for Louvain: a published map must be reproducible.
SEED = 20240101

NORMALIZATIONS = ("none", "association", "jaccard", "salton", "inclusion", "equivalence")


def _adjacency(G, order):
    """Weighted adjacency matrix, dense: these networks fit in memory."""
    import numpy as np

    index = {n: i for i, n in enumerate(order)}
    A = np.zeros((len(order), len(order)), dtype=float)
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        i, j = index[u], index[v]
        A[i, j] = A[j, i] = w
    return A


def _pagerank(A, alpha: float = 0.85, iters: int = 200, tol: float = 1e-10):
    """Weighted PageRank by power iteration.

    Written here rather than calling networkx: since version 3, `nx.pagerank`
    goes through scipy. Without scipy it raises, and a fallback block would
    return 1/n for everyone, a column that looks normal but is entirely
    meaningless, hence worse than a missing column.
    """
    import numpy as np

    n = A.shape[0]
    if n == 0:
        return np.zeros(0)
    out = A.sum(axis=1)
    dangling = out == 0
    M = np.zeros_like(A)
    nz = ~dangling
    M[nz] = A[nz] / out[nz, None]

    r = np.full(n, 1.0 / n)
    teleport = np.full(n, 1.0 / n)
    for _ in range(iters):
        new = alpha * (M.T @ r + dangling @ r * teleport) + (1 - alpha) * teleport
        if np.abs(new - r).sum() < tol:
            r = new
            break
        r = new
    total = r.sum()
    return r / total if total else r


def _eigenvector(A, iters: int = 2000, tol: float = 1e-12):
    """Eigenvector centrality by power iteration.

    The iteration is on **A + I**, not on A. Both have the same eigenvectors,
    but on a BIPARTITE graph (a star, a tree, frequent in co-authorship) A has
    two opposite dominant eigenvalues, +λ and -λ: the plain iteration then
    oscillates between two vectors without ever converging, and returned the
    one where it stopped. The +1 shift makes the dominant eigenvalue unique.
    """
    import numpy as np

    n = A.shape[0]
    if n == 0 or not A.any():
        return np.zeros(n)
    shifted = A + np.eye(n)
    x = np.full(n, 1.0 / np.sqrt(n))
    for _ in range(iters):
        y = shifted @ x
        norm = np.linalg.norm(y)
        if norm == 0:
            return np.zeros(n)
        y /= norm
        if np.abs(y - x).sum() < tol:
            x = y
            break
        x = y
    # The dominant eigenvector of a matrix with positive coefficients has a
    # constant sign (Perron-Frobenius): it is made positive.
    return np.abs(x)


def _to_networkx(graph: Dict[str, Any]):
    from .._stable import ordered_graph

    # Fixed insertion order: Louvain, even with its seed, depends on it.
    return ordered_graph(graph.get("nodes", []), graph.get("edges", []))


def normalize(graph: Dict[str, Any], method: str = "association") -> Dict[str, Any]:
    """Reweights the links according to a similarity measure.

    With ``c_ij`` the co-occurrence and ``s_i`` the total of entity *i*:

      - ``association``: c_ij / (s_i · s_j), the association strength,
        proportional to the ratio between observed and expected.
      - ``jaccard``      : c_ij / (s_i + s_j - c_ij)
      - ``salton``       : c_ij / √(s_i · s_j)
      - ``inclusion``    : c_ij / min(s_i, s_j)
      - ``equivalence``  : c_ij² / (s_i · s_j)

    Normalised weights are real numbers; the raw weight stays available in
    ``raw_weight``, because a reader often wants to know "how many documents"
    lie behind a value of 0.043.
    """
    if method == "none" or not graph.get("edges"):
        return graph

    totals = {n["id"]: float(n.get("occurrences", 0) or 0) for n in graph["nodes"]}
    out = dict(graph)
    edges = []
    for e in graph["edges"]:
        c = float(e.get("weight", 1))
        si, sj = totals.get(e["source"], 0.0), totals.get(e["target"], 0.0)
        if si <= 0 or sj <= 0:
            value = c
        elif method == "association":
            value = c / (si * sj)
        elif method == "jaccard":
            denom = si + sj - c
            value = c / denom if denom > 0 else 0.0
        elif method == "salton":
            value = c / math.sqrt(si * sj)
        elif method == "inclusion":
            value = c / min(si, sj)
        elif method == "equivalence":
            value = (c * c) / (si * sj)
        else:
            value = c
        edges.append({**e, "raw_weight": int(c), "weight": round(value, 6)})

    out["edges"] = sorted(edges, key=lambda x: -x["weight"])
    out["normalization"] = method
    return out


def annotate(graph: Dict[str, Any], communities: bool = True,
             overlay: Optional[Dict[str, float]] = None,
             resolution: float = 1.0) -> Dict[str, Any]:
    """Adds centralities, community and time overlay to the nodes.

    Every node receives ``degree_centrality``, ``betweenness``,
    ``closeness``, ``pagerank``, ``eigenvector``, ``clustering``, and if
    requested ``community`` and ``overlay_year``.

    ``overlay`` maps a node identifier to its mean year: it is the overlay
    view, which shows at once which areas of the network are recent and which
    are old, information that no node size can carry.
    """
    if not graph.get("nodes"):
        return graph

    import networkx as nx

    G = _to_networkx(graph)

    degree = nx.degree_centrality(G)
    clustering = nx.clustering(G, weight="weight")
    # The weights are SIMILARITIES: the stronger, the closer. Shortest paths
    # reason in DISTANCES, hence the inversion; without it, betweenness would
    # go through the weakest links.
    for _, _, d in G.edges(data=True):
        w = float(d.get("weight", 1)) or 1e-9
        d["distance"] = 1.0 / w

    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
    closeness = nx.closeness_centrality(G, distance="distance")
    order = list(G.nodes())
    A = _adjacency(G, order)
    pagerank = dict(zip(order, _pagerank(A)))
    eigen = dict(zip(order, _eigenvector(A)))

    membership = {}
    if communities and G.number_of_edges():
        # `resolution` is the modularity resolution: above 1 there are more and
        # smaller groups, below 1 larger groups. It has no "right" value; it is a
        # choice of granularity, which must therefore stay in the reader's hands.
        try:
            groups = nx.community.louvain_communities(
                G, weight="weight", seed=SEED, resolution=float(resolution))
        except Exception:
            groups = nx.community.greedy_modularity_communities(G, weight="weight")
        # Groups are numbered from the largest to the smallest: the number becomes
        # readable instead of arbitrary.
        from .._stable import ordered_communities
        for i, members in enumerate(ordered_communities(groups), start=1):
            for m in members:
                membership[m] = i

    out = dict(graph)
    out["nodes"] = [{
        **node,
        "degree_centrality": round(degree.get(node["id"], 0.0), 4),
        "betweenness": round(betweenness.get(node["id"], 0.0), 4),
        "closeness": round(closeness.get(node["id"], 0.0), 4),
        "pagerank": round(pagerank.get(node["id"], 0.0), 5),
        "eigenvector": round(float(eigen.get(node["id"], 0.0)), 4),
        "clustering": round(clustering.get(node["id"], 0.0), 4),
        **({"community": membership.get(node["id"], 0)} if communities else {}),
        **({"overlay_year": overlay.get(node["id"])} if overlay else {}),
    } for node in graph["nodes"]]
    return out


def _communities_on_nodes(graph: Dict[str, Any]) -> Optional[List[set]]:
    """The partition already set by `annotate`, if the nodes carry it.

    The modularity of the summary must be that of the DISPLAYED communities.
    It used to be recomputed by a Louvain at resolution 1: as soon as the user
    chose another resolution, the figure no longer described the groups they
    saw.
    """
    nodes = graph.get("nodes") or []
    if not nodes or any("community" not in n for n in nodes):
        return None
    groups: Dict[Any, set] = {}
    for n in nodes:
        groups.setdefault(n["community"], set()).add(n["id"])
    return list(groups.values())


def graph_summary(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Shape indicators of the whole network.

    ``density``: share of the possible links actually present.
    ``transitivity``: probability that two neighbours of a node are
    neighbours.
    ``components``: number of disjoint pieces; more than one signals a
    fragmented field, which a network drawing often hides.
    ``mean_path_length`` and ``diameter`` cover the **largest** component: on
    a disconnected graph they would be infinite.
    """
    if not graph.get("nodes"):
        return {"nodes": 0, "edges": 0, "density": 0.0, "transitivity": 0.0,
                "components": 0, "mean_degree": 0.0, "mean_path_length": None,
                "diameter": None, "modularity": None}

    import networkx as nx

    G = _to_networkx(graph)
    comps = list(nx.connected_components(G))
    degrees = [d for _, d in G.degree()]

    mean_path = diameter = None
    if comps:
        largest = G.subgraph(max(comps, key=len))
        if largest.number_of_nodes() > 1:
            for _, _, d in largest.edges(data=True):
                w = float(d.get("weight", 1)) or 1e-9
                d["distance"] = 1.0 / w
            mean_path = round(nx.average_shortest_path_length(largest,
                                                              weight="distance"), 3)
            diameter = int(nx.diameter(largest))

    modularity = None
    if G.number_of_edges():
        try:
            groups = _communities_on_nodes(graph)
            if groups is None:
                groups = nx.community.louvain_communities(G, weight="weight", seed=SEED)
            modularity = round(nx.community.modularity(G, groups, weight="weight"), 3)
        except Exception:
            modularity = None

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "transitivity": round(nx.transitivity(G), 4),
        "components": len(comps),
        "mean_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0.0,
        "mean_path_length": mean_path,
        "diameter": diameter,
        "modularity": modularity,
    }


def overlay_years(corpus, unit: str, level: str = "parent") -> Dict[str, float]:
    """Mean publication year associated with each node of a network.

    It is the data of the overlay view: it colours the network by time, and
    reveals at once the recent and the old areas. No node size can carry this
    information, because size is already taken by frequency.

    The returned identifiers follow EXACTLY those of `networks.build`,
    otherwise the join would be silently empty.
    """
    import pandas as pd

    years = corpus.documents[["eid", "year"]].copy()
    years["year"] = pd.to_numeric(years["year"], errors="coerce")
    years = years.dropna(subset=["year"])
    if years.empty:
        return {}

    if unit == "keywords":
        t = corpus.keywords[corpus.keywords["keyword"].notna()].copy()
        t["id"] = t["keyword"].map(str).str.strip().str.lower()
    elif unit == "authors":
        t = corpus.authors
        t = t[t["name"].notna() & (t["name"].map(str).str.strip() != "")].copy()
        t["id"] = t["scopus_id"].fillna("name:" + t["name"].map(str))
    elif unit == "countries":
        t = corpus.affiliations
        t = t[t["country"].notna() & (t["country"].map(str).str.strip() != "")].copy()
        t["id"] = t["country"]
    elif unit == "institutions":
        from ..metrics.production import _org_frame
        t = _org_frame(corpus, level).copy()
        t["id"] = t["org"]
    elif unit == "documents":
        t = years.copy()
        t["id"] = t["eid"]
    else:
        return {}

    t = t[["eid", "id"]].drop_duplicates().merge(years, on="eid", how="inner")
    if t.empty:
        return {}
    return {k: round(float(v), 1) for k, v in t.groupby("id")["year"].mean().items()}


# ---------------------------------------------------------------------------
# Layout and density map
# ---------------------------------------------------------------------------

def _shortest_paths(G, index: Dict[str, int]):
    """All shortest paths (``distance`` weights), as a matrix.

    Vectorised Floyd-Warshall: the same lengths as Dijkstra from every node,
    but computed by numpy. On a network of 250 references (7,000 links),
    Dijkstra in pure Python took 8 s while the screen waited.
    """
    import numpy as np

    size = len(index)
    D = np.full((size, size), np.inf)
    np.fill_diagonal(D, 0.0)
    for a, b, data in G.edges(data=True):
        i, j = index[a], index[b]
        if i != j:
            value = min(D[i, j], float(data["distance"]))
            D[i, j] = D[j, i] = value
    for k in range(size):
        D = np.minimum(D, D[:, k, None] + D[None, k, :])
    return D


def _mds_component(G, nodes: list) -> Dict[str, list]:
    """Places the nodes of ONE connected component, by MDS on its paths.

    A single component only has finite distances: there, the MDS works on
    gaps that all mean something.
    """
    if len(nodes) == 1:
        return {nodes[0]: [0.0, 0.0]}
    if len(nodes) == 2:
        return {nodes[0]: [-0.5, 0.0], nodes[1]: [0.5, 0.0]}

    index = {node: i for i, node in enumerate(nodes)}
    D = _shortest_paths(G.subgraph(nodes), index)
    D = (D + D.T) / 2.0

    from ..metrics.factorial import _classical_mds

    coords, _ = _classical_mds(D, 2)
    start = {node: [float(coords[i, 0]), float(coords[i, 1])]
             for node, i in index.items()}
    return _readable(G.subgraph(nodes), start)


#: FIXED seed: the spring layout is deterministic, the map does not move from
#: one opening to the next, nor between the screen and the figure.
_LAYOUT_SEED = 7


def _readable(G, start: Dict[str, list]) -> Dict[str, list]:
    """From MDS to a READABLE map.

    Classical MDS crushes a tightly linked group onto a single point: when all
    the distances of a group are almost equal (co-authors who all sign
    together), its nodes fall in the same place. Found on a real corpus: a
    co-authorship network drawn as a column of stacked discs, with
    overlapping names.

    Two passes, both deterministic:
      1. a spring layout STARTING from the MDS (fixed seed): it keeps the
         overall shape and loosens the groups;
      2. spreading of pairs that are too close, up to a minimum distance that
         depends on the number of nodes: no disc hides another.
    """
    import networkx as nx
    import numpy as np

    nodes = sorted(start)
    if len(nodes) < 3:
        return start
    pos = nx.spring_layout(G, pos={n: np.array(start[n]) for n in nodes},
                           weight="weight", iterations=150, seed=_LAYOUT_SEED)
    xy = np.array([pos[n] for n in nodes], dtype=float)
    xy = _spread(xy)
    return {n: [float(xy[i, 0]), float(xy[i, 1])] for i, n in enumerate(nodes)}


def _spread(xy, passes: int = 80):
    """Pushes apart points that are too close, without moving anything else.

    Coordinates brought to [-1, 1], minimum distance 1.6/√n: enough for a
    disc not to cover its neighbour, little enough to keep the shape. Two
    coincident points are separated along a direction fixed by their rank,
    never at random.
    """
    import numpy as np

    n = len(xy)
    span = np.ptp(xy, axis=0).max() or 1.0
    xy = (xy - xy.mean(axis=0)) / span * 2.0
    min_d = 1.6 / np.sqrt(n)
    angles = np.arange(n) * 2.399963          # golden angle: fixed directions
    fallback = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    for _ in range(passes):
        diff = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        close = dist < min_d
        if not close.any():
            break
        safe = np.where(dist == 0, 1.0, dist)
        unit = diff / safe[..., None]
        same = dist == 0
        unit[same] = (fallback[:, None, :] - fallback[None, :, :])[same]
        push = np.where(close, (min_d - np.where(np.isinf(dist), min_d, dist)) / 2, 0.0)
        xy = xy + (unit * push[..., None]).sum(axis=1)
    return xy


#: The frame targeted by the layout: the shape of a map on screen
#: (≈ 1100 × 600 px) and of an exported figure, a little wider than tall.
LAYOUT_WIDTH, LAYOUT_HEIGHT = 860.0, 480.0


def _pack(blocks: List[Dict[str, list]], gap: float = 0.6) -> Dict[str, list]:
    """Arranges the components in ROWS, in the shape of the map.

    Between two DISCONNECTED components, distance makes no sense: no path
    links them. Putting them in the same MDS meant asking it to encode a
    distance that does not exist, and it spent its first axis on it, folding
    the real structure onto a line. They are therefore placed side by side,
    which asserts nothing more than a graphical neighbourhood.

    Side by side, but not on ONE line: a co-authorship network often has
    fifteen or twenty small groups. Aligned, they formed a band twenty times
    wider than tall; framed on screen, the band became a string of stacked
    discs with unreadable names (found on a real corpus). The components
    therefore fill successive rows, from the largest to the smallest, up to
    the shape of the map.

    Each block is scaled by √n: a three-node component does not take the
    width of a twenty-node one.
    """
    import math

    import numpy as np

    boxes = []
    for block in blocks:
        ids = list(block)
        xy = np.array([block[i] for i in ids], dtype=float)
        span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-9)
        # An isolated component (all nodes at the same place) must stay visible: it
        # gets a minimum width rather than a point.
        scale = math.sqrt(len(ids)) / span if len(ids) > 1 else 1.0
        xy = (xy - xy.min(axis=0)) * scale
        width = max(float(xy[:, 0].max()), 0.4)
        height = max(float(xy[:, 1].max()), 0.4)
        boxes.append((ids, xy, width, height))

    aspect = LAYOUT_WIDTH / LAYOUT_HEIGHT
    area = sum((w + gap) * (h + gap) for _, _, w, h in boxes)
    row_width = max(max(w for _, _, w, _ in boxes), math.sqrt(area * aspect))

    rows, row, used = [], [], 0.0
    for box in boxes:
        if row and used + box[2] > row_width:
            rows.append(row)
            row, used = [], 0.0
        row.append(box)
        used += box[2] + gap
    rows.append(row)

    placed: Dict[str, list] = {}
    top = 0.0
    for row in rows:
        height = max(box[3] for box in row)
        total = sum(box[2] for box in row) + gap * (len(row) - 1)
        x = (row_width - total) / 2              # centred row
        for ids, xy, width, box_height in row:
            offset = top - (height - box_height) / 2 - box_height
            for node, (px, py) in zip(ids, xy):
                placed[node] = [x + float(px), offset + float(py)]
            x += width + gap
        top -= height + gap                      # the y axis goes up
    return placed


def _declutter(graph: Dict[str, Any], coords: Dict[str, list],
               rounds: int = 4, passes: int = 150, gap: float = 4.0) -> Dict[str, list]:
    """No disc on top of another, at the size at which the map is DRAWN.

    The spreading of `_readable` works inside a component, with the same
    minimum distance for all nodes. Yet a highly cited node is drawn four
    times wider than a small one: two large neighbouring discs still
    overlapped ("Navarro" on "Okafor").

    The coordinates are brought to the map frame (`LAYOUT_WIDTH` ×
    `LAYOUT_HEIGHT`, in screen pixels); each node there gets the radius the
    interface gives it (10 + 26·√(occurrences / max) px of diameter), and
    pairs that are too close are pushed apart, symmetrically, by what they
    lack. Deterministic: two coincident points move apart along a direction
    fixed by their rank.
    """
    import numpy as np

    ids = sorted(coords)
    if len(ids) < 2:
        return coords
    xy = np.array([coords[i] for i in ids], dtype=float)
    occurrences = {str(n["id"]): float(n.get("occurrences") or 1.0)
                   for n in graph.get("nodes", [])}
    top = max(occurrences.values(), default=1.0) or 1.0
    radius = np.array([(10.0 + 26.0 * np.sqrt(occurrences.get(i, 1.0) / top)) / 2.0
                       for i in ids])
    need = radius[:, None] + radius[None, :] + gap
    # Each pair only once (upper triangle): the others are never looked at.
    upper = np.triu(np.ones((len(ids), len(ids)), dtype=bool), 1)
    angles = np.arange(len(ids)) * 2.399963       # golden angle: fixed directions
    fallback = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    for _ in range(rounds):
        # Reframe at every round: spreading enlarges the map, and it is the
        # REFRAMED map that will be drawn.
        span = np.ptp(xy, axis=0)
        span[span == 0] = 1.0
        xy = (xy - xy.min(axis=0)) * min(LAYOUT_WIDTH / span[0], LAYOUT_HEIGHT / span[1])
        moved = False
        for _ in range(passes):
            diff = xy[:, None, :] - xy[None, :, :]
            dist2 = (diff ** 2).sum(-1)
            i, j = np.nonzero(upper & (dist2 < need ** 2))
            if not len(i):
                break
            moved = True
            dist = np.sqrt(dist2[i, j])
            unit = diff[i, j] / np.where(dist == 0, 1.0, dist)[:, None]
            same = dist == 0
            unit[same] = (fallback[i] - fallback[j])[same]
            # Each of the two covers half of the missing distance.
            push = unit * ((need[i, j] - dist) / 2.0)[:, None]
            np.add.at(xy, i, push)
            np.add.at(xy, j, -push)
        if not moved:
            break
    return {node: [round(float(xy[k, 0]), 2), round(float(xy[k, 1]), 2)]
            for k, node in enumerate(ids)}


def layout(graph: Dict[str, Any]) -> Dict[str, list]:
    """2D coordinates of the nodes, by MDS on the shortest paths.

    Why not let the browser place the nodes by force simulation? Because a
    simulation is **stochastic**: two openings of the same page give two
    different drawings, and a published map cannot move from one run to the
    next. MDS is deterministic.

    The weights are similarities; paths reason in distances, hence the
    inversion.

    **Each connected component is placed separately, then placed side by
    side.** Putting them all into a single MDS required giving a distance to
    pairs that no path connects: a large one was assigned, and this artificial
    contrast became the most salient fact of the cloud. The MDS devoted its
    first axis to it, and the real structure, the one just computed, folded
    onto a line. See `_pack`.

    The coordinates are in pixels of a `LAYOUT_WIDTH` × `LAYOUT_HEIGHT` map:
    no disc covers another there (`_declutter`).
    """
    import networkx as nx

    if not graph.get("nodes"):
        return {}

    G = _to_networkx(graph)
    for _, _, d in G.edges(data=True):
        w = float(d.get("weight", 1)) or 1e-9
        d["distance"] = 1.0 / w

    # STABLE order: components are sorted on their smallest node, otherwise two
    # runs could place them side by side in another order.
    components = sorted((sorted(c) for c in nx.connected_components(G)),
                        key=lambda c: (-len(c), c[0]))
    packed = _pack([_mds_component(G, nodes) for nodes in components])
    return _declutter(graph, packed)


def attach_layout(graph: Dict[str, Any]) -> Dict[str, Any]:
    """The network, each node carrying its ``x`` and ``y`` coordinates.

    They are exactly those of `layout`, hence those of `render_network`: the
    web interface draws the network at these positions instead of running its
    own force simulation. Without this, the map shown on screen and the
    figure produced by the package had two different shapes, and the on-screen
    map changed at every opening.

    The ``y`` axis follows the mathematical convention (upwards), like
    matplotlib; a screen rendering, whose axis goes down, must invert it.
    """
    if not graph.get("nodes"):
        return graph
    coords = layout(graph)
    out = dict(graph)
    out["nodes"] = [{**node,
                     "x": coords.get(node["id"], [0.0, 0.0])[0],
                     "y": coords.get(node["id"], [0.0, 0.0])[1]}
                    for node in graph["nodes"]]
    return out


def density_grid(graph: Dict[str, Any], coords: Dict[str, list],
                 size: int = 48, bandwidth: Optional[float] = None) -> Dict[str, Any]:
    """Density map: where is the network dense, and with what?

    Each node spreads its weight around it according to a Gaussian kernel;
    the contributions are summed into a surface. It answers a question the
    graph does not show: which **areas** of the field are saturated and which
    are deserted.

    Returns ``{"x": [...], "y": [...], "cells": [[i, j, value], ...], "max"}``,
    directly usable by a heat map.
    """
    import numpy as np

    nodes = [n for n in graph.get("nodes", []) if n["id"] in coords]
    if not nodes:
        return {"x": [], "y": [], "cells": [], "max": 0.0}

    P = np.array([coords[n["id"]] for n in nodes], dtype=float)
    w = np.array([float(n.get("occurrences", 1) or 1) for n in nodes])

    lo, hi = P.min(axis=0), P.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    # A margin: without it, the nodes at the edge are cut in half by the frame.
    lo, hi = lo - 0.08 * span, hi + 0.08 * span

    xs = np.linspace(lo[0], hi[0], size)
    ys = np.linspace(lo[1], hi[1], size)
    if bandwidth is None:
        # The typical gap between neighbours: too narrow a kernel gives a map of
        # dots, too wide a plain blob. The old rule (an eighth of the largest
        # extent) depended on where the small islands placed in the margin were: on
        # a network in several components, the extent swelled and the whole main
        # group melted into a single blob.
        if len(P) > 1:
            d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            bandwidth = float(np.median(d.min(axis=1)))
        else:
            bandwidth = float(max(hi - lo)) / 8.0
    bandwidth = max(bandwidth, 1e-6)

    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    field = np.zeros_like(gx)
    for (px, py), weight in zip(P, w):
        d2 = (gx - px) ** 2 + (gy - py) ** 2
        field += weight * np.exp(-d2 / (2.0 * bandwidth ** 2))

    peak = float(field.max()) or 1.0
    cells = [[i, j, round(float(field[i, j] / peak), 4)]
             for i in range(size) for j in range(size)]
    return {
        "x": [round(float(v), 4) for v in xs],
        "y": [round(float(v), 4) for v in ys],
        "cells": cells,
        "max": round(peak, 3),
    }
