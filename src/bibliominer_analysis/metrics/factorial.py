"""Conceptual structure: factorial analysis of keywords.

Co-word networks show *who is linked to whom*. Factorial analysis answers
another question: **along which axes is the field structured?** It
projects the terms onto a plane where distance has a meaning (two close
terms appear in the same documents), then groups are searched for there.

Two methods, two points of view:

  - **CA** (correspondence analysis) on the documents × terms table. The
    axes are the directions of greatest *inertia*, i.e. of greatest
    departure from independence. We know what share of the information
    each axis carries, which MDS does not give.
  - **MDS** (multidimensional scaling) on a dissimilarity matrix. It only
    tries to preserve the pairwise distances: more faithful locally, but
    the axes have no interpretation of their own.

Everything is computed in numpy: SVD for the CA, eigendecomposition for
the MDS, k-means for the groups. No additional dependency, and above all
**deterministic** results: an article must be reproducible identically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

#: Fixed seed: two runs must give exactly the same map.
SEED = 20240101


# ---------------------------------------------------------------------------
# Starting table
# ---------------------------------------------------------------------------

def _incidence(corpus, kind: str = "author", top_n: int = 50,
               min_documents: int = 2):
    """Binary documents × terms table, and the occurrences of each term."""
    k = corpus.keywords
    if kind in ("author", "index"):
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna() & (k["keyword"].map(str).str.strip() != "")]
    if k.empty:
        return None, None, None

    k = k.copy()
    k["norm"] = k["keyword"].map(str).str.strip().str.lower()
    k = k.drop_duplicates(subset=["eid", "norm"])

    counts = k.groupby("norm")["eid"].nunique()
    counts = counts[counts >= min_documents].sort_values(ascending=False, kind="stable").head(top_n)
    if counts.empty:
        return None, None, None

    k = k[k["norm"].isin(counts.index)]
    labels = (k.groupby("norm")["keyword"]
                .agg(lambda s: s.mode().iat[0]).reindex(counts.index))

    matrix = pd.crosstab(k["eid"], k["norm"]).reindex(columns=counts.index,
                                                      fill_value=0)
    matrix = (matrix > 0).astype(float)
    # A document without any of the retained terms brings nothing and would
    # break the CA (zero row mass).
    matrix = matrix.loc[matrix.sum(axis=1) > 0]
    if matrix.empty or matrix.shape[1] < 3:
        return None, None, None
    return matrix, labels, counts


# ---------------------------------------------------------------------------
# Correspondence analysis
# ---------------------------------------------------------------------------

def _correspondence(matrix: pd.DataFrame, n_dims: int = 2):
    """Classical CA by SVD. Returns (column coordinates, explained inertias)."""
    N = matrix.to_numpy(dtype=float)
    total = N.sum()
    if total <= 0:
        return None, None

    P = N / total
    r = P.sum(axis=1)          # row masses (documents)
    c = P.sum(axis=0)          # column masses (terms)
    ok_r, ok_c = r > 0, c > 0
    P, r, c = P[np.ix_(ok_r, ok_c)], r[ok_r], c[ok_c]

    # Matrix of standardised residuals: the departure from the independence
    # model, weighted by the masses. It is this matrix that the CA decomposes.
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)

    inertia = sigma ** 2
    total_inertia = inertia.sum()
    if total_inertia <= 0:
        return None, None
    explained = 100.0 * inertia / total_inertia

    k = min(n_dims, sigma.size)
    # Principal coordinates of the columns: D_c^{-1/2} V Σ
    coords = (Vt[:k].T * sigma[:k]) / np.sqrt(c)[:, None]

    full = np.full((matrix.shape[1], k), np.nan)
    full[np.where(ok_c)[0]] = coords
    return full, explained[:k]


# ---------------------------------------------------------------------------
# Positionnement multidimensionnel (MDS classique)
# ---------------------------------------------------------------------------

def _classical_mds(dissimilarity: np.ndarray, n_dims: int = 2):
    """Classical (Torgerson) MDS: double centring then eigendecomposition."""
    D2 = dissimilarity ** 2
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # B is symmetric: eigh is more stable and faster than eig, and returns the
    # eigenvalues already sorted.
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    positive = np.clip(values[:n_dims], 0, None)
    coords = vectors[:, :n_dims] * np.sqrt(positive)
    total = np.clip(values, 0, None).sum()
    explained = (100.0 * positive / total) if total > 0 else np.zeros(n_dims)
    return coords, explained


def _dissimilarity(matrix: pd.DataFrame) -> np.ndarray:
    """1 - Salton index between terms, from the co-occurrences."""
    X = matrix.to_numpy(dtype=float)
    co = X.T @ X
    diag = np.diag(co).astype(float)
    denom = np.sqrt(np.outer(diag, diag))
    with np.errstate(divide="ignore", invalid="ignore"):
        salton = np.where(denom > 0, co / denom, 0.0)
    np.fill_diagonal(salton, 1.0)
    return np.clip(1.0 - salton, 0.0, None)


# ---------------------------------------------------------------------------
# Deterministic k-means
# ---------------------------------------------------------------------------

def _kmeans(X: np.ndarray, k: int, seed: int = SEED, iters: int = 100):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if k >= n:
        return np.arange(n)

    # k-means++: the first centre at random, the following ones proportionally
    # to the squared distance to the nearest centre. Without it, two runs
    # sometimes converge to different partitions.
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1 / n)
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = X[labels == j]
            if members.size:
                C[j] = members.mean(axis=0)
    return labels


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette: used to choose k, not to judge absolute quality."""
    uniq = np.unique(labels)
    if uniq.size < 2 or uniq.size >= X.shape[0]:
        return -1.0
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    scores = []
    for i in range(X.shape[0]):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == other].mean()
                for other in uniq if other != labels[i])
        denom = max(a, b)
        if denom > 0:
            scores.append((b - a) / denom)
    return float(np.mean(scores)) if scores else -1.0


def _choose_k(X: np.ndarray, k_max: int = 8) -> int:
    best_k, best_score = 2, -2.0
    for k in range(2, min(k_max, X.shape[0] - 1) + 1):
        score = _silhouette(X, _kmeans(X, k))
        if score > best_score:
            best_k, best_score = k, score
    return best_k


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def conceptual_structure(corpus, method: str = "CA", kind: str = "author",
                         top_n: int = 50, min_documents: int = 2,
                         n_clusters: Optional[int] = None) -> Dict[str, Any]:
    """Factorial map of the keywords, with automatic clustering.

    `method` is ``"CA"`` (correspondence) or ``"MDS"``.
    `n_clusters` set to ``None`` lets the silhouette choose the number of
    groups.

    Returns ::

        {"method", "terms": DataFrame, "explained": [dim1 %, dim2 %],
         "clusters": [{"cluster", "label", "terms", "size", "occurrences"}]}

    ``terms`` carries ``keyword``, ``dim1``, ``dim2``, ``occurrences``,
    ``cluster``.
    """
    matrix, labels, counts = _incidence(corpus, kind, top_n, min_documents)
    empty = {"method": method, "terms": pd.DataFrame(
        columns=["keyword", "dim1", "dim2", "occurrences", "cluster"]),
        "explained": [], "clusters": []}
    if matrix is None:
        return empty

    if method.upper() == "MDS":
        coords, explained = _classical_mds(_dissimilarity(matrix), 2)
    else:
        coords, explained = _correspondence(matrix, 2)
        if coords is None:
            return empty

    keep = ~np.isnan(coords).any(axis=1)
    coords = coords[keep]
    terms = matrix.columns[keep]
    if coords.shape[0] < 3:
        return empty

    k = n_clusters if n_clusters and n_clusters >= 2 else _choose_k(coords)
    assign = _kmeans(coords, k)

    df = pd.DataFrame({
        "keyword": [labels.get(t, t) for t in terms],
        "dim1": np.round(coords[:, 0], 4),
        "dim2": np.round(coords[:, 1], 4),
        "occurrences": [int(counts.get(t, 0)) for t in terms],
        "cluster": assign.astype(int) + 1,
    }).sort_values(["cluster", "occurrences"], ascending=[True, False], kind="stable")

    groups = []
    for cid, g in df.groupby("cluster"):
        groups.append({
            "cluster": int(cid),
            # The most frequent term names the group, as on the thematic map: the two
            # views thus stay comparable.
            "label": str(g.iloc[0]["keyword"]),
            "terms": ", ".join(g["keyword"].head(8)),
            "size": int(len(g)),
            "occurrences": int(g["occurrences"].sum()),
        })

    return {
        "method": "MDS" if method.upper() == "MDS" else "CA",
        "terms": df.reset_index(drop=True),
        "explained": [round(float(x), 1) for x in np.asarray(explained)[:2]],
        "clusters": sorted(groups, key=lambda c: -c["occurrences"]),
    }


# ---------------------------------------------------------------------------
# Dendrogram of themes
# ---------------------------------------------------------------------------

def _average_linkage(D: np.ndarray):
    """Agglomerative hierarchical clustering, average linkage (UPGMA).

    Returns the list of merges ``(a, b, distance, size)``, in the order in
    which they happen, the format of a classical linkage matrix.

    **Average** linkage rather than single linkage: the latter produces
    chaining, where a group stretches step by step without ever being
    compact, and the dendrogram becomes unreadable.
    """
    n = D.shape[0]
    active = {i: [i] for i in range(n)}
    dist = D.astype(float).copy()
    np.fill_diagonal(dist, np.inf)
    merges = []
    next_id = n
    ids = {i: i for i in range(n)}

    while len(active) > 1:
        keys = sorted(active)
        sub = dist[np.ix_(keys, keys)]
        flat = int(np.argmin(sub))
        i, j = divmod(flat, len(keys))
        a, b = keys[i], keys[j]
        d = float(sub[i, j])

        members = active[a] + active[b]
        merges.append((ids[a], ids[b], d, len(members)))

        # Distance of the new group: mean weighted by the group sizes, which is
        # exactly the definition of average linkage.
        for k in keys:
            if k in (a, b):
                continue
            na, nb = len(active[a]), len(active[b])
            new = (na * dist[a, k] + nb * dist[b, k]) / (na + nb)
            dist[a, k] = dist[k, a] = new

        del active[b]
        active[a] = members
        ids[a] = next_id
        next_id += 1
        dist[b, :] = np.inf
        dist[:, b] = np.inf

    return merges


def topic_dendrogram(corpus, kind: str = "author", top_n: int = 40,
                     min_documents: int = 2,
                     max_clusters: int = 6) -> Dict[str, Any]:
    """Hierarchical tree of the terms, cut into `max_clusters` groups.

    The dendrogram shows what a factorial map hides: **at which level** two
    themes join. Two terms can be neighbours on the plane and only belong to
    the same group at the very last moment; the tree says so, the scatter
    plot does not.

    Returns ``{"tree": {...}, "clusters": [...], "n_terms": int}``, the tree
    being directly usable by a "tree" type rendering.
    """
    matrix, labels, counts = _incidence(corpus, kind, top_n, min_documents)
    empty = {"tree": None, "clusters": [], "n_terms": 0}
    if matrix is None:
        return empty

    terms = list(matrix.columns)
    D = _dissimilarity(matrix)
    if len(terms) < 3:
        return empty

    merges = _average_linkage(D)

    # The tree is rebuilt from the merges.
    nodes: Dict[int, Dict[str, Any]] = {
        i: {"name": str(labels.get(t, t)), "value": int(counts.get(t, 0)),
            "size": 1, "height": 0.0}
        for i, t in enumerate(terms)
    }
    next_id = len(terms)
    for a, b, d, _ in merges:
        nodes[next_id] = {
            "name": "",
            "height": round(d, 4),
            "size": nodes[a]["size"] + nodes[b]["size"],
            "value": nodes[a].get("value", 0) + nodes[b].get("value", 0),
            "children": [nodes[a], nodes[b]],
        }
        next_id += 1

    root = nodes[next_id - 1]

    # Cutting the tree: the last `max_clusters - 1` merges are undone; they are
    # the highest, hence those that separate best.
    groups: List[Dict[str, Any]] = []
    frontier = [root]
    while len(frontier) < max_clusters:
        # Open the highest node still merged.
        openable = [x for x in frontier if x.get("children")]
        if not openable:
            break
        target = max(openable, key=lambda x: x["height"])
        frontier.remove(target)
        frontier.extend(target["children"])

    for i, node in enumerate(sorted(frontier, key=lambda x: -x["size"]), start=1):
        leaves = _leaves(node)
        groups.append({
            "cluster": i,
            "label": leaves[0][0] if leaves else "",
            "terms": ", ".join(name for name, _ in leaves[:8]),
            "size": len(leaves),
            "occurrences": int(sum(v for _, v in leaves)),
        })

    return {"tree": root, "clusters": groups, "n_terms": len(terms)}


def _leaves(node: Dict[str, Any]):
    """Leaves of a subtree, the most frequent first."""
    if not node.get("children"):
        return [(node["name"], node.get("value", 0))]
    out = []
    for child in node["children"]:
        out.extend(_leaves(child))
    return sorted(out, key=lambda x: -x[1])
