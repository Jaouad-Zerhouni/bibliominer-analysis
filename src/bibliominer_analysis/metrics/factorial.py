"""Structure conceptuelle : analyse factorielle des mots-clés.

Les réseaux de co-mots montrent *qui est lié à qui*. L'analyse factorielle
répond à une autre question : **sur quels axes le domaine se structure-t-il ?**
Elle projette les termes dans un plan où la distance a un sens — deux termes
proches apparaissent dans les mêmes documents — puis on y cherche des groupes.

Deux méthodes, deux points de vue :

  - **AFC** (analyse factorielle des correspondances) sur le tableau
    documents × termes. Les axes sont les directions de plus forte *inertie*,
    c'est-à-dire de plus fort écart à l'indépendance. On sait quelle part de
    l'information chaque axe porte, ce que ne donne pas le MDS.
  - **MDS** (positionnement multidimensionnel) sur une matrice de
    dissimilarité. Il ne cherche qu'à préserver les distances deux à deux :
    plus fidèle localement, mais les axes n'ont pas d'interprétation propre.

Tout est calculé en numpy — SVD pour l'AFC, décomposition propre pour le MDS,
k-moyennes pour les groupes. Aucune dépendance supplémentaire, et surtout des
résultats **déterministes** : un article doit pouvoir être refait à l'identique.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

#: Graine fixe : deux exécutions doivent donner exactement la même carte.
SEED = 20240101


# ---------------------------------------------------------------------------
# Tableau de départ
# ---------------------------------------------------------------------------

def _incidence(corpus, kind: str = "author", top_n: int = 50,
               min_documents: int = 2):
    """Tableau binaire documents × termes, et les occurrences de chaque terme."""
    k = corpus.keywords
    if kind in ("author", "index"):
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna() & (k["keyword"].astype(str).str.strip() != "")]
    if k.empty:
        return None, None, None

    k = k.copy()
    k["norm"] = k["keyword"].astype(str).str.strip().str.lower()
    k = k.drop_duplicates(subset=["eid", "norm"])

    counts = k.groupby("norm")["eid"].nunique()
    counts = counts[counts >= min_documents].sort_values(ascending=False).head(top_n)
    if counts.empty:
        return None, None, None

    k = k[k["norm"].isin(counts.index)]
    labels = (k.groupby("norm")["keyword"]
                .agg(lambda s: s.mode().iat[0]).reindex(counts.index))

    matrix = pd.crosstab(k["eid"], k["norm"]).reindex(columns=counts.index,
                                                      fill_value=0)
    matrix = (matrix > 0).astype(float)
    # Un document sans aucun des termes retenus n'apporte rien et casserait
    # l'AFC (masse de ligne nulle).
    matrix = matrix.loc[matrix.sum(axis=1) > 0]
    if matrix.empty or matrix.shape[1] < 3:
        return None, None, None
    return matrix, labels, counts


# ---------------------------------------------------------------------------
# Analyse factorielle des correspondances
# ---------------------------------------------------------------------------

def _correspondence(matrix: pd.DataFrame, n_dims: int = 2):
    """AFC classique par SVD. Renvoie (coords colonnes, inerties expliquées)."""
    N = matrix.to_numpy(dtype=float)
    total = N.sum()
    if total <= 0:
        return None, None

    P = N / total
    r = P.sum(axis=1)          # masses des lignes (documents)
    c = P.sum(axis=0)          # masses des colonnes (termes)
    ok_r, ok_c = r > 0, c > 0
    P, r, c = P[np.ix_(ok_r, ok_c)], r[ok_r], c[ok_c]

    # Matrice des résidus standardisés : l'écart au modèle d'indépendance,
    # pondéré par les masses. C'est cette matrice que l'AFC décompose.
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)

    inertia = sigma ** 2
    total_inertia = inertia.sum()
    if total_inertia <= 0:
        return None, None
    explained = 100.0 * inertia / total_inertia

    k = min(n_dims, sigma.size)
    # Coordonnées principales des colonnes : D_c^{-1/2} V Σ
    coords = (Vt[:k].T * sigma[:k]) / np.sqrt(c)[:, None]

    full = np.full((matrix.shape[1], k), np.nan)
    full[np.where(ok_c)[0]] = coords
    return full, explained[:k]


# ---------------------------------------------------------------------------
# Positionnement multidimensionnel (MDS classique)
# ---------------------------------------------------------------------------

def _classical_mds(dissimilarity: np.ndarray, n_dims: int = 2):
    """MDS classique (Torgerson) : double centrage puis décomposition propre."""
    D2 = dissimilarity ** 2
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # B est symétrique : eigh est plus stable et plus rapide que eig, et rend
    # les valeurs propres déjà triées.
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    positive = np.clip(values[:n_dims], 0, None)
    coords = vectors[:, :n_dims] * np.sqrt(positive)
    total = np.clip(values, 0, None).sum()
    explained = (100.0 * positive / total) if total > 0 else np.zeros(n_dims)
    return coords, explained


def _dissimilarity(matrix: pd.DataFrame) -> np.ndarray:
    """1 − indice de Salton entre termes, à partir des co-occurrences."""
    X = matrix.to_numpy(dtype=float)
    co = X.T @ X
    diag = np.diag(co).astype(float)
    denom = np.sqrt(np.outer(diag, diag))
    with np.errstate(divide="ignore", invalid="ignore"):
        salton = np.where(denom > 0, co / denom, 0.0)
    np.fill_diagonal(salton, 1.0)
    return np.clip(1.0 - salton, 0.0, None)


# ---------------------------------------------------------------------------
# k-moyennes déterministes
# ---------------------------------------------------------------------------

def _kmeans(X: np.ndarray, k: int, seed: int = SEED, iters: int = 100):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if k >= n:
        return np.arange(n)

    # k-means++ : le premier centre au hasard, les suivants proportionnellement
    # au carré de la distance au centre le plus proche. Sans cela, deux
    # exécutions convergent parfois vers des partitions différentes.
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
    """Silhouette moyenne — sert à choisir k, pas à juger la qualité absolue."""
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
# Point d'entrée
# ---------------------------------------------------------------------------

def conceptual_structure(corpus, method: str = "CA", kind: str = "author",
                         top_n: int = 50, min_documents: int = 2,
                         n_clusters: Optional[int] = None) -> Dict[str, Any]:
    """Carte factorielle des mots-clés, avec regroupement automatique.

    `method` vaut ``"CA"`` (correspondances) ou ``"MDS"``.
    `n_clusters` à ``None`` laisse la silhouette choisir le nombre de groupes.

    Retour ::

        {"method", "terms": DataFrame, "explained": [dim1 %, dim2 %],
         "clusters": [{"cluster", "label", "terms", "size", "occurrences"}]}

    ``terms`` porte ``keyword``, ``dim1``, ``dim2``, ``occurrences``, ``cluster``.
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
    }).sort_values(["cluster", "occurrences"], ascending=[True, False])

    groups = []
    for cid, g in df.groupby("cluster"):
        groups.append({
            "cluster": int(cid),
            # Le terme le plus fréquent nomme le groupe, comme sur la carte
            # thématique : les deux vues restent ainsi comparables.
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
# Dendrogramme des thèmes
# ---------------------------------------------------------------------------

def _average_linkage(D: np.ndarray):
    """Classification ascendante hiérarchique, lien moyen (UPGMA).

    Renvoie la liste des fusions ``(a, b, distance, taille)``, dans l'ordre où
    elles se produisent — le format d'une matrice de liaison classique.

    Le lien **moyen** plutôt que le lien simple : ce dernier produit des
    chaînages, où un groupe s'étire de proche en proche sans jamais être
    compact, et le dendrogramme devient illisible.
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

        # Distance du nouveau groupe : moyenne pondérée par les effectifs,
        # ce qui est exactement la définition du lien moyen.
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
    """Arbre hiérarchique des termes, coupé en `max_clusters` groupes.

    Le dendrogramme montre ce qu'une carte factorielle cache : **à quel niveau**
    deux thèmes se rejoignent. Deux termes peuvent être voisins dans le plan et
    n'appartenir au même groupe qu'au tout dernier moment — l'arbre le dit, le
    nuage de points non.

    Retour ``{"tree": {...}, "clusters": [...], "n_terms": int}``, l'arbre
    étant directement consommable par un rendu de type « tree ».
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

    # On reconstruit l'arbre à partir des fusions.
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

    # Couper l'arbre : on défait les `max_clusters - 1` dernières fusions, qui
    # sont les plus hautes, donc celles qui séparent le mieux.
    groups: List[Dict[str, Any]] = []
    frontier = [root]
    while len(frontier) < max_clusters:
        # On ouvre le nœud le plus haut encore fusionné.
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
    """Feuilles d'un sous-arbre, les plus fréquentes d'abord."""
    if not node.get("children"):
        return [(node["name"], node.get("value", 0))]
    out = []
    for child in node["children"]:
        out.extend(_leaves(child))
    return sorted(out, key=lambda x: -x[1])
