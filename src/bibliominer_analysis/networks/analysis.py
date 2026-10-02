"""Mesures sur les réseaux : centralités, regroupement, normalisations.

Un réseau bibliométrique brut se lit mal : tout le monde voit le gros nœud au
centre, personne ne voit le nœud discret qui relie deux communautés. Ce module
calcule ce que l'œil ne voit pas.

**Les centralités ne disent pas la même chose**, et les confondre est l'erreur
classique :

  - **degré**, le nombre de liens. Qui est le plus actif.
  - **intermédiarité**, la fréquence à laquelle un nœud se trouve sur le
    chemin le plus court entre deux autres. Qui fait le **pont** entre des
    groupes qui ne se parlent pas. Un nœud peut avoir un degré médiocre et une
    intermédiarité énorme : c'est souvent le plus intéressant du corpus.
  - **proximité**, la distance moyenne au reste. Qui atteint tout le monde vite.
  - **PageRank**, être cité par des nœuds eux-mêmes centraux compte davantage
    qu'être cité par des nœuds isolés.

**Les normalisations** viennent de VOSviewer : un lien brut favorise
mécaniquement les entités fréquentes. Rapporter la co-occurrence à ce qu'on
attendrait par hasard (*force d'association*) corrige ce biais, et change
réellement la carte.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

#: Graine fixe pour Louvain : une carte publiée doit être reproductible.
SEED = 20240101

NORMALIZATIONS = ("none", "association", "jaccard", "salton", "inclusion", "equivalence")


def _adjacency(G, order):
    """Matrice d'adjacence pondérée, dense, nos réseaux tiennent en mémoire."""
    import numpy as np

    index = {n: i for i, n in enumerate(order)}
    A = np.zeros((len(order), len(order)), dtype=float)
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        i, j = index[u], index[v]
        A[i, j] = A[j, i] = w
    return A


def _pagerank(A, alpha: float = 0.85, iters: int = 200, tol: float = 1e-10):
    """PageRank pondéré par itération de la puissance.

    Écrit ici plutôt qu'appelé à networkx : depuis la version 3, `nx.pagerank`
    passe par scipy. Sans scipy il lève, et un bloc de secours renverrait 1/n
    pour tout le monde, une colonne d'apparence normale mais entièrement
    vide de sens, donc pire qu'une colonne absente.
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
    """Centralité de vecteur propre par itération de la puissance.

    On itère sur **A + I**, pas sur A. Les deux ont les mêmes vecteurs propres,
    mais sur un graphe BIPARTI (une étoile, un arbre, fréquents en
    co-signature) A a deux valeurs propres dominantes opposées, +λ et −λ :
    l'itération simple oscille alors entre deux vecteurs sans jamais converger,
    et renvoyait celui où elle s'arrêtait. Le décalage de +1 rend la valeur
    propre dominante unique.
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
    # Le vecteur propre dominant d'une matrice à coefficients positifs est de
    # signe constant (Perron-Frobenius) : on le rend positif.
    return np.abs(x)


def _to_networkx(graph: Dict[str, Any]):
    from .._stable import ordered_graph

    # Ordre d'insertion fixe : Louvain, même avec sa graine, en dépend.
    return ordered_graph(graph.get("nodes", []), graph.get("edges", []))


def normalize(graph: Dict[str, Any], method: str = "association") -> Dict[str, Any]:
    """Repondère les liens selon une mesure de similarité.

    Avec ``c_ij`` la co-occurrence et ``s_i`` le total de l'entité *i* :

      - ``association`` : c_ij / (s_i · s_j), la force d'association de
        VOSviewer, proportionnelle au rapport entre observé et attendu.
      - ``jaccard``      : c_ij / (s_i + s_j − c_ij)
      - ``salton``       : c_ij / √(s_i · s_j)
      - ``inclusion``    : c_ij / min(s_i, s_j)
      - ``equivalence``  : c_ij² / (s_i · s_j)

    Les poids normalisés sont des réels ; le poids brut reste disponible dans
    ``raw_weight``, parce qu'un lecteur veut souvent savoir « combien de
    documents » derrière une valeur de 0,043.
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
    """Ajoute centralités, communauté et superposition temporelle aux nœuds.

    Chaque nœud reçoit ``degree_centrality``, ``betweenness``, ``closeness``,
    ``pagerank``, ``eigenvector``, ``clustering``, et si demandé ``community``
    et ``overlay_year``.

    ``overlay`` associe un identifiant de nœud à son année moyenne : c'est la
    vue « overlay » de VOSviewer, qui montre d'un coup quelles zones du réseau
    sont récentes et lesquelles sont anciennes, information qu'aucune taille
    de nœud ne peut porter.
    """
    if not graph.get("nodes"):
        return graph

    import networkx as nx

    G = _to_networkx(graph)

    degree = nx.degree_centrality(G)
    clustering = nx.clustering(G, weight="weight")
    # Les poids sont des SIMILARITÉS : plus c'est fort, plus c'est proche. Les
    # chemins les plus courts raisonnent en DISTANCES, d'où l'inversion, sans
    # elle, l'intermédiarité passerait par les liens les plus faibles.
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
        # `resolution` est le paramètre de VOSviewer : au-dessus de 1 on
        # obtient des groupes plus nombreux et plus petits, en dessous des
        # groupes plus larges. Il n'a pas de valeur « juste », c'est un choix
        # de granularité, qui doit donc rester entre les mains du lecteur.
        try:
            groups = nx.community.louvain_communities(
                G, weight="weight", seed=SEED, resolution=float(resolution))
        except Exception:
            groups = nx.community.greedy_modularity_communities(G, weight="weight")
        # Les groupes sont numérotés du plus grand au plus petit : le numéro
        # devient lisible au lieu d'être arbitraire.
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
    """La partition déjà posée par `annotate`, si les nœuds la portent.

    La modularité du résumé doit être celle des communautés AFFICHÉES. Elle
    était recalculée par un Louvain à résolution 1 : dès que l'utilisateur
    choisissait une autre résolution, le chiffre ne décrivait plus les groupes
    qu'il voyait.
    """
    nodes = graph.get("nodes") or []
    if not nodes or any("community" not in n for n in nodes):
        return None
    groups: Dict[Any, set] = {}
    for n in nodes:
        groups.setdefault(n["community"], set()).add(n["id"])
    return list(groups.values())


def graph_summary(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Indicateurs de forme du réseau entier.

    ``density``, part des liens possibles réellement présents.
    ``transitivity``, probabilité que deux voisins d'un nœud soient voisins.
    ``components``, nombre de morceaux disjoints ; plus d'un signale un
    domaine fragmenté, ce qu'un dessin de réseau masque souvent.
    ``mean_path_length`` et ``diameter`` portent sur la **plus grande**
    composante : sur un graphe non connexe ils seraient infinis.
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
    """Année moyenne de publication associée à chaque nœud d'un réseau.

    C'est la donnée de la vue « overlay » : elle colore le réseau par le temps,
    et fait apparaître d'un coup les zones récentes et les zones anciennes.
    Aucune taille de nœud ne peut porter cette information, parce que la taille
    est déjà prise par la fréquence.

    Les identifiants renvoyés suivent EXACTEMENT ceux de `networks.build`,
    sinon la jointure serait silencieusement vide.
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
# Disposition et carte de densité (la vue « density » de VOSviewer)
# ---------------------------------------------------------------------------

def _shortest_paths(G, index: Dict[str, int]):
    """Tous les plus courts chemins (poids ``distance``), en matrice.

    Floyd-Warshall vectorisé : les mêmes longueurs que Dijkstra depuis chaque
    nœud, mais calculées par numpy. Sur un réseau de 250 références (7 000
    liens), Dijkstra en Python pur prenait 8 s, l'écran attendait.
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
    """Place les nœuds d'UNE composante connexe, par MDS sur ses chemins.

    Une composante seule n'a que des distances finies : le MDS y travaille
    sur des écarts qui veulent tous dire quelque chose.
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


#: Graine FIXE : la disposition à ressorts est déterministe, la carte ne
#: bouge pas d'une ouverture à l'autre, ni entre l'écran et la figure.
_LAYOUT_SEED = 7


def _readable(G, start: Dict[str, list]) -> Dict[str, list]:
    """Du MDS à une carte LISIBLE.

    Le MDS classique écrase un groupe très lié sur un seul point : quand
    toutes les distances d'un groupe sont presque égales (des co-auteurs qui
    signent tous ensemble), ses nœuds tombent au même endroit. Constaté sur
    un vrai corpus : un réseau de co-auteurs dessiné comme une colonne de
    disques empilés, noms superposés.

    Deux passes, toutes deux déterministes :
      1. une disposition à ressorts PARTANT du MDS (graine fixe), elle garde
         la forme d'ensemble et desserre les groupes ;
      2. un écartement des paires trop proches, jusqu'à une distance minimale
         qui dépend du nombre de nœuds, aucun disque ne cache un autre.
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
    """Écarte les points trop proches, sans rien déplacer d'autre.

    Coordonnées ramenées à [-1, 1], distance minimale 1.6/√n : assez pour
    qu'un disque ne recouvre pas son voisin, assez peu pour garder la forme.
    Deux points confondus sont séparés selon une direction fixée par leur
    rang, jamais au hasard.
    """
    import numpy as np

    n = len(xy)
    span = np.ptp(xy, axis=0).max() or 1.0
    xy = (xy - xy.mean(axis=0)) / span * 2.0
    min_d = 1.6 / np.sqrt(n)
    angles = np.arange(n) * 2.399963          # angle d'or : directions fixes
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


#: Le cadre visé par la disposition : le format d'une carte à l'écran
#: (≈ 1100 × 600 px) et d'une figure exportée, un peu plus large que haute.
LAYOUT_WIDTH, LAYOUT_HEIGHT = 860.0, 480.0


def _pack(blocks: List[Dict[str, list]], gap: float = 0.6) -> Dict[str, list]:
    """Range les composantes en RANGÉES, au format de la carte.

    Entre deux composantes DÉCONNECTÉES, la distance n'a aucun sens : aucun
    chemin ne les relie. Les faire entrer dans un même MDS revenait à lui
    demander de coder une distance qui n'existe pas, et il y dépensait son
    premier axe, repliant la vraie structure sur une droite. On les place
    donc côte à côte, ce qui n'affirme rien de plus qu'un voisinage
    graphique.

    Côte à côte, mais pas sur UNE ligne : un réseau de co-auteurs compte
    souvent quinze ou vingt petits groupes. Alignés, ils formaient une bande
    vingt fois plus large que haute ; cadrée à l'écran, la bande devenait
    un chapelet de disques empilés, noms illisibles (constaté sur un vrai
    corpus). Les composantes remplissent donc des rangées successives, de la
    plus grande à la plus petite, jusqu'au format de la carte.

    Chaque bloc est mis à l'échelle en √n : une composante de trois nœuds
    n'occupe pas la largeur d'une de vingt.
    """
    import math

    import numpy as np

    boxes = []
    for block in blocks:
        ids = list(block)
        xy = np.array([block[i] for i in ids], dtype=float)
        span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-9)
        # Une composante isolée (tous les nœuds au même endroit) doit rester
        # visible : on lui donne une largeur minimale plutôt qu'un point.
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
        x = (row_width - total) / 2              # rangée centrée
        for ids, xy, width, box_height in row:
            offset = top - (height - box_height) / 2 - box_height
            for node, (px, py) in zip(ids, xy):
                placed[node] = [x + float(px), offset + float(py)]
            x += width + gap
        top -= height + gap                      # l'axe y monte
    return placed


def _declutter(graph: Dict[str, Any], coords: Dict[str, list],
               rounds: int = 4, passes: int = 150, gap: float = 4.0) -> Dict[str, list]:
    """Aucun disque sur un autre, à la taille où la carte est DESSINÉE.

    L'écartement de `_readable` travaille à l'intérieur d'une composante,
    avec une distance minimale identique pour tous les nœuds. Or un nœud
    très cité est dessiné quatre fois plus large qu'un petit : deux gros
    disques voisins se recouvraient quand même (« Nassif » sur « Hosni »).

    Les coordonnées sont ramenées au cadre de la carte (`LAYOUT_WIDTH` ×
    `LAYOUT_HEIGHT`, en pixels d'écran) ; chaque nœud y reçoit le rayon que
    l'interface lui donne (10 + 26·√(occurrences / max) px de diamètre), et
    les paires trop proches sont écartées, symétriquement, de ce qui leur
    manque. Déterministe : deux points confondus s'écartent selon une
    direction fixée par leur rang.
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
    # Chaque paire une seule fois (triangle supérieur) : les autres ne sont
    # jamais regardées.
    upper = np.triu(np.ones((len(ids), len(ids)), dtype=bool), 1)
    angles = np.arange(len(ids)) * 2.399963       # angle d'or : directions fixes
    fallback = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    for _ in range(rounds):
        # Recadrer à chaque tour : écarter agrandit la carte, et c'est la
        # carte RECADRÉE qui sera dessinée.
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
            # Chacun des deux fait la moitié du chemin qui manque.
            push = unit * ((need[i, j] - dist) / 2.0)[:, None]
            np.add.at(xy, i, push)
            np.add.at(xy, j, -push)
        if not moved:
            break
    return {node: [round(float(xy[k, 0]), 2), round(float(xy[k, 1]), 2)]
            for k, node in enumerate(ids)}


def layout(graph: Dict[str, Any]) -> Dict[str, list]:
    """Coordonnées 2D des nœuds, par MDS sur les plus courts chemins.

    Pourquoi ne pas laisser le navigateur placer les nœuds par simulation de
    forces ? Parce qu'une simulation est **stochastique** : deux ouvertures de
    la même page donnent deux dessins différents, et une carte qu'on publie ne
    peut pas bouger d'une exécution à l'autre. Le MDS est déterministe.

    Les poids sont des similarités ; les chemins raisonnent en distances, d'où
    l'inversion.

    **Chaque composante connexe est placée séparément, puis juxtaposée.**
    Les faire entrer dans un seul MDS demandait de donner une distance à des
    paires qu'aucun chemin ne relie : on leur en attribuait une, grande, et
    ce contraste artificiel devenait le fait le plus saillant du nuage. Le
    MDS y consacrait son premier axe, et la structure réelle, celle qu'on
    vient de calculer, se repliait sur une droite. Voir `_pack`.

    Les coordonnées sont en pixels d'une carte de `LAYOUT_WIDTH` ×
    `LAYOUT_HEIGHT` : aucun disque n'y recouvre un autre (`_declutter`).
    """
    import networkx as nx

    if not graph.get("nodes"):
        return {}

    G = _to_networkx(graph)
    for _, _, d in G.edges(data=True):
        w = float(d.get("weight", 1)) or 1e-9
        d["distance"] = 1.0 / w

    # Ordre STABLE : les composantes sont triées sur leur plus petit nœud,
    # sinon deux exécutions pourraient les juxtaposer dans un autre ordre.
    components = sorted((sorted(c) for c in nx.connected_components(G)),
                        key=lambda c: (-len(c), c[0]))
    packed = _pack([_mds_component(G, nodes) for nodes in components])
    return _declutter(graph, packed)


def attach_layout(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Le réseau, chaque nœud portant ses coordonnées ``x`` et ``y``.

    Ce sont exactement celles de `layout`, donc celles de `render_network` :
    l'interface web dessine le réseau à ces positions au lieu de lancer sa
    propre simulation de forces. Sans cela, la carte affichée à l'écran et la
    figure produite par le package avaient deux formes différentes, et la
    carte de l'écran changeait à chaque ouverture.

    L'axe ``y`` suit la convention mathématique (vers le haut), comme
    matplotlib ; un rendu d'écran, dont l'axe descend, doit l'inverser.
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
    """Carte de densité : où le réseau est-il dense, et de quoi ?

    La vue « density » de VOSviewer. Chaque nœud diffuse son poids autour de lui
    selon un noyau gaussien ; on somme, et on obtient une surface. Elle répond à
    une question que le graphe ne montre pas : quelles **zones** du domaine sont
    saturées et lesquelles sont désertes.

    Retour ``{"x": [...], "y": [...], "cells": [[i, j, valeur], ...], "max"}``,
    directement consommable par une carte de chaleur.
    """
    import numpy as np

    nodes = [n for n in graph.get("nodes", []) if n["id"] in coords]
    if not nodes:
        return {"x": [], "y": [], "cells": [], "max": 0.0}

    P = np.array([coords[n["id"]] for n in nodes], dtype=float)
    w = np.array([float(n.get("occurrences", 1) or 1) for n in nodes])

    lo, hi = P.min(axis=0), P.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    # Une marge : sans elle les nœuds du bord sont coupés en deux par le cadre.
    lo, hi = lo - 0.08 * span, hi + 0.08 * span

    xs = np.linspace(lo[0], hi[0], size)
    ys = np.linspace(lo[1], hi[1], size)
    if bandwidth is None:
        # L'écart typique entre voisins : un noyau trop étroit rend une carte
        # de points, trop large une tache unie. L'ancienne règle (un huitième
        # de la plus grande étendue) dépendait de la place des petits îlots
        # posés en marge : sur un réseau en plusieurs composantes, l'étendue
        # gonflait et tout le groupe principal fondait en une seule tache.
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
