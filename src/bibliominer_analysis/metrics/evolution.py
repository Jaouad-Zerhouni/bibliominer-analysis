"""Évolution thématique : comment les thèmes se transmettent d'une période à l'autre.

La carte thématique (`themes.py`) est une photographie. Ici on en prend
plusieurs, à des périodes successives, et on relie les groupes entre eux : un
thème qui se scinde en deux, deux thèmes qui fusionnent, un thème qui
disparaît. C'est la seule vue qui distingue vraiment un sujet **émergent** d'un
sujet **déclinant** — le quadrant en bas à gauche de Callon confond les deux.

Le lien entre un groupe d'une période et un groupe de la suivante est l'**indice
d'inclusion pondéré** :

    inclusion = Σ_{t commun} min(occ_A(t), occ_B(t))
                ────────────────────────────────────────
                min(Σ occurrences de A, Σ occurrences de B)

Au numérateur, le MINIMUM des occurrences de chaque terme commun aux deux
périodes : un terme ne peut pas transmettre plus d'occurrences que la période
la plus pauvre n'en porte, ce qui garde l'indice entre 0 et 1.

On divise par le **plus petit** des deux, pas par l'union : un petit groupe
entièrement absorbé par un grand doit donner 1, parce qu'il a bel et bien été
absorbé. Diviser par l'union écraserait ce cas, qui est justement celui qu'on
cherche à voir.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


def _period_clusters(sub, top_n: int, min_weight: int,
                     min_cluster_size: int, kind: str) -> List[dict]:
    """Groupes de co-mots d'un sous-corpus, avec leurs termes complets."""
    from ..networks import build as nets

    graph = nets.co_word(sub, top_n=top_n, min_weight=min_weight, kind=kind)
    if graph["n_nodes"] == 0:
        return []

    import networkx as nx

    G = nx.Graph()
    for node in graph["nodes"]:
        G.add_node(node["id"], label=node["label"], occurrences=node["occurrences"])
    for e in graph["edges"]:
        G.add_edge(e["source"], e["target"], weight=e["weight"])

    out = []
    for members in nx.community.greedy_modularity_communities(G, weight="weight"):
        members = set(members)
        if len(members) < min_cluster_size:
            continue
        occ = {G.nodes[m]["label"]: int(G.nodes[m]["occurrences"]) for m in members}
        ordered = sorted(occ, key=lambda w: -occ[w])
        out.append({
            "label": ordered[0],
            "terms": occ,                       # terme -> occurrences
            "n_terms": len(occ),
            "occurrences": int(sum(occ.values())),
            "top_terms": ", ".join(ordered[:8]),
        })
    # Le plus gros groupe en premier : la lecture du Sankey suit ce poids.
    return sorted(out, key=lambda c: -c["occurrences"])


def _slice_bounds(corpus, cuts: Optional[Sequence[int]],
                  n_periods: int) -> List[tuple]:
    """Bornes (début, fin) incluses de chaque période."""
    years = pd.to_numeric(corpus.documents["year"], errors="coerce").dropna()
    if years.empty:
        return []
    lo, hi = int(years.min()), int(years.max())

    if cuts:
        pts = sorted({int(c) for c in cuts if lo <= int(c) < hi})
    else:
        # Sans point de coupe explicite : des tranches d'effectif comparable,
        # pas de durée égale. Un corpus dont la production explose donnerait
        # sinon une première tranche vide de sens.
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
    """Flux thématiques entre périodes successives.

    Retour ::

        {"periods": [{"label", "year_min", "year_max", "documents", "clusters"}],
         "nodes":   [{"name", "period", "label", "occurrences", "terms"}],
         "flows":   [{"source", "target", "value", "inclusion", "shared_terms"}]}

    ``source`` et ``target`` reprennent le ``name`` des nœuds, qui préfixe le
    groupe par sa période : deux périodes peuvent avoir un groupe du même nom,
    et il ne faut surtout pas les confondre en un seul.
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
    # Un nœud sans aucun flux n'a rien à montrer dans un Sankey : il ferait une
    # colonne orpheline. On ne garde que ceux qui sont reliés.
    linked = {f["source"] for f in flows} | {f["target"] for f in flows}
    nodes = [n for n in nodes if n["name"] in linked]
    return {"periods": periods, "nodes": nodes, "flows": flows}
