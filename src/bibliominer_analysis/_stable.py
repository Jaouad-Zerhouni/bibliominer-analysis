"""Un ordre qui ne dépend jamais du hasard.

Python mélange l'ordre des chaînes dans un ``set`` d'une exécution à l'autre
(hachage aléatoire). Tout ce qui s'en remet à cet ordre pour départager des
ex æquo change donc de résultat sans que les données aient changé : un groupe
de la carte thématique s'appelait « Software Testing Effort » à une exécution
et « ISBSG » à la suivante, deux auteurs à égalité échangeaient leur place
dans le diagramme à trois champs.

Règle unique, partout : le plus fréquent d'abord, puis l'ordre alphabétique.
Arbitraire, mais toujours le même — c'est ce qu'exige un résultat publié.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Set, Tuple


def top_by_count(counts: Mapping[Hashable, float],
                 n: Optional[int] = None) -> List[Tuple[Hashable, float]]:
    """``Counter.most_common`` sans son départage par ordre d'insertion."""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return items if n is None else items[:n]


def ordered_graph(nodes: Iterable[Dict[str, Any]], edges: Iterable[Dict[str, Any]],
                  node_attrs: bool = True):
    """Graphe networkx construit dans un ordre FIXE (nœuds puis liens triés).

    La modularité gloutonne et Louvain dépendent de l'ordre d'insertion des
    nœuds : à graine égale, un ordre différent donne des groupes différents.
    """
    import networkx as nx

    G = nx.Graph()
    for node in sorted(nodes, key=lambda n: str(n["id"])):
        attrs = {k: v for k, v in node.items() if k != "id"} if node_attrs else {}
        G.add_node(node["id"], **attrs)
    for e in sorted(edges, key=lambda e: (str(e["source"]), str(e["target"]))):
        G.add_edge(e["source"], e["target"], weight=float(e.get("weight", 1)))
    return G


def ordered_communities(groups: Iterable[Iterable[Hashable]]) -> List[Set[Hashable]]:
    """Le plus grand groupe d'abord ; à taille égale, par ses membres."""
    return sorted((set(g) for g in groups),
                  key=lambda g: (-len(g), sorted(str(m) for m in g)))
