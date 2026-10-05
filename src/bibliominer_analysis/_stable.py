"""An order that never depends on chance.

Python shuffles the order of strings in a ``set`` from one run to the next
(hash randomisation). Anything that relies on that order to break ties
therefore changes its result although the data did not change: a cluster
of the thematic map was called "Software Testing Effort" in one run and
"ISBSG" in the next, and two tied authors swapped places in the
three-field plot.

One rule, everywhere: most frequent first, then alphabetical order.
Arbitrary, but always the same, which is what a published result requires.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Set, Tuple


def top_by_count(counts: Mapping[Hashable, float],
                 n: Optional[int] = None) -> List[Tuple[Hashable, float]]:
    """``Counter.most_common`` without its tie-break by insertion order."""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return items if n is None else items[:n]


def ordered_graph(nodes: Iterable[Dict[str, Any]], edges: Iterable[Dict[str, Any]],
                  node_attrs: bool = True):
    """A networkx graph built in a FIXED order (sorted nodes, then edges).

    Greedy modularity and Louvain depend on the insertion order of the nodes:
    with the same seed, a different order gives different clusters.
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
    """The largest cluster first; at equal size, by its members."""
    return sorted((set(g) for g in groups),
                  key=lambda g: (-len(g), sorted(str(m) for m in g)))
