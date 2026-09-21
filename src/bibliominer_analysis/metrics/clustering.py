"""Regroupement des documents par **couplage bibliographique**.

La carte thématique de `themes.py` regroupe des *mots*. Ici on regroupe des
*documents* : deux articles qui puisent aux mêmes références traitent
probablement du même sujet, même s'ils n'emploient pas le même vocabulaire.
C'est la différence qui compte — le couplage attrape les proximités que les
mots-clés ratent, notamment entre communautés qui nomment différemment la même
chose.

Chaque groupe est placé sur deux axes, comme chez Callon :

  - **centralité** : la force des liens du groupe vers les AUTRES groupes ;
  - **impact** : les citations moyennes de ses documents.

Le lien est normalisé par la **force d'association** avant regroupement. Sans
cela, un article à trois cents références se coupleraient fortement avec tout
le monde par simple effet de taille, et formerait un groupe artificiel.
"""

from __future__ import annotations

from .._stable import ordered_communities, ordered_graph, top_by_count

from collections import Counter
from typing import Any, Dict

import pandas as pd


def clustering_by_coupling(corpus, top_n: int = 100, min_weight: int = 3,
                           min_cluster_size: int = 3,
                           impact: str = "local") -> Dict[str, Any]:
    """Groupes de documents couplés, avec centralité et impact.

    `impact` vaut ``"local"`` (citations reçues dans le corpus) ou
    ``"global"`` (citations Scopus). Le local est le plus parlant : il mesure
    l'influence du groupe **sur le domaine étudié**.

    Retour ``{"clusters": DataFrame, "medians": {...}, "n_documents": int}``.
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

    # Données par document, pour décrire les groupes ensuite.
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
        # Le document le plus cité du groupe ; à égalité, le plus petit
        # identifiant — `idxmax` prenait le premier rencontré, dont l'ordre
        # venait d'un ensemble (hachage aléatoire).
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
