"""Analyse thématique : carte stratégique de Callon et sujets émergents.

La **carte thématique** (Callon, 1991) place chaque groupe de mots-clés sur
deux axes :

  - **centralité** — l'intensité des liens du groupe avec les AUTRES groupes.
    Elle mesure à quel point le thème est relié au reste du domaine.
  - **densité** — l'intensité des liens INTERNES au groupe. Elle mesure à quel
    point le thème est développé, structuré.

En coupant aux médianes, on obtient quatre quadrants :

    densité ↑
      niches            │   thèmes moteurs
      (développés,      │   (développés,
       peu reliés)      │    centraux)
    ──────────────────── ┼────────────────────► centralité
      émergents ou      │   thèmes de base
      déclinants        │   (centraux,
                        │    peu développés)

Le quadrant en bas à gauche est ambigu par construction : un thème peu
développé et peu relié peut être en train d'apparaître **ou** de disparaître.
C'est la lecture chronologique qui tranche, pas la carte.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def trend_topics(corpus, n: int = 25, min_documents: int = 2,
                 kind: str = "author") -> pd.DataFrame:
    """Sujets et leur position dans le temps.

    Pour chaque terme : la **médiane** des années où il apparaît, encadrée par
    le premier et le troisième quartile. La médiane dit quand le terme a été
    le plus employé ; l'écart entre quartiles dit s'il est concentré sur une
    période ou étalé.

    On préfère la médiane à la moyenne : une seule occurrence ancienne ne doit
    pas tirer tout le terme vers le passé.
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
    """Carte stratégique de Callon, calculée sur le réseau de co-mots.

    Renvoie ``{"clusters": DataFrame, "medians": {...}}``.

    Formules de Callon et al. (1991) :

        centralité = 10 × Σ e_kh   (liens entre un terme du groupe et un terme hors groupe)
        densité    = 100 × Σ e_ij / w   (liens internes, w = nombre de termes)

    où **e_ij est l'indice d'équivalence** c_ij² / (c_i · c_j), compris entre
    0 et 1 — pas la co-occurrence brute c_ij. Les constantes 10 et 100 sont
    calibrées pour cet indice. Défaut corrigé : les poids bruts étaient
    utilisés, ce qui favorisait les termes fréquents et pouvait changer un
    thème de quadrant, pas seulement l'échelle des axes.

    Les groupes sont détectés par modularité gloutonne sur ce même réseau
    normalisé : c'est un algorithme déterministe, donc deux exécutions donnent
    la même carte — indispensable pour un résultat qu'on publie.
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

    # Ordre FIXE de construction et de numérotation : sans lui, la
    # modularité gloutonne départageait ses ex æquo selon le hachage des
    # chaînes, différent à chaque exécution (voir `_stable`).
    G = ordered_graph(graph["nodes"], graph["edges"])
    communities = ordered_communities(
        nx.community.greedy_modularity_communities(G, weight="weight"))

    rows = []
    for i, members in enumerate(communities, start=1):
        members = set(members)
        if len(members) < min_cluster_size:
            continue

        internal = external = 0.0
        for u, v, data in G.edges(data=True):
            w = float(data.get("weight", 1))
            u_in, v_in = u in members, v in members
            if u_in and v_in:
                internal += w
            elif u_in or v_in:
                external += w

        occ = {n: G.nodes[n]["occurrences"] for n in members}
        # À fréquence égale, l'ordre alphabétique : c'est lui qui choisit le
        # NOM du groupe quand deux termes sont ex æquo.
        ordered = sorted(members, key=lambda n: (-occ[n], str(G.nodes[n]["label"]).lower(), str(n)))
        labels = [G.nodes[n]["label"] for n in ordered]

        rows.append({
            "cluster": i,
            # Le groupe porte le nom de son terme le plus fréquent : c'est la
            # convention de lecture des cartes thématiques.
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
