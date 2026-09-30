"""Le package doit savoir DESSINER un réseau, pas seulement le calculer.

Sans ce rendu, quelqu'un qui installe `bibliominer-analysis` sans
l'application web obtenait des chiffres et aucune carte : les réseaux
n'existaient qu'en JSON, et leur dessin vivait dans le navigateur.

Deux exigences portent ces tests :

  - **la carte se rejoue**, deux exécutions sur les mêmes données donnent
    le même fichier, sans quoi une figure publiée bougerait d'un tirage à
    l'autre et deviendrait indéfendable ;
  - **rien n'est inventé**, un réseau vide lève plutôt que de rendre une
    image blanche, qu'on croirait valide.
"""
import pytest

from bibliominer_analysis.figures.network import (
    NetworkFigureError, render_network)
from bibliominer_analysis.networks.analysis import annotate

_GRAPH = {
    "nodes": [{"id": n, "label": n, "weight": w} for n, w in [
        ("machine learning", 30), ("neural network", 22),
        ("deep learning", 18), ("flood mapping", 25),
        ("remote sensing", 20), ("hydrology", 14)]],
    "edges": [
        {"source": "machine learning", "target": "neural network", "weight": 12},
        {"source": "machine learning", "target": "deep learning", "weight": 10},
        {"source": "neural network", "target": "deep learning", "weight": 8},
        {"source": "flood mapping", "target": "remote sensing", "weight": 11},
        {"source": "flood mapping", "target": "hydrology", "weight": 7},
        {"source": "machine learning", "target": "flood mapping", "weight": 4},
    ],
}


@pytest.fixture
def graph():
    return annotate(_GRAPH, communities=True)


def test_it_renders_a_png(graph):
    image = render_network(graph, title="Co-word network")
    assert image.startswith(b"\x89PNG"), "ce n'est pas un PNG"
    assert len(image) > 5_000, "image suspecte de vacuité"


@pytest.mark.parametrize("fmt", ["png", "jpg", "svg", "pdf"])
def test_the_same_data_gives_the_same_file(graph, fmt):
    """Une figure publiée ne peut pas bouger d'une exécution à l'autre.

    Deux sources de bruit ont dû être neutralisées : l'horodatage que
    matplotlib inscrit dans le fichier, et, en SVG, les identifiants
    d'éléments tirés d'un grain aléatoire.
    """
    assert render_network(graph, fmt=fmt) == render_network(graph, fmt=fmt)


def test_the_layout_is_not_redrawn_at_random(graph):
    """Le placement vient d'un MDS déterministe, pas d'une simulation."""
    from bibliominer_analysis.networks.analysis import layout
    assert layout(graph) == layout(graph)


def test_given_coordinates_are_used_as_is(graph):
    """On doit pouvoir dessiner EXACTEMENT la carte affichée à l'écran.

    L'application web calcule les coordonnées une fois ; la figure exportée
    doit être la même image, sinon l'article ne montre pas ce que
    l'utilisateur a vu.
    """
    coords = {n["id"]: [i * 1.0, -i * 1.0]
              for i, n in enumerate(graph["nodes"])}
    fixed = render_network(graph, coords=coords)
    assert fixed != render_network(graph)      # ce n'est pas la carte MDS
    assert fixed == render_network(graph, coords=coords)


def test_node_size_can_carry_a_centrality(graph):
    """`annotate` pose pagerank & co. : la taille doit pouvoir les porter."""
    assert render_network(graph, size_by="pagerank") != render_network(graph)


def test_an_empty_network_is_refused():
    with pytest.raises(NetworkFigureError):
        render_network({"nodes": [], "edges": []})


def test_an_unknown_format_is_refused(graph):
    with pytest.raises(NetworkFigureError):
        render_network(graph, fmt="webp")


def test_a_network_without_communities_still_draws():
    """Sans `annotate`, pas de communauté : couleur neutre, pas d'exception."""
    image = render_network(_GRAPH)
    assert image.startswith(b"\x89PNG")


# ---------------------------------------------------------------------------
# La disposition doit occuper le PLAN, pas une droite
# ---------------------------------------------------------------------------

_TWO_COMPONENTS = {
    "nodes": [{"id": n, "label": n, "weight": 10} for n in
              ["a", "b", "c", "d", "e", "x", "y", "z"]],
    "edges": [
        # une composante dense…
        {"source": "a", "target": "b", "weight": 9},
        {"source": "b", "target": "c", "weight": 7},
        {"source": "c", "target": "d", "weight": 8},
        {"source": "d", "target": "a", "weight": 6},
        {"source": "a", "target": "e", "weight": 5},
        {"source": "c", "target": "e", "weight": 4},
        # …et une autre, sans aucun lien avec la première.
        {"source": "x", "target": "y", "weight": 9},
        {"source": "y", "target": "z", "weight": 8},
        {"source": "x", "target": "z", "weight": 7},
    ],
}


def _span(coords, ids):
    xs = [coords[i][0] for i in ids]
    ys = [coords[i][1] for i in ids]
    return max(xs) - min(xs), max(ys) - min(ys)


def test_a_disconnected_component_does_not_flatten_the_map():
    """Le défaut qui rendait toute carte illisible.

    Les paires qu'aucun chemin ne relie recevaient une distance ARTIFICIELLE,
    grande, dans le même MDS que les autres. Ce contraste devenait le fait
    dominant du nuage : le premier axe servait à séparer les composantes, et
    la structure réelle se repliait sur une droite (mesuré sur un corpus
    réel : 0,18 d'étendue en X contre 1,32 en Y).
    """
    from bibliominer_analysis.networks.analysis import layout

    coords = layout(_TWO_COMPONENTS)
    width, height = _span(coords, ["a", "b", "c", "d", "e"])
    ratio = min(width, height) / max(width, height)

    assert ratio > 0.2, (
        f"la composante principale est aplatie (rapport {ratio:.2f}) : "
        "elle occupe une droite, pas le plan"
    )


def test_every_node_of_a_component_gets_its_own_place():
    """Les trois nœuds isolés se retrouvaient tous au MÊME point."""
    from bibliominer_analysis.networks.analysis import layout

    coords = layout(_TWO_COMPONENTS)
    small = [tuple(coords[i]) for i in ("x", "y", "z")]
    assert len(set(small)) == 3, "des nœuds distincts sont empilés au même endroit"


def test_components_do_not_sit_on_top_of_each_other():
    """Juxtaposées, pas superposées : sinon on lirait un seul amas. Côte à
    côte OU l'une sous l'autre (rangées) : leurs cadres sont disjoints."""
    from bibliominer_analysis.networks.analysis import layout

    coords = layout(_TWO_COMPONENTS)
    main = [coords[i] for i in ("a", "b", "c", "d", "e")]
    other = [coords[i] for i in ("x", "y", "z")]

    def apart(axis):
        return (min(p[axis] for p in other) > max(p[axis] for p in main)
                or min(p[axis] for p in main) > max(p[axis] for p in other))

    assert apart(0) or apart(1)


def _many_small_groups(groups=18, size=3):
    """Le cas d'un vrai réseau de co-auteurs : une vingtaine de petites
    équipes sans lien entre elles."""
    nodes, edges = [], []
    for g in range(groups):
        ids = [f"g{g}n{k}" for k in range(size)]
        nodes += [{"id": i, "label": i, "occurrences": 1 + (g + k) % 5}
                  for k, i in enumerate(ids)]
        edges += [{"source": a, "target": b, "weight": 1}
                  for a in ids for b in ids if a < b]
    return {"nodes": nodes, "edges": edges}


def test_many_components_fill_the_frame_not_a_strip():
    """Constaté sur un vrai corpus : dix-sept équipes alignées sur UNE ligne,
    une bande vingt fois plus large que haute, cadrée à l'écran, un chapelet
    de disques empilés. Elles remplissent maintenant des rangées."""
    import numpy as np
    from bibliominer_analysis.networks.analysis import layout

    xy = np.array(list(layout(_many_small_groups()).values()))
    width, height = np.ptp(xy, axis=0)
    assert width / height < 3.0, f"une bande {width / height:.1f} fois plus large que haute"


def test_no_disc_covers_another_at_drawing_size():
    """Chaque nœud est écarté de ses voisins de la somme de leurs rayons
    DESSINÉS (10 + 26·√(occ/max) px de diamètre, cadre de la carte)."""
    import numpy as np
    from bibliominer_analysis.networks.analysis import (LAYOUT_HEIGHT, LAYOUT_WIDTH,
                                                        layout)

    graph = _many_small_groups(groups=25, size=4)
    coords = layout(graph)
    ids = sorted(coords)
    xy = np.array([coords[i] for i in ids])
    span = np.ptp(xy, axis=0)
    xy = (xy - xy.min(axis=0)) * min(LAYOUT_WIDTH / span[0], LAYOUT_HEIGHT / span[1])
    occ = {n["id"]: n["occurrences"] for n in graph["nodes"]}
    top = max(occ.values())
    radius = np.array([(10 + 26 * np.sqrt(occ[i] / top)) / 2 for i in ids])
    dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    assert (dist >= radius[:, None] + radius[None] - 0.5).all()


def test_the_layout_stays_the_same_across_runs():
    """Y compris l'ORDRE dans lequel les composantes sont juxtaposées."""
    from bibliominer_analysis.networks.analysis import layout
    assert layout(_TWO_COMPONENTS) == layout(_TWO_COMPONENTS)


def test_a_dense_group_is_spread_out_not_stacked():
    """Constaté : le MDS classique empilait les co-auteurs d'un même groupe
    au même endroit (une colonne de disques, noms superposés). La carte est
    maintenant desserrée, et reste identique d'un calcul à l'autre."""
    import numpy as np
    from bibliominer_analysis.networks.analysis import layout
    # une clique de 12 nœuds : toutes les distances égales, le cas qui écrasait
    nodes = [{"id": f"n{i}", "label": f"N{i}", "occurrences": 3} for i in range(12)]
    edges = [{"source": f"n{i}", "target": f"n{j}", "weight": 2}
             for i in range(12) for j in range(i + 1, 12)]
    g = {"nodes": nodes, "edges": edges}
    a, b = layout(g), layout(g)
    assert a == b                                   # déterministe
    xy = np.array(list(a.values()))
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert d.min() > 0.3                            # aucun nœud sur un autre


def test_labels_are_short_and_printable():
    from bibliominer_analysis.figures.network import short_label
    from bibliominer_analysis.figures.palette import tick_label
    assert short_label("Ali Idri (2015) · Accuracy Comparison of Analogy-Based") == "Ali Idri (2015)"
    assert short_label("Fernández‐Alemán J.L.") == "Fernández-Alemán J.L."
    long_name = ("Proceedings of the Annual International Conference of the IEEE "
                 "Engineering in Medicine and Biology Society, EMBS")
    lines = tick_label(long_name, 34).split("\n")
    assert len(lines) == 2 and all(len(x) <= 34 for x in lines)
    assert lines[-1].endswith("…")
    assert tick_label("IEEE Software") == "IEEE Software"


def test_nodes_are_sized_by_occurrences_when_weight_is_absent():
    from bibliominer_analysis.figures.network import _node_areas
    areas = _node_areas([{"occurrences": 1}, {"occurrences": 10}], "weight")
    assert areas[1] > areas[0]
