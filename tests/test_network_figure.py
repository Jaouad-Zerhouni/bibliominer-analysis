"""Le package doit savoir DESSINER un réseau, pas seulement le calculer.

Sans ce rendu, quelqu'un qui installe `bibliominer-analysis` sans
l'application web obtenait des chiffres et aucune carte : les réseaux
n'existaient qu'en JSON, et leur dessin vivait dans le navigateur.

Deux exigences portent ces tests :

  - **la carte se rejoue** — deux exécutions sur les mêmes données donnent
    le même fichier, sans quoi une figure publiée bougerait d'un tirage à
    l'autre et deviendrait indéfendable ;
  - **rien n'est inventé** — un réseau vide lève plutôt que de rendre une
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
    matplotlib inscrit dans le fichier, et — en SVG — les identifiants
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
    """Juxtaposées, pas superposées : sinon on lirait un seul amas."""
    from bibliominer_analysis.networks.analysis import layout

    coords = layout(_TWO_COMPONENTS)
    main_x = [coords[i][0] for i in ("a", "b", "c", "d", "e")]
    other_x = [coords[i][0] for i in ("x", "y", "z")]
    assert min(other_x) > max(main_x) or min(main_x) > max(other_x)


def test_the_layout_stays_the_same_across_runs():
    """Y compris l'ORDRE dans lequel les composantes sont juxtaposées."""
    from bibliominer_analysis.networks.analysis import layout
    assert layout(_TWO_COMPONENTS) == layout(_TWO_COMPONENTS)
