"""The package must be able to DRAW a network, not only compute it.

Without this rendering, someone installing `bibliominer-analysis` without
the web application got figures and no map: networks only existed as
JSON, and their drawing lived in the browser.

Two requirements carry these tests:

  - **the map replays**: two runs on the same data give the same file,
    otherwise a published figure would move from one draw to the next and
    become indefensible;
  - **nothing is invented**: an empty network raises rather than returning
    a blank image that would be taken as valid.
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
    assert image.startswith(b"\x89PNG"), "this is not a PNG"
    assert len(image) > 5_000, "suspiciously empty image"


@pytest.mark.parametrize("fmt", ["png", "jpg", "svg", "pdf"])
def test_the_same_data_gives_the_same_file(graph, fmt):
    """A published figure cannot move from one run to the next.

    Two sources of noise had to be neutralised: the timestamp matplotlib
    writes into the file, and, in SVG, element identifiers drawn from a
    random seed.
    """
    assert render_network(graph, fmt=fmt) == render_network(graph, fmt=fmt)


def test_the_layout_is_not_redrawn_at_random(graph):
    """Placement comes from a deterministic MDS, not from a simulation."""
    from bibliominer_analysis.networks.analysis import layout
    assert layout(graph) == layout(graph)


def test_given_coordinates_are_used_as_is(graph):
    """It must be possible to draw EXACTLY the map shown on screen.

    The web application computes the coordinates once; the exported figure
    must be the same image, otherwise the article does not show what the
    user saw.
    """
    coords = {n["id"]: [i * 1.0, -i * 1.0]
              for i, n in enumerate(graph["nodes"])}
    fixed = render_network(graph, coords=coords)
    assert fixed != render_network(graph)      # this is not the MDS map
    assert fixed == render_network(graph, coords=coords)


def test_node_size_can_carry_a_centrality(graph):
    """`annotate` sets pagerank and the others: size must be able to carry them."""
    assert render_network(graph, size_by="pagerank") != render_network(graph)


def test_an_empty_network_is_refused():
    with pytest.raises(NetworkFigureError):
        render_network({"nodes": [], "edges": []})


def test_an_unknown_format_is_refused(graph):
    with pytest.raises(NetworkFigureError):
        render_network(graph, fmt="webp")


def test_a_network_without_communities_still_draws():
    """Without `annotate`, no community: neutral colour, no exception."""
    image = render_network(_GRAPH)
    assert image.startswith(b"\x89PNG")


# ---------------------------------------------------------------------------
# The layout must fill the PLANE, not a line
# ---------------------------------------------------------------------------

_TWO_COMPONENTS = {
    "nodes": [{"id": n, "label": n, "weight": 10} for n in
              ["a", "b", "c", "d", "e", "x", "y", "z"]],
    "edges": [
        # a dense component...
        {"source": "a", "target": "b", "weight": 9},
        {"source": "b", "target": "c", "weight": 7},
        {"source": "c", "target": "d", "weight": 8},
        {"source": "d", "target": "a", "weight": 6},
        {"source": "a", "target": "e", "weight": 5},
        {"source": "c", "target": "e", "weight": 4},
        # ...and another one, without any link to the first.
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
    """The defect that made every map unreadable.

    Pairs that no path connects received an ARTIFICIAL, large distance, in the
    same MDS as the others. That contrast became the dominant fact of the
    cloud: the first axis served to separate the components, and the real
    structure folded onto a line (measured on a real corpus: 0.18 of extent
    in X against 1.32 in Y).
    """
    from bibliominer_analysis.networks.analysis import layout

    coords = layout(_TWO_COMPONENTS)
    width, height = _span(coords, ["a", "b", "c", "d", "e"])
    ratio = min(width, height) / max(width, height)

    assert ratio > 0.2, (
        f"the main component is flattened (ratio {ratio:.2f}): "
        "it fills a line, not the plane"
    )


def test_every_node_of_a_component_gets_its_own_place():
    """The three isolated nodes all ended up at the SAME point."""
    from bibliominer_analysis.networks.analysis import layout

    coords = layout(_TWO_COMPONENTS)
    small = [tuple(coords[i]) for i in ("x", "y", "z")]
    assert len(set(small)) == 3, "distinct nodes are stacked at the same place"


def test_components_do_not_sit_on_top_of_each_other():
    """Side by side, not stacked: otherwise one would read a single cluster.
    Side by side OR one below the other (rows): their frames are disjoint."""
    from bibliominer_analysis.networks.analysis import layout

    coords = layout(_TWO_COMPONENTS)
    main = [coords[i] for i in ("a", "b", "c", "d", "e")]
    other = [coords[i] for i in ("x", "y", "z")]

    def apart(axis):
        return (min(p[axis] for p in other) > max(p[axis] for p in main)
                or min(p[axis] for p in main) > max(p[axis] for p in other))

    assert apart(0) or apart(1)


def _many_small_groups(groups=18, size=3):
    """The case of a real co-authorship network: about twenty small teams with
    no link between them."""
    nodes, edges = [], []
    for g in range(groups):
        ids = [f"g{g}n{k}" for k in range(size)]
        nodes += [{"id": i, "label": i, "occurrences": 1 + (g + k) % 5}
                  for k, i in enumerate(ids)]
        edges += [{"source": a, "target": b, "weight": 1}
                  for a in ids for b in ids if a < b]
    return {"nodes": nodes, "edges": edges}


def test_many_components_fill_the_frame_not_a_strip():
    """Found on a real corpus: seventeen teams aligned on ONE line, a band twenty
    times wider than tall, framed on screen, a string of stacked discs. They
    now fill rows."""
    import numpy as np
    from bibliominer_analysis.networks.analysis import layout

    xy = np.array(list(layout(_many_small_groups()).values()))
    width, height = np.ptp(xy, axis=0)
    assert width / height < 3.0, f"a band {width / height:.1f} times wider than tall"


def test_no_disc_covers_another_at_drawing_size():
    """Each node is kept away from its neighbours by the sum of their DRAWN radii
    (diameter of 10 + 26·√(occ/max) px, map frame)."""
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
    """Including the ORDER in which the components are placed side by side."""
    from bibliominer_analysis.networks.analysis import layout
    assert layout(_TWO_COMPONENTS) == layout(_TWO_COMPONENTS)


def test_a_dense_group_is_spread_out_not_stacked():
    """Found: classical MDS stacked the co-authors of the same group at the same
    place (a column of discs, overlapping names). The map is now loosened,
    and stays identical from one computation to the next."""
    import numpy as np
    from bibliominer_analysis.networks.analysis import layout
    # a clique of 12 nodes: all distances equal, the case that collapsed
    nodes = [{"id": f"n{i}", "label": f"N{i}", "occurrences": 3} for i in range(12)]
    edges = [{"source": f"n{i}", "target": f"n{j}", "weight": 2}
             for i in range(12) for j in range(i + 1, 12)]
    g = {"nodes": nodes, "edges": edges}
    a, b = layout(g), layout(g)
    assert a == b                                   # deterministic
    xy = np.array(list(a.values()))
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert d.min() > 0.3                            # no node on top of another


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
