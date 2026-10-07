"""Drawing a bibliometric network, without a browser.

Why this module exists
----------------------
The package knows how to BUILD networks (`networks.build`), annotate them
with communities and centralities (`networks.analysis.annotate`) and place
their nodes (`networks.analysis.layout`). But it could not DRAW them:
`figures.render` only covers bars, lines and scatter plots, and the
rendering of networks lived only in the browser of the web application.

Someone who installs only `bibliominer-analysis` (a notebook, a script, a
CI pipeline) therefore got figures and no map. That is the gap this module
fills.

What it guarantees
------------------
**The drawing is deterministic.** The coordinates come from
`networks.analysis.layout`, an MDS on shortest paths: two runs on the
same data give the same map. A figure published in an article cannot
move from one run to the next, which is precisely what a force simulation
cannot promise.

**Nothing is invented.** The size of a node carries a computed quantity
(its weight, or a centrality if asked), its colour carries its community.
A node without a community gets a neutral colour, never a hue picked at
random from the palette.
"""

from __future__ import annotations

import functools
import io
from typing import Any, Dict, List, Optional

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..networks.analysis import layout as compute_layout
from .palette import RENDER_RC, categorical, chrome, no_timestamp

__all__ = ["render_network", "NetworkFigureError"]

_FORMATS = {"png", "jpg", "svg", "pdf"}
_MPL_FORMAT = {"jpg": "jpeg"}

#: Bounds of the disc area, in points². Below, a node disappears; above, it
#: hides its neighbours and the map becomes unreadable.
_MIN_AREA, _MAX_AREA = 30.0, 700.0

#: Groups named in the legend; the following ones are summed up in one line.
_LEGEND_MAX = 10


class NetworkFigureError(ValueError):
    """The network cannot be drawn, and the figure says why."""


def _node_areas(nodes: List[Dict[str, Any]], size_by: str) -> List[float]:
    """Area of each disc, proportional to the requested quantity.

    The AREA, not the radius: the surface is what the eye compares. Doubling
    the radius would quadruple the spot for a value only twice as large, and
    the map would exaggerate its own differences.

    All values equal -> all discs identical, rather than a division by zero.
    """
    # The requested quantity does not exist on this network (``weight`` is only
    # set after `annotate`): the number of occurrences, which every node
    # carries, is used. Without this fallback, all the discs had the same size.
    if not any(node.get(size_by) is not None for node in nodes):
        size_by = "occurrences"
    values = [float(node.get(size_by) or 0.0) for node in nodes]
    low, high = min(values), max(values)
    if high <= low:
        return [(_MIN_AREA + _MAX_AREA) / 2] * len(values)
    span = high - low
    return [_MIN_AREA + (value - low) / span * (_MAX_AREA - _MIN_AREA)
            for value in values]


def _node_colours(nodes: List[Dict[str, Any]], palette: List[str],
                  neutral: str) -> List[str]:
    """One colour per community, the same from one drawing to the next.

    The community NUMBER set by `annotate` is followed, not the order in which
    nodes appear: otherwise two runs would colour the same groups differently,
    and comparing two maps would become impossible.
    """
    out: List[str] = []
    for node in nodes:
        community = node.get("community")
        if community is None:
            out.append(neutral)
        else:
            out.append(palette[int(community) % len(palette)])
    return out


def _draw_edges(ax, edges, positions, colour, n_nodes: int) -> None:
    """The links, under the nodes, with a width that follows their weight.

    A map where all links weigh the same visually says nothing about its own
    structure: it is precisely the inequality of the weights that reveals the
    groupings.

    Beyond three links per node, the background fades: eight hundred links at
    the same opacity made a grey tangle that hid the nodes. Strong links stay
    sharp.
    """
    weights = [float(edge.get("weight") or 0.0) for edge in edges]
    heaviest = max(weights) if weights else 0.0
    if heaviest <= 0:
        heaviest = 1.0
    floor = min(0.15, max(0.04, 0.15 * 3 * n_nodes / max(len(edges), 1)))

    for edge, weight in zip(edges, weights):
        source, target = str(edge.get("source")), str(edge.get("target"))
        if source not in positions or target not in positions:
            continue
        share = weight / heaviest
        ax.plot([positions[source][0], positions[target][0]],
                [positions[source][1], positions[target][1]],
                color=colour, zorder=1,
                linewidth=0.3 + 1.7 * share,
                alpha=floor + (0.6 - floor) * share)


def _place_labels(fig, ax, nodes, areas, positions, colour, limit: int) -> None:
    """Labels the largest nodes, leaving out those that overlap.

    Two overlapping labels do not make two unreadable ones: they make ONE
    wrong one, where the eye reads words that do not exist ("Class balance"
    and "Data preprocessing" printed on top of each other). A map that shows
    less and tells the truth is better.

    The rectangles actually rendered are measured rather than estimated: the
    width of a text depends on the font, the size, the DPI; an estimate goes
    wrong precisely where words are long.
    """
    ranked = sorted(zip(nodes, areas),
                    key=lambda pair: (-pair[1], str(pair[0].get("id"))))[:limit]
    fig.canvas.draw()                       # a rendering is needed to measure
    renderer = fig.canvas.get_renderer()

    kept = []
    for node, area in ranked:
        x, y = positions[str(node["id"])]
        # The offset follows the disc's RADIUS: otherwise a large bubble swallows
        # its label, and a small one leaves it floating far away.
        radius_pt = (area ** 0.5) / 2
        text = ax.annotate(
            short_label(node.get("label") or node.get("id")), (x, y),
            zorder=4, fontsize=7.5, color=colour, ha="center", va="bottom",
            xytext=(0, radius_pt + 3.5), textcoords="offset points")

        box = text.get_window_extent(renderer=renderer).expanded(1.03, 1.15)
        if any(box.overlaps(other) for other in kept):
            text.remove()                   # it would hide another one
        else:
            kept.append(box)


def short_label(label: Any, limit: int = 28) -> str:
    """The ON-SCREEN label of a node: short, readable.

    A co-cited reference is called "Ana Varela (2015), A Study of
    Analogy-Based...": on the map, "Ana Varela (2015)" is enough to recognise
    it; the full title stays in the network table. Beyond ``limit``
    characters, the text is cut with "…".
    """
    from .palette import printable
    text = printable(label).strip()
    if " · " in text:
        text = text.split(" · ", 1)[0].strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _add_community_legend(ax, nodes, palette, skin) -> None:
    """A legend as soon as there is more than one community.

    Colour carries an IDENTITY here (which group this node belongs to).
    Without a legend it cannot be read: one sees three families without
    knowing that they are communities, nor how many nodes each weighs.
    """
    from matplotlib.lines import Line2D

    counts: Dict[int, int] = {}
    for node in nodes:
        community = node.get("community")
        if community is not None:
            counts[int(community)] = counts.get(int(community), 0) + 1
    if len(counts) < 2:
        return

    # Beyond ten groups, the legend became taller than the map (seventeen
    # "Cluster 15 · 2 nodes" lines). The ten largest are named, the others
    # summed up in one line.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    shown, rest = sorted(ranked[:_LEGEND_MAX]), ranked[_LEGEND_MAX:]
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor=palette[community % len(palette)],
               markeredgecolor=skin["surface"],
               label=f"Cluster {community} · {size} nodes")
        for community, size in shown
    ]
    if rest:
        handles.append(Line2D([], [], linestyle="none", label=(
            f"+ {len(rest)} smaller clusters · {sum(s for _, s in rest)} nodes")))
    # BELOW the map, never inside it. Placed in a corner of the plot, it covered
    # nodes, and a legend that hides the data it explains defeats its purpose.
    legend = ax.legend(handles=handles, loc="upper left",
                       bbox_to_anchor=(0, -0.02), ncol=min(len(handles), 4),
                       frameon=False, fontsize=7.5,
                       handletextpad=0.4, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(skin["text_secondary"])


def _fit_margins(fig, ax, areas) -> None:
    """Widens the frame so that the LARGEST disc fits entirely.

    A fixed margin is expressed as a fraction of the data; a node's radius is
    in points. A large disc placed at the edge was therefore cut in half, and
    a truncated node reads as a rendering error, not as a node. The real size
    of the axes is measured to convert the radius into a fraction, and the
    necessary amount is added.
    """
    fig.canvas.draw()
    box = ax.get_window_extent()
    side_pt = min(box.width, box.height) * 72.0 / fig.dpi
    radius_pt = (max(areas) ** 0.5) / 2 if areas else 0.0
    ax.margins(0.06 + (radius_pt / side_pt if side_pt > 0 else 0.08))


def _in_render_rc(func):
    """Applies `RENDER_RC` for the duration of the rendering, and restores it
    untouched afterwards."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with matplotlib.rc_context(RENDER_RC):
            return func(*args, **kwargs)
    return wrapper


@_in_render_rc
def render_network(
    graph: Dict[str, Any],
    *,
    coords: Optional[Dict[str, List[float]]] = None,
    size_by: str = "weight",
    label_top: int = 15,
    title: str = "",
    fmt: str = "png",
    mode: str = "light",
    width: float = 9.0,
    height: float = 7.0,
    dpi: int = 200,
) -> bytes:
    """Draws a network and returns the image's bytes.

    ``graph``      the network as returned by `networks.build`, then
                   `networks.analysis.annotate` if communities are wanted.
    ``coords``     the coordinates; computed by MDS if none are given.
    ``size_by``    the quantity carried by the size of the nodes: ``weight``,
                   or any key set by `annotate` (``pagerank``,
                   ``betweenness``...).
    ``label_top``  only the N largest nodes are labelled. Labelling fifty
                   gives an unreadable tangle; labelling none leaves the map
                   mute.

    Raises `NetworkFigureError` on an empty network rather than returning a
    blank image that would be taken as valid.
    """
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    if not nodes:
        raise NetworkFigureError("Empty network: nothing to draw.")
    if fmt not in _FORMATS:
        raise NetworkFigureError(
            f"Unsupported format '{fmt}'. Use one of {sorted(_FORMATS)}.")

    # No communities yet: they are computed (deterministic), otherwise every
    # node stays grey and the map shows no grouping.
    if not any(node.get("community") is not None for node in nodes):
        from ..networks.analysis import annotate
        graph = annotate({"nodes": nodes, "edges": edges})
        nodes = list(graph.get("nodes") or [])

    positions = coords if coords else compute_layout(graph)
    placed = [node for node in nodes if str(node.get("id")) in positions]
    if not placed:
        raise NetworkFigureError(
            "No node could be placed: the layout returned no coordinates.")

    skin = chrome(mode)
    palette = categorical(mode)

    # A `Figure` with its Agg canvas, never `pyplot`: no window, no global
    # display backend changed on the user's side (see RENDER_RC).
    fig = Figure(figsize=(width, height), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.patch.set_facecolor(skin["surface"])
    ax.set_facecolor(skin["surface"])

    if edges:
        _draw_edges(ax, edges, positions, skin["muted"], len(placed))

    areas = _node_areas(placed, size_by)
    ax.scatter([positions[str(node["id"])][0] for node in placed],
               [positions[str(node["id"])][1] for node in placed],
               s=areas, c=_node_colours(placed, palette, skin["muted"]),
               zorder=2, edgecolors=skin["surface"], linewidths=0.8)

    # A network map is read in DISTANCES: two close nodes are close. Letting
    # matplotlib stretch one axis more than the other distorts exactly what was
    # just computed: a round cluster becomes a column, and the map lies about
    # its own structure.
    # `box` and not `datalim`: the FRAME is shrunk to the shape of the data
    # rather than stretching the data to fill the frame. With
    # `bbox_inches="tight"`, the figure is cropped to the map instead of
    # drowning it in white space.
    ax.set_aspect("equal", adjustable="box")

    if title:
        ax.set_title(title, fontsize=11, color=skin["text_primary"],
                     loc="left", pad=12)

    # A network map has NO axes: MDS coordinates have no unit, and graduating
    # them would suggest they can be read.
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    _fit_margins(fig, ax, areas)

    if label_top > 0:
        _place_labels(fig, ax, placed, areas, positions, skin["text_primary"],
                      label_top)

    _add_community_legend(ax, placed, palette, skin)

    buffer = io.BytesIO()
    fig.savefig(buffer, format=_MPL_FORMAT.get(fmt, fmt),
                facecolor=fig.get_facecolor(), bbox_inches="tight",
                **no_timestamp(fmt))
    return buffer.getvalue()
