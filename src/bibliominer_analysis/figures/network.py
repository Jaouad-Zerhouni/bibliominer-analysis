"""Dessin d'un réseau bibliométrique, sans navigateur.

Pourquoi ce module existe
-------------------------
Le package sait CONSTRUIRE les réseaux (`networks.build`), les annoter en
communautés et centralités (`networks.analysis.annotate`) et placer leurs
nœuds (`networks.analysis.layout`). Mais il ne savait pas les DESSINER :
`figures.render` ne couvre que barres, lignes et nuages de points, et le
rendu des réseaux vivait uniquement dans le navigateur de l'application web.

Quelqu'un qui installe seulement `bibliominer-analysis`, un carnet, un
script, une chaîne d'intégration, obtenait donc des chiffres et aucune
carte. C'est ce que ce module comble.

Ce qu'il garantit
-----------------
**Le dessin est déterministe.** Les coordonnées viennent de
`networks.analysis.layout`, un MDS sur les plus courts chemins : deux
exécutions sur les mêmes données donnent la même carte. Une figure publiée
dans un article ne peut pas bouger d'une exécution à l'autre, c'est
précisément ce qu'une simulation de forces ne sait pas promettre.

**Rien n'est inventé.** La taille d'un nœud porte une grandeur calculée (son
poids, ou une centralité si on la demande), sa couleur porte sa communauté.
Un nœud sans communauté reçoit une couleur neutre, jamais une teinte prise
au hasard dans la palette.
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

#: Bornes de l'aire des disques, en points². En dessous, un nœud disparaît ;
#: au-dessus, il masque ses voisins et la carte devient illisible.
_MIN_AREA, _MAX_AREA = 30.0, 700.0

#: Groupes nommés dans la légende ; les suivants sont résumés en une ligne.
_LEGEND_MAX = 10


class NetworkFigureError(ValueError):
    """Le réseau ne peut pas être dessiné, et on dit pourquoi."""


def _node_areas(nodes: List[Dict[str, Any]], size_by: str) -> List[float]:
    """Aire de chaque disque, proportionnelle à la grandeur demandée.

    L'AIRE, pas le rayon : c'est la surface que l'œil compare. Doubler le
    rayon quadruplerait la tache pour une valeur seulement deux fois plus
    grande, et la carte exagérerait ses propres écarts.

    Toutes les valeurs égales -> tous les disques identiques, plutôt qu'une
    division par zéro.
    """
    # La grandeur demandée n'existe pas sur ce réseau (``weight`` n'est posé
    # qu'après `annotate`) : on prend le nombre d'occurrences, que porte tout
    # nœud. Sans ce repli, tous les disques avaient la même taille.
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
    """Une couleur par communauté, la même d'un dessin à l'autre.

    On suit le NUMÉRO de communauté posé par `annotate`, pas l'ordre
    d'apparition des nœuds : sinon deux exécutions coloreraient
    différemment les mêmes groupes, et comparer deux cartes deviendrait
    impossible.
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
    """Les liens, sous les nœuds, avec une épaisseur qui suit leur poids.

    Une carte où tous les liens pèsent visuellement pareil ne dit rien de sa
    propre structure : c'est justement l'inégalité des poids qui fait
    apparaître les regroupements.

    Au-delà de trois liens par nœud, le fond s'efface : huit cents liens à
    la même opacité faisaient une pelote grise qui cachait les nœuds. Les
    liens forts restent nets.
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
    """Étiquette les plus gros nœuds, en écartant celles qui se recouvrent.

    Deux étiquettes superposées n'en font pas deux illisibles : elles en font
    UNE fausse, où l'œil lit des mots qui n'existent pas (« Class balance » et
    « Data preprocessing » imprimés l'un sur l'autre). Mieux vaut une carte
    qui en montre moins et dit vrai.

    On mesure les rectangles réellement rendus plutôt que de les estimer :
    la largeur d'un texte dépend de la police, de la taille, du DPI, une
    estimation se trompe précisément là où les mots sont longs.
    """
    ranked = sorted(zip(nodes, areas),
                    key=lambda pair: (-pair[1], str(pair[0].get("id"))))[:limit]
    fig.canvas.draw()                       # il faut un rendu pour mesurer
    renderer = fig.canvas.get_renderer()

    kept = []
    for node, area in ranked:
        x, y = positions[str(node["id"])]
        # Le décalage suit le RAYON du disque : sinon une grosse bulle
        # avale son étiquette, et une petite la laisse flotter loin.
        radius_pt = (area ** 0.5) / 2
        text = ax.annotate(
            short_label(node.get("label") or node.get("id")), (x, y),
            zorder=4, fontsize=7.5, color=colour, ha="center", va="bottom",
            xytext=(0, radius_pt + 3.5), textcoords="offset points")

        box = text.get_window_extent(renderer=renderer).expanded(1.03, 1.15)
        if any(box.overlaps(other) for other in kept):
            text.remove()                   # elle en cacherait une autre
        else:
            kept.append(box)


def short_label(label: Any, limit: int = 28) -> str:
    """L'étiquette À L'ÉCRAN d'un nœud : courte, lisible.

    Une référence co-citée s'appelle « Ali Idri (2015), Accuracy Comparison
    of Analogy-Based… » : sur la carte, « Ali Idri (2015) » suffit à la
    reconnaître, le titre complet reste dans la table du réseau. Au-delà de
    ``limit`` caractères, le texte est coupé d'un « … ».
    """
    from .palette import printable
    text = printable(label).strip()
    if " · " in text:
        text = text.split(" · ", 1)[0].strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _add_community_legend(ax, nodes, palette, skin) -> None:
    """Une légende dès qu'il y a plus d'une communauté.

    La couleur porte ici une IDENTITÉ (à quel regroupement appartient ce
    nœud). Sans légende, elle ne se lit pas : on voit trois familles sans
    savoir qu'on regarde des communautés, ni combien de nœuds pèse chacune.
    """
    from matplotlib.lines import Line2D

    counts: Dict[int, int] = {}
    for node in nodes:
        community = node.get("community")
        if community is not None:
            counts[int(community)] = counts.get(int(community), 0) + 1
    if len(counts) < 2:
        return

    # Au-delà de dix groupes, la légende devenait plus haute que la carte
    # (dix-sept lignes « Cluster 15 · 2 nodes »). Les dix plus gros sont
    # nommés, les autres résumés en une ligne.
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
    # SOUS la carte, jamais dedans. Posée dans un coin du graphe, elle
    # recouvrait des nœuds, et une légende qui cache la donnée qu'elle
    # explique est un contresens.
    legend = ax.legend(handles=handles, loc="upper left",
                       bbox_to_anchor=(0, -0.02), ncol=min(len(handles), 4),
                       frameon=False, fontsize=7.5,
                       handletextpad=0.4, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(skin["text_secondary"])


def _fit_margins(fig, ax, areas) -> None:
    """Élargit le cadre pour que le PLUS GROS disque tienne en entier.

    Une marge fixe est exprimée en fraction des données ; le rayon d'un nœud,
    lui, est en points. Un gros disque placé au bord se trouvait donc coupé
    par la moitié, et un nœud tronqué se lit comme une erreur de rendu, pas
    comme un nœud. On mesure la taille réelle des axes pour convertir le
    rayon en fraction, et on ajoute ce qu'il faut.
    """
    fig.canvas.draw()
    box = ax.get_window_extent()
    side_pt = min(box.width, box.height) * 72.0 / fig.dpi
    radius_pt = (max(areas) ** 0.5) / 2 if areas else 0.0
    ax.margins(0.06 + (radius_pt / side_pt if side_pt > 0 else 0.08))


def _in_render_rc(func):
    """Applique `RENDER_RC` le temps du rendu, et le rend ensuite intact."""
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
    """Dessine un réseau et renvoie les octets de l'image.

    ``graph``      le réseau tel que le rendent `networks.build`, puis
                   `networks.analysis.annotate` si l'on veut les communautés.
    ``coords``     les coordonnées ; calculées par MDS si on n'en donne pas.
    ``size_by``    la grandeur que porte la taille des nœuds : ``weight``, ou
                   toute clé posée par `annotate` (``pagerank``,
                   ``betweenness``…).
    ``label_top``  on n'étiquette que les N plus gros nœuds. En étiqueter
                   cinquante donne un enchevêtrement illisible ; n'en
                   étiqueter aucun rend la carte muette.

    Lève `NetworkFigureError` sur un réseau vide plutôt que de rendre une
    image blanche, qu'on croirait valide.
    """
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    if not nodes:
        raise NetworkFigureError("Empty network: nothing to draw.")
    if fmt not in _FORMATS:
        raise NetworkFigureError(
            f"Unsupported format '{fmt}'. Use one of {sorted(_FORMATS)}.")

    # Pas encore de communautés : on les calcule (déterministe), sinon tous
    # les nœuds restent gris et la carte ne montre aucun regroupement.
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

    # Une `Figure` avec son canevas Agg, jamais `pyplot` : ni fenêtre, ni
    # moteur d'affichage global changé chez l'utilisateur (voir RENDER_RC).
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

    # Une carte de réseau se lit en DISTANCES : deux nœuds proches sont
    # proches. Laisser matplotlib étirer un axe plus que l'autre déforme
    # exactement ce qu'on vient de calculer, un amas rond devient une
    # colonne, et la carte ment sur sa propre structure.
    # `box` et non `datalim` : on rétrécit le CADRE à la forme des données
    # plutôt que d.étirer les données pour remplir le cadre. Avec
    # `bbox_inches="tight"`, la figure se recadre sur la carte au lieu de
    # la noyer dans du blanc.
    ax.set_aspect("equal", adjustable="box")

    if title:
        ax.set_title(title, fontsize=11, color=skin["text_primary"],
                     loc="left", pad=12)

    # Une carte de réseau n'a PAS d'axes : les coordonnées d'un MDS n'ont
    # aucune unité, les graduer laisserait croire qu'elles se lisent.
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
