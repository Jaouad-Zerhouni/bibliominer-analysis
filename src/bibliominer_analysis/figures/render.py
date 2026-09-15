"""Rendu matplotlib — barres, lignes, nuages de points simples.

Portée VOLONTAIREMENT limitée pour l'instant : ce sont les trois formes les
plus utilisées de l'application (barres/lignes : plus de cinquante graphiques
à elles deux) et les plus fidèlement reproductibles telles quelles. Les
formes plus riches — réseaux à disposition de forces, carte géographique,
sankey, nuage de points à bulles multi-dimensionnelles (taille ET couleur
portant chacune une variable) — ne sont PAS couvertes : les reproduire
fidèlement demande un rendu dédié par forme, pas un mécanisme générique.
Un nuage de points dont les données ne sont qu'une liste de paires (x, y)
EST couvert ; un nuage de points à bulles ne l'est pas et lève `FigureError`
plutôt que de produire une figure qui mentirait sur les données.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .palette import NO_TIMESTAMP, RENDER_RC, categorical, chrome

_SUPPORTED_KINDS = {"bar", "line", "scatter"}
_SUPPORTED_FORMATS = {"png", "jpg", "svg", "pdf"}
# matplotlib ne connaît que "jpeg", pas "jpg" -- alias résolu au moment
# d'écrire le fichier, jamais avant : le NOM de fichier reste "jpg" partout
# ailleurs (extension, en-tête HTTP), ce que l'utilisateur reconnaît.
_MPL_FORMAT = {"jpg": "jpeg"}

# `family="sans-serif"` est un GROUPE : c'est la liste `RENDER_RC` (palette)
# que matplotlib consulte pour savoir laquelle du groupe utiliser.
_FONT_STACK = "sans-serif"


class FigureError(ValueError):
    """Donnée de figure que ce module ne sait PAS rendre fidèlement."""


@dataclass
class Series:
    name: str
    #: barres/lignes : une valeur par catégorie, dans le MÊME ordre.
    values: Optional[List[float]] = None
    #: nuage de points : paires (x, y). Jamais les deux à la fois qu'un
    #: `values` sur la même série -- une série est l'un OU l'autre.
    points: Optional[List[Tuple[float, float]]] = None
    #: Écrase `FigureSpec.kind` pour CETTE série -- "bar" | "line", jamais
    #: "scatter" (un nuage mélangé à des barres n'a pas de sens : les axes
    #: ne se lisent plus de la même façon). `None` = suit le type de la
    #: figure. Sert le motif le plus courant après les barres et lignes
    #: pures : un compte (barres) et sa tendance (ligne) sur le même axe
    #: catégoriel -- voir `_draw_mixed`.
    kind: Optional[str] = None


@dataclass
class FigureSpec:
    kind: str  # "bar" | "line" | "scatter"
    series: List[Series]
    categories: Optional[List[str]] = None  # barres/lignes
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    orientation: str = "vertical"  # "vertical" | "horizontal" -- barres seul.
    mode: str = "light"  # "light" | "dark"
    width_in: float = 7.2
    height_in: float = 4.2
    dpi: int = 200


def render_figure(spec: FigureSpec, fmt: str = "png") -> bytes:
    if spec.kind not in _SUPPORTED_KINDS:
        raise FigureError(
            f"« {spec.kind} » n'est pas encore couvert par l'export Python "
            "(seuls barres, lignes et nuages de points simples le sont)."
        )
    if fmt not in _SUPPORTED_FORMATS:
        raise FigureError(f"Format inconnu : « {fmt} ».")
    if not spec.series:
        raise FigureError("Aucune série à tracer.")

    with matplotlib.rc_context(RENDER_RC):
        return _render(spec, fmt)


def _render(spec: FigureSpec, fmt: str) -> bytes:
    c = chrome(spec.mode)
    colors = categorical(spec.mode)

    # `constrained_layout` -- pas `tight_layout` : il recalcule à CHAQUE
    # ajout (légende, rotation d'étiquette), `tight_layout` ne le fait
    # qu'une fois et coupe régulièrement un nom d'axe long.
    # Une `Figure` avec son canevas Agg, jamais `pyplot` : aucune fenêtre,
    # aucun moteur d'affichage global à changer — le rendu marche sur un
    # serveur sans écran comme dans un notebook, sans rien y dérégler.
    fig = Figure(figsize=(spec.width_in, spec.height_in), dpi=spec.dpi,
                 constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    series_kinds = {s.kind or spec.kind for s in spec.series}
    if len(series_kinds) > 1:
        if not series_kinds <= {"bar", "line"}:
            # Un nuage de points ne peut pas se mélanger à des barres/lignes
            # -- les axes ne se liraient plus de la même façon.
            raise FigureError(
                "Un nuage de points ne peut pas être mélangé à des barres ou "
                "des lignes sur la même figure."
            )
        _draw_mixed(ax, spec, colors)
    elif spec.kind == "bar":
        _draw_bars(ax, spec, colors)
    elif spec.kind == "line":
        _draw_lines(ax, spec, colors)
    else:
        _draw_scatter(ax, spec, colors)

    _apply_chrome(ax, spec, c)

    buf = io.BytesIO()
    fig.savefig(
        buf, format=_MPL_FORMAT.get(fmt, fmt), facecolor=fig.get_facecolor(),
        bbox_inches=None,  # `constrained_layout` gère déjà les marges.
        metadata=NO_TIMESTAMP.get(fmt),
    )
    return buf.getvalue()


def _apply_chrome(ax, spec: FigureSpec, c: Dict[str, str]) -> None:
    """La grille et les axes — communs aux trois formes.

    Pas de titre sur la figure elle-même : `spec.title` ne sert qu'à nommer
    le fichier téléchargé (voir `figures.py::_slug`), jamais à écrire sur
    l'image -- un article ou une diapositive porte déjà sa propre légende, et
    la dupliquer dans l'image fait doublon (ou pire, affiche le nom de
    fichier technique plutôt qu'un titre lisible).
    """
    font = {"family": _FONT_STACK}

    if spec.x_label:
        ax.set_xlabel(spec.x_label, color=c["text_secondary"], fontsize=10, **font)
    if spec.y_label:
        ax.set_ylabel(spec.y_label, color=c["text_secondary"], fontsize=10, **font)

    ax.tick_params(colors=c["muted"], labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(_FONT_STACK)
        label.set_color(c["text_secondary"])

    # Filet fin, jamais de cadre plein : la donnée porte le poids visuel, pas
    # la boîte qui la contient.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(0.8)

    # Le filet suit l'axe des VALEURS, pour aider à lire une hauteur/longueur
    # de barre — sur des barres horizontales, c'est l'axe X qui porte les
    # valeurs, Y n'étant plus qu'une liste de catégories.
    grid_axis = "x" if (spec.kind == "bar" and spec.orientation == "horizontal") else "y"
    ax.grid(axis=grid_axis, color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    if len(spec.series) > 1:
        legend = ax.legend(
            frameon=False, fontsize=9, labelcolor=c["text_secondary"],
            loc="upper left", bbox_to_anchor=(1.0, 1.0),
        )
        for text in legend.get_texts():
            text.set_fontfamily(_FONT_STACK)


def _draw_bars(ax, spec: FigureSpec, colors: List[str]) -> None:
    cats = spec.categories or []
    n = len(spec.series)
    width = 0.8 / max(n, 1)
    positions = range(len(cats))

    horizontal = spec.orientation == "horizontal"
    for i, s in enumerate(spec.series):
        if s.values is None or len(s.values) != len(cats):
            raise FigureError(
                f"« {s.name} » : {len(s.values or [])} valeur(s) pour "
                f"{len(cats)} catégorie(s) — les deux doivent s'accorder."
            )
        offset = (i - (n - 1) / 2) * width
        pos = [p + offset for p in positions]
        color = colors[i % len(colors)]
        if horizontal:
            ax.barh(pos, s.values, height=width * 0.92, color=color,
                    label=s.name, zorder=3)
        else:
            ax.bar(pos, s.values, width=width * 0.92, color=color,
                   label=s.name, zorder=3)

    if horizontal:
        ax.set_yticks(list(positions))
        ax.set_yticklabels(cats)
        ax.invert_yaxis()  # la première catégorie en haut, comme à l'écran.
    else:
        ax.set_xticks(list(positions))
        ax.set_xticklabels(cats, rotation=0 if len(cats) <= 12 else 45,
                           ha="right" if len(cats) > 12 else "center")


def _draw_lines(ax, spec: FigureSpec, colors: List[str]) -> None:
    cats = spec.categories or []
    x = range(len(cats))
    for i, s in enumerate(spec.series):
        if s.values is None or len(s.values) != len(cats):
            raise FigureError(
                f"« {s.name} » : {len(s.values or [])} valeur(s) pour "
                f"{len(cats)} catégorie(s) — les deux doivent s'accorder."
            )
        color = colors[i % len(colors)]
        ax.plot(x, s.values, color=color, linewidth=2, marker="o",
                markersize=5, label=s.name, zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=0 if len(cats) <= 12 else 45,
                       ha="right" if len(cats) > 12 else "center")


def _draw_mixed(ax, spec: FigureSpec, colors: List[str]) -> None:
    """Barres et lignes sur le MÊME axe catégoriel.

    Le motif le plus courant après les barres et lignes pures : un compte
    (barres) et sa tendance -- médiane glissante, moyenne mobile -- en
    ligne par-dessus. Les barres se partagent leur largeur ENTRE ELLES
    seulement ; une ligne n'occupe pas de créneau de largeur, elle trace
    juste au centre de chaque catégorie, exactement comme ECharts la
    positionne à l'écran.
    """
    cats = spec.categories or []
    bar_series = [s for s in spec.series if (s.kind or spec.kind) == "bar"]
    line_series = [s for s in spec.series if (s.kind or spec.kind) == "line"]
    x = range(len(cats))
    width = 0.8 / max(len(bar_series), 1)

    color_i = 0
    for i, s in enumerate(bar_series):
        if s.values is None or len(s.values) != len(cats):
            raise FigureError(
                f"« {s.name} » : {len(s.values or [])} valeur(s) pour "
                f"{len(cats)} catégorie(s) — les deux doivent s'accorder."
            )
        offset = (i - (len(bar_series) - 1) / 2) * width
        pos = [p + offset for p in x]
        ax.bar(pos, s.values, width=width * 0.92, color=colors[color_i % len(colors)],
               label=s.name, zorder=3)
        color_i += 1

    for s in line_series:
        if s.values is None or len(s.values) != len(cats):
            raise FigureError(
                f"« {s.name} » : {len(s.values or [])} valeur(s) pour "
                f"{len(cats)} catégorie(s) — les deux doivent s'accorder."
            )
        # zorder au-dessus des barres : la tendance reste lisible par-dessus
        # les colonnes, comme à l'écran.
        ax.plot(list(x), s.values, color=colors[color_i % len(colors)], linewidth=2,
                marker="o", markersize=4, label=s.name, zorder=4)
        color_i += 1

    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=0 if len(cats) <= 12 else 45,
                       ha="right" if len(cats) > 12 else "center")


def _draw_scatter(ax, spec: FigureSpec, colors: List[str]) -> None:
    for i, s in enumerate(spec.series):
        if not s.points:
            raise FigureError(
                f"« {s.name} » : un nuage de points a besoin de paires "
                "(x, y) — cette série n'en a pas."
            )
        xs = [p[0] for p in s.points]
        ys = [p[1] for p in s.points]
        ax.scatter(xs, ys, color=colors[i % len(colors)], s=42,
                   alpha=0.85, edgecolors="none", label=s.name, zorder=3)
