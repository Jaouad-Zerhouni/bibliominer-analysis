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
import math
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
    #: nuage de points : un nom par point, écrit à côté (carte thématique,
    #: structure conceptuelle). Facultatif ; seuls les points les plus
    #: éloignés du centre sont étiquetés si les noms se chevauchent.
    labels: Optional[List[str]] = None


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
    #: axes logarithmiques — Zipf (rang/fréquence) et Lotka se LISENT en
    #: log-log ; en échelle linéaire, la queue écrase tout le graphique.
    x_log: bool = False
    y_log: bool = False
    width_in: float = 7.2
    height_in: float = 4.2
    dpi: int = 200


def render_figure(spec: FigureSpec, fmt: str = "png") -> bytes:
    if spec.kind not in _SUPPORTED_KINDS:
        raise FigureError(
            f"'{spec.kind}' charts are not covered by the Python export yet "
            "(only bars, lines and simple scatter plots are)."
        )
    if fmt not in _SUPPORTED_FORMATS:
        raise FigureError(f"Unknown format '{fmt}'.")
    if not spec.series:
        raise FigureError("No series to draw.")

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
                "A scatter plot cannot be mixed with bars or lines on the "
                "same figure."
            )
        _draw_mixed(ax, spec, colors)
    elif spec.kind == "bar":
        _draw_bars(ax, spec, colors)
    elif spec.kind == "line":
        _draw_lines(ax, spec, colors)
    else:
        _draw_scatter(ax, spec, colors)

    _apply_chrome(ax, spec, c)
    if spec.x_log:
        ax.set_xscale("log")
    if spec.y_log:
        ax.set_yscale("log")
    _declutter(fig, ax)

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

    # Un COMPTE (documents, citations) n'a pas de graduation « 2,5 » : entre
    # deux documents il n'y en a pas un demi.
    counts = [v for s in spec.series for v in (s.values or [p[1] for p in s.points or []])
              if v is not None and not math.isnan(float(v))]
    value_log = spec.x_log if grid_axis == "x" else spec.y_log
    if counts and not value_log and all(float(v).is_integer() for v in counts):
        from matplotlib.ticker import MaxNLocator
        (ax.xaxis if grid_axis == "x" else ax.yaxis).set_major_locator(MaxNLocator(integer=True))

    if len(spec.series) > 1:
        # Des noms de revues de 120 caractères poussaient la légende hors de
        # l'image et écrasaient le graphique à rien. Chaque nom est coupé à
        # 40 caractères ; une légende large ou longue passe SOUS le tracé.
        from .palette import short_text
        handles, labels = ax.get_legend_handles_labels()
        labels = [short_text(label, _LEGEND_CHARS) for label in labels]
        below = len(labels) > 8 or max((len(label) for label in labels), default=0) > 24
        legend = ax.legend(
            handles, labels, frameon=False, fontsize=9, labelcolor=c["text_secondary"],
            **({"loc": "upper center", "bbox_to_anchor": (0.5, -0.16),
                "ncol": 2 if max(len(label) for label in labels) <= 32 else 1}
               if below else {"loc": "upper left", "bbox_to_anchor": (1.0, 1.0)}),
        )
        for text in legend.get_texts():
            text.set_fontfamily(_FONT_STACK)


#: Longueur maximale d'un nom dans une légende.
_LEGEND_CHARS = 40


#: au-delà, une graduation sur deux (trois, ...) : 77 rangs de Bradford
#: écrits côte à côte se chevauchaient en une bande illisible.
_MAX_X_TICKS = 25


def _category_ticks(ax, cats: List[str], width: int = 18) -> None:
    """Graduations catégorielles de l'axe x, clairsemées si elles sont trop
    nombreuses pour être lues (la première et la dernière restent)."""
    from .palette import tick_label
    step = max(1, -(-len(cats) // _MAX_X_TICKS))
    shown = list(range(0, len(cats), step))
    if shown and shown[-1] != len(cats) - 1 and len(cats) - 1 - shown[-1] >= step / 2:
        shown.append(len(cats) - 1)
    ax.set_xticks(shown)
    ax.set_xticklabels([tick_label(cats[i], width) for i in shown],
                       rotation=0 if len(shown) <= 12 else 45,
                       ha="right" if len(shown) > 12 else "center")


def _draw_bars(ax, spec: FigureSpec, colors: List[str]) -> None:
    cats = spec.categories or []
    n = len(spec.series)
    width = 0.8 / max(n, 1)
    positions = range(len(cats))

    horizontal = spec.orientation == "horizontal"
    for i, s in enumerate(spec.series):
        if s.values is None or len(s.values) != len(cats):
            raise FigureError(
                f"'{s.name}': {len(s.values or [])} value(s) for "
                f"{len(cats)} categorie(s) — the two must match."
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

    from .palette import tick_label
    if horizontal:
        ax.set_yticks(list(positions))
        ax.set_yticklabels([tick_label(c, 34) for c in cats])
        ax.invert_yaxis()  # la première catégorie en haut, comme à l'écran.
    else:
        _category_ticks(ax, cats)


def _draw_lines(ax, spec: FigureSpec, colors: List[str]) -> None:
    cats = spec.categories or []
    x = range(len(cats))
    for i, s in enumerate(spec.series):
        if s.values is None or len(s.values) != len(cats):
            raise FigureError(
                f"'{s.name}': {len(s.values or [])} value(s) for "
                f"{len(cats)} categorie(s) — the two must match."
            )
        color = colors[i % len(colors)]
        # Un marqueur par point se lit jusqu'à une quarantaine de points ; au-
        # delà (138 auteurs de la loi de Price), la courbe devient un chapelet
        # épais. Une série de quelques points seulement garde les siens, sinon
        # elle serait invisible.
        finite = sum(1 for v in s.values if v is not None and v == v)
        marker = "o" if len(cats) <= 40 or finite <= 3 else None
        ax.plot(x, s.values, color=color, linewidth=2, marker=marker,
                markersize=5 if len(cats) <= 40 else 7, label=s.name, zorder=3)

    _category_ticks(ax, [str(c) for c in cats])


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
                f"'{s.name}': {len(s.values or [])} value(s) for "
                f"{len(cats)} categorie(s) — the two must match."
            )
        offset = (i - (len(bar_series) - 1) / 2) * width
        pos = [p + offset for p in x]
        ax.bar(pos, s.values, width=width * 0.92, color=colors[color_i % len(colors)],
               label=s.name, zorder=3)
        color_i += 1

    for s in line_series:
        if s.values is None or len(s.values) != len(cats):
            raise FigureError(
                f"'{s.name}': {len(s.values or [])} value(s) for "
                f"{len(cats)} categorie(s) — the two must match."
            )
        # zorder au-dessus des barres : la tendance reste lisible par-dessus
        # les colonnes, comme à l'écran.
        ax.plot(list(x), s.values, color=colors[color_i % len(colors)], linewidth=2,
                marker="o", markersize=4, label=s.name, zorder=4)
        color_i += 1

    _category_ticks(ax, cats)


def _point_labels(ax) -> list:
    """Les étiquettes de points de ce graphique, à trier après le rendu."""
    if not hasattr(ax, "_bibliominer_labels"):
        ax._bibliominer_labels = []
    return ax._bibliominer_labels


def _declutter(fig, ax) -> None:
    """Retire les étiquettes de points qui en chevauchent une autre.

    Les points les plus éloignés du centre gardent leur nom en priorité :
    au centre d'une carte factorielle, les termes s'entassent et se lisent
    mal de toute façon, alors que les termes excentrés sont ceux qui
    donnent leur sens aux axes. Se fait APRÈS les échelles (log) et la mise
    en page, quand les positions à l'écran sont définitives.
    """
    labels = getattr(ax, "_bibliominer_labels", [])
    if len(labels) < 2:
        return
    xs = [x for _, x, _ in labels]
    ys = [y for _, _, y in labels]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = (max(xs) - min(xs)) or 1.0
    sy = (max(ys) - min(ys)) or 1.0
    order = sorted(labels, key=lambda l: -(((l[1] - cx) / sx) ** 2 + ((l[2] - cy) / sy) ** 2))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = ax.get_legend()
    # La légende compte comme déjà placée : une étiquette ne passe pas dessous.
    kept = [legend.get_window_extent(renderer)] if legend is not None else []
    for label, _, _ in order:
        box = label.get_window_extent(renderer).expanded(1.05, 1.15)
        if any(box.overlaps(other) for other in kept):
            label.remove()
        else:
            kept.append(box)


def _draw_scatter(ax, spec: FigureSpec, colors: List[str]) -> None:
    for i, s in enumerate(spec.series):
        if not s.points:
            raise FigureError(
                f"'{s.name}': a scatter plot needs (x, y) pairs — this "
                "series has none."
            )
        xs = [p[0] for p in s.points]
        ys = [p[1] for p in s.points]
        ax.scatter(xs, ys, color=colors[i % len(colors)], s=42,
                   alpha=0.85, edgecolors="none", label=s.name, zorder=3)
        if s.labels:
            # Des points au même endroit (thèmes de même centralité et
            # densité) écrivaient leurs noms l'un sur l'autre : une seule
            # étiquette par position, « premier nom +n ».
            from .palette import short_text
            at: Dict[Tuple[float, float], List[str]] = {}
            for (px, py), text in zip(s.points, s.labels):
                # Une étiquette vide : ce point-là n'est pas nommé.
                if text is not None and str(text).strip():
                    at.setdefault((round(px, 6), round(py, 6)), []).append(str(text))
            for (px, py), names in at.items():
                text = short_text(names[0], 24)
                if len(names) > 1:
                    text += f" +{len(names) - 1}"
                label = ax.annotate(text, (px, py), xytext=(4, 3),
                                    textcoords="offset points", fontsize=7,
                                    color="#3b3f45", zorder=4)
                _point_labels(ax).append((label, px, py))
