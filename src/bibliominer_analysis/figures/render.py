"""Rendu matplotlib, barres, lignes, aires, sucettes, secteurs, nuages.

Une figure exportée ou versée au rapport doit se lire SEULE, hors de
l'écran : titre et sous-titre (la période) écrits dessus, axes nommés,
couleurs qui portent un sens (dégradé selon la valeur, une teinte par
catégorie) et, au choix, la valeur au bout de chaque barre. L'utilisateur
choisit aussi la FORME : un classement se lit en barres, une part d'un tout
en secteurs.

Portée VOLONTAIREMENT limitée pour l'instant : ce sont les trois formes les
plus utilisées de l'application (barres/lignes : plus de cinquante graphiques
à elles deux) et les plus fidèlement reproductibles telles quelles. Les
formes plus riches, réseaux à disposition de forces, carte géographique,
sankey, nuage de points à bulles multi-dimensionnelles (taille ET couleur
portant chacune une variable), ne sont PAS couvertes : les reproduire
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

from .palette import RENDER_RC, categorical, chrome, no_timestamp

_SUPPORTED_KINDS = {"bar", "line", "scatter", "area", "lollipop", "pie", "donut"}
#: Formes qui n'ont qu'une série et pas d'axes : des PARTS d'un tout.
_PART_KINDS = {"pie", "donut"}
#: Au-delà, les plus petites parts se regroupent en « Other » : huit teintes
#: se distinguent, pas davantage (palette validée à huit).
_MAX_SLICES = 8
_PALETTES = {"series", "gradient", "category"}
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
    #: axes logarithmiques, Zipf (rang/fréquence) et Lotka se LISENT en
    #: log-log ; en échelle linéaire, la queue écrase tout le graphique.
    x_log: bool = False
    y_log: bool = False
    width_in: float = 7.2
    height_in: float = 4.2
    dpi: int = 200
    #: Écrire `title` (et `subtitle`, la période, les filtres) SUR la figure.
    #: Faux par défaut : `title` nomme aussi le fichier, et une figure
    #: destinée à un article porte souvent sa légende ailleurs. L'interface
    #: le propose, titre modifiable.
    show_title: bool = False
    subtitle: str = ""
    #: "series"   une teinte par série (une seule série : une seule teinte) ;
    #: "gradient" une série : du clair au foncé selon la VALEUR ;
    #: "category" une teinte par barre, par catégorie.
    palette: str = "series"
    #: Teinte de base (hex) d'une série unique, "series" ou "gradient".
    color: str = ""
    #: La valeur écrite au bout de chaque barre, ou sur chaque part.
    value_labels: bool = False


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
    if spec.palette not in _PALETTES:
        raise FigureError(f"Unknown palette '{spec.palette}'.")

    with matplotlib.rc_context(RENDER_RC):
        return _render(spec, fmt)


def _render(spec: FigureSpec, fmt: str) -> bytes:
    c = chrome(spec.mode)
    colors = categorical(spec.mode)

    # `constrained_layout` -- pas `tight_layout` : il recalcule à CHAQUE
    # ajout (légende, rotation d'étiquette), `tight_layout` ne le fait
    # qu'une fois et coupe régulièrement un nom d'axe long.
    # Une `Figure` avec son canevas Agg, jamais `pyplot` : aucune fenêtre,
    # aucun moteur d'affichage global à changer, le rendu marche sur un
    # serveur sans écran comme dans un notebook, sans rien y dérégler.
    fig = Figure(figsize=(spec.width_in, spec.height_in), dpi=spec.dpi,
                 constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    if spec.kind in _PART_KINDS:
        _draw_parts(ax, spec, colors, c)
        _write_title(fig, ax, spec, c)
        return _save(fig, fmt)

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
    elif spec.kind == "lollipop":
        _draw_lollipops(ax, spec, colors)
    elif spec.kind in ("line", "area"):
        _draw_lines(ax, spec, colors, fill=spec.kind == "area")
    else:
        _draw_scatter(ax, spec, colors)

    _apply_chrome(ax, spec, c)
    if spec.x_log:
        ax.set_xscale("log")
    if spec.y_log:
        ax.set_yscale("log")
    _write_title(fig, ax, spec, c)
    _declutter(fig, ax)
    return _save(fig, fmt)


def _save(fig, fmt: str) -> bytes:
    for ax in fig.axes:
        _align_heading(fig, ax)
    buf = io.BytesIO()
    fig.savefig(
        buf, format=_MPL_FORMAT.get(fmt, fmt), facecolor=fig.get_facecolor(),
        bbox_inches=None,  # `constrained_layout` gère déjà les marges.
        **no_timestamp(fmt),
    )
    return buf.getvalue()


def _write_title(fig, ax, spec: FigureSpec, c: Dict[str, str]) -> None:
    """Titre et sous-titre, calés à gauche au-dessus du tracé.

    Seulement si `show_title` : `spec.title` nomme aussi le fichier (voir
    `figures.py::_slug`), et une figure d'article porte souvent sa légende
    hors de l'image. Le sous-titre dit le PÉRIMÈTRE (période, filtres) : une
    figure sans lui se cite avec le mauvais chiffre.
    """
    if not spec.show_title:
        return
    from .palette import printable
    title = printable(spec.title).strip()
    subtitle = printable(spec.subtitle).strip()
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold",
                     color=c["text_primary"], fontfamily=_FONT_STACK)
    if subtitle:
        ax._bibliominer_subtitle = ax.set_title(
            subtitle, loc="left", fontsize=9.5, color=c["text_secondary"],
            fontfamily=_FONT_STACK, pad=10)


def _align_heading(fig, ax) -> None:
    """Cale le sous-titre sur le bord gauche de la FIGURE, sous le titre.

    Posé au-dessus du tracé, il commençait au bord de la zone de tracé :
    décalé à droite du titre de toute la largeur des noms d'auteurs. La mise
    en page est calculée une fois, puis figée, pour que l'enregistrement ne
    la déplace plus sous le sous-titre recalé."""
    subtitle = getattr(ax, "_bibliominer_subtitle", None)
    if subtitle is None:
        return
    fig.canvas.draw()
    fig.set_layout_engine("none")
    pos = ax.get_position()
    subtitle.set_x((0.01 - pos.x0) / pos.width)


def _apply_chrome(ax, spec: FigureSpec, c: Dict[str, str]) -> None:
    """La grille et les axes, communs aux formes à axes.

    Le titre n'est écrit que sur demande (`_write_title`) : `spec.title`
    nomme aussi le fichier téléchargé.
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
    # de barre, sur des barres horizontales, c'est l'axe X qui porte les
    # valeurs, Y n'étant plus qu'une liste de catégories.
    grid_axis = "x" if (spec.kind in ("bar", "lollipop")
                        and spec.orientation == "horizontal") else "y"
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
                f"{len(cats)} categorie(s), the two must match."
            )
        offset = (i - (n - 1) / 2) * width
        pos = [p + offset for p in positions]
        color = _bar_colors(spec, s, i, colors)
        if horizontal:
            bars = ax.barh(pos, s.values, height=width * 0.92, color=color,
                           label=s.name, zorder=3)
        else:
            bars = ax.bar(pos, s.values, width=width * 0.92, color=color,
                          label=s.name, zorder=3)
        if spec.value_labels:
            _label_bars(ax, bars, s.values, chrome(spec.mode)["text_secondary"])

    from .palette import tick_label
    if horizontal:
        ax.set_yticks(list(positions))
        ax.set_yticklabels([tick_label(c, 34) for c in cats])
        ax.invert_yaxis()  # la première catégorie en haut, comme à l'écran.
    else:
        _category_ticks(ax, cats)


def _base_color(spec: FigureSpec, index: int, colors: List[str]) -> str:
    """La teinte d'une série : celle choisie pour une série unique, sinon
    celle de son rang dans la palette."""
    if spec.color and len(spec.series) == 1:
        return spec.color
    return colors[index % len(colors)]


def _spread(colors: List[str], n: int) -> List[str]:
    """``n`` teintes distinctes, dans l'ORDRE de la palette validée.

    Jusqu'à huit, la palette telle quelle. Au-delà, des teintes intermédiaires
    entre deux couleurs voisines de la palette, jamais un recyclage : deux
    barres de même couleur se liraient comme la même catégorie.
    """
    if n <= len(colors):
        return colors[:n]
    from matplotlib.colors import to_hex, to_rgb
    rgb = [to_rgb(col) for col in colors]
    out = []
    for k in range(n):
        pos = k * (len(rgb) - 1) / max(n - 1, 1)
        lo = int(pos)
        hi = min(lo + 1, len(rgb) - 1)
        f = pos - lo
        out.append(to_hex(tuple(a + (b - a) * f for a, b in zip(rgb[lo], rgb[hi]))))
    return out


def _gradient(base: str, values: List[float], surface: str = "#ffffff") -> List[str]:
    """Une teinte, du clair (petite valeur) au foncé (grande valeur).

    Le plus clair garde 35 % de la teinte : une barre presque blanche sur
    fond blanc disparaîtrait."""
    from matplotlib.colors import to_hex, to_rgb
    b, s = to_rgb(base), to_rgb(surface)
    finite = [v for v in values if _finite(v)]
    lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
    out = []
    for v in values:
        f = 1.0 if hi == lo or not _finite(v) else (v - lo) / (hi - lo)
        mix = 0.35 + 0.65 * f
        out.append(to_hex(tuple(sv + (bv - sv) * mix for bv, sv in zip(b, s))))
    return out


def _bar_colors(spec: FigureSpec, s: Series, index: int, colors: List[str]):
    """Les couleurs d'une série de barres : une, un dégradé, ou une par
    catégorie. Dégradé et « une par catégorie » ne valent que pour une
    série seule : à plusieurs séries, la couleur désigne la SÉRIE."""
    base = _base_color(spec, index, colors)
    if len(spec.series) > 1 or spec.palette == "series":
        return base
    if spec.palette == "gradient":
        return _gradient(base, _num_list(s.values), chrome(spec.mode)["surface"])
    return _spread(colors, len(s.values or []))


def _finite(v) -> bool:
    """Une valeur présente : ni None ni NaN."""
    return v is not None and not math.isnan(float(v))


def _num_list(values) -> List[float]:
    out = []
    for v in values or []:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _fmt(value) -> str:
    """Un nombre écrit sur la figure : entier sans décimale, sinon deux."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(v):
        return ""
    if v.is_integer():
        return f"{int(v):,}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _label_bars(ax, bars, values, color: str) -> None:
    ax.bar_label(bars, labels=[_fmt(v) for v in values], padding=3, fontsize=8,
                 color=color, fontfamily=_FONT_STACK)
    # La valeur la plus grande ne doit pas sortir du cadre.
    ax.margins(x=0.08, y=0.08)


def _draw_lollipops(ax, spec: FigureSpec, colors: List[str]) -> None:
    """Une « sucette » par catégorie : une tige fine, un point au bout.

    Même lecture qu'une barre (la longueur), avec moins d'encre : un
    classement de trente noms reste léger."""
    cats = spec.categories or []
    if len(spec.series) != 1:
        raise FigureError("A lollipop chart shows a single series.")
    s = spec.series[0]
    if s.values is None or len(s.values) != len(cats):
        raise FigureError(
            f"'{s.name}': {len(s.values or [])} value(s) for "
            f"{len(cats)} categorie(s), the two must match."
        )
    color = _bar_colors(spec, s, 0, colors)
    dots = color if isinstance(color, list) else [color] * len(cats)
    positions = list(range(len(cats)))
    from .palette import tick_label
    if spec.orientation == "horizontal":
        ax.hlines(positions, 0, s.values, color=dots, linewidth=1.6, zorder=2)
        ax.scatter(s.values, positions, color=dots, s=46, zorder=3, label=s.name)
        ax.set_yticks(positions)
        ax.set_yticklabels([tick_label(c, 34) for c in cats])
        ax.invert_yaxis()
        if spec.value_labels:
            for y, v in zip(positions, s.values):
                ax.annotate(_fmt(v), (v, y), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=chrome(spec.mode)["text_secondary"])
            ax.margins(x=0.1)
        if min(_num_list(s.values), default=0) >= 0:
            ax.set_xlim(left=0)
    else:
        ax.vlines(positions, 0, s.values, color=dots, linewidth=1.6, zorder=2)
        ax.scatter(positions, s.values, color=dots, s=46, zorder=3, label=s.name)
        _category_ticks(ax, cats)
        if spec.value_labels:
            for x, v in zip(positions, s.values):
                ax.annotate(_fmt(v), (x, v), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=8, color=chrome(spec.mode)["text_secondary"])
            ax.margins(y=0.1)
        if min(_num_list(s.values), default=0) >= 0:
            ax.set_ylim(bottom=0)


def _draw_parts(ax, spec: FigureSpec, colors: List[str], c: Dict[str, str]) -> None:
    """Secteurs (« pie ») ou anneau (« donut ») : les PARTS d'un tout.

    Une seule série, des valeurs positives. Au-delà de huit parts, les plus
    petites se regroupent en « Other » : une neuvième teinte ne se
    distinguerait plus. La part en % est toujours écrite (un secteur sans
    son pourcentage ne se lit pas) ; la valeur, sur demande."""
    if len(spec.series) != 1:
        raise FigureError("A pie or donut chart shows a single series.")
    s = spec.series[0]
    cats = [str(x) for x in (spec.categories or [])]
    values = _num_list(s.values)
    if len(values) != len(cats):
        raise FigureError(
            f"'{s.name}': {len(values)} value(s) for {len(cats)} categorie(s), "
            "the two must match."
        )
    pairs = [(cat, v) for cat, v in zip(cats, values) if _finite(v) and v > 0]
    if any(v < 0 for v in values if _finite(v)):
        raise FigureError("A pie or donut chart needs positive values.")
    if not pairs:
        raise FigureError("Nothing to share out: every value is zero.")
    pairs.sort(key=lambda p: -p[1])
    if len(pairs) > _MAX_SLICES:
        head = pairs[: _MAX_SLICES - 1]
        pairs = head + [("Other", sum(v for _, v in pairs[_MAX_SLICES - 1:]))]
    labels = [p[0] for p in pairs]
    sizes = [p[1] for p in pairs]
    total = sum(sizes)
    palette = _spread(colors, len(sizes))

    def pct(share: float) -> str:
        if share < 4:
            return ""
        value = share * total / 100
        return f"{share:.0f}%\n{_fmt(value)}" if spec.value_labels else f"{share:.0f}%"

    donut = spec.kind == "donut"
    wedges, _, autotexts = ax.pie(
        sizes, colors=palette, startangle=90, counterclock=False, autopct=pct,
        pctdistance=0.79 if donut else 0.68,
        wedgeprops={"linewidth": 2, "edgecolor": c["surface"],
                    **({"width": 0.42} if donut else {})},
        textprops={"fontsize": 8.5, "color": "#ffffff", "fontweight": "semibold"},
    )
    # Blanc sur une part foncée, noir sur une part claire (le jaune) : un
    # pourcentage illisible ne sert à rien.
    from matplotlib.colors import to_rgb
    for wedge, text in zip(wedges, autotexts):
        r, g, b = to_rgb(wedge.get_facecolor())
        text.set_color("#0b0b0b" if 0.299 * r + 0.587 * g + 0.114 * b > 0.62 else "#ffffff")
    if donut:
        ax.text(0, 0, f"{_fmt(total)}\n{s.name or 'total'}", ha="center", va="center",
                fontsize=10, color=c["text_primary"], fontfamily=_FONT_STACK)
    ax.set_aspect("equal")
    from .palette import short_text
    legend = ax.legend(wedges, [f"{short_text(label, _LEGEND_CHARS)} ({size / total:.0%})"
                                for label, size in zip(labels, sizes)],
                       frameon=False, fontsize=9, labelcolor=c["text_secondary"],
                       loc="center left", bbox_to_anchor=(1.0, 0.5))
    for text in legend.get_texts():
        text.set_fontfamily(_FONT_STACK)


def _draw_lines(ax, spec: FigureSpec, colors: List[str], fill: bool = False) -> None:
    cats = spec.categories or []
    x = range(len(cats))
    for i, s in enumerate(spec.series):
        if s.values is None or len(s.values) != len(cats):
            raise FigureError(
                f"'{s.name}': {len(s.values or [])} value(s) for "
                f"{len(cats)} categorie(s), the two must match."
            )
        color = _base_color(spec, i, colors)
        # Un marqueur par point se lit jusqu'à une quarantaine de points ; au-
        # delà (138 auteurs de la loi de Price), la courbe devient un chapelet
        # épais. Une série de quelques points seulement garde les siens, sinon
        # elle serait invisible.
        finite = sum(1 for v in s.values if v is not None and v == v)
        marker = "o" if len(cats) <= 40 or finite <= 3 else None
        ax.plot(x, s.values, color=color, linewidth=2, marker=marker,
                markersize=5 if len(cats) <= 40 else 7, label=s.name, zorder=3)
        if fill:
            # L'aire se lit comme un VOLUME : transparente, pour que deux
            # séries superposées restent visibles l'une sous l'autre.
            ax.fill_between(list(x), [v if _finite(v) else 0 for v in _num_list(s.values)],
                            color=color, alpha=0.18 if len(spec.series) > 1 else 0.28,
                            linewidth=0, zorder=2)
        if spec.value_labels and len(cats) <= 40:
            for xi, v in zip(x, s.values):
                ax.annotate(_fmt(v), (xi, v), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=7.5, color=chrome(spec.mode)["text_secondary"])

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
                f"{len(cats)} categorie(s), the two must match."
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
                f"{len(cats)} categorie(s), the two must match."
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
                f"'{s.name}': a scatter plot needs (x, y) pairs, this "
                "series has none."
            )
        xs = [p[0] for p in s.points]
        ys = [p[1] for p in s.points]
        ax.scatter(xs, ys, color=_base_color(spec, i, colors), s=42,
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
