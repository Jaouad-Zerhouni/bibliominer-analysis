"""Cartes du monde publiables : carte des pays (choroplèthe) et carte des
collaborations internationales.

Le fond de carte est celui de l'interface web — `world-atlas`
(countries-110m, Natural Earth, licence ISC dans ``data_ref``) — pour que la
figure du package et la carte de l'écran montrent les mêmes frontières. Le
TopoJSON est décodé ici, sans dépendance cartographique (ni geopandas, ni
cartopy) : projection équirectangulaire, simple et lisible.

Les noms de pays du corpus ne sont pas ceux du fond de carte : Scopus écrit
« United States », « Russian Federation », « Viet Nam » ; la carte « United
States of America », « Russia », « Vietnam ». Sans table de correspondance,
ces pays restaient BLANCS sur la carte. `atlas_name` fait ce passage (voir
`io.countries`) ; l'interface le reçoit tout fait de `country_map`
(``map_name``, ``lon``, ``lat``), sans table à maintenir de son côté.
"""

from __future__ import annotations

import io
from typing import Dict, Iterable, List, Optional, Tuple

from ..io.countries import (ATLAS_ALIASES, _atlas, _key, atlas_name, centroids,  # noqa: F401
                            is_country, position)

#: Couleur des pays au-delà des huit teintes de la palette.
_OTHER = "#9aa0a6"


def _locate(country: object) -> Optional[Tuple[str, Tuple[float, float], bool]]:
    """(clé, position, est-un-point) d'un pays : son territoire sur le fond de
    carte, ou un point à sa capitale s'il est trop petit pour y figurer."""
    name = atlas_name(country)
    if name:
        return name, centroids()[name], False
    where = position(country)
    if where:
        return "point:" + _key(country), where, True
    return None


def _base(ax, fills: Dict[str, str], empty: str, edge: str) -> None:
    from matplotlib.collections import PolyCollection
    shapes, _ = _atlas()
    polys, colors = [], []
    for name, rings in shapes.items():
        if name == "Antarctica":
            continue
        for r in rings:
            polys.append(r)
            colors.append(fills.get(name, empty))
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=edge,
                                     linewidths=0.3))
    ax.set_xlim(-170, 190)
    ax.set_ylim(-58, 85)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


class _Countries:
    """Les pays d'une carte, rangés : clé (nom du fond de carte, ou « point:… »
    pour un petit pays), valeur cumulée, nom affiché, position."""

    def __init__(self, values: Dict[str, float]):
        self.value: Dict[str, float] = {}
        self.shown: Dict[str, str] = {}
        self.where: Dict[str, Tuple[float, float]] = {}
        self.dots = set()
        self.missing: List[str] = []
        for country, value in values.items():
            self.add(country, value)

    def add(self, country: object, value: object) -> None:
        found = _locate(country)
        if found is None:
            self.missing.append(str(country))
            return
        k, xy, is_dot = found
        self.value[k] = self.value.get(k, 0.0) + float(value or 0)
        self.shown.setdefault(k, str(country))
        self.where[k] = xy
        if is_dot:
            self.dots.add(k)

    @property
    def top(self) -> float:
        return max(self.value.values(), default=1.0) or 1.0

    def ranked(self) -> List[Tuple[str, float]]:
        return sorted(self.value.items(), key=lambda kv: (-kv[1], kv[0]))


def _tint(color: str) -> str:
    from matplotlib import colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((0.8 * r + 0.2, 0.8 * g + 0.2, 0.8 * b + 0.2))


def _arc(ax, start: Tuple[float, float], end: Tuple[float, float], width: float) -> None:
    """Un arc plutôt qu'un segment : les liens qui se superposent restent
    distincts. Arc neutre : la couleur désigne les pays, pas les liens."""
    (x0, y0), (x1, y1) = start, end
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.15 * abs(x1 - x0)
    steps = [i / 30 for i in range(31)]
    xs = [(1 - s) ** 2 * x0 + 2 * (1 - s) * s * mx + s ** 2 * x1 for s in steps]
    ys = [(1 - s) ** 2 * y0 + 2 * (1 - s) * s * my + s ** 2 * y1 for s in steps]
    ax.plot(xs, ys, color="#6b7280", alpha=0.45, linewidth=width, zorder=3)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _choropleth(fig, ax, countries: _Countries, value_label: str) -> None:
    """Carte des pays : une seule teinte, du clair au foncé."""
    import matplotlib
    from matplotlib import cm, colors as mcolors
    cmap = matplotlib.colormaps["Blues"]
    top = countries.top
    fills = {k: mcolors.to_hex(cmap(0.15 + 0.85 * v / top)) for k, v in countries.value.items()}
    _base(ax, fills, "#eceef1", "#ffffff")
    for k in countries.dots:
        ax.scatter([countries.where[k][0]], [countries.where[k][1]], s=36, color=fills[k],
                   edgecolors="#5f6368", linewidths=0.6, zorder=4)
    bar = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=top), cmap=cmap),
                       ax=ax, fraction=0.025, pad=0.01)
    bar.set_label(value_label)
    if countries.missing:
        fig.text(0.01, 0.01, "Not a recognised country: " + ", ".join(sorted(countries.missing)),
                 fontsize=7, color="#5f6368")


def _collaboration(ax, countries: _Countries, links: List[Tuple[str, str, float]],
                   value_label: str) -> None:
    """Carte des collaborations.

    Une couleur par pays, sur son territoire (ou sur un point pour un petit
    pays), définie dans la légende sous la carte : les noms écrits sur la
    carte se chevauchaient dès que des pays voisins collaborent. La palette
    validée a huit teintes distinguables ; au-delà, les pays partagent le
    gris « Other » plutôt qu'une neuvième couleur illisible.
    """
    from .palette import categorical
    palette = categorical("light")
    ranked = countries.ranked()
    colour = {k: palette[i] if i < len(palette) else _OTHER for i, (k, _) in enumerate(ranked)}
    _base(ax, {k: (_tint(colour[k]) if colour[k] != _OTHER else "#c3c7cd")
               for k in countries.value if k not in countries.dots}, "#eceef1", "#ffffff")
    for k in countries.dots:
        ax.scatter([countries.where[k][0]], [countries.where[k][1]], s=30, color=colour[k],
                   edgecolors="white", linewidths=0.6, zorder=5)

    placed = []
    for a, b, w in links:
        start, end = _locate(a), _locate(b)
        if start and end and start[0] != end[0]:
            placed.append((start[1], end[1], float(w)))
    wmax = max((w for _, _, w in placed), default=1.0) or 1.0
    for start, end, w in sorted(placed, key=lambda l: l[2]):
        _arc(ax, start, end, 0.6 + 3.4 * w / wmax)
    _legend(ax, countries, ranked, colour, len(palette), value_label)


def _legend(ax, countries: _Countries, ranked: List[Tuple[str, float]],
            colour: Dict[str, str], n_colours: int, value_label: str) -> None:
    import textwrap
    from matplotlib.lines import Line2D
    square = dict(marker="s", linestyle="none", markersize=9)
    handles = [Line2D([], [], markerfacecolor=_tint(colour[k]), markeredgecolor=colour[k],
                      label=f"{countries.shown[k]} ({_fmt(v)})", **square)
               for k, v in ranked[:n_colours]]
    others = ranked[n_colours:]
    if others:
        handles.append(Line2D([], [], markerfacecolor="#c3c7cd", markeredgecolor=_OTHER,
                              label="Other: " + ", ".join(
                                  f"{countries.shown[k]} ({_fmt(v)})" for k, v in others),
                              **square))
    if countries.missing:
        handles.append(Line2D([], [], linestyle="none", label="Not a recognised country: "
                              + ", ".join(sorted(countries.missing))))
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                       ncol=4, frameon=False, fontsize=8, handletextpad=0.3,
                       columnspacing=1.4, title_fontsize=8,
                       title=f"Colour: country ({value_label.lower()}) · "
                             "arc: co-authored documents, thicker = more")
    legend.get_title().set_color("#5f6368")
    for text in legend.get_texts():
        text.set_color("#1f2328")
        if len(text.get_text()) > 90:
            text.set_text(chr(10).join(textwrap.wrap(text.get_text(), 90)))


def render_world_map(values: Dict[str, float], value_label: str = "Documents",
                     links: Optional[Iterable[Tuple[str, str, float]]] = None,
                     fmt: str = "png", dpi: int = 200) -> bytes:
    """Carte du monde.

    ``values`` : {pays du corpus -> valeur}. Sans ``links``, carte des pays :
    chaque pays est coloré selon sa valeur (une seule teinte, du clair au
    foncé ; gris clair = absent du corpus). Avec ``links`` — des triplets
    (pays, pays, poids) — carte des collaborations : chaque pays a sa
    couleur, définie en légende, et un arc relie chaque paire, d'autant plus
    épais que le poids est fort.

    Un pays trop petit pour le fond de carte (Singapour, Bahreïn, Malte…) est
    dessiné par un point à sa capitale. Un nom qui n'est pas un pays connu
    n'est pas perdu en silence : il est listé sous la carte.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    countries = _Countries(values)
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=dpi)
    fig.patch.set_facecolor("white")
    if links is None:
        _choropleth(fig, ax, countries, value_label)
    else:
        _collaboration(ax, countries, list(links), value_label)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()
