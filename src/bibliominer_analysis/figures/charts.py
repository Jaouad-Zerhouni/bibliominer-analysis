"""Les graphiques de l'interface qui ne sont ni des barres, ni des lignes, ni
des nuages de points : treemap, nuage de mots, diagramme de Sankey (trois
champs, évolution thématique), dendrogramme, carte de densité.

Chacun prend le RÉSULTAT du calcul correspondant (celui que l'interface
affiche) et rend une figure matplotlib, PNG, SVG ou PDF, avec la même
palette que `render_figure`. Ainsi tout graphique de l'application a son
équivalent publiable depuis Python, sans navigateur.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .palette import RENDER_RC, categorical, no_timestamp, short_text

_INK = "#3b3f45"


def _figure(width: float, height: float, dpi: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("white")
    return plt, fig, ax


def _bytes(plt, fig, fmt: str) -> bytes:
    """Les octets de la figure, identiques d'un export à l'autre : ni date
    (`no_timestamp`), ni identifiants SVG tirés au hasard (graine fixe, celle
    de `RENDER_RC`). Sans la graine, le dendrogramme, les cartes de densité,
    le treemap et les Sankey changeaient à chaque export SVG."""
    buf = io.BytesIO()
    with plt.rc_context({"svg.hashsalt": RENDER_RC["svg.hashsalt"]}):
        fig.savefig(buf, format=fmt, bbox_inches="tight", facecolor="white",
                    **no_timestamp(fmt))
    plt.close(fig)
    return buf.getvalue()


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


# -- treemap -----------------------------------------------------------------

def _squarify(values: Sequence[float]) -> List[Tuple[float, float, float, float]]:
    """Rectangles (x, y, w, h) du treemap « squarified » (Bruls et al., 2000)
    dans le carré unité, dans l'ordre des valeurs (décroissantes)."""
    total = float(sum(values)) or 1.0
    scaled = [v / total for v in values]
    x, y, w, h = 0.0, 0.0, 1.0, 1.0
    out: List[Tuple[float, float, float, float]] = []
    row: List[float] = []

    def worst(r: List[float], side: float) -> float:
        s = sum(r)
        return max(max(side * side * v / (s * s), (s * s) / (side * side * v)) for v in r)

    i = 0
    while i < len(scaled):
        side = min(w, h)
        v = scaled[i]
        if not row or worst(row + [v], side) <= worst(row, side):
            row.append(v)
            i += 1
            if i < len(scaled):
                continue
        s = sum(row)
        if w >= h:
            dx, cy = s / h, y
            for v_ in row:
                out.append((x, cy, dx, v_ / dx))
                cy += v_ / dx
            x, w = x + dx, w - dx
        else:
            dy, cx = s / w, x
            for v_ in row:
                out.append((cx, y, v_ / dy, dy))
                cx += v_ / dy
            y, h = y + dy, h - dy
        row = []
    return out


def render_treemap(table: pd.DataFrame, label: str = "keyword", value: str = "documents",
                   n: int = 60, fmt: str = "png", dpi: int = 200) -> bytes:
    """Treemap d'un classement (mots-clés par défaut) : l'aire est la valeur."""
    t = table[[label, value]].dropna()
    t = t[t[value] > 0].sort_values(value, ascending=False, kind="stable").head(n)
    plt, fig, ax = _figure(10, 6, dpi)
    pal = categorical("light")
    from matplotlib.patches import Rectangle
    for i, ((rx, ry, rw, rh), name) in enumerate(zip(_squarify(list(t[value])), t[label])):
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor=pal[i % len(pal)],
                               edgecolor="white", linewidth=2))
        if rw * rh > 0.004:
            size = max(6, min(13, 60 * min(rw, rh)))
            ax.text(rx + rw / 2, ry + rh / 2, short_text(name, int(rw * 60) + 4),
                    ha="center", va="center", fontsize=size, color="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _bare(ax)
    return _bytes(plt, fig, fmt)


# -- nuage de mots -----------------------------------------------------------

def render_word_cloud(table: pd.DataFrame, label: str = "keyword", value: str = "documents",
                      n: int = 70, fmt: str = "png", dpi: int = 200) -> bytes:
    """Nuage de mots : le plus fréquent au centre, puis en spirale, sans
    chevauchement ; un mot qui ne trouve pas de place est omis."""
    t = table[[label, value]].dropna().sort_values(value, ascending=False,
                                                   kind="stable").head(n)
    plt, fig, ax = _figure(10, 6, dpi)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    _bare(ax)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    top = float(t[value].max() or 1)
    pal = categorical("light")
    placed: List[Tuple[float, float, float, float]] = []
    for i, (word, weight) in enumerate(zip(t[label], t[value])):
        text = ax.text(0, 0, str(word), fontsize=8 + 26 * (float(weight) / top) ** 0.7,
                       ha="center", va="center", color=pal[i % len(pal)])
        # La boîte du mot est mesurée UNE fois, au centre : déplacer un texte
        # ne change pas sa taille. Chaque position de la spirale se teste
        # alors par un simple décalage, la mesurer à chaque essai (jusqu'à
        # 2 500 par mot) prenait dix secondes pour cent vingt mots.
        bb = text.get_window_extent(renderer)
        (bx0, by0), (bx1, by1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        for step in range(2500):
            angle, radius = step * 0.35, step * 0.0022
            cx, cy = radius * math.cos(angle), 0.7 * radius * math.sin(angle)
            x0, y0, x1, y1 = cx + bx0, cy + by0, cx + bx1, cy + by1
            box = (x0 - 0.01, y0 - 0.01, x1 + 0.01, y1 + 0.01)
            inside = x0 > -1 and x1 < 1 and y0 > -1 and y1 < 1
            if inside and all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]
                              for b in placed):
                text.set_position((cx, cy))
                placed.append(box)
                break
        else:
            text.remove()
    return _bytes(plt, fig, fmt)


# -- Sankey ------------------------------------------------------------------

def _sankey(ax, columns: List[List[str]], links: List[Tuple[str, str, float]],
            titles: List[str], weight: Optional[Dict[str, float]] = None) -> None:
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch, Rectangle
    out_w: Dict[str, float] = {}
    in_w: Dict[str, float] = {}
    for s, t, v in links:
        out_w[s] = out_w.get(s, 0) + v
        in_w[t] = in_w.get(t, 0) + v
    weight = weight or {}
    size = {n: max(out_w.get(n, 0), in_w.get(n, 0), weight.get(n, 0), 1e-9)
            for col in columns for n in col}
    gap = 0.02
    scale = min((1 - gap * (len(col) - 1)) / max(sum(size[n] for n in col), 1)
                for col in columns if col)
    pos: Dict[str, list] = {}
    for ci, col in enumerate(columns):
        y = 1.0
        for n in sorted(col, key=lambda n: -size[n]):
            hgt = size[n] * scale
            pos[n] = [ci, y - hgt, y, y, y]  # colonne, bas, haut, sortie, entrée
            y -= hgt + gap
    pal = categorical("light")
    colour = {n: pal[i % len(pal)]
              for i, n in enumerate(sorted(columns[0], key=lambda n: -size[n]))}
    width = 0.06
    for s, t, v in sorted(links, key=lambda l: -l[2]):
        if s not in pos or t not in pos:
            continue
        hs, ht = pos[s], pos[t]
        h = v * scale
        x0, x1 = hs[0] + width, ht[0]
        y0, y1 = hs[3], ht[4]
        hs[3] -= h
        ht[4] -= h
        c = colour.get(s) or colour.get(t) or "#9aa0a6"
        colour.setdefault(t, c)
        xm = (x0 + x1) / 2
        verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 - h), (xm, y1 - h),
                 (xm, y0 - h), (x0, y0 - h), (x0, y0)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=c, alpha=0.35,
                               edgecolor="none"))
    last = len(columns) - 1
    for n, (ci, bottom, top, _, _) in pos.items():
        ax.add_patch(Rectangle((ci, bottom), width, top - bottom, facecolor=_INK))
        right = ci < last
        ax.text(ci + width + 0.02 if right else ci - 0.02, (bottom + top) / 2,
                short_text(n.split(": ", 1)[-1], 30), fontsize=7,
                ha="left" if right else "right", va="center", color=_INK)
    for ci, title in enumerate(titles):
        ax.text(ci + width / 2, 1.04, title, ha="center", fontsize=9, weight="bold",
                color=_INK)
    ax.set_xlim(-0.6, last + 0.6)
    ax.set_ylim(-0.02, 1.08)
    _bare(ax)


def render_three_fields(links: pd.DataFrame,
                        titles: Sequence[str] = ("Authors", "Keywords", "Sources"),
                        fmt: str = "png", dpi: int = 200) -> bytes:
    """Diagramme à trois champs, depuis le résultat de `Corpus.three_fields`."""
    # Un nœud est identifié par sa colonne ET son nom : un mot-clé et une
    # revue peuvent porter le même nom (« Energies ») et ne doivent pas
    # fusionner. Le préfixe « colonne: » est retiré à l'affichage.
    first, second = links[links["depth"] == 0], links[links["depth"] == 1]
    tag = lambda col, names: [f"{col}: {v}" for v in names]
    columns = [list(dict.fromkeys(tag(0, first["source"]))),
               list(dict.fromkeys(tag(1, first["target"]) + tag(1, second["source"]))),
               list(dict.fromkeys(tag(2, second["target"])))]
    flows = [(f"{d}: {s}", f"{d + 1}: {t}", v)
             for s, t, v, d in zip(links["source"], links["target"], links["value"],
                                   links["depth"].astype(int))]
    plt, fig, ax = _figure(11, 7, dpi)
    _sankey(ax, columns, flows, list(titles))
    return _bytes(plt, fig, fmt)


def render_thematic_evolution(evolution: Dict[str, Any], fmt: str = "png",
                              dpi: int = 200) -> bytes:
    """Évolution thématique, depuis le résultat de `Corpus.thematic_evolution`.
    Un thème sans successeur garde la hauteur de ses occurrences."""
    periods = [p["label"] for p in evolution.get("periods", [])]
    nodes = pd.DataFrame(evolution.get("nodes", []))
    if nodes.empty:
        raise ValueError("thematic evolution without themes")
    columns = [list(nodes[nodes["period"] == p]["name"]) for p in periods]
    flows = [(f["source"], f["target"], f["value"]) for f in evolution.get("flows", [])]
    plt, fig, ax = _figure(11, 7, dpi)
    _sankey(ax, columns, flows, periods, dict(zip(nodes["name"], nodes["occurrences"])))
    return _bytes(plt, fig, fmt)


# -- dendrogramme ------------------------------------------------------------

def render_dendrogram(dendrogram: Dict[str, Any], fmt: str = "png", dpi: int = 200) -> bytes:
    """Dendrogramme des mots-clés, depuis `Corpus.topic_dendrogram`."""
    tree = dendrogram.get("tree")
    if not tree:
        raise ValueError("dendrogram without tree")
    plt, fig, ax = _figure(12, 6, dpi)
    leaves: List[str] = []

    def place(node) -> Tuple[float, float]:
        kids = node.get("children") or []
        if not kids:
            leaves.append(node.get("name", ""))
            return len(leaves) - 1, 0.0
        xs = [place(k) for k in kids]
        h = float(node.get("height", 0))
        for x, y in xs:
            ax.plot([x, x], [y, h], color=_INK, linewidth=1)
        ax.plot([min(x for x, _ in xs), max(x for x, _ in xs)], [h, h],
                color=_INK, linewidth=1)
        return sum(x for x, _ in xs) / len(xs), h

    place(tree)
    ax.set_xticks(range(len(leaves)))
    ax.set_xticklabels([short_text(l, 26) for l in leaves], rotation=90, fontsize=7)
    ax.set_xlabel("Keyword")
    ax.set_ylabel("Distance (1 - association)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _bytes(plt, fig, fmt)


# -- carte de densité --------------------------------------------------------

def render_density_map(density: Dict[str, Any], graph: Dict[str, Any], labels: int = 15,
                       fmt: str = "png", dpi: int = 200) -> bytes:
    """Carte de densité à la VOSviewer, depuis `Corpus.network_density` ;
    ``graph`` doit porter les positions (`Corpus.attach_layout`) pour écrire
    les termes principaux à leur place."""
    import numpy as np
    xs, ys = density["x"], density["y"]
    if not xs:
        raise ValueError("empty density map")
    grid = np.zeros((len(ys), len(xs)))
    for i, j, v in density["cells"]:
        grid[j][i] = v
    plt, fig, ax = _figure(10, 7, dpi)
    ax.imshow(grid, origin="lower", cmap="YlOrRd", aspect="equal",
              extent=(xs[0], xs[-1], ys[0], ys[-1]))
    for node in sorted(graph.get("nodes", []), key=lambda n: -n.get("occurrences", 0))[:labels]:
        if "x" in node and "y" in node:
            ax.text(node["x"], node["y"], short_text(node["label"], 22), fontsize=7,
                    ha="center", va="center", color="#1f2328")
    _bare(ax)
    return _bytes(plt, fig, fmt)
