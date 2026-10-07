"""matplotlib rendering: bars, lines, areas, lollipops, pies, scatters.

A figure exported or added to the report must read ON ITS OWN, away from
the screen: title and subtitle (the period) written on it, named axes,
colours that carry a meaning (a gradient by value, one hue per category)
and, optionally, the value at the end of each bar. The user also chooses
the FORM: a ranking reads as bars, a share of a whole as a pie.

Scope DELIBERATELY limited for now: these are the three most used forms of
the application (bars/lines: more than fifty charts between them) and the
most faithfully reproducible as they are. Richer forms (force-directed
networks, geographic maps, Sankey diagrams, multi-dimensional bubble
scatters where size AND colour each carry a variable) are NOT covered:
reproducing them faithfully requires a dedicated rendering per form, not a
generic mechanism. A scatter whose data is only a list of (x, y) pairs IS
covered; a bubble scatter is not, and raises `FigureError` rather than
producing a figure that would lie about the data.
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
#: Forms with a single series and no axes: the SHARES of a whole.
_PART_KINDS = {"pie", "donut"}
#: Beyond this, the smallest shares are grouped into "Other": eight hues can
#: be told apart, no more (palette validated at eight).
_MAX_SLICES = 8
_PALETTES = {"series", "gradient", "category"}
_SUPPORTED_FORMATS = {"png", "jpg", "svg", "pdf"}
# matplotlib only knows "jpeg", not "jpg" -- an alias resolved when writing
# the file, never before: the file NAME stays "jpg" everywhere else
# (extension, HTTP header), which is what users recognise.
_MPL_FORMAT = {"jpg": "jpeg"}

# `family="sans-serif"` is a GROUP: it is the `RENDER_RC` list (palette)
# that matplotlib consults to know which member of the group to use.
_FONT_STACK = "sans-serif"


class FigureError(ValueError):
    """Figure data that this module CANNOT render faithfully."""


@dataclass
class Series:
    name: str
    #: bars/lines: one value per category, in the SAME order.
    values: Optional[List[float]] = None
    #: scatter: (x, y) pairs. Never both together with a `values` on the same
    #: series -- a series is one OR the other.
    points: Optional[List[Tuple[float, float]]] = None
    #: Overrides `FigureSpec.kind` for THIS series -- "bar" | "line", never
    #: "scatter" (a scatter mixed with bars makes no sense: the axes no longer
    #: read the same way). `None` = follows the figure's type. Serves the most
    #: common pattern after pure bars and lines: a count (bars) and its trend
    #: (line) on the same categorical axis -- see `_draw_mixed`.
    kind: Optional[str] = None
    #: scatter: one name per point, written next to it (thematic map, conceptual
    #: structure). Optional; only the points furthest from the centre are
    #: labelled if the names overlap.
    labels: Optional[List[str]] = None


@dataclass
class FigureSpec:
    kind: str  # "bar" | "line" | "scatter"
    series: List[Series]
    categories: Optional[List[str]] = None  # bars/lines
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    orientation: str = "vertical"  # "vertical" | "horizontal" -- bars only.
    mode: str = "light"  # "light" | "dark"
    #: logarithmic axes: Zipf (rank/frequency) and Lotka READ in log-log; on a
    #: linear scale, the tail crushes the whole chart.
    x_log: bool = False
    y_log: bool = False
    width_in: float = 7.2
    height_in: float = 4.2
    dpi: int = 200
    #: Write `title` (and `subtitle`: the period, the filters) ON the figure.
    #: False by default: `title` also names the file, and a figure meant for an
    #: article often carries its caption elsewhere. The interface offers it,
    #: with an editable title.
    show_title: bool = False
    subtitle: str = ""
    #: "series"   one hue per series (a single series: a single hue);
    #: "gradient" one series: from light to dark according to the VALUE;
    #: "category" one hue per bar, per category.
    palette: str = "series"
    #: Base hue (hex) of a single series, "series" or "gradient".
    color: str = ""
    #: The value written at the end of each bar, or on each share.
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

    # `constrained_layout` -- not `tight_layout`: it recomputes on EVERY
    # addition (legend, label rotation); `tight_layout` does it only once and
    # regularly cuts a long axis name.
    # A `Figure` with its Agg canvas, never `pyplot`: no window, no global
    # display backend to change; rendering works on a server without a screen
    # as in a notebook, without disturbing anything.
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
            # A scatter cannot be mixed with bars/lines -- the axes would no longer read
            # the same way.
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
        bbox_inches=None,  # `constrained_layout` already handles the margins.
        **no_timestamp(fmt),
    )
    return buf.getvalue()


def _write_title(fig, ax, spec: FigureSpec, c: Dict[str, str]) -> None:
    """Title and subtitle, aligned left above the plot.

    Only if `show_title`: `spec.title` also names the file (see
    `figures.py::_slug`), and an article figure often carries its caption
    outside the image. The subtitle states the SCOPE (period, filters): a
    figure without it gets cited with the wrong number.
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
    """Aligns the subtitle on the left edge of the FIGURE, below the title.

    Placed above the plot, it started at the edge of the plotting area:
    shifted to the right of the title by the whole width of the author names.
    The layout is computed once, then frozen, so that saving does not move it
    again under the realigned subtitle."""
    subtitle = getattr(ax, "_bibliominer_subtitle", None)
    if subtitle is None:
        return
    fig.canvas.draw()
    fig.set_layout_engine("none")
    pos = ax.get_position()
    subtitle.set_x((0.01 - pos.x0) / pos.width)


def _apply_chrome(ax, spec: FigureSpec, c: Dict[str, str]) -> None:
    """The grid and the axes, common to the forms with axes.

    The title is only written on request (`_write_title`): `spec.title` also
    names the downloaded file.
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

    # A thin rule, never a full frame: the data carries the visual weight, not
    # the box that holds it.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(0.8)

    # The rule follows the VALUES axis, to help read a bar's height/length; on
    # horizontal bars, the X axis carries the values, Y being only a list of
    # categories.
    grid_axis = "x" if (spec.kind in ("bar", "lollipop")
                        and spec.orientation == "horizontal") else "y"
    ax.grid(axis=grid_axis, color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # A COUNT (documents, citations) has no "2.5" tick: there is no half
    # document between two documents.
    counts = [v for s in spec.series for v in (s.values or [p[1] for p in s.points or []])
              if v is not None and not math.isnan(float(v))]
    value_log = spec.x_log if grid_axis == "x" else spec.y_log
    if counts and not value_log and all(float(v).is_integer() for v in counts):
        from matplotlib.ticker import MaxNLocator
        (ax.xaxis if grid_axis == "x" else ax.yaxis).set_major_locator(MaxNLocator(integer=True))

    if len(spec.series) > 1:
        # 120-character journal names pushed the legend out of the image and
        # crushed the chart to nothing. Each name is cut at 40 characters; a wide
        # or long legend goes BELOW the plot.
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


#: Maximum length of a name in a legend.
_LEGEND_CHARS = 40


#: beyond this, one tick out of two (three, ...): 77 Bradford ranks written
#: side by side overlapped into an unreadable band.
_MAX_X_TICKS = 25


def _category_ticks(ax, cats: List[str], width: int = 18) -> None:
    """Categorical ticks of the x axis, thinned out if they are too many to be
    read (the first and the last stay)."""
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
        ax.invert_yaxis()  # the first category at the top, as on screen.
    else:
        _category_ticks(ax, cats)


def _base_color(spec: FigureSpec, index: int, colors: List[str]) -> str:
    """The hue of a series: the one chosen for a single series, otherwise that of
    its rank in the palette."""
    if spec.color and len(spec.series) == 1:
        return spec.color
    return colors[index % len(colors)]


def _spread(colors: List[str], n: int) -> List[str]:
    """``n`` distinct hues, in the ORDER of the validated palette.

    Up to eight, the palette as it is. Beyond that, intermediate hues between
    two neighbouring colours of the palette, never recycling: two bars of the
    same colour would read as the same category.
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
    """One hue, from light (small value) to dark (large value).

    The lightest keeps 35 % of the hue: an almost white bar on a white
    background would disappear."""
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
    """The colours of a bar series: one, a gradient, or one per category.
    Gradient and "one per category" only apply to a single series: with
    several series, colour designates the SERIES."""
    base = _base_color(spec, index, colors)
    if len(spec.series) > 1 or spec.palette == "series":
        return base
    if spec.palette == "gradient":
        return _gradient(base, _num_list(s.values), chrome(spec.mode)["surface"])
    return _spread(colors, len(s.values or []))


def _finite(v) -> bool:
    """A present value: neither None nor NaN."""
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
    """A number written on the figure: an integer without decimals, otherwise
    two."""
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
    # The largest value must not go out of the frame.
    ax.margins(x=0.08, y=0.08)


def _draw_lollipops(ax, spec: FigureSpec, colors: List[str]) -> None:
    """One "lollipop" per category: a thin stem, a dot at the end.

    Same reading as a bar (the length), with less ink: a ranking of thirty
    names stays light."""
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
    """Pie or donut: the SHARES of a whole.

    A single series, positive values. Beyond eight shares, the smallest are
    grouped into "Other": a ninth hue would no longer be distinguishable. The
    share in % is always written (a slice without its percentage cannot be
    read); the value, on request."""
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
    # White on a dark slice, black on a light one (the yellow): an unreadable
    # percentage is useless.
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
        # A marker per point reads up to about forty points; beyond that (138
        # authors for Price's law), the curve becomes a thick string of beads. A
        # series of only a few points keeps its markers, otherwise it would be
        # invisible.
        finite = sum(1 for v in s.values if v is not None and v == v)
        marker = "o" if len(cats) <= 40 or finite <= 3 else None
        ax.plot(x, s.values, color=color, linewidth=2, marker=marker,
                markersize=5 if len(cats) <= 40 else 7, label=s.name, zorder=3)
        if fill:
            # An area reads as a VOLUME: transparent, so that two overlapping series
            # stay visible one under the other.
            ax.fill_between(list(x), [v if _finite(v) else 0 for v in _num_list(s.values)],
                            color=color, alpha=0.18 if len(spec.series) > 1 else 0.28,
                            linewidth=0, zorder=2)
        if spec.value_labels and len(cats) <= 40:
            for xi, v in zip(x, s.values):
                ax.annotate(_fmt(v), (xi, v), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=7.5, color=chrome(spec.mode)["text_secondary"])

    _category_ticks(ax, [str(c) for c in cats])


def _draw_mixed(ax, spec: FigureSpec, colors: List[str]) -> None:
    """Bars and lines on the SAME categorical axis.

    The most common pattern after pure bars and lines: a count (bars) and its
    trend -- rolling median, moving average -- as a line on top. The bars
    share their width AMONG THEMSELVES only; a line takes no width slot, it
    just draws at the centre of each category, exactly as ECharts positions it
    on screen.
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
        # zorder above the bars: the trend stays readable on top of the columns, as
        # on screen.
        ax.plot(list(x), s.values, color=colors[color_i % len(colors)], linewidth=2,
                marker="o", markersize=4, label=s.name, zorder=4)
        color_i += 1

    _category_ticks(ax, cats)


def _point_labels(ax) -> list:
    """The point labels of this chart, to be sorted out after rendering."""
    if not hasattr(ax, "_bibliominer_labels"):
        ax._bibliominer_labels = []
    return ax._bibliominer_labels


def _declutter(fig, ax) -> None:
    """Removes the point labels that overlap another one.

    The points furthest from the centre keep their name first: at the centre
    of a factorial map, terms pile up and read badly anyway, while the
    off-centre terms are the ones that give the axes their meaning. Done AFTER
    the scales (log) and the layout, when the on-screen positions are final.
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
    # The legend counts as already placed: a label does not go under it.
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
            # Points at the same place (themes with the same centrality and density)
            # wrote their names on top of each other: a single label per position,
            # "first name +n".
            from .palette import short_text
            at: Dict[Tuple[float, float], List[str]] = {}
            for (px, py), text in zip(s.points, s.labels):
                # An empty label: this point is not named.
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
