"""The figure catalogue: every figure of the interface, in Python.

The interface shows about sixty charts. The package could compute all
their data and DRAW bars, lines, scatters, networks and maps, but it was
up to the user to redo the assembly: which method, which column, which
axis. Here, every figure on screen has an entry:

    >>> corpus.figure_catalog()                       # the list, by page
    >>> corpus.figure("top-authors", n=10, fmt="svg", path="top_authors.svg")
    >>> corpus.filter(years=(2023, 2025)).figure("documents-per-year")

The NAME is that of the file exported by the interface (``top-authors``,
``collaboration-map``, ``network-keywords``...); ``top_authors`` is
accepted. The default OPTIONS are those of the screen (the top 20 authors,
the top 15 countries...): without options, you get the interface's
figure; with them, the same figure set up differently.

Two kinds of entries:
  - a "simple" figure (bars, lines, scatter): `figure_spec` returns its
    description (`FigureSpec`), editable before rendering;
  - a figure drawn by a dedicated renderer (network, world map, treemap,
    word cloud, Sankey, dendrogram, density map): only `figure` produces
    it.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .charts import (render_dendrogram, render_density_map, render_thematic_evolution,
                     render_three_fields, render_treemap, render_word_cloud)
from .network import render_network
from .render import FigureSpec, Series, render_figure
from .world import render_world_map

__all__ = ["CATALOG", "Entry", "catalog_table", "figure_spec", "figure_bytes", "entry"]

#: Axis titles: every figure names BOTH its axes.
AXIS = {
    "doc_type": "Document type", "year": "Year", "documents": "Documents",
    "citations": "Citations", "cumulative": "Cumulative documents",
    "h_index": "h-index", "local_citations": "Local citations",
    "global_citations": "Global citations", "self_rate": "Self-citation rate (%)",
    "self_citations": "Self-citations", "external_citations": "External citations",
    "year_median": "Median year", "rank": "Source rank (by documents)",
    "documents_written": "Documents written per author", "age": "Reference age (years)",
    "period": "Period", "authors": "Authors per document",
    "affiliations": "Affiliations per document", "label": "Document", "author": "Author",
    "source": "Source", "institution": "Institution", "city": "City", "country": "Country",
    "keyword": "Keyword", "term": "Term", "area": "Subject area", "quartile": "SCImago quartile",
    "scale": "Collaboration scale", "group": "Authorship group", "zone": "Bradford zone",
    "price_index": "Price index (%)", "references": "Cited references",
    "deviation": "Deviation from the 5-year median", "fitted": "Fitted documents",
    "open_access": "Open access documents", "share_observed": "Observed",
    "share_fitted": "Fitted (power law)", "share_lotka": "Lotka (exponent 2)",
    "median_5": "5-year median", "cai_single": "Single-authored", "cai_two": "Two authors",
    "cai_three": "Three authors", "cai_multi": "Four or more", "rgr": "RGR",
    "doubling_time": "Doubling time (years)", "scp": "Single-country (SCP)",
    "mcp": "Multi-country (MCP)",
}

#: The groups of the interface menu, by page.
_SECTION = {
    "Overview": "Corpus", "Production": "Corpus", "Documents": "Corpus",
    "Corpus profile": "Corpus", "Quality": "Corpus",
    "Authors": "Actors", "Institutions": "Actors", "Sources": "Actors",
    "Cities": "Actors", "Countries": "Actors",
    "Local impact": "Impact", "Spectroscopy": "Impact", "Indicators": "Impact",
    "Laws": "Impact",
    "Themes": "Concepts", "Text mining": "Concepts", "Thematic map": "Concepts",
    "Thematic evolution": "Concepts", "Conceptual structure": "Concepts",
    "Three fields": "Concepts",
    "Networks": "Networks", "Network lab": "Networks", "Citation network": "Networks",
    "Coupling clusters": "Networks",
}

#: A figure drawn by a dedicated renderer: ``(fmt, dpi) -> bytes``.
Renderer = Callable[[str, int], bytes]
Built = Union[FigureSpec, Renderer, None]


@dataclasses.dataclass(frozen=True)
class Entry:
    name: str
    title: str
    page: str
    build: Callable[..., Built]
    #: on-screen options, the defaults of `figure`
    options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    #: the `Corpus` method that carries the data, for the README
    source: str = ""

    @property
    def section(self) -> str:
        return _SECTION.get(self.page, "Other")


# ---------------------------------------------------------------------------
# Generic forms
# ---------------------------------------------------------------------------

def axis(column: str) -> str:
    return AXIS.get(column, column.replace("_", " ").capitalize())


def _num(values, keep_gaps: bool = False) -> List[float]:
    out = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = math.nan
        out.append(x if (keep_gaps or not math.isnan(x)) else 0.0)
    return out


def _empty(table: Optional[pd.DataFrame], *columns: str) -> bool:
    return table is None or table.empty or not set(columns) <= set(table.columns)


def hbar(table, label: str, values: Union[str, List[str]], n: Optional[int] = None,
         value_label: str = "") -> Optional[FigureSpec]:
    """A ranking: horizontal bars, the first at the top."""
    values = [values] if isinstance(values, str) else values
    if _empty(table, label, values[0]):
        return None
    t = table.sort_values(values[0], ascending=False, kind="stable")
    t = t.head(n) if n else t
    return FigureSpec(kind="bar", orientation="horizontal",
                      categories=[str(x) for x in t[label]],
                      series=[Series(name=axis(v), values=_num(t[v])) for v in values if v in t],
                      x_label=value_label or axis(values[0]), y_label=axis(label),
                      height_in=max(3.6, 0.32 * len(t) * max(1, len(values) * 0.7) + 1.2))


def vbar(table, cat: str, values: List[str], y_label: str = "") -> Optional[FigureSpec]:
    """Vertical bars, one series per column."""
    cols = [v for v in values if table is not None and v in table]
    if _empty(table, cat) or not cols:
        return None
    return FigureSpec(kind="bar", categories=[str(x) for x in table[cat]],
                      series=[Series(name=axis(v), values=_num(table[v])) for v in cols],
                      x_label=axis(cat), y_label=y_label or axis(cols[0]))


def lines(table, cat: str, values: List[str], y_label: str = "") -> Optional[FigureSpec]:
    cols = [v for v in values if table is not None and v in table]
    if _empty(table, cat) or not cols:
        return None
    return FigureSpec(kind="line", categories=[str(x) for x in table[cat]],
                      series=[Series(name=axis(v), values=_num(table[v], keep_gaps=True))
                              for v in cols],
                      x_label=axis(cat), y_label=y_label or axis(cols[0]))


def mixed(table, cat: str, bars: List[str], trend: List[str],
          y_label: str = "") -> Optional[FigureSpec]:
    """A count as bars, its trend as a line, on the same axis."""
    if _empty(table, cat):
        return None
    series = ([Series(name=axis(b), values=_num(table[b]), kind="bar") for b in bars if b in table]
              + [Series(name=axis(t), values=_num(table[t], keep_gaps=True), kind="line")
                 for t in trend if t in table])
    if not series:
        return None
    return FigureSpec(kind="bar", categories=[str(x) for x in table[cat]], series=series,
                      x_label=axis(cat), y_label=y_label or series[0].name)


def over_time(table, entity: str, value: str = "documents", top: int = 8,
              y_label: str = "") -> Optional[FigureSpec]:
    """A long table (entity, year, value) -> one line per leading entity."""
    if _empty(table, entity, "year", value):
        return None
    leaders = (table.groupby(entity)[value].sum().sort_values(ascending=False, kind="stable")
               .head(top).index.tolist())
    wide = (table[table[entity].isin(leaders)]
            .pivot_table(index="year", columns=entity, values=value, aggfunc="sum")
            .fillna(0).sort_index())
    return FigureSpec(kind="line", categories=[str(y) for y in wide.index],
                      series=[Series(name=str(e), values=_num(wide[e])) for e in leaders if e in wide],
                      x_label="Year", y_label=y_label or axis(value))


def points(groups: Dict[str, pd.DataFrame], x: str, y: str, label: Optional[str] = None,
           x_label: str = "", y_label: str = "", x_log: bool = False,
           y_log: bool = False) -> Optional[FigureSpec]:
    """A scatter, one series per group, labelled if ``label``."""
    series = []
    for name, t in groups.items():
        if t is None or t.empty or not {x, y} <= set(t.columns):
            continue
        t = t.dropna(subset=[x, y])
        if x_log:
            t = t[t[x] > 0]
        if y_log:
            t = t[t[y] > 0]
        if len(t):
            series.append(Series(name=name, points=list(zip(_num(t[x]), _num(t[y]))),
                                 labels=[str(v) for v in t[label]] if label else None))
    if not series:
        return None
    return FigureSpec(kind="scatter", series=series, x_label=x_label or axis(x),
                      y_label=y_label or axis(y), x_log=x_log, y_log=y_log)


def _network(graph_of: Callable[[], Dict[str, Any]], title: str,
             size_by: str = "occurrences") -> Optional[Renderer]:
    graph = graph_of()
    if not graph.get("nodes"):
        return None
    return lambda fmt, dpi: render_network(graph, title=title, fmt=fmt, dpi=dpi,
                                           size_by=size_by)


def _world(values: Dict[str, float], label: str, links=None) -> Optional[Renderer]:
    if not values:
        return None
    return lambda fmt, dpi: render_world_map(values, label, links=links, fmt=fmt, dpi=dpi)


# ---------------------------------------------------------------------------
# The interface's figures, page by page
# ---------------------------------------------------------------------------

CATALOG: Dict[str, Entry] = {}


def _add(name: str, title: str, page: str, build, source: str = "", **options) -> None:
    CATALOG[name] = Entry(name, title, page, build, options, source)


# --- Corpus ---------------------------------------------------------------------
_add("document-types", "Document types", "Overview",
     lambda c: hbar(c.document_types(), "doc_type", "documents"), "document_types()")
_add("documents-per-year", "Documents per year", "Production",
     lambda c: vbar(c.production_by_year(), "year", ["documents"]), "production_by_year()")
_add("citations-per-year", "Citations per year", "Production",
     lambda c: lines(c.production_by_year(), "year", ["citations"]), "production_by_year()")
_add("production-per-country", "Most productive countries", "Production",
     lambda c, n: hbar(c.production_by_country(n), "country", "documents"),
     "production_by_country(n)", n=15)
_add("price-index", "Price index by year", "Corpus profile",
     lambda c, window: lines(c.price_index_by_year(window), "year", ["price_index"]),
     "price_index_by_year(window)", window=5)
_add("reference-ages", "Age of cited references", "Corpus profile",
     lambda c, max_age: vbar(c.reference_age_distribution(max_age), "age", ["references"]),
     "reference_age_distribution(max_age)", max_age=40)
_add("lorenz", "Lorenz curve", "Corpus profile",
     lambda c, unit: points({"Lorenz": pd.DataFrame(c.concentration(unit)["lorenz"])},
                            "population_share", "value_share",
                            x_label=f"Cumulative share of {unit}",
                            y_label="Cumulative share of documents"),
     "concentration(unit)", unit="authors")
_add("self-citation-authors", "Self-citations by author", "Corpus profile",
     lambda c, n: hbar(c.authors_self_citation(n), "author",
                       ["self_citations", "external_citations"], value_label="Citations"),
     "authors_self_citation(n)", n=15)
_add("subject-areas", "Subject areas", "Corpus profile",
     lambda c: hbar(c.subject_areas(), "area", "documents"), "subject_areas()")
_add("open-access-over-time", "Open access over time", "Corpus profile",
     lambda c: vbar(c.access_over_time(), "year", ["documents", "open_access"], "Documents"),
     "access_over_time()")

# --- Actors -----------------------------------------------------------------------
_add("top-authors", "Authors ranked by h-index", "Authors",
     lambda c, n: hbar(c.authors_impact(n), "author", "h_index"), "authors_impact(n)", n=20)
_add("authorship-groups", "Authorship groups", "Authors",
     lambda c: vbar(c.authorship_groups(), "group", ["documents"]), "authorship_groups()")
_add("authors-over-time", "Authors' production over time", "Authors",
     lambda c, n: over_time(c.authors_over_time(n), "author", top=n),
     "authors_over_time(n)", n=10)
for _level, _who in (("parent", "institutions"), ("subparent", "units")):
    _add(f"top-{_who}", f"{_who.capitalize()} ranked by h-index", "Institutions",
         lambda c, n, level=_level: hbar(c.institutions_impact(n, level), "institution",
                                         "h_index"),
         "institutions_impact(n, level)", n=20)
    _add(f"{_who}-over-time", f"{_who.capitalize()} over time", "Institutions",
         lambda c, n, value, level=_level: over_time(c.institutions_over_time(n, level),
                                                     "institution", value, top=n),
         "institutions_over_time(n, level)", n=8, value="cumulative")
    _add(f"collaboration-between-{_who}", f"Collaboration between {_who}", "Institutions",
         lambda c, top_n, min_weight, level=_level, who=_who: _network(
             lambda: c.network("institutions", top_n, min_weight, level=level),
             f"Collaboration between {who}"),
         "network('institutions', top_n, min_weight, level=level)", top_n=40, min_weight=1)
_add("documents-by-affiliation-count", "Documents by number of affiliations", "Institutions",
     lambda c: vbar(c.documents_by_affiliation_count(), "affiliations", ["documents"]),
     "documents_by_affiliation_count()")
_add("scimago-quartiles", "SCImago quartiles", "Sources",
     lambda c: vbar(c.quartile_distribution(), "quartile", ["documents"]),
     "quartile_distribution()")
_add("quartiles-over-time", "SCImago quartiles over time", "Sources",
     lambda c: over_time(c.quartile_over_time(), "quartile", top=5), "quartile_over_time()")
_add("source-impact", "Sources ranked by h-index", "Sources",
     lambda c, n: hbar(c.sources_impact(n), "source", "h_index"), "sources_impact(n)", n=15)
_add("sources-over-time", "Sources over time", "Sources",
     lambda c, n: over_time(c.sources_over_time(n), "source", top=n),
     "sources_over_time(n)", n=6)
_add("local-cited-sources", "Most locally cited sources", "Sources",
     lambda c, n: hbar(c.most_local_cited_sources(n), "source", "local_citations"),
     "most_local_cited_sources(n)", n=15)
_add("collaboration-scale", "Collaboration scale", "Cities",
     lambda c: vbar(c.collaboration_scale(), "scale", ["documents"]), "collaboration_scale()")
_add("cities", "Most productive cities", "Cities",
     lambda c, n: hbar(c.top_cities(n), "city", "documents"), "top_cities(n)", n=15)
_add("cities-over-time", "Cities over time", "Cities",
     lambda c, n: over_time(c.cities_over_time(n), "city", top=n), "cities_over_time(n)", n=6)
_add("city-network", "Inter-city collaboration", "Cities",
     lambda c, top_n, min_weight: _network(lambda: c.network("cities", top_n, min_weight),
                                           "Inter-city collaboration"),
     "network('cities', top_n, min_weight)", top_n=30, min_weight=1)


def _country_values(c, column: str) -> Dict[str, float]:
    t = c.country_map()
    return {} if t.empty else dict(zip(t["country"], _num(t[column])))


def _collaboration_links(c, top_n: int, min_weight: int):
    g = c.co_country(top_n, min_weight)
    label = {n["id"]: n["label"] for n in g["nodes"]}
    return [(label[e["source"]], label[e["target"]], e.get("weight", 1)) for e in g["edges"]]


_add("country-map", "Country map", "Countries",
     lambda c, value: _world(_country_values(c, value), axis(value)),
     "country_map()", value="documents")
_add("collaboration-map", "International collaboration map", "Countries",
     lambda c, top_n, min_weight: _world(_country_values(c, "documents"), "Documents",
                                         links=_collaboration_links(c, top_n, min_weight)),
     "country_map(), co_country(top_n, min_weight)", top_n=60, min_weight=1)
_add("scp-mcp", "First author's country: SCP / MCP", "Countries",
     lambda c, n: hbar(c.corresponding_author_countries(n), "country", ["scp", "mcp"],
                       value_label="Documents"),
     "corresponding_author_countries(n)", n=20)

# --- Impact -----------------------------------------------------------------------
_add("local-cited-documents", "Most locally cited documents", "Local impact",
     lambda c, n: hbar(c.most_local_cited_documents(n), "label",
                       ["local_citations", "global_citations"], value_label="Citations"),
     "most_local_cited_documents(n)", n=20)
_add("local-cited-authors", "Most locally cited authors", "Local impact",
     lambda c, n: hbar(c.most_local_cited_authors(n), "author", "local_citations"),
     "most_local_cited_authors(n)", n=15)
_add("historiograph", "Historiograph", "Local impact",
     lambda c, n: _network(lambda: c.historiograph(n), "Historiograph",
                           size_by="local_citations"),
     "historiograph(n)", n=25)
_add("rpys-counts", "Reference publication year spectroscopy", "Spectroscopy",
     lambda c: mixed(c.reference_spectroscopy(), "year", ["references"], ["median_5"]),
     "reference_spectroscopy()")
_add("rpys-deviation", "Deviation from the 5-year median", "Spectroscopy",
     lambda c: vbar(c.reference_spectroscopy(), "year", ["deviation"]),
     "reference_spectroscopy()")
_add("rgr", "Relative growth rate", "Indicators",
     lambda c: lines(c.rgr_doubling_time(), "year", ["rgr"]), "rgr_doubling_time()")
_add("doubling-time", "Doubling time", "Indicators",
     lambda c: lines(c.rgr_doubling_time(), "year", ["doubling_time"]), "rgr_doubling_time()")
_add("trend-and-projection", "Trend and projection", "Indicators",
     lambda c, horizon, model: lines(c.trend_forecast(horizon, model).get("table"), "year",
                                     ["documents", "fitted"], "Documents"),
     "trend_forecast(horizon, model)", horizon=5, model="linear")
_add("authorship-pattern", "Documents by number of authors", "Indicators",
     lambda c: vbar(c.authorship_pattern(), "authors", ["documents"]), "authorship_pattern()")
_add("cai-by-period", "Co-authorship index (CAI) by period", "Indicators",
     lambda c, block_years: lines(c.cai(block_years), "period",
                                  ["cai_single", "cai_two", "cai_three", "cai_multi"], "CAI"),
     "cai(block_years)", block_years=5)
_add("lotka", "Lotka: author productivity", "Laws",
     lambda c: mixed(c.lotka()["table"], "documents_written", ["share_observed"],
                     ["share_fitted", "share_lotka"], "Share of authors"), "lotka()")
_add("bradford", "Bradford: cumulative documents by source rank", "Laws",
     lambda c: lines(c.bradford()["table"], "rank", ["cumulative"]), "bradford()")


def _bradford_zones(c) -> Optional[FigureSpec]:
    zones = pd.DataFrame(c.bradford()["zones"])
    if zones.empty:
        return None
    return FigureSpec(kind="bar", categories=[f"Zone {z}" for z in zones["zone"]],
                      series=[Series(name="Sources", values=_num(zones["sources"]), kind="bar"),
                              Series(name="Cumulative documents",
                                     values=_num(zones["documents"].cumsum()), kind="line")],
                      x_label="Bradford zone", y_label="Count")


_add("bradford-zones", "Bradford zones", "Laws", _bradford_zones, "bradford()")
_add("bradford-core-journals", "Bradford core journals (zone 1)", "Laws",
     lambda c: hbar(c.bradford()["table"].query("zone == 1"), "source", "documents"),
     "bradford()")
_add("zipf", "Zipf: term frequency (log-log)", "Laws",
     lambda c, n: points({"Observed": c.zipf(n)["table"].rename(columns={"frequency": "f"}),
                          "Predicted": c.zipf(n)["table"].drop(columns=["frequency"])
                                                        .rename(columns={"predicted": "f"})},
                         "rank", "f", x_label="Rank", y_label="Frequency",
                         x_log=True, y_log=True),
     "zipf(n)", n=60)

# --- Concepts ---------------------------------------------------------------------
_add("word-cloud", "Word cloud", "Themes",
     lambda c, n, kind: (lambda t: None if t.empty else
                         (lambda fmt, dpi: render_word_cloud(t, fmt=fmt, dpi=dpi)))(
         c.top_keywords(n, kind)),
     "top_keywords(n, kind)", n=120, kind="author")
_add("keyword-treemap", "Keyword treemap", "Themes",
     lambda c, n, kind: (lambda t: None if t.empty else
                         (lambda fmt, dpi: render_treemap(t, fmt=fmt, dpi=dpi)))(
         c.top_keywords(n, kind)),
     "top_keywords(n, kind)", n=120, kind="author")
_add("word-dynamics", "Keyword dynamics", "Themes",
     lambda c, n, kind, value: over_time(c.word_dynamics(n, kind), "keyword", value, top=n,
                                         y_label="Cumulative occurrences"
                                         if value == "cumulative" else "Occurrences"),
     "word_dynamics(n, kind)", n=8, kind="author", value="cumulative")
_add("co-word-network", "Co-word network", "Themes",
     lambda c, top_n, min_weight, kind: _network(
         lambda: c.network("keywords", top_n, min_weight, kind=kind), "Co-word network"),
     "network('keywords', top_n, min_weight, kind=kind)", top_n=50, min_weight=2, kind="author")
_add("text-terms", "Most frequent terms", "Text mining",
     lambda c, field, ngram, n, min_documents: hbar(
         c.top_terms(field, ngram, n, min_documents), "term", "documents"),
     "top_terms(field, ngram, n, min_documents)", field="abstract", ngram=2, n=30,
     min_documents=3)
_add("text-trend", "Terms by median year", "Text mining",
     lambda c, field, ngram, n: hbar(c.text_trend(field, ngram, n), "term", "year_median"),
     "text_trend(field, ngram, n)", field="abstract", ngram=2, n=20)
_add("text-network", "Term co-occurrence network", "Text mining",
     lambda c, field, ngram, top_n, min_weight, normalization: _network(
         lambda: c.term_network(field, ngram, top_n, min_weight, normalization=normalization),
         "Term co-occurrence network"),
     "term_network(field, ngram, top_n, min_weight, normalization=...)",
     field="abstract", ngram=2, top_n=35, min_weight=3, normalization="association")
_add("thematic-map", "Thematic map", "Thematic map",
     lambda c: points({"Themes": c.thematic_map()["clusters"]}, "centrality", "density",
                      "label", "Centrality (relevance)", "Density (development)"),
     "thematic_map()")
_add("topics-over-time", "Topics over time", "Thematic map",
     lambda c, n: hbar(c.trend_topics(n), "keyword", "year_median"), "trend_topics(n)", n=25)
_add("thematic-evolution", "Thematic evolution", "Thematic evolution",
     lambda c, n_periods, top_n, kind: (lambda e: None if not e.get("nodes") else
                                        (lambda fmt, dpi: render_thematic_evolution(
                                            e, fmt=fmt, dpi=dpi)))(
         c.thematic_evolution(n_periods=n_periods, top_n=top_n, kind=kind)),
     "thematic_evolution(n_periods, top_n, kind)", n_periods=3, top_n=60, kind="author")


def _conceptual(c, method, kind, top_n) -> Optional[FigureSpec]:
    terms = pd.DataFrame(c.conceptual_structure(method, kind, top_n)["terms"])
    if terms.empty:
        return None
    return points({f"Cluster {k}": g for k, g in terms.groupby("cluster")},
                  "dim1", "dim2", "keyword", "Dimension 1", "Dimension 2")


_add("conceptual-map", "Conceptual structure", "Conceptual structure", _conceptual,
     "conceptual_structure(method, kind, top_n)", method="CA", kind="author", top_n=50)
_add("dendrogram", "Topic dendrogram", "Conceptual structure",
     lambda c, kind, top_n, max_clusters: (lambda d: None if not d.get("tree") else
                                           (lambda fmt, dpi: render_dendrogram(
                                               d, fmt=fmt, dpi=dpi)))(
         c.topic_dendrogram(kind, top_n, max_clusters=max_clusters)),
     "topic_dendrogram(kind, top_n, max_clusters=...)", kind="author", top_n=50, max_clusters=5)
_THREE = {"authors": "Authors", "keywords": "Keywords", "sources": "Sources",
          "countries": "Countries", "institutions": "Institutions", "references": "References"}
_add("three-fields", "Three-field plot", "Three fields",
     lambda c, left, middle, right, n: (lambda t: None if t.empty else
                                        (lambda fmt, dpi: render_three_fields(
                                            t, tuple(_THREE.get(x, x.capitalize())
                                                     for x in (left, middle, right)),
                                            fmt=fmt, dpi=dpi)))(
         c.three_fields(left, middle, right, n)),
     "three_fields(left, middle, right, n)", left="authors", middle="keywords",
     right="sources", n=10)

# --- Networks -----------------------------------------------------------------------
for _unit, _label, _mw in (("authors", "authors", 1), ("institutions", "institutions", 1),
                           ("countries", "countries", 1)):
    _add(f"collaboration-{_label}", f"Collaboration: {_label}", "Networks",
         lambda c, top_n, min_weight, unit=_unit, label=_label: _network(
             lambda: c.network(unit, top_n, min_weight), f"Collaboration: {label}"),
         f"network('{_unit}', top_n, min_weight)", top_n=50, min_weight=_mw)
for _unit, _label in (("references", "references"), ("coupling", "coupling"),
                      ("cited-authors", "cited authors")):
    _slug = _label.replace(" ", "-")
    _add(f"co-citation-{_slug}", f"Co-citation: {_label}", "Networks",
         lambda c, top_n, min_weight, unit=_unit, label=_label: _network(
             lambda: c.network(unit, top_n, min_weight), f"Co-citation: {label}"),
         f"network('{_unit}', top_n, min_weight)", top_n=50, min_weight=2)

_LAB_UNITS = ("authors", "keywords", "institutions", "countries", "references",
              "cited-authors", "coupling")
for _unit in _LAB_UNITS:
    _add(f"network-{_unit}", f"Network lab: {_unit.replace('-', ' ')}", "Network lab",
         lambda c, top_n, min_weight, normalization, resolution, unit=_unit: _network(
             lambda: c.network(unit, top_n, min_weight, normalization, resolution=resolution),
             f"{unit.replace('-', ' ').capitalize()} network"),
         f"network('{_unit}', top_n, min_weight, normalization, resolution=...)",
         top_n=50, min_weight=1, normalization="association", resolution=1.0)


def _density(c, unit, top_n, min_weight) -> Optional[Renderer]:
    from ..views import _unit_graph
    graph = _unit_graph(c, unit, top_n, min_weight, "author", "parent")
    if not graph.get("nodes"):
        return None
    laid = c.attach_layout(graph)
    grid = c.network_density(graph)
    return lambda fmt, dpi: render_density_map(grid, laid, fmt=fmt, dpi=dpi)


for _unit in _LAB_UNITS:
    _add(f"density-{_unit}", f"Density map: {_unit.replace('-', ' ')}", "Network lab",
         lambda c, top_n, min_weight, unit=_unit: _density(c, unit, top_n, min_weight),
         f"density_map('{_unit}', top_n, min_weight)", top_n=50, min_weight=1)

for _unit in ("sources", "authors", "countries", "institutions", "documents"):
    _add(f"citation-network-{_unit}", f"{_unit.capitalize()} citing {_unit}", "Citation network",
         lambda c, top_n, min_weight, unit=_unit: _network(
             lambda: c.citation_graph(unit, top_n, min_weight), f"{unit.capitalize()} citing {unit}"),
         f"citation_graph('{_unit}', top_n, min_weight)", top_n=30, min_weight=2)
_add("coupling-map", "Coupling clusters", "Coupling clusters",
     lambda c, top_n, min_weight, impact: points(
         {"Clusters": c.clustering_by_coupling(top_n, min_weight, impact)["clusters"]},
         "centrality", "impact", "label", "Centrality", f"Impact ({impact} citations)"),
     "clustering_by_coupling(top_n, min_weight, impact)", top_n=100, min_weight=3,
     impact="local")


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------

def entry(name: str) -> Entry:
    """The entry of a figure; ``top_authors`` means ``top-authors``."""
    key = str(name).strip().lower().replace("_", "-")
    if key not in CATALOG:
        close = sorted(k for k in CATALOG if key.split("-")[0] in k)[:8]
        hint = f" Close names: {', '.join(close)}." if close else ""
        raise KeyError(f"no figure named {name!r}.{hint} See Corpus.figure_catalog().")
    return CATALOG[key]


def _build(corpus, name: str, options: Dict[str, Any]) -> tuple:
    e = entry(name)
    unknown = set(options) - set(e.options)
    if unknown:
        accepted = ", ".join(sorted(e.options)) or "none"
        raise TypeError(f"{e.name}: unknown option(s) {sorted(unknown)}; accepted: {accepted}")
    built = e.build(corpus, **{**e.options, **options})
    if built is None:
        raise ValueError(f"{e.name}: no data to draw for this corpus and these options")
    return e, built


def figure_spec(corpus, name: str, **options) -> FigureSpec:
    """The description of a simple figure (bars, lines, scatter), with its title."""
    e, built = _build(corpus, name, options)
    if not isinstance(built, FigureSpec):
        raise TypeError(f"{e.name} is drawn by a dedicated renderer (network, map…): "
                        "use Corpus.figure() to get the image")
    return dataclasses.replace(built, title=e.title)


#: What ``style`` can change: the dressing, never the data.
STYLE_FIELDS = ("kind", "orientation", "title", "show_title", "subtitle", "palette",
                "color", "value_labels", "x_label", "y_label", "width_in", "height_in",
                "mode")


def _styled(spec: FigureSpec, style: Optional[Dict[str, Any]]) -> FigureSpec:
    """The figure dressed as in the interface's export dialog: form
    (``kind="pie"``), title written on it, subtitle, colours."""
    if not style:
        return spec
    unknown = set(style) - set(STYLE_FIELDS)
    if unknown:
        raise TypeError(f"unknown style key(s) {sorted(unknown)}; accepted: "
                        + ", ".join(STYLE_FIELDS))
    return dataclasses.replace(spec, **style)


def figure_bytes(corpus, name: str, fmt: Optional[str] = None,
                 path: Union[str, Path, None] = None, dpi: int = 300,
                 style: Optional[Dict[str, Any]] = None, **options) -> bytes:
    """The image of a figure; written to ``path`` if given (the format is then
    deduced from the extension). ``style``: the export dialog's settings,
    ``{"kind": "pie", "show_title": True, "subtitle": "Years 2010-2013",
    "palette": "gradient", "value_labels": True}``."""
    fmt = (fmt or (Path(path).suffix.lstrip(".") if path else "") or "png").lower()
    fmt = "jpg" if fmt == "jpeg" else fmt
    e, built = _build(corpus, name, options)
    if isinstance(built, FigureSpec):
        spec = _styled(dataclasses.replace(built, title=e.title, dpi=dpi), style)
        data = render_figure(spec, fmt=fmt)
    elif style:
        raise TypeError(f"{e.name} is drawn by a dedicated renderer (network, map…): "
                        "its style cannot be changed")
    else:
        data = built(fmt, dpi)
    if path is not None:
        Path(path).write_bytes(data)
    return data


def catalog_table() -> pd.DataFrame:
    """All the figures: name, title, page and section of the interface, options
    (with their on-screen value) and the method that carries the data."""
    return pd.DataFrame([{
        "name": e.name, "title": e.title, "page": e.page, "section": e.section,
        "options": ", ".join(f"{k}={v!r}" for k, v in e.options.items()),
        "data": f"corpus.{e.source}" if e.source else "",
    } for e in CATALOG.values()])


def _check_signatures() -> None:
    """An entry whose function does not accept its options is a programming
    error: it is refused at import time rather than at the first call."""
    for e in CATALOG.values():
        params = inspect.signature(e.build).parameters
        missing = [k for k in e.options if k not in params]
        if missing:
            raise AssertionError(f"{e.name}: options {missing} not accepted by its builder")


_check_signatures()
