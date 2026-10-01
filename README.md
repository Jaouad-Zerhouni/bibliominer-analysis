# bibliominer-analysis

Bibliometric analysis of a Scopus corpus, as a **library**: notebook, script,
CI pipeline, no interface required.

It is the computation layer behind [Bibliominer](https://github.com/Jaouad-Zerhouni/bibliominer).
The web application consumes it rather than reimplementing it: one source of
truth, one place to test.

```bash
pip install bibliominer-analysis
```

## What it takes as input

A **cleaned** Scopus export (CSV). Cleaning, recovering DOIs, naming source
titles, resolving affiliations, reconciling references, is the job of the
companion package `bibliominer-cleaning`. This one starts where that one stops.

## What it computes

| | |
|---|---|
| **Production & impact** | documents per year, growth, citations, h-index, and the usual per-author / per-source / per-country breakdowns |
| **Networks** | co-citation (references *and* authors), co-word, bibliographic coupling, collaboration |
| **Network analysis** | degree, betweenness, closeness, PageRank, eigenvector, clustering, and community detection |
| **Journals** | SCImago quartiles and categories, the reference table ships **inside** the package, so nothing is fetched at runtime |
| **Figures** | every figure of the web interface, bars, lines, scatters, networks, world maps, treemap, word cloud, Sankey, dendrogram, density map, in PNG, SVG, PDF or JPG |
| **Reports** | figures, tables and indicators gathered in one ZIP: a folder per section, PNG at 300 dpi + SVG, Excel tables |

## Everything the interface shows, in Python

The web interface calls this package for every number it displays, so a
notebook gets the same tables, the same figures and the same options.

**Filters.** The filter bar of the interface is `Corpus.filter`, which returns
a new corpus, every method then works on it:

```python
corpus = Corpus.from_csv("corpus_cleaned.csv")        # say 2020-2025
recent = corpus.filter(years=(2023, 2025))              # only 2023-2025
articles = recent.filter(doc_types=["Article"], countries=["Morocco"])
```

**Top N.** Every ranking takes `n` (or `top_n` for networks): the default is
what the interface shows, any value works.

```python
recent.authors_impact(n=10)          # top 10 authors by h-index
recent.production_by_country(n=5)    # top 5 countries
```

**Figures.** Each figure of the interface has a name, the one its export
file carries in the interface, and the interface's settings as defaults:

```python
Corpus.figure_catalog()                                  # every figure, by page
recent.figure("top-authors", n=10, path="top_authors.svg")        # vector
recent.figure("collaboration-map", path="collaboration.png")      # 300 dpi
recent.figure("network-keywords", top_n=80, normalization="association",
              path="keywords.pdf")
spec = recent.figure_spec("documents-per-year")          # editable before drawing
```

The format follows the file extension (`png`, `svg`, `pdf`, `jpg`), or pass
`fmt=`. PNG is rendered at 300 dpi by default (`dpi=` to change it); SVG is
vector and stays sharp at any size. Network maps keep every disc apart and
lay the separate groups out in rows, so a map with twenty teams stays
readable.

`style` dresses a bar, line or scatter figure as the interface's export
dialog does: the form (`bar`, `lollipop`, `line`, `area`, `pie`, `donut`), a
title and subtitle written on the figure, the colours (`gradient` by value,
`category` for one colour per bar, `series`), and the values on the bars:

```python
recent.figure("top-authors", n=10, path="top_authors.png",
              style={"show_title": True, "subtitle": "Years 2023-2025",
                     "palette": "gradient", "value_labels": True})
recent.figure("document-types", path="types.svg", style={"kind": "donut"})
```

**Reports.** The interface's *Add to report* basket is `Report`:

```python
from bibliominer_analysis import Report

report = Report(filters={"years": "2023-2025"})
report.add_corpus_figure(recent, "top-authors", n=10,       # filed under Actors
                         period="2023-2025")                 # top-authors_2023-2025.png
report.add_table("Actors", "Top 10 authors", recent.authors_impact(10))
report.add_indicators("Impact", "Collaboration", recent.collaboration_indicators())
report.save("report.zip")
```

```
report.zip
  README.txt                   each file, its title, and the filters it was computed under
  all_tables.xlsx              every table, one sheet each
  1-corpus/figures/*.png|svg   2-actors/…   3-impact/…   4-concepts/…   5-networks/…
  1-corpus/corpus_tables.xlsx  the section's tables, one sheet each, after a Contents sheet
```

An entry computed over a period carries it in its file or sheet name
(`period=`), so the same figure over two periods gives two files. Excel files are written by
the package itself: no extra dependency.

### Interface page → Python call

| Page | Tables | Figures (`corpus.figure(name)`) |
|---|---|---|
| **Corpus** · Overview | `summary()`, `main_information()`, `most_cited_documents(10)`, `document_types()` | `document-types` |
| Production | `production_by_year()`, `production_by_country(15)` | `documents-per-year`, `citations-per-year`, `production-per-country` |
| Documents | `document_list(200, sort)`, `most_normalized_documents(20)` | n/a |
| Corpus profile | `price_index(window)`, `price_index_by_year()`, `reference_age_distribution()`, `concentration(unit)`, `concentration_summary()`, `self_citation_summary()`, `authors_self_citation(15)`, `interdisciplinarity()`, `subject_areas()`, `access_status()`, `access_routes()`, `access_over_time()` | `price-index`, `reference-ages`, `lorenz`, `self-citation-authors`, `subject-areas`, `open-access-over-time` |
| Quality | `quality_summary()`, `indicator_readiness()`, `field_completeness()`, `table_completeness()`, `anomalies()`, `duplicates()` | n/a |
| **Actors** · Authors | `impact()`, `authors_impact(20)`, `authorship_groups()`, `authors_over_time(10)` | `top-authors`, `authorship-groups`, `authors-over-time` |
| Institutions | `institutions_impact(20, level)`, `institutions_over_time(8, level)`, `institutions_by_country(20, level)`, `org_hierarchy(40)`, `affiliation_profile()`, `documents_by_affiliation_count()`, `authors_by_affiliation_count(20)` | `top-institutions`, `top-units`, `institutions-over-time`, `units-over-time`, `documents-by-affiliation-count`, `collaboration-between-institutions`, `collaboration-between-units` |
| Sources | `scimago_coverage()`, `quartile_distribution()`, `quartile_over_time()`, `scimago_sources()`, `sources_impact(15)`, `sources_over_time(6)`, `most_local_cited_sources(15)` | `scimago-quartiles`, `quartiles-over-time`, `source-impact`, `sources-over-time`, `local-cited-sources` |
| Cities | `top_cities(15)`, `collaboration_scale()`, `cities_impact(20)`, `cities_over_time(6)`, `city_hierarchy(60)` | `cities`, `collaboration-scale`, `cities-over-time`, `city-network` |
| Countries | `country_map()`, `corresponding_author_countries(20)`, `countries_impact(20)` | `country-map`, `collaboration-map`, `scp-mcp` |
| **Impact** · Local impact | `most_local_cited_documents(20)`, `most_local_cited_authors(15)`, `historiograph(25)` | `local-cited-documents`, `local-cited-authors`, `historiograph` |
| Spectroscopy | `reference_spectroscopy()` | `rpys-counts`, `rpys-deviation` |
| Indicators | `growth_summary()`, `rgr_doubling_time()`, `trend_forecast(5, model)`, `collaboration_indicators()`, `authorship_pattern()`, `cai(5)`, `price_law()`, `cochran_sample_size()` | `rgr`, `doubling-time`, `trend-and-projection`, `authorship-pattern`, `cai-by-period` |
| Laws | `lotka()`, `bradford()`, `zipf(60)` | `lotka`, `bradford`, `bradford-zones`, `bradford-core-journals`, `zipf` |
| **Concepts** · Themes | `top_keywords(120, kind)`, `word_dynamics(8, kind)` | `word-cloud`, `keyword-treemap`, `word-dynamics`, `co-word-network` |
| Text mining | `top_terms(field, ngram, n)`, `text_trend(field, ngram)`, `term_network(...)` | `text-terms`, `text-trend`, `text-network` |
| Thematic map | `thematic_map()`, `trend_topics(25)` | `thematic-map`, `topics-over-time` |
| Thematic evolution | `thematic_evolution(n_periods=3)` | `thematic-evolution` |
| Conceptual structure | `conceptual_structure(method, kind)`, `topic_dendrogram(kind)` | `conceptual-map`, `dendrogram` |
| Three fields | `three_fields(left, middle, right, n)` | `three-fields` |
| **Networks** · Networks | `network(unit, top_n, min_weight)` | `collaboration-authors`, `collaboration-institutions`, `collaboration-countries`, `co-citation-references`, `co-citation-coupling`, `co-citation-cited-authors` |
| Network lab | `network(unit, top_n, min_weight, normalization, overlay, resolution=…)`, `density_map(unit)` | `network-<unit>`, `density-<unit>` |
| Citation network | `citation_graph(unit)`, `citation_balance(unit)` | `citation-network-<unit>` |
| Coupling clusters | `clustering_by_coupling(100, 3, impact)` | `coupling-map` |

## Drawing a network

Networks are built, annotated and **drawn** without a browser:

```python
from bibliominer_analysis import Corpus
from bibliominer_analysis.networks.build import co_word
from bibliominer_analysis.networks.analysis import annotate
from bibliominer_analysis.figures.network import render_network

corpus = Corpus.from_csv("corpus_cleaned.csv")

graph = co_word(corpus, top_n=60, min_weight=2)   # the thematic structure
graph = annotate(graph, communities=True)          # clusters + centralities

png = render_network(
    graph,
    title="Co-word network",
    size_by="pagerank",   # or "weight", "betweenness"…
    label_top=15,         # label only the biggest nodes
)
open("co_word.png", "wb").write(png)
```

`render_network` returns the image **bytes**, in `png`, `jpg`, `svg` or `pdf`.
Nothing is written to disk unless you write it.

### The map does not move

Two runs on the same data produce the **same file, byte for byte**. Node
positions come from a deterministic MDS on shortest paths, refined with a
fixed seed, not from a free force simulation, a force layout is stochastic,
and a figure published in an article cannot change between runs. The timestamp matplotlib normally writes
into the file is stripped for the same reason.

That also means you can commit a figure and let a diff tell you whether the
*data* changed.

### Drawing exactly what the screen showed

Pass the coordinates you already have, and the exported figure is the same map
the user was looking at:

```python
from bibliominer_analysis.networks.analysis import layout

coords = layout(graph)                       # compute once…
svg = render_network(graph, coords=coords, fmt="svg")   # …draw many times
```

## Reading the numbers

```python
from bibliominer_analysis.metrics.production import by_year

corpus = Corpus.from_csv("corpus_cleaned.csv")
print(by_year(corpus))
```

Every metric takes the same `Corpus` and returns plain Python / pandas
objects, nothing to unwrap, nothing to serialise.

## Requirements

Python ≥ 3.9. Depends on pandas, numpy, networkx and matplotlib, all pure
Python or shipped as wheels, so there is nothing to compile.

## Data shipped with the package

The SCImago journal list (`data_ref/scimagojr_2025.csv`) is third-party data,
not covered by the MIT licence: SCImago allows its use for non-commercial
purposes as long as it is cited, *SCImago (n.d.). SJR - SCImago Journal &
Country Rank [Portal]. Retrieved from https://www.scimagojr.com*. The terms
travel with the file, in `data_ref/scimagojr_2025.NOTICE`.

## License

MIT, the `LICENSE` file ships with the package. The world map
(`data_ref/countries-110m.json`) keeps its own ISC licence, next to it.
