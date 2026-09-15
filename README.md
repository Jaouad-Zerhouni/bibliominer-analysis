# bibliominer-analysis

Bibliometric analysis of a Scopus corpus, as a **library**: notebook, script,
CI pipeline — no interface required.

It is the computation layer behind [Bibliominer](https://github.com/jaouad/bibliominer).
The web application consumes it rather than reimplementing it: one source of
truth, one place to test.

```bash
pip install bibliominer-analysis
```

## What it takes as input

A **cleaned** Scopus export (CSV). Cleaning — recovering DOIs, naming source
titles, resolving affiliations, reconciling references — is the job of the
companion package `bibliominer-cleaning`. This one starts where that one stops.

## What it computes

| | |
|---|---|
| **Production & impact** | documents per year, growth, citations, h-index, and the usual per-author / per-source / per-country breakdowns |
| **Networks** | co-citation (references *and* authors), co-word, bibliographic coupling, collaboration |
| **Network analysis** | degree, betweenness, closeness, PageRank, eigenvector, clustering, and community detection |
| **Journals** | SCImago quartiles and categories — the reference table ships **inside** the package, so nothing is fetched at runtime |
| **Figures** | bar, line, scatter, and network maps, rendered with matplotlib |

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
positions come from a deterministic MDS on shortest paths, not from a force
simulation — a force layout is stochastic, and a figure published in an
article cannot change between runs. The timestamp matplotlib normally writes
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
objects — nothing to unwrap, nothing to serialise.

## Requirements

Python ≥ 3.9. Depends on pandas, numpy, networkx and matplotlib — all pure
Python or shipped as wheels, so there is nothing to compile.

## License

MIT — see [LICENSE](LICENSE).
