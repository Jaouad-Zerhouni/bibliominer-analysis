"""Results EXACTLY AS the interface shows them.

The building blocks all existed in the package (networks, normalisation,
centralities, local citations...), but it was the web API, or even the
browser, that assembled them: the document list, the citation balance,
the complete "Network lab" network. A package user had to guess the
assembly, and could get something other than the screen.

Here, each function IS what the interface displays, for the same options;
the API only calls them. Available through the `Corpus`
(`corpus.document_list()`, `corpus.network("keywords")`...).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .networks.analysis import normalize

#: The "Network lab" networks: screen unit -> Corpus method.
NETWORK_UNITS = ("authors", "keywords", "institutions", "countries", "cities",
                 "references", "cited-authors", "coupling")

#: Overlay by mean year only makes sense for these units (and for coupling,
#: carried by the documents).
_OVERLAY = {"authors": "authors", "keywords": "keywords",
            "institutions": "institutions", "countries": "countries",
            "coupling": "documents"}


def _first_authors(corpus) -> pd.DataFrame:
    authors = corpus.authors
    first = authors[pd.to_numeric(authors["position"], errors="coerce").eq(1)]
    return (first[["eid", "name"]].drop_duplicates("eid")
            .rename(columns={"name": "first_author"}))


def document_list(corpus, n: Optional[int] = 100, sort: str = "citations") -> pd.DataFrame:
    """The documents themselves: title, first author, year, source, type, global
    and local citations, DOI. `sort`: ``citations`` (descending), ``year``
    (most recent first) or ``title``."""
    if sort not in ("citations", "year", "title"):
        raise ValueError("sort must be 'citations', 'year' or 'title'")
    d = corpus.documents[["eid", "title", "year", "source", "doc_type",
                          "doi", "cited_by"]].copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    d["citations"] = pd.to_numeric(d["cited_by"], errors="coerce").fillna(0).astype(int)
    d = d.drop(columns=["cited_by"])
    d = d.merge(corpus.local_citations(), on="eid", how="left")
    d["local_citations"] = d["local_citations"].fillna(0).astype(int)
    d = d.merge(_first_authors(corpus), on="eid", how="left")
    ascending = sort == "title"
    d = d.sort_values(sort, ascending=ascending, na_position="last")
    cols = ["title", "first_author", "year", "source", "doc_type", "citations",
            "local_citations", "doi"]
    d = d[cols].reset_index(drop=True)
    return d if n is None else d.head(n)


def most_normalized_documents(corpus, n: Optional[int] = 25) -> pd.DataFrame:
    """Documents ranked by NORMALISED citations, relative to the mean of the
    documents of the same year. The only measure that can rank an article
    from 2016 and one from 2024 together."""
    d = corpus.documents[["eid", "title", "year", "source"]].copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    d = d.merge(corpus.normalized_citations(), on="eid", how="left")
    d = d.merge(_first_authors(corpus), on="eid", how="left")
    d = d.sort_values("normalized_citations", ascending=False)
    cols = ["title", "first_author", "year", "source", "citations",
            "year_mean_citations", "normalized_citations"]
    d = d[cols].reset_index(drop=True)
    return d if n is None else d.head(n)


def _unit_graph(corpus, unit: str, top_n: int, min_weight: int,
                kind: str, level: str) -> Dict[str, Any]:
    if unit == "authors":
        return corpus.co_authorship(top_n, min_weight)
    if unit == "keywords":
        return corpus.co_word(top_n, min_weight, kind)
    if unit == "institutions":
        return corpus.co_institution(top_n, min_weight, level)
    if unit == "countries":
        return corpus.co_country(top_n, min_weight)
    if unit == "cities":
        return corpus.co_city(top_n, min_weight)
    if unit == "references":
        return corpus.co_citation(top_n, min_weight)
    if unit == "cited-authors":
        return corpus.co_citation_authors(top_n, min_weight)
    if unit == "coupling":
        return corpus.bibliographic_coupling(top_n, min_weight)
    raise ValueError(f"unit must be one of {', '.join(NETWORK_UNITS)}")


def network(corpus, unit: str = "authors", top_n: int = 50, min_weight: int = 1,
            normalization: str = "none", overlay: bool = False,
            kind: str = "author", level: str = "parent",
            resolution: float = 1.0) -> Dict[str, Any]:
    """The "Network lab" network: normalised links, communities, centralities,
    overlay by mean year (`overlay`) and summary (density, transitivity,
    modularity...). `resolution`: granularity of the communities (above 1,
    more clusters)."""
    graph = normalize(_unit_graph(corpus, unit, top_n, min_weight, kind, level),
                      normalization)
    overlay_unit = _OVERLAY.get(unit) if overlay else None
    graph = corpus.network_metrics(graph, overlay_unit=overlay_unit,
                                   level=level, resolution=resolution)
    graph["summary"] = corpus.network_summary(graph)
    return graph


def density_map(corpus, unit: str = "keywords", top_n: int = 50,
                min_weight: int = 1, size: int = 48, kind: str = "author",
                level: str = "parent") -> Dict[str, Any]:
    """The density map of a network: the grid, and every node at its position,
    deterministic, the same from one opening to the next."""
    graph = _unit_graph(corpus, unit, top_n, min_weight, kind, level)
    coords = corpus.network_layout(graph)
    return {
        "nodes": [{"id": n["id"], "label": n["label"],
                   "occurrences": n["occurrences"],
                   "x": coords.get(n["id"], [0, 0])[0],
                   "y": coords.get(n["id"], [0, 0])[1]}
                  for n in graph["nodes"] if n["id"] in coords],
        "grid": corpus.network_density(graph, size=size),
        "n_nodes": graph["n_nodes"],
        "n_edges": graph["n_edges"],
    }


def term_network(corpus, field: str = "abstract", ngram: int = 2, top_n: int = 40,
                 min_weight: int = 2, min_documents: int = 2,
                 normalization: str = "none",
                 stopwords: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """The co-occurrence network of the text terms, measured (communities,
    centralities, summary), the one of the Text mining page."""
    graph = corpus.text_co_occurrence(field, ngram, top_n, min_weight,
                                      min_documents, stopwords)
    graph = corpus.network_metrics(normalize(graph, normalization))
    graph["summary"] = corpus.network_summary(graph)
    return graph


def citation_graph(corpus, unit: str = "sources", top_n: int = 40,
                   min_weight: int = 1, level: str = "parent") -> Dict[str, Any]:
    """The direct citation network, measured, the one of the Citation network
    page."""
    graph = corpus.network_metrics(corpus.citation_network(unit, top_n, min_weight, level))
    graph["summary"] = corpus.network_summary(graph)
    return graph


def citation_balance(corpus, unit: str = "sources", top_n: int = 40,
                     min_weight: int = 1, level: str = "parent") -> pd.DataFrame:
    """Citations received minus citations given, per entity of the citation
    network. Positive: the entity is a SOURCE of ideas for the corpus;
    negative: it consumes more than it provides."""
    graph = corpus.citation_network(unit, top_n, min_weight, level)
    rows = {n["id"]: {"label": n["label"], "received": 0, "emitted": 0}
            for n in graph["nodes"]}
    for edge in graph["edges"]:
        weight = edge.get("weight", 1)
        if edge["target"] in rows:
            rows[edge["target"]]["received"] += weight
        if edge["source"] in rows:
            rows[edge["source"]]["emitted"] += weight
    table = pd.DataFrame(list(rows.values()),
                         columns=["label", "received", "emitted"])
    table["balance"] = table["received"] - table["emitted"]
    return table.sort_values("balance", ascending=False, kind="stable").reset_index(drop=True)
