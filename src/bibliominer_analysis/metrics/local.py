"""**Local** citations: the citations received within the corpus.

It is the most misunderstood distinction of bibliometrics, and the most
useful one.

  - **GC** (*global citations*): what Scopus counts, all sources
    combined. An article highly cited by a neighbouring community shines
    there.
  - **LC** (*local citations*): how many documents **of this corpus** cite
    it. It is the measure of influence *within the field under study*.

An article can have 800 global citations and 0 local ones: it matters
elsewhere, not here. The reverse also exists, and signals a founding work
for the precise community being analysed. Without LC one can build
neither the historiograph, nor the "local cited" rankings, nor the direct
citation network, hence this module as a foundation.

Matching a reference to a document is done in two steps:

  1. **by DOI**, when both sides have one. It is an identifier, hence
     unambiguous.
  2. **by normalised title** otherwise (lower case, no punctuation or
     accents, collapsed spaces). A title of at least 25 characters is
     required: below that, generic titles ("Introduction", "Machine
     learning") would create false matches, and a false citation is worse
     than a missing one.
"""

from __future__ import annotations

from .._stable import top_by_count

import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

#: Below this length, a title is not discriminating enough.
MIN_TITLE_LEN = 25


def _norm_doi(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None
    # Exports mix "10.1000/x", "https://doi.org/10.1000/x" and
    # "doi:10.1000/x": only the part starting at "10." is kept.
    m = re.search(r"10\.\d{4,9}/\S+", s)
    return m.group(0).rstrip(".,;)") if m else None


def _norm_title(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def citation_pairs(corpus) -> pd.DataFrame:
    """The citation links **internal** to the corpus.

    Columns: ``citing`` (eid of the citing document), ``cited`` (eid of the
    cited one), ``via`` ("doi" or "title", how the match was made).

    A pair appears only once even if the citing document lists the same
    reference twice, and a document citing itself is left out.
    """
    empty = pd.DataFrame(columns=["citing", "cited", "via"])
    docs, refs = corpus.documents, corpus.references
    if docs.empty or refs.empty:
        return empty

    # Index of the corpus documents, on the "cited" side.
    by_doi, by_title = {}, {}
    for eid, doi, title in zip(docs["eid"], docs.get("doi"), docs.get("title")):
        d = _norm_doi(doi)
        if d and d not in by_doi:
            by_doi[d] = eid
        t = _norm_title(title)
        if t and len(t) >= MIN_TITLE_LEN and t not in by_title:
            by_title[t] = eid

    if not by_doi and not by_title:
        return empty

    rows = []
    for citing, rdoi, rtitle in zip(refs["eid"], refs.get("ref_doi"),
                                    refs.get("ref_title")):
        cited, via = None, None
        d = _norm_doi(rdoi)
        if d is not None:
            cited = by_doi.get(d)
            via = "doi"
        if cited is None:
            t = _norm_title(rtitle)
            if t is not None and len(t) >= MIN_TITLE_LEN:
                cited = by_title.get(t)
                via = "title"
        if cited is not None and cited != citing:
            rows.append((citing, cited, via))

    if not rows:
        return empty
    out = pd.DataFrame(rows, columns=["citing", "cited", "via"])
    # The DOI takes precedence over the title when both exist for the same pair.
    out["_rank"] = (out["via"] == "title").astype(int)
    out = (out.sort_values("_rank", kind="stable")
              .drop_duplicates(subset=["citing", "cited"])
              .drop(columns="_rank")
              .reset_index(drop=True))
    return out


def local_citations(corpus) -> pd.DataFrame:
    """Count of local citations per document. Columns ``eid``, ``local_citations``."""
    pairs = citation_pairs(corpus)
    base = pd.DataFrame({"eid": corpus.documents["eid"]})
    if pairs.empty:
        base["local_citations"] = 0
        return base
    counts = pairs.groupby("cited").size().rename("local_citations")
    base = base.merge(counts, left_on="eid", right_index=True, how="left")
    base["local_citations"] = base["local_citations"].fillna(0).astype(int)
    return base


def _doc_frame(corpus) -> pd.DataFrame:
    d = corpus.documents[["eid", "title", "year", "source", "doi"]].copy()
    d["global_citations"] = pd.to_numeric(
        corpus.documents["cited_by"], errors="coerce").fillna(0).astype(int)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    first = (corpus.authors[pd.to_numeric(corpus.authors["position"],
                                          errors="coerce").eq(1)]
             [["eid", "name"]].drop_duplicates("eid"))
    d = d.merge(first.rename(columns={"name": "first_author"}), on="eid", how="left")
    return d


def most_local_cited_documents(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """The documents most cited **by the other documents of the corpus**.

    Columns: ``label``, ``title``, ``first_author``, ``year``, ``source``,
    ``local_citations``, ``global_citations``, ``lc_gc_ratio``,
    ``local_citations_per_year``.

    ``lc_gc_ratio`` is the percentage of global citations that come from the
    corpus: high, the work is specific to the field; close to zero, its
    renown comes from elsewhere.
    """
    lc = local_citations(corpus)
    d = _doc_frame(corpus).merge(lc, on="eid", how="left")
    d["local_citations"] = d["local_citations"].fillna(0).astype(int)
    d = d[d["local_citations"] > 0]
    if d.empty:
        return pd.DataFrame(columns=["label", "title", "first_author", "year",
                                     "source", "local_citations",
                                     "global_citations", "lc_gc_ratio",
                                     "local_citations_per_year"])

    last_year = pd.to_numeric(corpus.documents["year"], errors="coerce").max()
    age = (last_year - d["year"] + 1).clip(lower=1)
    d["local_citations_per_year"] = (d["local_citations"] / age).round(2)
    d["lc_gc_ratio"] = np.where(
        d["global_citations"] > 0,
        (100 * d["local_citations"] / d["global_citations"]).round(1), 0.0)
    d["label"] = _labels(d)

    out = d.sort_values(["local_citations", "global_citations", "title", "eid"],
                        ascending=[False, False, True, True], kind="stable")
    out = out[["label", "title", "first_author", "year", "source",
               "local_citations", "global_citations", "lc_gc_ratio",
               "local_citations_per_year"]].reset_index(drop=True)
    out["year"] = out["year"].astype("Int64")
    return out.head(n) if n else out


def _labels(d: pd.DataFrame) -> pd.Series:
    '''"OKAFOR M., 2019", the usual short label in bibliometrics.'''
    author = d["first_author"].fillna("ANONYMOUS").map(str).str.upper()
    year = d["year"].astype("Int64").map(str).replace("<NA>", "n.d.")
    base = author + ", " + year
    # Two documents by the same author in the same year: suffixes a, b, c...
    # in the order of the TITLES, not of the rows: otherwise "2018-b" changed
    # document when the Scopus export was sorted differently.
    dup = base.duplicated(keep=False)
    if dup.any():
        order = pd.DataFrame({"base": base,
                              "title": d.get("title", pd.Series("", index=d.index)).fillna("").map(str),
                              "eid": d.get("eid", pd.Series("", index=d.index)).map(str)})
        order = order.sort_values(["base", "title", "eid"], kind="stable")
        suffix = order.groupby("base").cumcount().reindex(base.index)
        letters = suffix.map(lambda i: "" if i == 0 else "-" + chr(97 + min(i, 25)))
        base = base + letters.where(dup, "")
    return base


def most_local_cited_authors(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """Authors ranked by citations received **from within the corpus**.

    A document cited locally 5 times gives 5 to each of its authors: that is
    full counting.
    """
    lc = local_citations(corpus)
    a = corpus.authors[["eid", "name"]].dropna(subset=["name"])
    a = a[a["name"].map(str).str.strip() != ""].drop_duplicates(["eid", "name"])
    if a.empty:
        return pd.DataFrame(columns=["author", "local_citations", "documents"])
    a = a.merge(lc, on="eid", how="left")
    a["local_citations"] = a["local_citations"].fillna(0).astype(int)
    out = (a.groupby("name")
             .agg(local_citations=("local_citations", "sum"),
                  documents=("eid", "nunique"))
             .reset_index().rename(columns={"name": "author"}))
    out = out[out["local_citations"] > 0]
    out = out.sort_values(["local_citations", "documents"],
                          ascending=False, kind="stable").reset_index(drop=True)
    return out.head(n) if n else out


def most_local_cited_sources(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """Journals ranked by the cumulative local citations of their articles."""
    lc = local_citations(corpus)
    d = corpus.documents[["eid", "source"]].copy()
    d = d[d["source"].notna() & (d["source"].map(str).str.strip() != "")]
    if d.empty:
        return pd.DataFrame(columns=["source", "local_citations", "documents"])
    d = d.merge(lc, on="eid", how="left")
    d["local_citations"] = d["local_citations"].fillna(0).astype(int)
    out = (d.groupby("source")
             .agg(local_citations=("local_citations", "sum"),
                  documents=("eid", "nunique"))
             .reset_index())
    out = out[out["local_citations"] > 0]
    out = out.sort_values(["local_citations", "documents"],
                          ascending=False, kind="stable").reset_index(drop=True)
    return out.head(n) if n else out


def historiograph(corpus, n: int = 25) -> dict:
    """**Direct citation** graph between the most influential documents.

    Garfield's historiograph: the `n` most locally cited documents are kept
    and only the citations linking them are drawn. Read from left to right
    (time), it shows the lineage of ideas, which work builds on which other,
    something no ranking gives.

    Returns: ``{"nodes": [...], "edges": [...], "n_nodes", "n_edges"}``.
    Every node carries ``id``, ``label``, ``year``, ``local_citations``,
    ``global_citations``, ``title``.
    """
    pairs = citation_pairs(corpus)
    lc = local_citations(corpus)
    d = _doc_frame(corpus).merge(lc, on="eid", how="left")
    d["local_citations"] = d["local_citations"].fillna(0).astype(int)

    keep = d[d["local_citations"] > 0].sort_values(
        ["local_citations", "global_citations", "eid"],
        ascending=[False, False, True], kind="stable").head(n)
    if keep.empty:
        return {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}

    keep = keep.copy()
    keep["label"] = _labels(keep)
    ids = set(keep["eid"])
    edges = pairs[pairs["citing"].isin(ids) & pairs["cited"].isin(ids)]

    nodes = [{
        "id": str(r.eid),
        "label": str(r.label),
        "year": int(r.year) if pd.notna(r.year) else None,
        "local_citations": int(r.local_citations),
        "global_citations": int(r.global_citations),
        "title": None if pd.isna(r.title) else str(r.title),
    } for r in keep.itertuples()]

    return {
        "nodes": nodes,
        "edges": [{"source": str(s), "target": str(t)}
                  for s, t in zip(edges["citing"], edges["cited"])],
        "n_nodes": len(nodes),
        "n_edges": int(len(edges)),
    }


#: Units on which a direct citation network can be aggregated.
CITATION_UNITS = ("documents", "authors", "sources", "countries", "institutions")


def citation_network(corpus, unit: str = "sources", top_n: int = 40,
                     min_weight: int = 1, level: str = "parent") -> dict:
    """**Direct citation** network, aggregated to the requested unit.

    Co-citation says "these two works are cited together". Coupling says
    "these two works cite the same things". Direct citation says something
    else entirely, and it is the only one of the three that has a
    **direction**: *who cites whom*. It is therefore the only one producing a
    directed graph.

    Aggregated to authors, journals or countries, it shows relations of
    intellectual dependence: a journal that feeds the whole field without
    ever citing it back is seen at once.

    Self-citations of an entity to itself are left out: at country level
    they would crush everything else, and they say nothing about an exchange.
    """
    from collections import Counter

    pairs = citation_pairs(corpus)
    empty = {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}
    if pairs.empty:
        return empty

    members = _unit_members(corpus, unit, level)
    if not members:
        return empty

    flows: Counter = Counter()
    for citing, cited in zip(pairs["citing"], pairs["cited"]):
        for a in members.get(citing, ()):
            for b in members.get(cited, ()):
                if a != b:
                    flows[(a, b)] += 1

    if not flows:
        return empty

    weight_of: Counter = Counter()
    for (a, b), w in flows.items():
        weight_of[a] += w
        weight_of[b] += w
    keep = {k for k, _ in top_by_count(weight_of, top_n)}

    edges = [{"source": a, "target": b, "weight": int(w)}
             for (a, b), w in flows.items()
             if w >= min_weight and a in keep and b in keep]
    if not edges:
        return empty

    linked = {e["source"] for e in edges} | {e["target"] for e in edges}
    # `occurrences` = citations exchanged, which gives the node its size.
    nodes = [{"id": k, "label": k, "occurrences": int(weight_of[k]),
              "degree": int(sum(e["weight"] for e in edges
                                if e["source"] == k or e["target"] == k))}
             for k in sorted(linked, key=lambda x: (-weight_of[x], str(x)))]

    return {"nodes": nodes,
            "edges": sorted(edges, key=lambda e: (-e["weight"], str(e["source"]),
                                                  str(e["target"]))),
            "n_nodes": len(nodes), "n_edges": len(edges)}


def _unit_members(corpus, unit: str, level: str = "parent") -> dict:
    """eid -> entities of the document, according to the aggregation unit."""
    out: dict = {}

    if unit == "documents":
        titles = corpus.documents.set_index("eid")["title"].to_dict()
        return {str(k): (str(v)[:70],) for k, v in titles.items() if pd.notna(v)}

    if unit == "authors":
        frame = corpus.authors[["eid", "name"]].dropna()
        frame = frame[frame["name"].map(str).str.strip() != ""]
        column = "name"
    elif unit == "sources":
        frame = corpus.documents[["eid", "source"]].dropna()
        frame = frame[frame["source"].map(str).str.strip() != ""]
        column = "source"
    elif unit == "countries":
        frame = corpus.affiliations[["eid", "country"]].dropna()
        frame = frame[frame["country"].map(str).str.strip() != ""]
        column = "country"
    elif unit == "institutions":
        from .production import _org_frame
        frame = _org_frame(corpus, level)[["eid", "org"]].dropna()
        column = "org"
    else:
        return {}

    for eid, value in zip(frame["eid"], frame[column]):
        out.setdefault(str(eid), set()).add(str(value))
    return {k: tuple(v) for k, v in out.items()}
