"""Main information about the corpus.

It is the table placed at the top of a bibliometric article: it describes
the corpus before any analysis. Every value is defined without ambiguity,
because the same label covers different computations from one tool to
another.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def main_information(corpus) -> Dict[str, Any]:
    """Descriptive indicators of the corpus.

    A few definitions, so that one knows what one reads:

    - **annual growth rate**: compound rate between the first and the last
      complete year, ``(N_end/N_start)^(1/years) - 1``. It only makes sense
      over at least two years.
    - **average document age**: years elapsed since publication, counted from
      the most recent year OF THE CORPUS, not from today; otherwise the value
      would change every year.
    - **authors per document**: mean number of authors, duplicates removed.
    - **single-authored documents**: a single author.
    - **international collaboration**: share of documents signed by at least
      two countries.
    """
    docs = corpus.documents
    n_docs = int(len(docs))
    years = pd.to_numeric(docs["year"], errors="coerce").dropna()
    cites = pd.to_numeric(docs["cited_by"], errors="coerce").fillna(0).astype(int)

    y_min = int(years.min()) if not years.empty else None
    y_max = int(years.max()) if not years.empty else None

    # --- compound annual growth -------------------------------------------
    growth = None
    if y_min is not None and y_max is not None and y_max > y_min:
        per_year = years.astype(int).value_counts().sort_index()
        first, last = per_year.iloc[0], per_year.iloc[-1]
        span = y_max - y_min
        if first > 0 and span > 0:
            growth = round(((last / first) ** (1 / span) - 1) * 100, 2)

    # --- average age -------------------------------------------------------
    age = None
    if not years.empty and y_max is not None:
        age = round(float((y_max - years).mean()), 2)

    # --- authors -----------------------------------------------------------
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    per_doc = a.drop_duplicates(subset=["eid", "name"]).groupby("eid").size()
    single = int((per_doc == 1).sum())
    authors_per_doc = round(float(per_doc.mean()), 2) if not per_doc.empty else 0.0

    # Collaboration index: authors per document, computed ONLY on co-authored
    # documents. A corpus full of single-author articles would otherwise drag
    # the indicator down, while that is not what it measures.
    multi = per_doc[per_doc > 1]
    collab_index = round(float(multi.mean()), 2) if not multi.empty else 0.0

    # --- collaboration internationale --------------------------------------
    aff = corpus.affiliations
    aff = aff[aff["country"].notna() & (aff["country"].map(str).str.strip() != "")]
    countries_per_doc = aff.groupby("eid")["country"].nunique()
    intl = int((countries_per_doc > 1).sum())
    docs_with_country = int(len(countries_per_doc))
    intl_share = round(100 * intl / docs_with_country, 1) if docs_with_country else 0.0

    # --- keywords and references -------------------------------------------
    kw = corpus.keywords
    n_kw_author = int(kw.loc[kw["kind"] == "author", "keyword"]
                        .map(str).str.lower().nunique())
    n_kw_index = int(kw.loc[kw["kind"] == "index", "keyword"]
                       .map(str).str.lower().nunique())

    refs = corpus.references
    refs_per_doc = (round(float(refs.groupby("eid").size().mean()), 1)
                    if not refs.empty else 0.0)

    return {
        "documents": n_docs,
        "year_min": y_min,
        "year_max": y_max,
        "timespan": ("%d-%d" % (y_min, y_max)) if y_min is not None else None,
        "annual_growth_rate": growth,
        "document_average_age": age,
        "sources": int(docs["source"].nunique(dropna=True)),
        "document_types": int(docs["doc_type"].nunique(dropna=True)),
        "total_citations": int(cites.sum()),
        "citations_per_document": round(float(cites.mean()), 2) if n_docs else 0.0,
        "references": int(len(refs)),
        "references_per_document": refs_per_doc,
        "keywords_author": n_kw_author,
        "keywords_index": n_kw_index,
        "authors": corpus.n_authors(),
        "authors_per_document": authors_per_doc,
        "single_authored_documents": single,
        # Mean number of authors of CO-AUTHORED documents only (a collaboration
        # index). Not to be confused with Lawani's CI
        # (`collaboration.collaboration_indicators`), which averages over ALL
        # documents and equals `authors_per_document`. The interface therefore
        # names them "Authors / co-authored document" and "CI (Lawani)".
        "collaboration_index": collab_index,
        "countries": int(aff["country"].nunique()) if not aff.empty else 0,
        "international_documents": intl,
        "international_share": intl_share,
    }


def most_cited_documents(corpus, n: int = 20) -> pd.DataFrame:
    """Most cited documents of the corpus.

    ``citations_per_year`` relates citations to age: without this column, an
    article from 2016 systematically crushes an article from 2024 that may be
    far more striking.
    """
    d = corpus.documents.copy()
    d["citations"] = pd.to_numeric(d["cited_by"], errors="coerce").fillna(0).astype(int)
    years = pd.to_numeric(d["year"], errors="coerce")
    if years.notna().any():
        latest = int(years.max())
        # +1 year: an article published in the most recent year has already "lived"
        # one year; otherwise we would divide by zero.
        d["citations_per_year"] = (d["citations"] / (latest - years + 1)).round(2)
    else:
        d["citations_per_year"] = np.nan

    first = (corpus.authors[corpus.authors["position"] == 1]
             .drop_duplicates(subset=["eid"])[["eid", "name"]]
             .rename(columns={"name": "first_author"}))
    d = d.merge(first, on="eid", how="left")

    cols = ["title", "first_author", "year", "source", "doc_type",
            "citations", "citations_per_year", "doi"]
    return (d.sort_values("citations", ascending=False, kind="stable")
             .head(n)[cols].reset_index(drop=True))


def most_cited_references(corpus, n: int = 20) -> pd.DataFrame:
    """References most cited BY the corpus (local impact).

    Not to be confused with the number of global citations: here we count how
    many documents of the corpus cite that work. That is what identifies the
    foundations of the field as this corpus practises it.
    """
    refs = corpus.references
    empty = pd.DataFrame(columns=["reference", "ref_year", "ref_doi",
                                  "local_citations"])
    if refs.empty:
        return empty

    r = refs.copy()
    # Identity: DOI if present, otherwise the normalised title, the same rule as
    # the co-citation network, so that the two views agree.
    key = r["ref_doi"].fillna("")
    key = key.where(key.map(str).str.strip() != "",
                    "t:" + r["ref_title"].fillna("").map(str).str.lower().str.strip())
    r["key"] = key.map(str).str.lower().str.strip()
    r = r[r["key"] != ""]
    if r.empty:
        return empty

    r = r.drop_duplicates(subset=["eid", "key"])
    g = (r.groupby("key")
           .agg(reference=("ref_title", lambda s: s.dropna().mode().iat[0]
                           if not s.dropna().empty else None),
                ref_year=("ref_year", "first"),
                ref_doi=("ref_doi", "first"),
                local_citations=("eid", "nunique"))
           .reset_index(drop=True)
           .sort_values("local_citations", ascending=False, kind="stable")
           .reset_index(drop=True))
    return g.head(n)
