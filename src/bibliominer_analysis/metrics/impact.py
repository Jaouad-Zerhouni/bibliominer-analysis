"""Impact indices: h, g, and derived ones.

The definitions used are the canonical ones, not variants:

  - **h** (Hirsch, 2005): the largest integer *h* such that *h* documents
    each have **at least** *h* citations.

  - **g** (Egghe, 2006): the largest integer *g* such that the *g* most
    cited documents total **at least** *g²* citations. The g-index is
    always ≥ h: it accounts for highly cited articles, which the h-index
    caps.

Citations are those of Scopus (`Cited by`), hence the article's **global**
citations, not only those received within the corpus. That is the
convention of usual bibliometric rankings.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def h_index(citations: Iterable[int]) -> int:
    """h-index of a series of citation counts."""
    c = np.sort(np.asarray([x for x in citations if x is not None and x >= 0],
                           dtype=float))[::-1]
    if c.size == 0:
        return 0
    ranks = np.arange(1, c.size + 1)
    return int(np.max(np.where(c >= ranks, ranks, 0)))


def g_index(citations: Iterable[int]) -> int:
    """g-index of a series of citation counts."""
    c = np.sort(np.asarray([x for x in citations if x is not None and x >= 0],
                           dtype=float))[::-1]
    if c.size == 0:
        return 0
    ranks = np.arange(1, c.size + 1)
    cumulative = np.cumsum(c)
    ok = cumulative >= ranks ** 2
    return int(np.max(np.where(ok, ranks, 0)))


def i10_index(citations: Iterable[int]) -> int:
    """Number of documents with at least 10 citations."""
    return int(sum(1 for x in citations if x is not None and x >= 10))


def e_index(citations: Iterable[int]) -> float:
    """e-index (Zhang, 2009): the EXCESS citations of the h-core.

    The h-index ignores everything beyond. Two authors with h = 10 are
    indistinguishable, whether their ten articles have 10 citations each or
    500: the first has 100 citations in their core, the second 5000, and h is
    10 in both cases. The e-index measures exactly what h throws away:

        e² = (sum of the citations of the h core articles) - h²
        e  = √(e²)

    It does not replace h, it **complements** it, which is the sense of the
    word in Zhang. Read with h, it tells a steady body of work from one
    carried by a few highly cited works.

    e is always ≥ 0: by definition of the h-index, each of the h core
    articles has at least h citations, so their sum reaches at least h².
    """
    c = np.sort(np.asarray([x for x in citations if x is not None and x >= 0],
                           dtype=float))[::-1]
    if c.size == 0:
        return 0.0
    h = h_index(c)
    if h == 0:
        return 0.0
    excess = float(c[:h].sum()) - h ** 2
    return round(float(np.sqrt(max(excess, 0.0))), 2)


def m_index(h: int, first_year, last_year) -> Optional[float]:
    """h relative to seniority: m = h / number of years of activity.

    The h-index can only grow over time: comparing a researcher with thirty
    years of career to one with five years by their h alone makes no sense.
    The m corrects exactly this bias.

    `last_year` is the last year **of the corpus**, never the current year:
    otherwise every m of a corpus that stops in 2020 would drop every 1
    January without any data having changed.
    """
    if first_year is None or last_year is None:
        return None
    if pd.isna(first_year) or pd.isna(last_year):
        return None
    span = int(last_year) - int(first_year) + 1
    if span <= 0:
        return None
    return round(h / span, 2)


# ---------------------------------------------------------------------------
# At corpus level
# ---------------------------------------------------------------------------

def corpus_impact(corpus) -> dict:
    """Impact indices of the whole corpus."""
    cites = pd.to_numeric(corpus.documents["cited_by"],
                          errors="coerce").fillna(0).astype(int)
    total = int(cites.sum())
    n = int(len(cites))
    return {
        "documents": n,
        "citations": total,
        "citations_per_doc": round(total / n, 2) if n else 0.0,
        "h_index": h_index(cites),
        "g_index": g_index(cites),
        "i10_index": i10_index(cites),
        "e_index": e_index(cites),
        "uncited": int((cites == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Per author
# ---------------------------------------------------------------------------

def _rank_counts(ranks, total: int) -> dict:
    """Distribution of an author's signatures by RANK: 1st, 2nd, 3rd, 4th and
    later.

    The authorship rank is not a detail: in most disciplines it encodes the
    role. First author means having carried the work; seventh out of eight
    means having contributed to it. Two authors with 50 documents can have
    opposite careers, and only this breakdown shows it.

    The four counts ALWAYS add up to the number of documents: "4th and later"
    is computed by difference. An unreadable rank therefore falls into it
    too, it is the catch-all bucket, and a table that adds up is better than
    a silent column. In practice Scopus exports always number the authors, so
    this case does not occur.
    """
    r = pd.to_numeric(ranks, errors="coerce")
    first = int((r == 1).sum())
    second = int((r == 2).sum())
    third = int((r == 3).sum())
    return {
        "first_author": first,
        "second_author": second,
        "third_author": third,
        "later_author": int(total - first - second - third),
    }


def authors_impact(corpus, n: Optional[int] = 20,
                   min_documents: int = 1) -> pd.DataFrame:
    """Ranking of the authors with their impact indices.

    Columns: ``author``, ``scopus_id``, ``documents``, ``citations``,
    ``h_index``, ``g_index``, ``first_author``, ``years``, ``first_year``,
    ``last_year``.

    An author is identified by their **Scopus identifier** if it exists;
    otherwise by their name. Two namesakes without an identifier stay
    indistinguishable: a limit of the data, not of the computation, and it is
    better to know it than to hide it.

    The computation is done on the documents OF THE CORPUS: an h-index of 6
    here means "6 documents of this corpus, each cited at least 6 times". It
    is not the person's global h-index, which would cover their whole work.
    """
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    empty = pd.DataFrame(columns=["author", "scopus_id", "documents", "citations",
                                  "h_index", "g_index", "first_author",
                                  "first_year", "last_year"])
    if a.empty:
        return empty

    docs = corpus.documents[["eid", "year"]].copy()
    docs["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                      errors="coerce").fillna(0).astype(int)
    a = a.merge(docs, on="eid", how="left")
    a["citations"] = a["citations"].fillna(0).astype(int)
    a["rank"] = pd.to_numeric(a["position"], errors="coerce")
    a["key"] = a["scopus_id"].fillna("name:" + a["name"].map(str))

    corpus_last = pd.to_numeric(corpus.documents["year"], errors="coerce").max()

    # Everything is computed in ONE pass over the table, not author by author:
    # the loop did five pandas operations per author (deduplication,
    # conversions, mode), 9 s on 10,000 documents and 2,500 authors, for
    # computations that fit in a few groupings. Same rules as before, same
    # result (checked on real corpora).
    #
    # An author can appear twice on the same document (rare, but exports do
    # it): duplicates are removed, otherwise citations and indices would be
    # inflated.
    per_doc = a.drop_duplicates(subset=["key", "eid"])
    per_doc = per_doc.assign(year_num=pd.to_numeric(per_doc["year"], errors="coerce"),
                             r1=(per_doc["rank"] == 1).astype(int),
                             r2=(per_doc["rank"] == 2).astype(int),
                             r3=(per_doc["rank"] == 3).astype(int))
    grouped = per_doc.groupby("key", sort=False)
    stats = pd.DataFrame({
        "documents": grouped.size(),
        "first_year": grouped["year_num"].min(),
        "last_year": grouped["year_num"].max(),
        "first_author": grouped["r1"].sum(),
        "second_author": grouped["r2"].sum(),
        "third_author": grouped["r3"].sum(),
    })
    cites = grouped["citations"].agg(list)
    # The most frequent name, the first in alphabetical order on a tie, exactly
    # what `Series.mode().iat[0]` gave.
    counts = a.groupby(["key", "name"], sort=False).size().rename("n").reset_index()
    names = (counts.sort_values(["key", "n", "name"], ascending=[True, False, True])
             .drop_duplicates("key").set_index("key")["name"])
    scopus = a.groupby("key", sort=False)["scopus_id"].first()

    rows = []
    for key in grouped.size().index:
        c = cites[key]
        h = h_index(c)
        s = stats.loc[key]
        first_year = s["first_year"]
        documents = int(s["documents"])
        first, second, third = int(s["first_author"]), int(s["second_author"]), int(s["third_author"])
        rows.append({
            "m_index": m_index(h, first_year if pd.notna(first_year) else None, corpus_last),
            "author": names.get(key),
            "scopus_id": scopus.get(key) if pd.notna(scopus.get(key)) else None,
            "documents": documents,
            "citations": int(sum(c)),
            "h_index": h,
            "g_index": g_index(c),
            "i10_index": i10_index(c),
            "e_index": e_index(c),
            "first_author": first,
            "second_author": second,
            "third_author": third,
            "later_author": documents - first - second - third,
            "first_year": int(first_year) if pd.notna(first_year) else None,
            "last_year": int(s["last_year"]) if pd.notna(s["last_year"]) else None,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return empty
    out = out[out["documents"] >= min_documents]
    out = out.sort_values(["h_index", "citations", "documents"],
                          ascending=False, kind="stable").reset_index(drop=True)
    return out.head(n) if n else out


def institutions_impact(corpus, n: Optional[int] = 20,
                        level: str = "parent") -> pd.DataFrame:
    """The same indices, aggregated per organisation.

    `level` is "parent" (the institution) or "subparent" (the internal unit:
    laboratory, school, department).

    A document counts ONCE per organisation, even if three of its authors are
    affiliated with it; otherwise the citations would be counted three times.
    """
    from .production import _org_frame

    aff = _org_frame(corpus, level)
    empty = pd.DataFrame(columns=["institution", "documents", "citations",
                                  "h_index", "g_index", "country"])
    if aff.empty:
        return empty

    pairs = aff[["eid", "org", "country"]].drop_duplicates(subset=["eid", "org"])
    pairs = pairs.rename(columns={"org": "parent1"})
    docs = corpus.documents[["eid"]].copy()
    docs["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                      errors="coerce").fillna(0).astype(int)
    docs["year"] = pd.to_numeric(corpus.documents["year"], errors="coerce")
    pairs = pairs.merge(docs, on="eid", how="left")
    pairs["citations"] = pairs["citations"].fillna(0).astype(int)
    corpus_last = docs["year"].max()

    rows = []
    for inst, g in pairs.groupby("parent1", sort=False):
        cites = g["citations"].tolist()
        countries = g["country"].dropna()
        years = g["year"].dropna()
        h = h_index(cites)
        rows.append({
            "institution": inst,
            "documents": int(g["eid"].nunique()),
            "citations": int(sum(cites)),
            "h_index": h,
            "g_index": g_index(cites),
            "m_index": m_index(h, years.min() if not years.empty else None,
                               corpus_last),
            "first_year": int(years.min()) if not years.empty else None,
            "country": countries.mode().iat[0] if not countries.empty else None,
        })

    out = pd.DataFrame(rows).sort_values(
        ["h_index", "citations", "documents"], ascending=False, kind="stable").reset_index(drop=True)
    return out.head(n) if n else out


def normalized_citations(corpus) -> pd.DataFrame:
    """Citations relative to the mean of their **publication year**.

    Columns: ``eid``, ``citations``, ``year_mean_citations``,
    ``normalized_citations``.

    Comparing the raw citations of two articles published eight years apart
    makes no sense: the older one almost always wins, and that says nothing
    about its quality. The normalised score divides by the mean of the
    cohort:

        1.0  = exactly the mean of its year
        3.0  = three times better than its contemporaries

    It is the only column that can rank an article from 2016 and one from
    2024 together. A year without any citation gives 0 rather than a division
    by zero.
    """
    d = corpus.documents[["eid"]].copy()
    d["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                   errors="coerce").fillna(0).astype(int)
    d["year"] = pd.to_numeric(corpus.documents["year"], errors="coerce")

    means = d.groupby("year")["citations"].transform("mean")
    d["year_mean_citations"] = means.round(2)
    d["normalized_citations"] = np.where(
        means > 0, (d["citations"] / means).round(3), 0.0)
    return d[["eid", "citations", "year_mean_citations", "normalized_citations"]]
