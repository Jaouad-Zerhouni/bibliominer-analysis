"""Collaboration and productivity indicators.

Formulas, as established by their authors:

  - **Degree of collaboration C** (Subramanyam, 1983)
        ``C = Nm / (Nm + Ns)``
    share of co-authored documents. Between 0 (all single-authored) and 1
    (none).

  - **Collaborative index CI** (Lawani, 1980)
        ``CI = Σ(j · f_j) / N``
    mean number of authors per document, over all documents.

  - **Collaboration coefficient CC** (Ajiferuke, Burrell & Tague, 1988)
        ``CC = 1 - [Σ(f_j / j)] / N``
    fixes the CI's flaw: the CI grows without bound with fifty-author
    articles, while the CC stays between 0 and 1.

  - **Modified collaboration coefficient MCC** (Savanur & Srikanth, 2010)
        ``MCC = (A / (A - 1)) × [1 - Σ(f_j / j) / N]``
    where **A is the total number of distinct authors in the collection**.
    The CC cannot reach 1: if all A authors sign every article, it equals
    1 - 1/A. The factor A/(A - 1) makes that maximum collaboration equal 1.
    On a real corpus A is large, and MCC ≈ CC: that is the behaviour
    reported by the studies that apply it.

  - **CAI, co-authorship index** (Garg & Padhi, 2001)
        ``CAI = [(N_ij / N_i0) / (N_0j / N_00)] × 100``
    compares, period by period, the share of an authorship type with its
    overall share. **100 = in line with the average**, above =
    over-represented.

  - **AAPP, average author productivity**
        ``AAPP = number of documents / number of distinct authors``

  - **Price's law**: the square root of the authors produces half of the
    signatures. The gap between theory and observation is given.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd


def _authors_per_doc(corpus) -> pd.Series:
    """Number of DISTINCT authors per document."""
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    if a.empty:
        return pd.Series(dtype=int)
    return a.drop_duplicates(subset=["eid", "name"]).groupby("eid").size()


def collaboration_indicators(corpus) -> Dict[str, Any]:
    """All the collaboration indicators in one call."""
    per_doc = _authors_per_doc(corpus)
    n = int(len(per_doc))
    if n == 0:
        return {"documents": 0, "single_authored": 0, "multi_authored": 0,
                "degree_of_collaboration": None, "collaboration_index": None,
                "collaborative_coefficient": None,
                "modified_collaborative_coefficient": None,
                "authors": 0, "aapp": None, "max_authors": 0}

    single = int((per_doc == 1).sum())
    multi = n - single

    # f_j: number of documents with j authors.
    f = per_doc.value_counts().sort_index()
    j = f.index.to_numpy(dtype=float)
    fj = f.to_numpy(dtype=float)

    ci = float((j * fj).sum() / n)
    cc = float(1 - (fj / j).sum() / n)
    a_max = int(per_doc.max())
    n_authors = corpus.n_authors()
    # A = DISTINCT authors of the collection, not the maximum per document.
    # Fixed defect: with the maximum, a corpus where every article has exactly
    # two authors got MCC = 1, "maximum collaboration", while each author there
    # signs with a single partner only.
    mcc = float(cc * n_authors / (n_authors - 1)) if n_authors > 1 else 0.0
    return {
        "documents": n,
        "single_authored": single,
        "multi_authored": multi,
        "degree_of_collaboration": round(multi / n, 4),
        "collaboration_index": round(ci, 4),
        "collaborative_coefficient": round(cc, 4),
        "modified_collaborative_coefficient": round(mcc, 4),
        "authors": n_authors,
        # Average productivity: documents per distinct author.
        "aapp": round(n / n_authors, 4) if n_authors else None,
        "max_authors": a_max,
    }


def authorship_pattern(corpus) -> pd.DataFrame:
    """Distribution of documents by number of authors."""
    per_doc = _authors_per_doc(corpus)
    empty = pd.DataFrame(columns=["authors", "documents", "share"])
    if per_doc.empty:
        return empty
    f = per_doc.value_counts().sort_index()
    df = pd.DataFrame({"authors": f.index.astype(int),
                       "documents": f.to_numpy(dtype=int)})
    df["share"] = (100 * df["documents"] / df["documents"].sum()).round(2)
    return df.reset_index(drop=True)


#: Authorship categories of the CAI, as used in the literature.
_CAI_BINS = [("single", 1, 1), ("two", 2, 2), ("three", 3, 3),
             ("multi", 4, None)]


def cai(corpus, block_years: int = 5) -> pd.DataFrame:
    """Co-authorship index per period.

    `block_years` groups the years into blocks: on a short corpus, a 5-year
    period gives usable counts where year by year would produce indices drawn
    from one or two documents.

    Reading: **100 = in line with the corpus average**. 150 means that the
    period produces 1.5 times more documents of this type than expected.
    """
    per_doc = _authors_per_doc(corpus)
    empty = pd.DataFrame(columns=["period", "documents", "single", "two",
                                  "three", "multi", "cai_single", "cai_two",
                                  "cai_three", "cai_multi"])
    if per_doc.empty:
        return empty

    years = pd.to_numeric(corpus.documents.set_index("eid")["year"],
                          errors="coerce")
    df = pd.DataFrame({"n_authors": per_doc})
    df["year"] = years.reindex(df.index)
    df = df.dropna(subset=["year"])
    if df.empty:
        return empty

    y0 = int(df["year"].min())
    df["block"] = ((df["year"].astype(int) - y0) // block_years).astype(int)

    def band(k: int) -> str:
        for name, lo, hi in _CAI_BINS:
            if k >= lo and (hi is None or k <= hi):
                return name
        return "multi"

    df["band"] = df["n_authors"].map(band)

    n00 = len(df)                                   # all documents
    n0j = df["band"].value_counts()                 # per type, all blocks

    rows = []
    for block, g in df.groupby("block"):
        ni0 = len(g)                                # documents of the block
        start = y0 + block * block_years
        end = min(start + block_years - 1, int(df["year"].max()))
        row: Dict[str, Any] = {
            "period": "%d-%d" % (start, end) if end > start else str(start),
            "documents": ni0,
        }
        counts = g["band"].value_counts()
        for name, _lo, _hi in _CAI_BINS:
            nij = int(counts.get(name, 0))
            row[name] = nij
            total_j = int(n0j.get(name, 0))
            # CAI undefined if this authorship type exists nowhere.
            row["cai_" + name] = (round(((nij / ni0) / (total_j / n00)) * 100, 1)
                                  if ni0 and total_j else None)
        rows.append(row)

    return pd.DataFrame(rows)


def price_law(corpus) -> Dict[str, Any]:
    """Price's law: √N authors should produce half of the signatures.

    Theory and observation are compared. A large gap signals a corpus more
    (or less) concentrated than Price predicts; that is the point of the
    indicator, not whether it "comes out right".
    """
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    if a.empty:
        return {"authors": 0, "expected_core": 0, "observed_share": None,
                "half_reached_with": None}

    key = a["scopus_id"].fillna("name:" + a["name"].map(str))
    per_author = (a.assign(key=key).drop_duplicates(subset=["key", "eid"])
                   .groupby("key").size().sort_values(ascending=False, kind="stable"))

    n_authors = int(len(per_author))
    total = int(per_author.sum())
    core = int(round(math.sqrt(n_authors)))

    observed = int(per_author.head(core).sum()) if core else 0
    cum = per_author.cumsum()
    reached = int((cum < total / 2).sum() + 1) if total else 0

    return {
        "authors": n_authors,
        "signatures": total,
        # Number of authors the law designates as the "core".
        "expected_core": core,
        # Share actually produced by this core (the law predicts 50 %).
        "observed_share": round(100 * observed / total, 1) if total else None,
        # Number of authors actually needed to reach half.
        "half_reached_with": reached,
    }


def authorship_groups(corpus) -> pd.DataFrame:
    """Distribution of documents by NUMBER of authors: 1, 2, 3, 4+.

    Columns: ``group``, ``min_authors``, ``documents``, ``share``,
    ``citations``, ``citations_per_document``.

    Grouping stops at "4 and more" because beyond that the counts crumble:
    telling 7 authors from 8 teaches nothing, while going from 1 to 2 authors
    is the boundary that matters, that of collaboration.

    The four rows are ALWAYS present, at zero if needed. A missing category
    would read as missing data, while a corpus without a single
    single-authored article is information in itself.
    """
    labels = [(1, "1 author"), (2, "2 authors"), (3, "3 authors"), (4, "4+ authors")]
    cols = ["group", "min_authors", "documents", "share", "citations",
            "citations_per_document"]

    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    docs = corpus.documents[["eid"]].copy()
    docs["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                      errors="coerce").fillna(0).astype(int)
    if a.empty or docs.empty:
        return pd.DataFrame([{"group": lab, "min_authors": k, "documents": 0,
                              "share": 0.0, "citations": 0,
                              "citations_per_document": 0.0}
                             for k, lab in labels], columns=cols)

    per_doc = (a.drop_duplicates(subset=["eid", "name"])
                .groupby("eid").size().rename("n_authors"))
    docs = docs.merge(per_doc, left_on="eid", right_index=True, how="left")
    docs["n_authors"] = docs["n_authors"].fillna(0).astype(int)
    # A document without an identified author is neither "single-authored" nor
    # "collaborative": it is left out rather than filed arbitrarily.
    docs = docs[docs["n_authors"] > 0]
    total = len(docs)

    rows = []
    for k, label in labels:
        sub = docs[docs["n_authors"] == k] if k < 4 else docs[docs["n_authors"] >= 4]
        n = int(len(sub))
        cites = int(sub["citations"].sum())
        rows.append({
            "group": label,
            "min_authors": k,
            "documents": n,
            "share": round(100.0 * n / total, 1) if total else 0.0,
            "citations": cites,
            "citations_per_document": round(cites / n, 2) if n else 0.0,
        })
    return pd.DataFrame(rows, columns=cols)
