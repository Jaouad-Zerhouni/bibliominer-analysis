"""The AFFILIATION profile of a corpus: how many institutions per article,
and which authors carry several.

Three questions a reader asks about a corpus, which the number of authors
does not answer:

  - how many articles carry ONLY ONE affiliation, a single team, against
    two, three, more;
  - how many articles are signed by a SINGLE author;
  - which authors are attached to several institutions, on the same
    article (declared double affiliation) or from one article to the next
    (mobility, or secondary affiliation).

Double affiliation is counted with **full counting**: an author with two
institutions counts for each, as a co-signed article counts for each
country. It is the convention of the rest of the package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd



def _author_key(authors: pd.DataFrame) -> pd.Series:
    """The identity of an author: their Scopus identifier, otherwise their name,
    the SAME key as `top_authors` and `n_authors`."""
    return authors["scopus_id"].fillna("name:" + authors["name"].map(str))


def _links(corpus) -> pd.DataFrame:
    """(document, author, affiliation), one row per declared affiliation."""
    link = corpus.author_affiliations
    cols = ["eid", "position", "aff_pos", "key", "name", "institution", "city",
            "country"]
    if link.empty or corpus.authors.empty:
        return pd.DataFrame(columns=cols)

    authors = corpus.authors.copy()
    authors["key"] = _author_key(authors)
    out = link.merge(authors[["eid", "position", "key", "name"]],
                     on=["eid", "position"], how="inner")

    aff = corpus.affiliations
    if not aff.empty:
        pick = [c for c in ("eid", "aff_pos", "parent1", "city", "country")
                if c in aff.columns]
        out = out.merge(aff[pick], on=["eid", "aff_pos"], how="left")
    out["institution"] = out.get("parent1")
    if "institution" not in out.columns:
        out["institution"] = None
    # An affiliation without a named institution is still an affiliation: it is
    # kept under its rank, otherwise raw corpora would count zero everywhere.
    out["institution"] = out["institution"].fillna(
        "aff#" + out["aff_pos"].astype("string").fillna("?"))
    return out[[c for c in cols if c in out.columns]]


def documents_by_affiliation_count(corpus) -> pd.DataFrame:
    """Distribution of documents by number of DISTINCT affiliations.

    Columns: ``affiliations``, ``documents``, ``share``.
    """
    cols = ["affiliations", "documents", "share"]
    aff = corpus.affiliations
    if aff.empty or "eid" not in aff.columns:
        return pd.DataFrame(columns=cols)

    key = "parent1" if "parent1" in aff.columns else "raw"
    named = aff[aff[key].notna() & (aff[key].map(str).str.strip() != "")]
    per_doc = named.groupby("eid")[key].nunique()
    # A document without any usable affiliation counts as zero: hiding it would
    # suggest that the whole corpus is filled in.
    missing = set(corpus.documents["eid"]) - set(per_doc.index)
    per_doc = pd.concat([per_doc, pd.Series(0, index=sorted(missing), dtype=int)])
    if per_doc.empty:
        return pd.DataFrame(columns=cols)

    counts = per_doc.value_counts()
    out = pd.DataFrame({"affiliations": counts.index.astype(int),
                        "documents": counts.to_numpy(dtype=int)})
    out["share"] = (100 * out["documents"] / out["documents"].sum()).round(2)
    return out.sort_values("affiliations", kind="stable").reset_index(drop=True)


def authors_by_affiliation_count(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """Authors attached to SEVERAL institutions, the most attached first.

    Columns: ``author``, ``scopus_id``, ``documents``, ``institutions``,
    ``max_in_one_document``, ``affiliations``.

    ``institutions`` counts the distinct institutions over the WHOLE corpus;
    ``max_in_one_document`` the maximum carried on a SINGLE article. That one
    is a declared double affiliation; the other may only be a change of
    institution over the years.
    """
    cols = ["author", "scopus_id", "documents", "institutions",
            "max_in_one_document", "affiliations"]
    links = _links(corpus)
    if links.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for key, g in links.groupby("key", sort=False):
        institutions = sorted(set(g["institution"].dropna().map(str)))
        if len(institutions) < 2:
            continue
        per_doc = g.groupby("eid")["institution"].nunique()
        names = g["name"].dropna()
        rows.append({
            "author": names.mode().iat[0] if not names.empty else str(key),
            "scopus_id": "" if str(key).startswith("name:") else str(key),
            "documents": int(g["eid"].nunique()),
            "institutions": len(institutions),
            "max_in_one_document": int(per_doc.max()),
            "affiliations": "; ".join(institutions[:6]),
        })

    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out = out.sort_values(["max_in_one_document", "institutions", "documents",
                           "author"],
                          ascending=[False, False, False, True], kind="stable")
    out = out.reset_index(drop=True)
    return out.head(n) if n else out


def affiliation_profile(corpus) -> Dict[str, Any]:
    """The key figures of the affiliation profile, for a summary sheet.

    Keys: ``documents``, ``single_author_documents``,
    ``single_affiliation_documents``, ``two_affiliation_documents``,
    ``many_affiliation_documents``, their percentage shares, ``authors``,
    ``authors_with_several_affiliations`` and
    ``authors_with_double_affiliation`` (on the same article).
    """
    from .collaboration import _authors_per_doc

    documents = int(len(corpus.documents))
    per_doc = _authors_per_doc(corpus)
    single_author = int((per_doc == 1).sum()) if not per_doc.empty else 0

    by_count = documents_by_affiliation_count(corpus)
    def docs_where(mask) -> int:
        return int(by_count.loc[mask, "documents"].sum()) if not by_count.empty else 0

    one = docs_where(by_count["affiliations"] == 1) if not by_count.empty else 0
    two = docs_where(by_count["affiliations"] == 2) if not by_count.empty else 0
    many = docs_where(by_count["affiliations"] >= 3) if not by_count.empty else 0

    multi = authors_by_affiliation_count(corpus, n=None)
    double = int((multi["max_in_one_document"] >= 2).sum()) if not multi.empty else 0

    def share(value: int) -> float:
        return round(100.0 * value / documents, 2) if documents else 0.0

    return {
        "documents": documents,
        "single_author_documents": single_author,
        "single_author_share": share(single_author),
        "single_affiliation_documents": one,
        "single_affiliation_share": share(one),
        "two_affiliation_documents": two,
        "two_affiliation_share": share(two),
        "many_affiliation_documents": many,
        "many_affiliation_share": share(many),
        "authors": corpus.n_authors(),
        "authors_with_several_affiliations": int(len(multi)),
        "authors_with_double_affiliation": double,
    }
