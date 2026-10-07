"""Open access: what the export says, what SCImago says, and why they differ.

Two sources of information, two distinct questions; confusing them is the
common mistake:

  - **The imported file** carries a status **per article** ("Gold",
    "Green", "Bronze", "Hybrid"). It is the truth about THIS document: can
    it be read freely, and through which route.
  - **SCImago** carries a status **per journal**. It is a property of the
    venue: is the journal fully open.

An open access article in a subscription journal exists: that is
precisely hybrid, and green (archive deposit). The reverse too: an open
journal whose field Scopus left empty. Neither source replaces the other.

The routes, in the usual terminology:

  - **Gold**: published open in a fully open journal.
  - **Hybrid**: made open by payment in a subscription journal.
  - **Bronze**: free to read on the publisher's site, without an explicit
    licence: access can be withdrawn at any time.
  - **Green**: deposited by the author in an archive.

A **missing status is not a closed status**. Scopus fills the field for
only part of the documents; counting the empty ones as "subscription"
would mechanically lower the open access share.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import pandas as pd

#: Recognised routes, in the usual reading order.
ROUTES = ("Gold", "Hybrid", "Bronze", "Green")

_ROUTE_PATTERNS = {
    "Gold": re.compile(r"\bgold\b", re.I),
    "Hybrid": re.compile(r"\bhybrid\b", re.I),
    "Bronze": re.compile(r"\bbronze\b", re.I),
    "Green": re.compile(r"\bgreen\b", re.I),
}
_ANY_OA = re.compile(r"\bopen\s*access\b", re.I)


def routes_of(value: Any) -> list:
    """Open access routes declared for a document.

    The same article can combine several, "Gold" AND "Green" when it is
    published open and then deposited in an archive. All are kept: keeping
    only one would lose the most interesting information, the deposit.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    found = [name for name, pat in _ROUTE_PATTERNS.items() if pat.search(value)]
    if not found and _ANY_OA.search(value):
        # "All Open Access" without a specified route: open, route unknown.
        return ["Unspecified"]
    return found


def _document_access(corpus) -> pd.DataFrame:
    """A document, its raw status, its routes, its citations."""
    docs = corpus.documents
    cols = ["eid", "raw", "is_open", "routes", "citations", "year"]
    if docs.empty:
        return pd.DataFrame(columns=cols)

    d = pd.DataFrame({"eid": docs["eid"].to_numpy()})
    raw = docs["open_access"] if "open_access" in docs.columns else pd.Series(
        [None] * len(docs))
    d["raw"] = raw.to_numpy()
    d["routes"] = [routes_of(v) for v in d["raw"]]
    # Scopus fills this field ONLY for open articles: it never writes "closed".
    # An empty field therefore means "not reported as open", which covers closed
    # articles AND those whose status is simply missing. They cannot be told
    # apart, and pretending otherwise would distort the open access share.
    d["is_open"] = [bool(r) if r else None for r in d["routes"]]
    d["citations"] = pd.to_numeric(docs["cited_by"], errors="coerce").fillna(0).astype(int)
    d["year"] = pd.to_numeric(docs["year"], errors="coerce")
    return d[cols]


def access_status(corpus) -> pd.DataFrame:
    """Documents by access status, **according to the imported file**.

    Columns: ``status``, ``documents``, ``share``, ``citations``,
    ``citations_per_document``.

    Only two rows, on purpose. Scopus only flags OPEN articles; it never
    writes "closed". "Not reported" therefore groups closed articles and those
    whose status is missing; separating them would assume information that is
    not available.
    """
    cols = ["status", "documents", "share", "citations", "citations_per_document"]
    order = ["Open access", "Not flagged"]
    d = _document_access(corpus)
    if d.empty:
        return pd.DataFrame([{"status": s, "documents": 0, "share": 0.0,
                              "citations": 0, "citations_per_document": 0.0}
                             for s in order], columns=cols)

    def bucket(v) -> str:
        return "Open access" if v else "Not flagged"

    d = d.copy()
    d["status"] = d["is_open"].map(bucket)
    total = len(d)

    rows = []
    for s in order:
        sub = d[d["status"] == s]
        n = int(len(sub))
        cites = int(sub["citations"].sum())
        rows.append({
            "status": s,
            "documents": n,
            "share": round(100.0 * n / total, 1) if total else 0.0,
            "citations": cites,
            "citations_per_document": round(cites / n, 2) if n else 0.0,
        })
    return pd.DataFrame(rows, columns=cols)


def access_routes(corpus) -> pd.DataFrame:
    """Declared open access routes.

    Columns: ``route``, ``documents``, ``share_of_open``, ``citations``,
    ``citations_per_document``.

    An article combining two routes counts for each: the sum therefore
    deliberately exceeds the number of open articles, and ``share_of_open``
    is read against the open documents ONLY, not against the whole corpus.
    """
    cols = ["route", "documents", "share_of_open", "citations",
            "citations_per_document"]
    d = _document_access(corpus)
    labels = list(ROUTES) + ["Unspecified"]
    if d.empty:
        return pd.DataFrame([{"route": r, "documents": 0, "share_of_open": 0.0,
                              "citations": 0, "citations_per_document": 0.0}
                             for r in labels], columns=cols)

    open_docs = int(sum(1 for v in d["is_open"] if v))
    rows = []
    for route in labels:
        sub = d[[route in r for r in d["routes"]]]
        n = int(len(sub))
        cites = int(sub["citations"].sum())
        rows.append({
            "route": route,
            "documents": n,
            "share_of_open": round(100.0 * n / open_docs, 1) if open_docs else 0.0,
            "citations": cites,
            "citations_per_document": round(cites / n, 2) if n else 0.0,
        })
    return pd.DataFrame(rows, columns=cols)


def access_over_time(corpus) -> pd.DataFrame:
    """Yearly evolution of the open access share.

    Columns: ``year``, ``documents``, ``open_access``, ``share``.

    The denominator is ALL the documents of the year, not only the flagged
    ones: since Scopus never writes "closed", restricting to the filled-in
    documents would give 100 % every year, an exact and perfectly useless
    figure.
    """
    cols = ["year", "documents", "open_access", "share"]
    d = _document_access(corpus).dropna(subset=["year"])
    if d.empty:
        return pd.DataFrame(columns=cols)

    d = d.copy()
    d["open_flag"] = [bool(v) for v in d["is_open"]]
    out = (d.groupby("year")
             .agg(documents=("eid", "nunique"),
                  open_access=("open_flag", "sum"))
             .reset_index())
    out["open_access"] = out["open_access"].astype(int)
    out["share"] = (100.0 * out["open_access"] / out["documents"]).round(1)
    out["year"] = out["year"].astype(int)
    return out.sort_values("year", kind="stable").reset_index(drop=True)[cols]


def access_summary(corpus, path: Optional[Any] = None) -> Dict[str, Any]:
    """The two sources side by side, with what separates them.

    Returns: ``documents``, ``reported``, ``open_documents``,
    ``open_share_of_reported``, ``journal_open_documents``, ``disagreement``.

    ``disagreement`` counts the documents declared open while their journal
    is not: hybrids and archive deposits. It is the figure that justifies
    keeping both sources rather than one.
    """
    from .scimago import enrich_sources

    d = _document_access(corpus)
    base = {"documents": int(len(d)), "reported": 0, "open_documents": 0,
            "open_share_of_reported": None, "journal_open_documents": 0,
            "disagreement": 0}
    if d.empty:
        return base

    reported = int(d["is_open"].notna().sum())
    open_docs = int(sum(1 for v in d["is_open"] if v))
    base["reported"] = reported
    base["open_documents"] = open_docs
    base["open_share_of_reported"] = (round(100.0 * open_docs / reported, 1)
                                      if reported else None)

    enriched = enrich_sources(corpus, path)
    if enriched.empty:
        return base

    flag = enriched["open_access"].map(lambda v: bool(v) if v == v and v is not None else False)
    journal_open = set(enriched.loc[flag, "source"])
    src = corpus.documents[["eid", "source"]].copy()
    src["source"] = src["source"].map(str).str.strip()
    in_open_journal = set(src.loc[src["source"].isin(journal_open), "eid"].map(str))

    base["journal_open_documents"] = len(in_open_journal)
    base["disagreement"] = int(sum(
        1 for eid, is_open in zip(d["eid"].map(str), d["is_open"])
        if bool(is_open) and eid not in in_open_journal))
    return base
