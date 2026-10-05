"""Self-citations: what share of the influence comes from oneself.

A citation count does not say where citations come from. An author cited
twenty times by their own articles and an author cited twenty times by
others have the same total, and entirely different reputations. It is one
of the few indicators that reviewers explicitly ask for.

A citation is a **self-citation** at a given level when the citing and
the cited document share at least one entity at that level: an author, an
organisation, a country. The three levels do not overlap: two distinct
teams from the same country produce a *national* self-citation without
any author self-citation.

**Limit to state before any use**: the computation only covers citations
INTERNAL to the corpus. Self-citations from articles outside the corpus
are invisible here. The measured rate is therefore a floor, never the real
rate, and presenting it otherwise would be misleading.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

LEVELS = ("authors", "institutions", "countries")


def _members(corpus, level: str) -> Dict[str, set]:
    """eid -> set of the document's entities, at the requested level."""
    out: Dict[str, set] = {}

    if level == "authors":
        a = corpus.authors
        a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
        if a.empty:
            return out
        key = a["scopus_id"].fillna("name:" + a["name"].map(str))
        frame = pd.DataFrame({"eid": a["eid"].to_numpy(), "key": key.to_numpy()})
    elif level == "institutions":
        from .production import _org_frame
        org = _org_frame(corpus, "parent")
        if org.empty:
            return out
        frame = pd.DataFrame({"eid": org["eid"].to_numpy(),
                              "key": org["org"].to_numpy()})
    elif level == "countries":
        aff = corpus.affiliations
        aff = aff[aff["country"].notna()
                  & (aff["country"].map(str).str.strip() != "")]
        if aff.empty:
            return out
        frame = pd.DataFrame({"eid": aff["eid"].to_numpy(),
                              "key": aff["country"].to_numpy()})
    else:
        return out

    for eid, key in zip(frame["eid"], frame["key"]):
        if pd.notna(key):
            out.setdefault(str(eid), set()).add(str(key))
    return out


def self_citation_rate(corpus, level: str = "authors") -> Dict[str, Any]:
    """Overall self-citation rate at a given level.

    Returns: ``level``, ``citations``, ``self_citations``, ``external``,
    ``self_rate`` (%), and ``coverage_note``.
    """
    from .local import citation_pairs

    pairs = citation_pairs(corpus)
    base = {"level": level, "citations": 0, "self_citations": 0,
            "external": 0, "self_rate": None}
    if pairs.empty:
        return base

    members = _members(corpus, level)
    if not members:
        return base

    total = len(pairs)
    self_count = 0
    for citing, cited in zip(pairs["citing"], pairs["cited"]):
        a, b = members.get(str(citing)), members.get(str(cited))
        if a and b and (a & b):
            self_count += 1

    return {
        "level": level,
        "citations": int(total),
        "self_citations": int(self_count),
        "external": int(total - self_count),
        "self_rate": round(100.0 * self_count / total, 1) if total else None,
    }


def self_citation_summary(corpus) -> pd.DataFrame:
    """The three levels side by side.

    Columns: ``level``, ``citations``, ``self_citations``, ``external``,
    ``self_rate``.

    The levels are NOT strictly nested: an author who changed institution, or
    country, between the cited and the citing article produces an author
    self-citation without an institution self-citation. In practice the
    "country" rate almost always exceeds the "author" rate, and the gap
    between the two is what informs; but it is not a guarantee, and an
    institutional rate lower than the author rate is possible.
    """
    cols = ["level", "citations", "self_citations", "external", "self_rate"]
    rows = [self_citation_rate(corpus, lv) for lv in LEVELS]
    return pd.DataFrame(rows, columns=cols)


def _received_citations(pairs: pd.DataFrame,
                        authors_of: Dict[str, set]) -> tuple:
    """Citations received per author, and how many came from their own
    documents.

    One pass per citation, over the cited document's authors ONLY: the
    previous version went through all the authors of the corpus for every
    citation, and became unusable on a few thousand documents.
    """
    received: Dict[str, int] = {}
    self_hits: Dict[str, int] = {}
    for citing, cited in zip(pairs["citing"].map(str), pairs["cited"].map(str)):
        citing_authors = authors_of.get(citing, set())
        for key in authors_of.get(cited, ()):
            received[key] = received.get(key, 0) + 1
            if key in citing_authors:
                self_hits[key] = self_hits.get(key, 0) + 1
    return received, self_hits


def authors_self_citation(corpus, n: Optional[int] = 20,
                          min_citations: int = 1) -> pd.DataFrame:
    """Self-citation per author.

    Columns: ``author``, ``documents``, ``local_citations``,
    ``self_citations``, ``external_citations``, ``self_rate``.

    ``self_citations`` counts the citations an author received from
    documents they signed themselves.

    An author is identified as everywhere else in the package: by their
    Scopus identifier, otherwise by their name. Fixed defect: this one
    computation went by NAME, and merged two namesakes; one then "cited
    themselves" by citing the other.
    """
    from .local import citation_pairs

    cols = ["author", "documents", "local_citations", "self_citations",
            "external_citations", "self_rate"]
    pairs = citation_pairs(corpus)
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    if pairs.empty or a.empty:
        return pd.DataFrame(columns=cols)

    a = a.assign(key=a["scopus_id"].fillna("name:" + a["name"].map(str)))
    docs_of: Dict[str, set] = {}
    authors_of: Dict[str, set] = {}
    for eid, key in zip(a["eid"].map(str), a["key"]):
        docs_of.setdefault(key, set()).add(eid)
        authors_of.setdefault(eid, set()).add(key)
    name_of = a.groupby("key")["name"].agg(lambda s: s.mode().iat[0]).to_dict()

    received, self_hits = _received_citations(pairs, authors_of)

    rows = []
    for key, docs in docs_of.items():
        total = received.get(key, 0)
        if total < min_citations:
            continue
        own = self_hits.get(key, 0)
        rows.append({
            "author": name_of.get(key, key),
            "documents": len(docs),
            "local_citations": total,
            "self_citations": own,
            "external_citations": total - own,
            "self_rate": round(100.0 * own / total, 1) if total else 0.0,
        })

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols).sort_values(
        ["local_citations", "self_rate"], ascending=False, kind="stable").reset_index(drop=True)
    return out.head(n) if n else out
