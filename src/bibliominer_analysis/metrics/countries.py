"""Countries: first-author production, SCP/MCP.

The "Corresponding Author's Countries" table is a classic of bibliometric
articles. It distributes documents by the country of the **main** author,
then distinguishes:

  - **SCP** (*single country publications*): all authors from the same
    country;
  - **MCP** (*multiple country publications*): at least two countries.

The MCP/total ratio measures a country's international openness, which
the plain number of documents does not say: a country can publish a lot
in isolation.

**Important caveat**: exports do not always carry the correspondence
address. The **first author** is then taken as the document's
representative. That is the usual fallback convention, but it is not
strictly the corresponding author, and it must be said rather than
suggesting a precision that is not there.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def corresponding_author_countries(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """Documents by country of the first author, with SCP and MCP.

    Columns: ``country``, ``documents``, ``scp``, ``mcp``, ``mcp_ratio``,
    ``citations``, ``citations_per_document``.
    """
    cols = ["country", "documents", "scp", "mcp", "mcp_ratio", "citations",
            "citations_per_document"]
    aff = corpus.affiliations
    aff = aff[aff["country"].notna() & (aff["country"].map(str).str.strip() != "")]
    if aff.empty:
        return pd.DataFrame(columns=cols)

    # How many distinct countries sign each document: that decides SCP versus
    # MCP, regardless of who is first author.
    per_doc = aff.groupby("eid")["country"].nunique()

    link = corpus.author_affiliations
    lead = pd.DataFrame(columns=["eid", "country"])
    if not link.empty:
        first = link[pd.to_numeric(link["position"], errors="coerce").eq(1)]
        if not first.empty:
            lead = (first.merge(aff[["eid", "aff_pos", "country"]],
                                on=["eid", "aff_pos"], how="inner")
                         [["eid", "country"]].drop_duplicates("eid"))

    # Fallback: no affiliation for the first author (frequent when the cleaning
    # could not link author and affiliation). The first affiliation of the
    # document is then used, better than leaving the document out.
    missing = set(aff["eid"]) - set(lead["eid"])
    if missing:
        fallback = (aff[aff["eid"].isin(missing)]
                    .sort_values("aff_pos", kind="stable")
                    .drop_duplicates("eid")[["eid", "country"]])
        lead = pd.concat([lead, fallback], ignore_index=True)

    if lead.empty:
        return pd.DataFrame(columns=cols)

    docs = corpus.documents[["eid"]].copy()
    docs["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                      errors="coerce").fillna(0).astype(int)
    lead = lead.merge(docs, on="eid", how="left")
    lead["citations"] = lead["citations"].fillna(0).astype(int)
    lead["n_countries"] = lead["eid"].map(per_doc).fillna(1).astype(int)
    lead["is_mcp"] = lead["n_countries"] > 1

    out = (lead.groupby("country")
               .agg(documents=("eid", "nunique"),
                    mcp=("is_mcp", "sum"),
                    citations=("citations", "sum"))
               .reset_index())
    out["mcp"] = out["mcp"].astype(int)
    out["scp"] = out["documents"] - out["mcp"]
    out["mcp_ratio"] = (100 * out["mcp"] / out["documents"]).round(1)
    out["citations_per_document"] = (out["citations"] / out["documents"]).round(2)

    out = out.sort_values(["documents", "citations"],
                          ascending=False, kind="stable").reset_index(drop=True)
    return (out[cols].head(n) if n else out[cols])


def countries_impact(corpus, n=20, min_documents=1):
    """Impact indices by country: h, g, m, citations.

    The counterpart, for countries, of what `impact.py` does for authors and
    organisations. A document co-signed by two countries counts ONCE for each:
    that is full counting, so the sum deliberately exceeds the number of
    documents.
    """
    from .impact import g_index, h_index, m_index

    cols = ["country", "documents", "citations", "h_index", "g_index",
            "m_index", "first_year", "last_year"]
    aff = corpus.affiliations
    aff = aff[aff["country"].notna() & (aff["country"].map(str).str.strip() != "")]
    if aff.empty:
        return pd.DataFrame(columns=cols)

    pairs = aff[["eid", "country"]].drop_duplicates()
    docs = corpus.documents[["eid"]].copy()
    docs["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                      errors="coerce").fillna(0).astype(int)
    docs["year"] = pd.to_numeric(corpus.documents["year"], errors="coerce")
    pairs = pairs.merge(docs, on="eid", how="left")
    pairs["citations"] = pairs["citations"].fillna(0).astype(int)
    corpus_last = docs["year"].max()

    rows = []
    for country, g in pairs.groupby("country", sort=False):
        cites = g["citations"].tolist()
        years = g["year"].dropna()
        h = h_index(cites)
        first = int(years.min()) if not years.empty else None
        rows.append({
            "country": country,
            "documents": int(g["eid"].nunique()),
            "citations": int(sum(cites)),
            "h_index": h,
            "g_index": g_index(cites),
            "m_index": m_index(h, first, corpus_last),
            "first_year": first,
            "last_year": int(years.max()) if not years.empty else None,
        })

    out = pd.DataFrame(rows)
    out = out[out["documents"] >= min_documents]
    out = out.sort_values(["h_index", "citations", "documents"],
                          ascending=False, kind="stable").reset_index(drop=True)
    return out.head(n) if n else out
