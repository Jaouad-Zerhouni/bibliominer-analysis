"""Pays : production du premier auteur, SCP/MCP.

Le tableau « Corresponding Author's Countries » est un classique des articles
bibliométriques. Il répartit les documents selon le pays de l'auteur
**principal**, puis distingue :

  - **SCP** (*single country publications*) — tous les signataires du même pays ;
  - **MCP** (*multiple country publications*) — au moins deux pays.

Le rapport MCP/total mesure l'ouverture internationale d'un pays, ce que le
simple nombre de documents ne dit pas : un pays peut beaucoup publier en vase
clos.

**Réserve importante** : les exports ne portent pas toujours l'adresse de
correspondance. On prend alors le **premier auteur** comme représentant du
document — c'est la convention de repli usuelle, mais ce n'est pas
rigoureusement l'auteur correspondant, et il faut le dire plutôt que de laisser
croire à une précision qu'on n'a pas.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def corresponding_author_countries(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """Documents par pays du premier auteur, avec SCP et MCP.

    Colonnes : ``country``, ``documents``, ``scp``, ``mcp``, ``mcp_ratio``,
    ``citations``, ``citations_per_document``.
    """
    cols = ["country", "documents", "scp", "mcp", "mcp_ratio", "citations",
            "citations_per_document"]
    aff = corpus.affiliations
    aff = aff[aff["country"].notna() & (aff["country"].map(str).str.strip() != "")]
    if aff.empty:
        return pd.DataFrame(columns=cols)

    # Combien de pays distincts signent chaque document : c'est ce qui décide
    # SCP contre MCP, indépendamment de qui est premier auteur.
    per_doc = aff.groupby("eid")["country"].nunique()

    link = corpus.author_affiliations
    lead = pd.DataFrame(columns=["eid", "country"])
    if not link.empty:
        first = link[pd.to_numeric(link["position"], errors="coerce").eq(1)]
        if not first.empty:
            lead = (first.merge(aff[["eid", "aff_pos", "country"]],
                                on=["eid", "aff_pos"], how="inner")
                         [["eid", "country"]].drop_duplicates("eid"))

    # Repli : aucun rattachement pour le premier auteur (fréquent quand le
    # nettoyage n'a pas pu relier auteur et affiliation). On prend alors la
    # première affiliation du document — mieux qu'écarter le document.
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
    """Indices d'impact par pays : h, g, m, citations.

    Le pendant, pour les pays, de ce que `impact.py` fait pour les auteurs et
    les organisations. Un document co-signé par deux pays compte UNE fois pour
    chacun : c'est le comptage entier, et la somme dépasse donc volontairement
    le nombre de documents.
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
