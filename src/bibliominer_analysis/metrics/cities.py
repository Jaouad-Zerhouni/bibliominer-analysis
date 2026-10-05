"""**Cities**: the geographic scale that the country crushes.

The cleaning resolves every affiliation down to the city. Stopping at the
country loses the essential: a 90 % "Moroccan" corpus can be a highly
structured Rabat-Meknes-Oujda network, or three isolated teams that never
talk to each other. The country does not distinguish these two
situations; the city does.

Hence the central indicator of this module, the **collaboration scale**,
which refines the usual binary split (national / international) into three
levels:

  - **local**: all authors in the same city. Corridor collaboration, the
    kind that costs nothing.
  - **national**: several cities, one country. It requires a real
    organisational effort, and stays invisible in an SCP/MCP count, which
    files it with local collaboration.
  - **international**: several countries.

The national level is precisely the one that usual bibliometrics loses.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .impact import g_index, h_index, m_index
from .production import institution_rows


def _city_frame(corpus) -> pd.DataFrame:
    """Unique (document, city) pairs, with country, citations and year."""
    aff = corpus.affiliations
    cols = ["eid", "city", "country"]
    if aff.empty or "city" not in aff.columns:
        return pd.DataFrame(columns=cols + ["citations", "year"])

    a = aff[cols].copy()
    a["city"] = a["city"].map(str).str.strip()
    # "None" comes from a missing column converted to a string: without this
    # filter, a corpus without cities would produce a city named "None".
    a = a[(a["city"] != "") & (~a["city"].str.lower().isin({"nan", "none"}))]
    if a.empty:
        return pd.DataFrame(columns=cols + ["citations", "year"])

    a = a.drop_duplicates(subset=["eid", "city"])
    docs = corpus.documents[["eid"]].copy()
    docs["citations"] = pd.to_numeric(corpus.documents["cited_by"],
                                      errors="coerce").fillna(0).astype(int)
    docs["year"] = pd.to_numeric(corpus.documents["year"], errors="coerce")
    a = a.merge(docs, on="eid", how="left")
    a["citations"] = a["citations"].fillna(0).astype(int)
    return a


def top_cities(corpus, n: Optional[int] = 20) -> pd.DataFrame:
    """Cities by number of documents.

    Columns: ``city``, ``country``, ``documents``, ``citations``,
    ``institutions``, ``citations_per_document``.

    A document co-signed by two cities counts once for each: the sum
    therefore deliberately exceeds the number of documents, as for countries.
    """
    cols = ["city", "country", "documents", "citations", "institutions",
            "citations_per_document"]
    a = _city_frame(corpus)
    if a.empty:
        return pd.DataFrame(columns=cols)

    # Distinct institutions per city: that is what distinguishes a university
    # hub from an isolated laboratory with the same production.
    # Parent 1 AND parent 2: a double affiliation names two institutions.
    inst = pd.DataFrame(columns=["city", "institutions"])
    rows = institution_rows(corpus.affiliations)
    if not rows.empty and "city" in rows.columns:
        i = rows[["city", "institution"]].dropna()
        i["city"] = i["city"].map(str).str.strip()
        i = i[i["city"] != ""]
        inst = (i.drop_duplicates().groupby("city")["institution"].nunique()
                 .rename("institutions").reset_index())

    out = (a.groupby("city")
             .agg(documents=("eid", "nunique"),
                  citations=("citations", "sum"),
                  country=("country", lambda s: s.mode().iat[0]
                           if not s.dropna().empty else None))
             .reset_index())
    out = out.merge(inst, on="city", how="left")
    out["institutions"] = out["institutions"].fillna(0).astype(int)
    out["citations_per_document"] = (out["citations"] / out["documents"]).round(2)
    out = out.sort_values(["documents", "citations"],
                          ascending=False, kind="stable").reset_index(drop=True)
    return (out[cols].head(n) if n else out[cols])


def cities_impact(corpus, n: Optional[int] = 20,
                  min_documents: int = 1) -> pd.DataFrame:
    """h, g and m per city."""
    cols = ["city", "country", "documents", "citations", "h_index", "g_index",
            "m_index", "first_year", "last_year"]
    a = _city_frame(corpus)
    if a.empty:
        return pd.DataFrame(columns=cols)

    corpus_last = a["year"].max()
    rows = []
    for city, g in a.groupby("city", sort=False):
        cites = g["citations"].tolist()
        years = g["year"].dropna()
        first = int(years.min()) if not years.empty else None
        h = h_index(cites)
        countries = g["country"].dropna()
        rows.append({
            "city": city,
            "country": countries.mode().iat[0] if not countries.empty else None,
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
    return (out[cols].head(n) if n else out[cols])


def collaboration_scale(corpus) -> pd.DataFrame:
    """Documents distributed by the **geographic scope** of the collaboration.

    Columns: ``scale``, ``documents``, ``share``, ``citations``,
    ``citations_per_document``.

    The four rows are always present, at zero if needed. ``single`` = a
    single identified affiliation: it is not local collaboration, and
    confusing the two would artificially inflate the latter.
    """
    cols = ["scale", "documents", "share", "citations", "citations_per_document"]
    order = [("single", "Single affiliation"), ("local", "Local (same city)"),
             ("national", "National (same country)"),
             ("international", "International")]

    a = _city_frame(corpus)
    if a.empty:
        return pd.DataFrame([{"scale": lab, "documents": 0, "share": 0.0,
                              "citations": 0, "citations_per_document": 0.0}
                             for _, lab in order], columns=cols)

    per_doc = a.groupby("eid").agg(cities=("city", "nunique"),
                                   countries=("country", "nunique"),
                                   citations=("citations", "first"))

    # The number of AFFILIATIONS is counted on the original table, never on
    # `a`: that one is deduplicated per city, so two laboratories of the same
    # city form a single row there. Counting on it made local collaboration
    # structurally impossible to detect.
    raw = corpus.affiliations
    key = "parent1" if "parent1" in raw.columns else "raw"
    affs = (raw[["eid", key]].dropna().drop_duplicates()
               .groupby("eid").size().rename("affiliations"))
    per_doc = per_doc.join(affs, how="left")
    per_doc["affiliations"] = per_doc["affiliations"].fillna(1).astype(int)

    def classify(r) -> str:
        if r["countries"] > 1:
            return "international"
        if r["cities"] > 1:
            return "national"
        # A single city: local collaboration only if at least two distinct
        # affiliations appear in it.
        return "local" if r["affiliations"] > 1 else "single"

    per_doc["scale"] = per_doc.apply(classify, axis=1)
    total = len(per_doc)

    rows = []
    for key, label in order:
        sub = per_doc[per_doc["scale"] == key]
        count = int(len(sub))
        cites = int(sub["citations"].sum())
        rows.append({
            "scale": label,
            "documents": count,
            "share": round(100.0 * count / total, 1) if total else 0.0,
            "citations": cites,
            "citations_per_document": round(cites / count, 2) if count else 0.0,
        })
    return pd.DataFrame(rows, columns=cols)


def cities_over_time(corpus, n: int = 8) -> pd.DataFrame:
    """Yearly and cumulative production of the top `n` cities.

    Every year of the range is present, at zero if needed: without it the
    cumulative count would be wrong.
    """
    import numpy as np

    cols = ["year", "city", "documents", "cumulative"]
    a = _city_frame(corpus).dropna(subset=["year"])
    if a.empty:
        return pd.DataFrame(columns=cols)

    top = (a.groupby("city")["eid"].nunique()
             .sort_values(ascending=False, kind="stable").head(n).index)
    a = a[a["city"].isin(top)]

    counts = (a.groupby(["city", "year"])["eid"].nunique()
                .rename("documents").reset_index())
    years = np.arange(int(a["year"].min()), int(a["year"].max()) + 1)
    grid = pd.MultiIndex.from_product([top, years], names=["city", "year"])
    counts = (counts.set_index(["city", "year"]).reindex(grid, fill_value=0)
                    .reset_index())
    counts["cumulative"] = counts.groupby("city")["documents"].cumsum()
    counts["year"] = counts["year"].astype(int)

    rank = {c: i for i, c in enumerate(top)}
    counts = counts.sort_values(
        ["city", "year"], key=lambda s: s.map(rank) if s.name == "city" else s, kind="stable")
    return counts[cols].reset_index(drop=True)


def _institutions_by_affiliations(parents: pd.Series) -> list:
    """[(institution, affiliations)], the most frequent first; on a tie, in
    alphabetical order, so that the same corpus always gives the same order
    (and the same leader)."""
    names = parents.dropna().map(str).str.strip()
    names = names[names != ""]
    counts = names.value_counts()
    return sorted(((name, int(c)) for name, c in counts.items()),
                  key=lambda item: (-item[1], item[0]))


def city_hierarchy(corpus, n: Optional[int] = 40) -> pd.DataFrame:
    """Country -> city -> institutions.

    Columns: ``country``, ``city``, ``institutions``, ``documents``,
    ``top_institution``, ``share_of_country``, ``institution_affiliations``.

    ``share_of_country`` says whether a city carries most of its country's
    production or is only one part of it: the same reading as the
    institution -> units hierarchy, transposed to geography.

    ``institution_affiliations`` lists ALL the institutions of the city with
    their number of affiliations, from the most to the least frequent:
    "Mohammed V University (26); National School of Mineral Industry (12)".
    The leader alone hid the others: a school with 12 documents disappeared
    behind the university of its city. ``top_institution`` is its first
    element.
    """
    cols = ["country", "city", "institutions", "documents", "top_institution",
            "share_of_country", "institution_affiliations"]
    aff = corpus.affiliations
    if aff.empty or "city" not in aff.columns:
        return pd.DataFrame(columns=cols)

    a = aff.copy()
    a["city"] = a["city"].map(str).str.strip()
    a = a[(a["city"] != "") & (~a["city"].str.lower().isin({"nan", "none"}))]
    if a.empty:
        return pd.DataFrame(columns=cols)

    # Total per country computed over ALL its cities, before any truncation:
    # otherwise the share would relate to an incomplete denominator.
    by_country = a.drop_duplicates(["eid", "country"]).groupby("country")["eid"].nunique()

    # The institutions of a city: parent 1 AND parent 2 of its affiliations.
    institutions = {key: g["institution"] for key, g in
                    institution_rows(a).groupby(["country", "city"], sort=False)}
    empty = pd.Series(dtype=object)

    rows = []
    for (country, city), g in a.groupby(["country", "city"], sort=False):
        docs = int(g["eid"].nunique())
        ranked = _institutions_by_affiliations(
            institutions.get((country, city), empty))
        rows.append({
            "country": country,
            "city": city,
            "institutions": len(ranked),
            "documents": docs,
            "top_institution": ranked[0][0] if ranked else None,
            "share_of_country": round(100.0 * docs / by_country.get(country, docs), 1),
            "institution_affiliations": "; ".join(
                f"{name} ({count})" for name, count in ranked),
        })

    # The city breaks ties: without it, the cities kept under the `n` cut-off
    # depended on the order of the rows in the file.
    out = pd.DataFrame(rows).sort_values(
        ["country", "documents", "city"],
        ascending=[True, False, True], kind="stable").reset_index(drop=True)
    return (out[cols].head(n) if n else out[cols])
