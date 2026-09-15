"""Les **villes** : l'échelle géographique que le pays écrase.

Le nettoyage résout chaque affiliation jusqu'à la ville. Ne s'arrêter au pays,
c'est perdre l'essentiel : un corpus « marocain » à 90 % peut être un réseau
Rabat–Meknès–Oujda très structuré, ou trois équipes isolées qui ne se parlent
jamais. Le pays ne distingue pas ces deux situations ; la ville, si.

D'où l'indicateur central de ce module, **l'échelle de collaboration**, qui
raffine le partage binaire habituel (national / international) en trois
niveaux :

  - **locale** — tous les signataires dans la même ville. La collaboration de
    couloir, celle qui ne coûte rien.
  - **nationale** — plusieurs villes, un seul pays. Elle demande un effort réel
    d'organisation, et reste invisible dans un décompte SCP/MCP, qui la range
    avec la collaboration locale.
  - **internationale** — plusieurs pays.

Le niveau national est précisément celui que la bibliométrie usuelle perd.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .impact import g_index, h_index, m_index


def _city_frame(corpus) -> pd.DataFrame:
    """Couples (document, ville) uniques, avec pays, citations et année."""
    aff = corpus.affiliations
    cols = ["eid", "city", "country"]
    if aff.empty or "city" not in aff.columns:
        return pd.DataFrame(columns=cols + ["citations", "year"])

    a = aff[cols].copy()
    a["city"] = a["city"].astype(str).str.strip()
    # « None » vient d'une colonne absente convertie en chaîne : sans ce filtre,
    # un corpus sans ville produirait une ville nommée « None ».
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
    """Villes par nombre de documents.

    Colonnes : ``city``, ``country``, ``documents``, ``citations``,
    ``institutions``, ``citations_per_document``.

    Un document co-signé par deux villes compte une fois pour chacune : la
    somme dépasse donc volontairement le nombre de documents, comme pour les
    pays.
    """
    cols = ["city", "country", "documents", "citations", "institutions",
            "citations_per_document"]
    a = _city_frame(corpus)
    if a.empty:
        return pd.DataFrame(columns=cols)

    # Institutions distinctes par ville : c'est ce qui distingue un pôle
    # universitaire d'un laboratoire isolé à production égale.
    aff = corpus.affiliations
    inst = pd.DataFrame(columns=["city", "institutions"])
    if "parent1" in aff.columns:
        i = aff[["city", "parent1"]].dropna()
        i["city"] = i["city"].astype(str).str.strip()
        i = i[i["city"] != ""]
        inst = (i.drop_duplicates().groupby("city")["parent1"].nunique()
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
                          ascending=False).reset_index(drop=True)
    return (out[cols].head(n) if n else out[cols])


def cities_impact(corpus, n: Optional[int] = 20,
                  min_documents: int = 1) -> pd.DataFrame:
    """h, g et m par ville."""
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
                          ascending=False).reset_index(drop=True)
    return (out[cols].head(n) if n else out[cols])


def collaboration_scale(corpus) -> pd.DataFrame:
    """Documents répartis par **portée géographique** de la collaboration.

    Colonnes : ``scale``, ``documents``, ``share``, ``citations``,
    ``citations_per_document``.

    Les quatre lignes sont toujours présentes, à zéro si besoin. ``single`` =
    une seule affiliation identifiée : ce n'est pas de la collaboration locale,
    et les confondre gonflerait artificiellement cette dernière.
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

    # Le nombre d'AFFILIATIONS se compte sur la table d'origine, jamais sur
    # `a` : celle-ci est dédoublonnée par ville, donc deux laboratoires d'une
    # même ville n'y forment qu'une ligne. Compter là-dessus rendait la
    # collaboration locale structurellement impossible à détecter.
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
        # Une seule ville : collaboration locale seulement si au moins deux
        # affiliations distinctes y figurent.
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
    """Production annuelle et cumulée des `n` villes principales.

    Toutes les années de l'intervalle sont présentes, à zéro si besoin : sans
    cela le cumul serait faux.
    """
    import numpy as np

    cols = ["year", "city", "documents", "cumulative"]
    a = _city_frame(corpus).dropna(subset=["year"])
    if a.empty:
        return pd.DataFrame(columns=cols)

    top = (a.groupby("city")["eid"].nunique()
             .sort_values(ascending=False).head(n).index)
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
        ["city", "year"], key=lambda s: s.map(rank) if s.name == "city" else s)
    return counts[cols].reset_index(drop=True)


def _institutions_by_affiliations(parents: pd.Series) -> list:
    """[(institution, affiliations)], la plus fréquente d'abord ; à égalité,
    par ordre alphabétique, pour qu'un même corpus donne toujours le même
    ordre (et le même leader)."""
    names = parents.dropna().astype(str).str.strip()
    names = names[names != ""]
    counts = names.value_counts()
    return sorted(((name, int(c)) for name, c in counts.items()),
                  key=lambda item: (-item[1], item[0]))


def city_hierarchy(corpus, n: Optional[int] = 40) -> pd.DataFrame:
    """Pays → ville → institutions.

    Colonnes : ``country``, ``city``, ``institutions``, ``documents``,
    ``top_institution``, ``share_of_country``, ``institution_affiliations``.

    ``share_of_country`` dit si une ville porte l'essentiel de la production de
    son pays ou n'en est qu'une composante — la même lecture que la hiérarchie
    établissement → unités, transposée à la géographie.

    ``institution_affiliations`` liste TOUTES les institutions de la ville avec
    leur nombre d'affiliations, de la plus fréquente à la moins fréquente :
    « Mohammed V University (26); National School of Mineral Industry (12) ».
    Le seul leader cachait les autres — une école de 12 documents disparaissait
    derrière l'université de sa ville. ``top_institution`` en est le premier
    élément.
    """
    cols = ["country", "city", "institutions", "documents", "top_institution",
            "share_of_country", "institution_affiliations"]
    aff = corpus.affiliations
    if aff.empty or "city" not in aff.columns:
        return pd.DataFrame(columns=cols)

    a = aff[["eid", "city", "country", "parent1"]].copy()
    a["city"] = a["city"].astype(str).str.strip()
    a = a[(a["city"] != "") & (~a["city"].str.lower().isin({"nan", "none"}))]
    if a.empty:
        return pd.DataFrame(columns=cols)

    # Total par pays calculé sur TOUTES ses villes, avant toute troncature :
    # sinon la part serait rapportée à un dénominateur incomplet.
    by_country = a.drop_duplicates(["eid", "country"]).groupby("country")["eid"].nunique()

    rows = []
    for (country, city), g in a.groupby(["country", "city"], sort=False):
        docs = int(g["eid"].nunique())
        ranked = _institutions_by_affiliations(g["parent1"])
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

    out = pd.DataFrame(rows).sort_values(
        ["country", "documents"], ascending=[True, False]).reset_index(drop=True)
    return (out[cols].head(n) if n else out[cols])
