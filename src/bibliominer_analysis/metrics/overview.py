"""Fiche signalétique du corpus — l'équivalent du « Main Information ».

C'est le tableau qu'on met en tête d'un article bibliométrique : il décrit le
corpus avant toute analyse. Chaque valeur y est définie sans ambiguïté, parce
que la même étiquette recouvre des calculs différents selon les outils.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def main_information(corpus) -> Dict[str, Any]:
    """Indicateurs descriptifs du corpus.

    Quelques définitions, pour qu'on sache ce qu'on lit :

    - **taux de croissance annuel** : taux composé entre la première et la
      dernière année complète, ``(N_fin/N_début)^(1/années) − 1``. Il n'a de
      sens que sur au moins deux années.
    - **âge moyen des documents** : années écoulées depuis la publication,
      comptées par rapport à l'année la plus récente DU CORPUS — pas par
      rapport à aujourd'hui, sinon la valeur changerait tous les ans.
    - **auteurs par document** : moyenne des signataires, doublons retirés.
    - **documents à auteur unique** : un seul signataire.
    - **collaboration internationale** : part des documents signés par au
      moins deux pays.
    """
    docs = corpus.documents
    n_docs = int(len(docs))
    years = pd.to_numeric(docs["year"], errors="coerce").dropna()
    cites = pd.to_numeric(docs["cited_by"], errors="coerce").fillna(0).astype(int)

    y_min = int(years.min()) if not years.empty else None
    y_max = int(years.max()) if not years.empty else None

    # --- croissance annuelle composée -------------------------------------
    growth = None
    if y_min is not None and y_max is not None and y_max > y_min:
        per_year = years.astype(int).value_counts().sort_index()
        first, last = per_year.iloc[0], per_year.iloc[-1]
        span = y_max - y_min
        if first > 0 and span > 0:
            growth = round(((last / first) ** (1 / span) - 1) * 100, 2)

    # --- âge moyen ---------------------------------------------------------
    age = None
    if not years.empty and y_max is not None:
        age = round(float((y_max - years).mean()), 2)

    # --- auteurs -----------------------------------------------------------
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    per_doc = a.drop_duplicates(subset=["eid", "name"]).groupby("eid").size()
    single = int((per_doc == 1).sum())
    authors_per_doc = round(float(per_doc.mean()), 2) if not per_doc.empty else 0.0

    # Index de collaboration : auteurs par document, calculé UNIQUEMENT sur
    # les documents co-signés. Un corpus plein d'articles solos ferait
    # autrement chuter l'indicateur alors qu'il ne mesure pas ça.
    multi = per_doc[per_doc > 1]
    collab_index = round(float(multi.mean()), 2) if not multi.empty else 0.0

    # --- collaboration internationale --------------------------------------
    aff = corpus.affiliations
    aff = aff[aff["country"].notna() & (aff["country"].map(str).str.strip() != "")]
    countries_per_doc = aff.groupby("eid")["country"].nunique()
    intl = int((countries_per_doc > 1).sum())
    docs_with_country = int(len(countries_per_doc))
    intl_share = round(100 * intl / docs_with_country, 1) if docs_with_country else 0.0

    # --- mots-clés et références -------------------------------------------
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
        "collaboration_index": collab_index,
        "countries": int(aff["country"].nunique()) if not aff.empty else 0,
        "international_documents": intl,
        "international_share": intl_share,
    }


def most_cited_documents(corpus, n: int = 20) -> pd.DataFrame:
    """Documents les plus cités du corpus.

    ``citations_per_year`` rapporte les citations à l'ancienneté : sans cette
    colonne, un article de 2016 écrase systématiquement un article de 2024 qui
    peut être bien plus percutant.
    """
    d = corpus.documents.copy()
    d["citations"] = pd.to_numeric(d["cited_by"], errors="coerce").fillna(0).astype(int)
    years = pd.to_numeric(d["year"], errors="coerce")
    if years.notna().any():
        latest = int(years.max())
        # +1 an : un article paru l'année la plus récente a déjà « vécu » un an,
        # sinon on diviserait par zéro.
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
    """Références les plus citées PAR le corpus (impact local).

    À distinguer du nombre de citations mondiales : ici on compte combien de
    documents du corpus citent ce travail. C'est ce qui identifie les
    fondations du domaine tel que ce corpus le pratique.
    """
    refs = corpus.references
    empty = pd.DataFrame(columns=["reference", "ref_year", "ref_doi",
                                  "local_citations"])
    if refs.empty:
        return empty

    r = refs.copy()
    # Identité : DOI si présent, sinon titre normalisé — même règle que le
    # réseau de co-citation, pour que les deux vues concordent.
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
