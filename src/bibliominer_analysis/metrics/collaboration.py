"""Indicateurs de collaboration et de productivité.

Formules, telles qu'établies par leurs auteurs :

  - **Degré de collaboration C** (Subramanyam, 1983)
        ``C = Nm / (Nm + Ns)``
    part des documents co-signés. Entre 0 (tout en solo) et 1 (aucun solo).

  - **Indice de collaboration CI** (Lawani, 1980)
        ``CI = Σ(j · f_j) / N``
    nombre moyen d'auteurs par document, tous documents confondus.

  - **Coefficient de collaboration CC** (Ajiferuke, Burrell & Tague, 1988)
        ``CC = 1 − [Σ(f_j / j)] / N``
    corrige le défaut du CI : celui-ci croît sans borne avec les articles à
    cinquante signataires, alors que le CC reste entre 0 et 1.

  - **Coefficient modifié MCC** (Savanur & Srikanth, 2010)
        ``MCC = (A / (A − 1)) × [1 − Σ(f_j / j) / N]``
    où **A est le nombre total d'auteurs distincts de la collection**. Le CC
    ne peut pas atteindre 1 : si les A auteurs signent tous chaque article, il
    vaut 1 − 1/A. Le facteur A/(A − 1) fait valoir 1 à cette collaboration
    maximale. Sur un corpus réel A est grand, et MCC ≈ CC : c'est le
    comportement rapporté par les études qui l'appliquent.

  - **CAI — indice de co-autorat** (Garg & Padhi, 2001)
        ``CAI = [(N_ij / N_i0) / (N_0j / N_00)] × 100``
    compare, période par période, la part d'un type de signature à sa part
    d'ensemble. **100 = conforme à la moyenne**, au-dessus = sur-représenté.

  - **AAPP — productivité moyenne des auteurs**
        ``AAPP = nombre de documents / nombre d'auteurs distincts``

  - **Loi de Price** — la racine carrée des auteurs produit la moitié des
    signatures. On donne l'écart entre le théorique et l'observé.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _authors_per_doc(corpus) -> pd.Series:
    """Nombre d'auteurs DISTINCTS par document."""
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].astype(str).str.strip() != "")]
    if a.empty:
        return pd.Series(dtype=int)
    return a.drop_duplicates(subset=["eid", "name"]).groupby("eid").size()


def collaboration_indicators(corpus) -> Dict[str, Any]:
    """Tous les indicateurs de collaboration en un appel."""
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

    # f_j : nombre de documents ayant j auteurs.
    f = per_doc.value_counts().sort_index()
    j = f.index.to_numpy(dtype=float)
    fj = f.to_numpy(dtype=float)

    ci = float((j * fj).sum() / n)
    cc = float(1 - (fj / j).sum() / n)
    a_max = int(per_doc.max())
    n_authors = corpus.n_authors()
    # A = auteurs DISTINCTS de la collection, pas le maximum par document.
    # Défaut corrigé : avec le maximum, un corpus où chaque article a
    # exactement deux auteurs obtenait MCC = 1, « collaboration maximale »,
    # alors que chacun n'y signe qu'avec un seul partenaire.
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
        # Productivité moyenne : documents par auteur distinct.
        "aapp": round(n / n_authors, 4) if n_authors else None,
        "max_authors": a_max,
    }


def authorship_pattern(corpus) -> pd.DataFrame:
    """Répartition des documents par nombre de signataires."""
    per_doc = _authors_per_doc(corpus)
    empty = pd.DataFrame(columns=["authors", "documents", "share"])
    if per_doc.empty:
        return empty
    f = per_doc.value_counts().sort_index()
    df = pd.DataFrame({"authors": f.index.astype(int),
                       "documents": f.to_numpy(dtype=int)})
    df["share"] = (100 * df["documents"] / df["documents"].sum()).round(2)
    return df.reset_index(drop=True)


#: Catégories de signature du CAI, telles qu'employées dans la littérature.
_CAI_BINS = [("single", 1, 1), ("two", 2, 2), ("three", 3, 3),
             ("multi", 4, None)]


def cai(corpus, block_years: int = 5) -> pd.DataFrame:
    """Indice de co-autorat par période.

    `block_years` regroupe les années en blocs : sur un corpus court, une
    période de 5 ans donne des effectifs exploitables là où l'année par année
    produirait des indices tirés d'un ou deux documents.

    Lecture : **100 = conforme à la moyenne du corpus**. 150 signifie que la
    période produit 1,5 fois plus de documents de ce type qu'attendu.
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

    n00 = len(df)                                   # tous documents
    n0j = df["band"].value_counts()                 # par type, tous blocs

    rows = []
    for block, g in df.groupby("block"):
        ni0 = len(g)                                # documents du bloc
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
            # CAI indéfini si ce type de signature n'existe nulle part.
            row["cai_" + name] = (round(((nij / ni0) / (total_j / n00)) * 100, 1)
                                  if ni0 and total_j else None)
        rows.append(row)

    return pd.DataFrame(rows)


def price_law(corpus) -> Dict[str, Any]:
    """Loi de Price : √N auteurs devraient produire la moitié des signatures.

    On compare le théorique et l'observé. Un écart important signale un corpus
    plus (ou moins) concentré que ce que Price prédit — c'est l'intérêt de
    l'indicateur, pas le fait qu'il « tombe juste ».
    """
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].astype(str).str.strip() != "")]
    if a.empty:
        return {"authors": 0, "expected_core": 0, "observed_share": None,
                "half_reached_with": None}

    key = a["scopus_id"].fillna("name:" + a["name"].astype(str))
    per_author = (a.assign(key=key).drop_duplicates(subset=["key", "eid"])
                   .groupby("key").size().sort_values(ascending=False))

    n_authors = int(len(per_author))
    total = int(per_author.sum())
    core = int(round(math.sqrt(n_authors)))

    observed = int(per_author.head(core).sum()) if core else 0
    cum = per_author.cumsum()
    reached = int((cum < total / 2).sum() + 1) if total else 0

    return {
        "authors": n_authors,
        "signatures": total,
        # Nombre d'auteurs que la loi désigne comme « noyau ».
        "expected_core": core,
        # Part réellement produite par ce noyau (la loi prédit 50 %).
        "observed_share": round(100 * observed / total, 1) if total else None,
        # Nombre d'auteurs réellement nécessaires pour atteindre la moitié.
        "half_reached_with": reached,
    }


def authorship_groups(corpus) -> pd.DataFrame:
    """Répartition des documents par NOMBRE de signataires : 1, 2, 3, 4+.

    Colonnes : ``group``, ``min_authors``, ``documents``, ``share``,
    ``citations``, ``citations_per_document``.

    Le regroupement s'arrête à « 4 et plus » parce qu'au-delà les effectifs
    s'émiettent : distinguer 7 signataires de 8 n'apprend rien, alors que le
    passage de 1 à 2 auteurs est la frontière qui compte — celle de la
    collaboration.

    Les quatre lignes sont TOUJOURS présentes, à zéro si besoin. Une catégorie
    absente se lirait comme une donnée manquante, alors qu'un corpus sans aucun
    article à auteur unique est une information en soi.
    """
    labels = [(1, "1 author"), (2, "2 authors"), (3, "3 authors"), (4, "4+ authors")]
    cols = ["group", "min_authors", "documents", "share", "citations",
            "citations_per_document"]

    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].astype(str).str.strip() != "")]
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
    # Un document sans auteur identifié n'est ni « à auteur unique » ni
    # « collaboratif » : il est écarté plutôt que rangé arbitrairement.
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
