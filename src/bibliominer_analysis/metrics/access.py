"""Accès ouvert : ce que dit l'export, ce que dit SCImago, et pourquoi c'est différent.

Deux sources d'information, deux questions distinctes — les confondre est
l'erreur courante :

  - **Le fichier importé** porte un statut **par article** (« Gold », « Green »,
    « Bronze », « Hybrid »). C'est la vérité sur CE document : est-il librement
    lisible, et par quelle voie.
  - **SCImago** porte un statut **par revue**. C'est une propriété du support :
    la revue est-elle entièrement ouverte.

Un article en accès ouvert dans une revue sur abonnement existe — c'est
précisément l'hybride, et le vert (dépôt en archive). L'inverse aussi :
une revue ouverte dont Scopus n'a pas renseigné le champ. Aucune des deux
sources ne remplace l'autre.

Les voies, dans la terminologie usuelle :

  - **Or** — publié ouvert dans une revue entièrement ouverte.
  - **Hybride** — ouvert par paiement dans une revue sur abonnement.
  - **Bronze** — lisible gratuitement sur le site de l'éditeur, sans licence
    explicite : l'accès peut être retiré à tout moment.
  - **Vert** — déposé par l'auteur dans une archive.

Un statut **absent n'est pas un statut fermé**. Scopus ne renseigne le champ
que pour une partie des documents ; compter les vides comme « sur abonnement »
ferait mécaniquement chuter la part d'accès ouvert.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import pandas as pd

#: Voies reconnues, dans l'ordre de lecture usuel.
ROUTES = ("Gold", "Hybrid", "Bronze", "Green")

_ROUTE_PATTERNS = {
    "Gold": re.compile(r"\bgold\b", re.I),
    "Hybrid": re.compile(r"\bhybrid\b", re.I),
    "Bronze": re.compile(r"\bbronze\b", re.I),
    "Green": re.compile(r"\bgreen\b", re.I),
}
_ANY_OA = re.compile(r"\bopen\s*access\b", re.I)


def routes_of(value: Any) -> list:
    """Voies d'accès ouvert déclarées pour un document.

    Un même article peut en cumuler plusieurs — « Gold » ET « Green » quand il
    est publié ouvert puis déposé en archive. On les garde toutes : n'en
    retenir qu'une perdrait l'information la plus intéressante, celle du dépôt.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    found = [name for name, pat in _ROUTE_PATTERNS.items() if pat.search(value)]
    if not found and _ANY_OA.search(value):
        # « All Open Access » sans voie précisée : ouvert, voie inconnue.
        return ["Unspecified"]
    return found


def _document_access(corpus) -> pd.DataFrame:
    """Un document, son statut brut, ses voies, ses citations."""
    docs = corpus.documents
    cols = ["eid", "raw", "is_open", "routes", "citations", "year"]
    if docs.empty:
        return pd.DataFrame(columns=cols)

    d = pd.DataFrame({"eid": docs["eid"].to_numpy()})
    raw = docs["open_access"] if "open_access" in docs.columns else pd.Series(
        [None] * len(docs))
    d["raw"] = raw.to_numpy()
    d["routes"] = [routes_of(v) for v in d["raw"]]
    # Scopus ne renseigne ce champ QUE pour les articles ouverts : il n'ecrit
    # jamais « fermé ». Un champ vide signifie donc « non signalé comme
    # ouvert », ce qui recouvre les articles fermés ET ceux dont le statut est
    # simplement absent. On ne peut pas les distinguer, et pretendre le
    # contraire fausserait la part d'acces ouvert.
    d["is_open"] = [bool(r) if r else None for r in d["routes"]]
    d["citations"] = pd.to_numeric(docs["cited_by"], errors="coerce").fillna(0).astype(int)
    d["year"] = pd.to_numeric(docs["year"], errors="coerce")
    return d[cols]


def access_status(corpus) -> pd.DataFrame:
    """Documents par statut d'accès, **d'après le fichier importé**.

    Colonnes : ``status``, ``documents``, ``share``, ``citations``,
    ``citations_per_document``.

    Deux lignes seulement, et c'est volontaire. Scopus ne signale que les
    articles OUVERTS ; il n'ecrit jamais « fermé ». « Non signalé » regroupe
    donc les articles fermés et ceux dont le statut manque — les separer
    supposerait une information qu'on n'a pas.
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
    """Voies d'accès ouvert déclarées.

    Colonnes : ``route``, ``documents``, ``share_of_open``, ``citations``,
    ``citations_per_document``.

    Un article cumulant deux voies compte pour chacune : la somme dépasse donc
    volontairement le nombre d'articles ouverts, et ``share_of_open`` se lit
    par rapport aux SEULS documents ouverts, pas au corpus entier.
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
    """Évolution annuelle de la part d'accès ouvert.

    Colonnes : ``year``, ``documents``, ``open_access``, ``share``.

    Le denominateur est l'ENSEMBLE des documents de l'annee, pas seulement
    ceux signales : comme Scopus n'ecrit jamais « ferme », se restreindre aux
    documents renseignes donnerait 100 % chaque annee — un chiffre exact et
    parfaitement inutile.
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
    return out.sort_values("year").reset_index(drop=True)[cols]


def access_summary(corpus, path: Optional[Any] = None) -> Dict[str, Any]:
    """Les deux sources côte à côte, avec ce qui les sépare.

    Retour : ``documents``, ``reported``, ``open_documents``,
    ``open_share_of_reported``, ``journal_open_documents``, ``disagreement``.

    ``disagreement`` compte les documents déclarés ouverts alors que leur revue
    ne l'est pas — les hybrides et les dépôts en archive. C'est le chiffre qui
    justifie de garder les deux sources plutôt qu'une seule.
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
    src["source"] = src["source"].astype(str).str.strip()
    in_open_journal = set(src.loc[src["source"].isin(journal_open), "eid"].astype(str))

    base["journal_open_documents"] = len(in_open_journal)
    base["disagreement"] = int(sum(
        1 for eid, is_open in zip(d["eid"].astype(str), d["is_open"])
        if bool(is_open) and eid not in in_open_journal))
    return base
