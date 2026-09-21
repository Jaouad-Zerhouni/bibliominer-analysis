"""Qualité des données : ce que le corpus permet vraiment de calculer.

Toutes les analyses de ce paquet reposent sur des champs qui peuvent manquer.
Sans références, pas de citations locales ni d'historiographe. Sans ISSN, pas
de quartile. Sans résumé, pas de fouille de texte. Ces limites existent déjà,
page par page — ce module les rassemble **avant** l'analyse, pour qu'on sache
quels chiffres on a le droit de citer.

Un piège précis motive tout ce fichier. Une valeur manquante convertie en
chaîne donne ``"nan"`` ou ``"None"`` selon qu'elle vient de numpy ou de Python.
Un test naïf ``valeur != ""`` les compte donc comme remplies, et un tableau de
complétude annonce 100 % partout. Le contrôle est faux d'une façon
particulièrement traîtresse : il rassure.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

import pandas as pd

#: Toutes les écritures d'une valeur absente, quelle que soit son origine.
_MISSING = frozenset(["", "nan", "none", "na", "null", "<na>", "nat", "-"])

#: Seuils de lecture. Ce sont des repères de prudence, pas des vérités : un
#: indicateur « partiel » reste calculable, il demande seulement d'être cité
#: avec sa couverture.
READY, PARTIAL, LIMITED = "ready", "partial", "limited"
_READY_AT = 90.0
_PARTIAL_AT = 50.0


def is_filled(series: pd.Series) -> pd.Series:
    """Masque des valeurs réellement renseignées.

    Le seul test fiable : convertir en chaîne, réduire, et comparer à TOUTES
    les écritures d'une absence. `notna()` seul laisserait passer les chaînes
    vides ; `!= ""` seul laisserait passer « nan » et « None ».
    """
    if series is None or len(series) == 0:
        return pd.Series([], dtype=bool)
    text = series.map(str).str.strip().str.lower()
    return ~text.isin(_MISSING)


def _share(series: pd.Series) -> float:
    n = len(series)
    return round(100.0 * int(is_filled(series).sum()) / n, 1) if n else 0.0


def _status(share: float) -> str:
    if share >= _READY_AT:
        return READY
    if share >= _PARTIAL_AT:
        return PARTIAL
    return LIMITED


#: Champ → ce qu'il rend possible. Sans cette colonne, un taux de remplissage
#: n'est qu'une curiosité ; avec elle, c'est une décision.
_DOCUMENT_FIELDS = [
    ("title", "Title", "Document lists, local citation matching by title"),
    ("year", "Year", "Everything time-based: production, growth, trends"),
    ("source", "Source", "Sources, Bradford, SCImago"),
    ("doc_type", "Document type", "Type filter and breakdown"),
    ("cited_by", "Citations", "All impact indices: h, g, i10, e, m"),
    ("doi", "DOI", "Local citation matching, deduplication"),
    ("issn", "ISSN", "SCImago matching, hence quartiles and SJR"),
    ("abstract", "Abstract", "Text mining on abstracts"),
    ("open_access", "Open access", "Open access status and routes"),
    ("publisher", "Publisher", "Publisher breakdown"),
    ("language", "Language", "Language breakdown"),
]


def field_completeness(corpus) -> pd.DataFrame:
    """Remplissage de chaque champ, et ce qu'il conditionne.

    Colonnes : ``field``, ``label``, ``filled``, ``total``, ``share``,
    ``status``, ``unlocks``.
    """
    cols = ["field", "label", "filled", "total", "share", "status", "unlocks"]
    docs = corpus.documents
    if docs.empty:
        return pd.DataFrame(columns=cols)

    total = len(docs)
    rows = []
    for field, label, unlocks in _DOCUMENT_FIELDS:
        if field not in docs.columns:
            rows.append({"field": field, "label": label, "filled": 0,
                         "total": total, "share": 0.0, "status": LIMITED,
                         "unlocks": unlocks})
            continue
        filled = int(is_filled(docs[field]).sum())
        share = round(100.0 * filled / total, 1) if total else 0.0
        rows.append({"field": field, "label": label, "filled": filled,
                     "total": total, "share": share, "status": _status(share),
                     "unlocks": unlocks})
    return pd.DataFrame(rows, columns=cols)


def table_completeness(corpus) -> pd.DataFrame:
    """Remplissage des tables liées : auteurs, affiliations, mots-clés, références.

    Colonnes : ``table``, ``field``, ``filled``, ``total``, ``share``, ``status``.
    """
    cols = ["table", "field", "filled", "total", "share", "status"]
    checks = [
        ("authors", "name"), ("authors", "scopus_id"),
        ("affiliations", "parent1"), ("affiliations", "subparent"),
        ("affiliations", "city"), ("affiliations", "country"),
        ("keywords", "keyword"),
        ("references", "ref_doi"), ("references", "ref_year"),
        ("references", "ref_title"),
    ]
    rows = []
    for table_name, field in checks:
        table = getattr(corpus, table_name, None)
        if table is None or table.empty or field not in table.columns:
            rows.append({"table": table_name, "field": field, "filled": 0,
                         "total": 0 if table is None else len(table),
                         "share": 0.0, "status": LIMITED})
            continue
        filled = int(is_filled(table[field]).sum())
        share = round(100.0 * filled / len(table), 1)
        rows.append({"table": table_name, "field": field, "filled": filled,
                     "total": len(table), "share": share,
                     "status": _status(share)})
    return pd.DataFrame(rows, columns=cols)


def _norm_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    s = unicodedata.normalize("NFKD", value)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def duplicates(corpus) -> pd.DataFrame:
    """Doublons probables, par DOI puis par titre normalisé.

    Colonnes : ``kind``, ``key``, ``documents``, ``titles``.

    Un doublon n'est pas forcément une erreur — un article peut exister en
    version conférence et en version revue. Mais il compte deux fois dans tous
    les indicateurs, et mieux vaut le savoir.
    """
    cols = ["kind", "key", "documents", "titles"]
    docs = corpus.documents
    if docs.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    if "doi" in docs.columns:
        d = docs[is_filled(docs["doi"])].copy()
        d["key"] = d["doi"].map(str).str.strip().str.lower()
        for key, g in d.groupby("key"):
            if len(g) > 1:
                rows.append({"kind": "DOI", "key": key, "documents": len(g),
                             "titles": " | ".join(str(t)[:60] for t in g["title"])})

    if "title" in docs.columns:
        d = docs.copy()
        d["key"] = d["title"].map(_norm_title)
        d = d[d["key"].str.len() >= 25]
        seen = {r["key"] for r in rows}
        for key, g in d.groupby("key"):
            if len(g) > 1 and key not in seen:
                rows.append({"kind": "Title", "key": key[:80], "documents": len(g),
                             "titles": " | ".join(str(t)[:60] for t in g["title"])})

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(
        "documents", ascending=False, kind="stable").reset_index(drop=True)


def anomalies(corpus) -> pd.DataFrame:
    """Incohérences qui faussent les indicateurs sans lever d'erreur.

    Colonnes : ``check``, ``documents``, ``share``, ``severity``, ``detail``.
    """
    cols = ["check", "documents", "share", "severity", "detail"]
    docs = corpus.documents
    if docs.empty:
        return pd.DataFrame(columns=cols)

    total = len(docs)
    years = pd.to_numeric(docs.get("year"), errors="coerce")
    cites = pd.to_numeric(docs.get("cited_by"), errors="coerce")
    last = int(years.max()) if years.notna().any() else None

    with_authors = set(corpus.authors["eid"]) if not corpus.authors.empty else set()
    with_aff = set(corpus.affiliations["eid"]) if not corpus.affiliations.empty else set()
    with_refs = set(corpus.references["eid"]) if not corpus.references.empty else set()
    with_kw = set(corpus.keywords["eid"]) if not corpus.keywords.empty else set()
    eids = set(docs["eid"])

    def row(check, n, severity, detail):
        return {"check": check, "documents": int(n),
                "share": round(100.0 * n / total, 1) if total else 0.0,
                "severity": severity, "detail": detail}

    rows = [
        row("Missing year", int(years.isna().sum()), "high",
            "Excluded from every time-based indicator."),
        row("Year in the future", int((years > (last or 0) + 1).sum()) if last else 0,
            "low", "Usually an « in press » record; harmless but it stretches the axis."),
        row("Negative or unreadable citations", int((cites < 0).sum()), "high",
            "Impact indices would be wrong."),
        row("No author", len(eids - with_authors), "high",
            "Absent from author rankings and from co-authorship."),
        row("No affiliation", len(eids - with_aff), "medium",
            "Absent from institutions, cities and countries."),
        row("No reference", len(eids - with_refs), "medium",
            "Cannot cite locally, nor take part in coupling."),
        row("No keyword", len(eids - with_kw), "medium",
            "Absent from co-word, thematic map and evolution."),
    ]
    return pd.DataFrame(rows, columns=cols)


def indicator_readiness(corpus) -> pd.DataFrame:
    """**Le tableau qui compte** : quelle analyse est fiable sur CE corpus.

    Colonnes : ``analysis``, ``status``, ``coverage``, ``basis``, ``note``.

    Chaque ligne relie une famille d'analyses au champ dont elle dépend, avec
    son taux de couverture réel. C'est ce qui permet de savoir, avant d'écrire
    une phrase, si le chiffre qu'on s'apprête à citer porte sur tout le corpus
    ou sur les deux tiers.
    """
    cols = ["analysis", "status", "coverage", "basis", "note"]
    docs = corpus.documents
    if docs.empty:
        return pd.DataFrame(columns=cols)

    def share_of(table, field) -> float:
        t = getattr(corpus, table, None)
        if t is None or t.empty or field not in t.columns:
            return 0.0
        return round(100.0 * int(is_filled(t[field]).sum()) / len(t), 1)

    rows: List[Dict[str, Any]] = []

    def add(analysis, coverage, basis, note):
        rows.append({"analysis": analysis, "status": _status(coverage),
                     "coverage": coverage, "basis": basis, "note": note})

    add("Production and growth", _share(docs.get("year", pd.Series(dtype=object))),
        "Publication year",
        "Documents without a year are excluded from every time series.")

    add("Impact indices (h, g, i10, e, m)",
        _share(docs.get("cited_by", pd.Series(dtype=object))), "Citation count",
        "Computed on THIS corpus, not on each person's whole body of work.")

    add("Author analysis", share_of("authors", "name"), "Author names",
        "Scopus identifier present on %.0f %% of rows — homonyms are only "
        "separable where it is." % share_of("authors", "scopus_id"))

    add("Institutions", share_of("affiliations", "parent1"), "Parent organisation",
        "Internal units filled on %.0f %% of affiliations."
        % share_of("affiliations", "subparent"))

    add("Cities and geography", share_of("affiliations", "city"), "City",
        "Drives the local / national / international collaboration scale.")

    add("Keywords and thematic map", share_of("keywords", "keyword"), "Keywords",
        "Co-word, thematic map and evolution all rest on this field.")

    add("Text mining", _share(docs.get("abstract", pd.Series(dtype=object))),
        "Abstract", "Terms are extracted from titles and abstracts.")

    # Les citations locales dépendent du DOI des références, et surtout du
    # nombre de rapprochements RÉELLEMENT trouvés — le seul chiffre honnête.
    ref_doi = share_of("references", "ref_doi")
    try:
        from .local import citation_pairs
        pairs = len(citation_pairs(corpus))
    except Exception:
        pairs = 0
    add("Local citations and historiograph", ref_doi, "Reference DOIs",
        "%d internal citation links actually matched." % pairs)

    try:
        from .scimago import scimago_coverage
        cov = scimago_coverage(corpus)
        add("SCImago quartiles and SJR", cov["document_share"], "ISSN matching",
            "%d of %d sources matched." % (cov["matched_sources"], cov["sources"]))
    except Exception:
        add("SCImago quartiles and SJR", 0.0, "ISSN matching",
            "Reference file unavailable.")

    add("Open access", _share(docs.get("open_access", pd.Series(dtype=object))),
        "Open access flag",
        "Scopus only flags OPEN articles; a blank is not a closed article.")

    return pd.DataFrame(rows, columns=cols)


def quality_summary(corpus) -> Dict[str, Any]:
    """Vue d'ensemble : combien d'analyses sont pleinement exploitables."""
    readiness = indicator_readiness(corpus)
    dup = duplicates(corpus)
    anom = anomalies(corpus)
    if readiness.empty:
        return {"documents": 0, "analyses": 0, "ready": 0, "partial": 0,
                "limited": 0, "duplicate_groups": 0, "high_severity_issues": 0,
                "mean_coverage": 0.0}

    counts = readiness["status"].value_counts()
    high = anom[(anom["severity"] == "high") & (anom["documents"] > 0)]
    return {
        "documents": int(len(corpus.documents)),
        "analyses": int(len(readiness)),
        "ready": int(counts.get(READY, 0)),
        "partial": int(counts.get(PARTIAL, 0)),
        "limited": int(counts.get(LIMITED, 0)),
        "duplicate_groups": int(len(dup)),
        "high_severity_issues": int(len(high)),
        "mean_coverage": round(float(readiness["coverage"].mean()), 1),
    }
