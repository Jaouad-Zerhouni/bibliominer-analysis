"""Enrichissement des revues par **SCImago** (SJR, quartile, et le reste).

Le corpus dit combien un article a été cité. Il ne dit rien du **support** :
publier dans une revue Q1 et dans une revue Q4 ne se compare pas, et cette
information n'existe nulle part dans un export Scopus. SCImago la fournit, avec
bien plus que le seul quartile.

L'appariement se fait en deux temps, du plus sûr au moins sûr :

  1. **par ISSN**, un identifiant, donc sans ambiguïté. SCImago en liste
     souvent plusieurs par revue (imprimé et électronique) : on les indexe tous.
  2. **par titre normalisé**, quand l'ISSN manque ou ne correspond à rien.
     Moins sûr, donc la méthode retenue est TOUJOURS renvoyée dans la colonne
     ``matched_by`` : un lecteur doit pouvoir écarter les rapprochements
     faibles lui-même.

Ce qui n'est pas trouvé reste **vide**, jamais deviné. Une revue absente de
SCImago (actes de conférence non indexés, revue trop récente) n'a pas de
quartile, écrire « Q4 » par défaut serait une invention.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

#: Emplacements fouillés dans l'ordre. Le premier est celui livré avec le
#: package ; les suivants permettent à une application de fournir sa propre
#: version sans réinstaller.
_SEARCH_PATHS = (
    Path(__file__).resolve().parent.parent / "data_ref" / "scimagojr_2025.csv",
)

#: Colonnes conservées, avec leur nom de sortie. Tout ce que SCImago publie
#: n'est pas utile ici : on garde ce qui décrit la revue, pas ce qui décrit le
#: fichier.
_COLUMNS = {
    "Sourceid": "scimago_id",
    "Title": "scimago_title",
    "Type": "source_type",
    "Publisher": "publisher",
    "Country": "publisher_country",
    "Region": "publisher_region",
    "Coverage": "coverage",
    "Categories": "categories",
    "Areas": "areas",
}

#: Colonnes numériques, SCImago écrit les décimales à la VIRGULE.
_NUMERIC = {
    "SJR": "sjr",
    "H index": "source_h_index",
    "Total Docs. (2025)": "docs_year",
    "Total Docs. (3years)": "docs_3y",
    "Total Refs.": "total_refs",
    "Total Citations (3years)": "citations_3y",
    "Citable Docs. (3years)": "citable_docs_3y",
    "Citations / Doc. (2years)": "citations_per_doc_2y",
    "Ref. / Doc.": "refs_per_doc",
    "%Female": "female_share",
    "Overton": "overton",
}

QUARTILES = ("Q1", "Q2", "Q3", "Q4")

_cache: Optional[pd.DataFrame] = None
_index_cache: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def default_path() -> Optional[Path]:
    for p in _SEARCH_PATHS:
        if p.exists():
            return p
    return None


def normalize_issn(raw: Any) -> str:
    """ISSN réduit à ses huit caractères, mêmes règles que le cleaning."""
    if raw is None:
        return ""
    s = re.sub(r"[^0-9Xx]", "", str(raw)).upper()
    return s if len(s) == 8 else ""


def _split_issn_field(cell: Any) -> List[str]:
    """SCImago liste plusieurs ISSN par revue, séparés par des virgules."""
    if not isinstance(cell, str):
        return []
    out = []
    for part in cell.replace('"', "").split(","):
        n = normalize_issn(part)
        if n:
            out.append(n)
    return out


def normalize_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    s = unicodedata.normalize("NFKD", value)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _to_number(series: pd.Series) -> pd.Series:
    """Décimales à la virgule, séparateur de milliers absent."""
    return pd.to_numeric(
        series.map(str).str.replace(".", "", regex=False)
                          .str.replace(",", ".", regex=False)
                          .str.strip(),
        errors="coerce")


def load_scimago(path: Optional[Any] = None, force: bool = False) -> pd.DataFrame:
    """Charge le référentiel SCImago. Le résultat est mis en cache.

    Renvoie un DataFrame vide si le fichier est absent : l'absence du
    référentiel doit dégrader l'analyse, jamais l'interrompre.
    """
    global _cache, _index_cache
    if _cache is not None and not force and path is None:
        return _cache

    target = Path(path) if path else default_path()
    if target is None or not target.exists():
        empty = pd.DataFrame(columns=list(_COLUMNS.values()) + list(_NUMERIC.values())
                             + ["quartile", "issns"])
        if path is None:
            _cache, _index_cache = empty, None
        return empty

    df = pd.read_csv(target, sep=";", dtype=str, encoding="utf-8",
                     on_bad_lines="skip")
    # « Publisher » apparaît DEUX fois dans le fichier ; pandas suffixe la
    # seconde. On garde la première et on ignore le doublon.
    df = df.loc[:, ~df.columns.duplicated()]

    out = pd.DataFrame(index=df.index)
    for src, dst in _COLUMNS.items():
        out[dst] = df[src].map(str).str.strip().str.strip('"') if src in df.columns else None
    for src, dst in _NUMERIC.items():
        out[dst] = _to_number(df[src]) if src in df.columns else np.nan

    quartile = (df["SJR Best Quartile"].map(str).str.strip().str.upper()
                if "SJR Best Quartile" in df.columns else pd.Series("", index=df.index))
    out["quartile"] = quartile.where(quartile.isin(QUARTILES), None)
    out["issns"] = df["Issn"].map(_split_issn_field) if "Issn" in df.columns else [[]] * len(df)
    out["scimago_rank"] = _to_number(df["Rank"]) if "Rank" in df.columns else np.nan

    if "Open Access" in df.columns:
        out["open_access"] = df["Open Access"].map(str).str.strip().str.lower().eq("yes")
    if "Open Access Diamond" in df.columns:
        out["open_access_diamond"] = (df["Open Access Diamond"].map(str)
                                      .str.strip().str.lower().eq("yes"))

    if path is None:
        _cache, _index_cache = out, None
    return out


def _index(path: Optional[Any] = None) -> Dict[str, Any]:
    """Index ISSN → ligne et titre normalisé → ligne."""
    global _index_cache
    if _index_cache is not None and path is None:
        return _index_cache

    table = load_scimago(path)
    by_issn: Dict[str, int] = {}
    by_title: Dict[str, int] = {}
    if not table.empty:
        for pos, (issns, title) in enumerate(zip(table["issns"], table["scimago_title"])):
            for issn in issns or []:
                by_issn.setdefault(issn, pos)
            t = normalize_title(title)
            # Un titre court est trop ambigu pour servir de clé.
            if len(t) >= 8:
                by_title.setdefault(t, pos)

    idx = {"table": table, "by_issn": by_issn, "by_title": by_title}
    if path is None:
        _index_cache = idx
    return idx


# ---------------------------------------------------------------------------
# Appariement
# ---------------------------------------------------------------------------

_OUT_COLS = ["source", "documents", "citations", "matched_by", "quartile", "sjr",
             "scimago_rank", "source_h_index", "citations_per_doc_2y",
             "source_type", "publisher", "publisher_country", "publisher_region",
             "open_access", "coverage", "categories", "areas", "docs_3y",
             "citations_3y", "refs_per_doc", "female_share", "scimago_title"]


def enrich_sources(corpus, path: Optional[Any] = None) -> pd.DataFrame:
    """Chaque revue du corpus, enrichie de ce que SCImago en sait.

    Colonnes : celles du corpus (``source``, ``documents``, ``citations``),
    puis ``matched_by`` (« issn », « title » ou vide) et les mesures SCImago.

    Une revue non trouvée garde ses colonnes SCImago **vides**. C'est le cas
    normal pour les actes de conférence, que SCImago n'indexe que
    partiellement, et le dire vaut mieux que de le masquer.
    """
    docs = corpus.documents
    if docs.empty or "source" not in docs.columns:
        return pd.DataFrame(columns=_OUT_COLS)

    d = docs[["eid", "source"]].copy()
    d["source"] = d["source"].map(str).str.strip()
    d = d[(d["source"] != "") & (d["source"].str.lower() != "nan")]
    if d.empty:
        return pd.DataFrame(columns=_OUT_COLS)

    d["citations"] = pd.to_numeric(docs["cited_by"], errors="coerce").fillna(0).astype(int)
    d["issn"] = docs["issn"].map(normalize_issn) if "issn" in docs.columns else ""

    grouped = (d.groupby("source")
                 .agg(documents=("eid", "nunique"),
                      citations=("citations", "sum"),
                      issn=("issn", lambda s: next((x for x in s if x), "")))
                 .reset_index())

    idx = _index(path)
    table, by_issn, by_title = idx["table"], idx["by_issn"], idx["by_title"]

    positions: List[Optional[int]] = []
    methods: List[str] = []
    for issn, title in zip(grouped["issn"], grouped["source"]):
        pos = by_issn.get(issn) if issn else None
        method = "issn" if pos is not None else ""
        if pos is None:
            t = normalize_title(title)
            if len(t) >= 8:
                pos = by_title.get(t)
                method = "title" if pos is not None else ""
        positions.append(pos)
        methods.append(method)

    grouped["matched_by"] = methods
    extra = [c for c in _OUT_COLS if c not in ("source", "documents", "citations",
                                               "matched_by")]
    for col in extra:
        if col in table.columns:
            grouped[col] = [table.iloc[p][col] if p is not None else None
                            for p in positions]
        else:
            grouped[col] = None

    grouped = grouped.sort_values(["documents", "citations"],
                                  ascending=False, kind="stable").reset_index(drop=True)
    return grouped[_OUT_COLS]


def quartile_distribution(corpus, path: Optional[Any] = None) -> pd.DataFrame:
    """Documents répartis par quartile SCImago de leur revue.

    Colonnes : ``quartile``, ``sources``, ``documents``, ``share``,
    ``citations``, ``citations_per_document``.

    Les cinq lignes sont toujours présentes, ``Not indexed`` compris. C'est la
    ligne la plus importante du tableau : elle dit quelle part du corpus échappe
    au classement, et donc à quel point les quatre autres sont représentatives.
    """
    cols = ["quartile", "sources", "documents", "share", "citations",
            "citations_per_document"]
    enriched = enrich_sources(corpus, path)
    order = list(QUARTILES) + ["Not indexed"]
    if enriched.empty:
        return pd.DataFrame([{"quartile": q, "sources": 0, "documents": 0,
                              "share": 0.0, "citations": 0,
                              "citations_per_document": 0.0} for q in order],
                            columns=cols)

    e = enriched.copy()
    e["bucket"] = e["quartile"].where(e["quartile"].isin(QUARTILES), "Not indexed")
    total = int(e["documents"].sum())

    rows = []
    for q in order:
        sub = e[e["bucket"] == q]
        docs = int(sub["documents"].sum())
        cites = int(sub["citations"].sum())
        rows.append({
            "quartile": q,
            "sources": int(len(sub)),
            "documents": docs,
            "share": round(100.0 * docs / total, 1) if total else 0.0,
            "citations": cites,
            "citations_per_document": round(cites / docs, 2) if docs else 0.0,
        })
    return pd.DataFrame(rows, columns=cols)


def quartile_over_time(corpus, path: Optional[Any] = None) -> pd.DataFrame:
    """Évolution annuelle du profil de publication par quartile.

    Colonnes : ``year``, ``quartile``, ``documents``, ``share``.

    C'est la lecture qui montre une **montée en gamme** : un laboratoire qui
    passe de Q3 à Q1 en cinq ans, ou l'inverse. Un total par quartile ne le dit
    pas.

    **Réserve à énoncer avec la figure** : le quartile vient d'UNE édition de
    SCImago (celle du fichier chargé) et s'applique à toutes les années. Une
    revue Q1 aujourd'hui pouvait être Q2 lors de la publication. La courbe
    montre donc dans quelles revues, classées selon leur rang ACTUEL, le
    corpus a publié chaque année, pas le quartile qu'elles avaient à l'époque.
    """
    cols = ["year", "quartile", "documents", "share"]
    enriched = enrich_sources(corpus, path)
    if enriched.empty:
        return pd.DataFrame(columns=cols)

    bucket = dict(zip(enriched["source"],
                      enriched["quartile"].where(
                          enriched["quartile"].isin(QUARTILES), "Not indexed")))

    d = corpus.documents[["eid", "source"]].copy()
    d["year"] = pd.to_numeric(corpus.documents["year"], errors="coerce")
    d = d.dropna(subset=["year"])
    d["quartile"] = d["source"].map(str).str.strip().map(bucket).fillna("Not indexed")
    if d.empty:
        return pd.DataFrame(columns=cols)

    counts = (d.groupby(["year", "quartile"])["eid"].nunique()
                .rename("documents").reset_index())
    order = list(QUARTILES) + ["Not indexed"]
    years = sorted(counts["year"].unique())
    grid = pd.MultiIndex.from_product([years, order], names=["year", "quartile"])
    counts = (counts.set_index(["year", "quartile"]).reindex(grid, fill_value=0)
                    .reset_index())
    totals = counts.groupby("year")["documents"].transform("sum")
    counts["share"] = np.where(totals > 0,
                               (100.0 * counts["documents"] / totals).round(1), 0.0)
    counts["year"] = counts["year"].astype(int)
    return counts[cols].reset_index(drop=True)


def scimago_coverage(corpus, path: Optional[Any] = None) -> Dict[str, Any]:
    """Ce que l'appariement a réussi à faire, et ce qu'il a manqué.

    Un tableau enrichi sans ce compte se lit comme s'il couvrait tout le
    corpus. Il ne le couvre jamais entièrement.
    """
    enriched = enrich_sources(corpus, path)
    available = default_path() is not None or not load_scimago(path).empty
    if enriched.empty:
        return {"available": available, "sources": 0, "matched_sources": 0,
                "matched_by_issn": 0, "matched_by_title": 0,
                "documents": 0, "matched_documents": 0, "document_share": 0.0}

    total_docs = int(enriched["documents"].sum())
    matched = enriched[enriched["matched_by"] != ""]
    matched_docs = int(matched["documents"].sum())
    return {
        "available": available,
        "sources": int(len(enriched)),
        "matched_sources": int(len(matched)),
        "matched_by_issn": int((enriched["matched_by"] == "issn").sum()),
        "matched_by_title": int((enriched["matched_by"] == "title").sum()),
        "documents": total_docs,
        "matched_documents": matched_docs,
        "document_share": round(100.0 * matched_docs / total_docs, 1) if total_docs else 0.0,
    }


# ---------------------------------------------------------------------------
# Interdisciplinarite, possible seulement grace au referentiel
# ---------------------------------------------------------------------------

def _split_areas(cell: Any) -> List[str]:
    """« Business, Management and Accounting; Computer Science » -> deux domaines.

    Le point-virgule separe les domaines ; la virgule appartient au NOM du
    domaine. Decouper sur la virgule casserait « Biochemistry, Genetics and
    Molecular Biology » en trois faux domaines.
    """
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [p.strip() for p in cell.split(";") if p.strip()]


def _strip_quartile(label: str) -> str:
    """« Oncology (Q1) » -> « Oncology » : la categorie, sans son rang."""
    return re.sub(r"\s*\(Q[1-4]\)\s*$", "", label).strip()


def subject_areas(corpus, path: Optional[Any] = None) -> pd.DataFrame:
    """Domaines scientifiques du corpus, d'apres les revues.

    Colonnes : ``area``, ``sources``, ``documents``, ``share``, ``citations``.

    Un document dont la revue couvre trois domaines compte pour chacun : la
    somme depasse donc volontairement le nombre de documents. C'est cette
    multi-appartenance qui FAIT l'interdisciplinarite.
    """
    cols = ["area", "sources", "documents", "share", "citations"]
    enriched = enrich_sources(corpus, path)
    if enriched.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for r in enriched.itertuples():
        for area in _split_areas(getattr(r, "areas", None)):
            rows.append({"area": area, "source": r.source,
                         "documents": r.documents, "citations": r.citations})
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    total = int(enriched["documents"].sum())
    out = (df.groupby("area")
             .agg(sources=("source", "nunique"),
                  documents=("documents", "sum"),
                  citations=("citations", "sum"))
             .reset_index())
    out["share"] = (100.0 * out["documents"] / total).round(1) if total else 0.0
    return out.sort_values(["documents", "citations"],
                           ascending=False, kind="stable").reset_index(drop=True)[cols]


def subject_categories(corpus, n: Optional[int] = 25,
                       path: Optional[Any] = None) -> pd.DataFrame:
    """Categories fines, plus precises que les domaines, avec leur quartile.

    Colonnes : ``category``, ``sources``, ``documents``, ``best_quartile``.
    """
    cols = ["category", "sources", "documents", "best_quartile"]
    enriched = enrich_sources(corpus, path)
    if enriched.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for r in enriched.itertuples():
        for raw in _split_areas(getattr(r, "categories", None)):
            m = re.search(r"\((Q[1-4])\)\s*$", raw)
            rows.append({"category": _strip_quartile(raw), "source": r.source,
                         "documents": r.documents,
                         "quartile": m.group(1) if m else None})
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    out = (df.groupby("category")
             .agg(sources=("source", "nunique"),
                  documents=("documents", "sum"),
                  # Le MEILLEUR quartile atteint : « Q1 » se trie avant « Q4 »,
                  # d'ou le min sur la chaine.
                  best_quartile=("quartile",
                                 lambda x: x.dropna().min() if x.notna().any() else None))
             .reset_index())
    out = out.sort_values(["documents", "sources"],
                          ascending=False, kind="stable").reset_index(drop=True)
    return (out[cols].head(n) if n else out[cols])


def interdisciplinarity(corpus, path: Optional[Any] = None) -> Dict[str, Any]:
    """Diversite disciplinaire du corpus.

    Retour : ``areas``, ``shannon``, ``simpson``, ``evenness``, ``top_area``,
    ``top_area_share``, ``coverage``.

    - **Simpson** : probabilite que deux documents tires au hasard relevent de
      domaines differents. Directement interpretable.
    - **Shannon** : entropie de la distribution.
    - **Regularite** : Shannon rapporte a son maximum. C'est la mesure a
      comparer entre corpus, parce qu'elle ne depend pas du NOMBRE de domaines,
      l'entropie brute, elle, monte mecaniquement avec.
    """
    areas = subject_areas(corpus, path)
    base = {"areas": 0, "shannon": None, "simpson": None, "evenness": None,
            "top_area": None, "top_area_share": None, "coverage": 0.0}
    if areas.empty:
        return base

    counts = areas["documents"].to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        return base

    p = counts / total
    shannon = float(-(p * np.log(p)).sum())
    k = int(p.size)
    cov = scimago_coverage(corpus, path)
    return {
        "areas": k,
        "shannon": round(shannon, 3),
        "simpson": round(float(1.0 - (p ** 2).sum()), 3),
        "evenness": round(shannon / np.log(k), 3) if k > 1 else 0.0,
        "top_area": str(areas.iloc[0]["area"]),
        "top_area_share": float(areas.iloc[0]["share"]),
        # La diversite ne se lit QUE sur la part appariee : sans ce chiffre on
        # croirait qu'elle decrit tout le corpus.
        "coverage": cov["document_share"],
    }


def journal_open_access(corpus, path: Optional[Any] = None) -> pd.DataFrame:
    """Documents selon l'acces de leur REVUE (propriete du support).

    A distinguer de `access.access_status`, qui porte sur l'ARTICLE. Un article
    ouvert dans une revue sur abonnement existe, c'est l'hybride.

    « Inconnu » n'est pas « payant » : une revue absente du referentiel n'a pas
    de statut connu.
    """
    cols = ["access", "sources", "documents", "share", "citations",
            "citations_per_document"]
    enriched = enrich_sources(corpus, path)
    order = ["Open access", "Subscription", "Unknown"]
    if enriched.empty:
        return pd.DataFrame([{"access": a, "sources": 0, "documents": 0,
                              "share": 0.0, "citations": 0,
                              "citations_per_document": 0.0} for a in order],
                            columns=cols)

    def bucket(v) -> str:
        # `is True` echoue sur un booleen numpy : np.bool_(True) n'EST pas le
        # singleton True. Toutes les revues tombaient donc dans « Inconnu ».
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "Unknown"
        try:
            return "Open access" if bool(v) else "Subscription"
        except (TypeError, ValueError):
            return "Unknown"

    e = enriched.copy()
    e["access"] = e["open_access"].map(bucket)
    total = int(e["documents"].sum())

    rows = []
    for a in order:
        sub = e[e["access"] == a]
        docs = int(sub["documents"].sum())
        cites = int(sub["citations"].sum())
        rows.append({
            "access": a,
            "sources": int(len(sub)),
            "documents": docs,
            "share": round(100.0 * docs / total, 1) if total else 0.0,
            "citations": cites,
            "citations_per_document": round(cites / docs, 2) if docs else 0.0,
        })
    return pd.DataFrame(rows, columns=cols)
