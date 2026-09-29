"""Indicateurs de production et d'acteurs.

Chaque fonction prend un `Corpus` et renvoie un DataFrame prêt à être affiché
ou exporté — jamais une figure. Les colonnes sont nommées de façon stable :
l'interface s'appuie dessus.

Une règle vaut partout : on ne compte JAMAIS deux fois le même document. Une
institution citée par trois auteurs du même article compte pour un document,
pas trois — sinon les classements sont faux.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _citations(corpus) -> pd.Series:
    """eid -> nombre de citations, en entier, jamais NaN."""
    d = corpus.documents
    return pd.to_numeric(d["cited_by"], errors="coerce").fillna(0).astype(int)


def by_year(corpus) -> pd.DataFrame:
    """Documents et citations par année, sans trou dans la série.

    Les années manquantes sont ajoutées à zéro : une courbe de production avec
    des années absentes se lit de travers.
    """
    d = corpus.documents.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["citations"] = _citations(corpus)
    d = d.dropna(subset=["year"])
    if d.empty:
        return pd.DataFrame(columns=["year", "documents", "citations",
                                     "citations_per_doc", "cumulative"])

    g = (d.groupby(d["year"].astype(int))
           .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
           .reset_index())

    full = pd.DataFrame({"year": range(int(g["year"].min()),
                                       int(g["year"].max()) + 1)})
    g = full.merge(g, on="year", how="left").fillna({"documents": 0, "citations": 0})
    g["documents"] = g["documents"].astype(int)
    g["citations"] = g["citations"].astype(int)
    # Une année SANS document (ajoutée à zéro ci-dessus) n'a pas de moyenne :
    # vide, pas zéro. `where` garde une colonne NUMÉRIQUE ; `replace(0, pd.NA)`
    # la rendait « object », et `round` échouait (erreur 500 sur Production
    # dès qu'une année manquait dans la série).
    per_doc = g["citations"] / g["documents"].where(g["documents"] > 0)
    g["citations_per_doc"] = per_doc.astype(float).round(2)
    g["cumulative"] = g["documents"].cumsum()
    return g


def by_country(corpus, n: Optional[int] = None) -> pd.DataFrame:
    """Documents par pays d'affiliation.

    Un document dont deux auteurs sont marocains compte UNE fois pour le
    Maroc. Un document maroco-espagnol compte une fois pour chaque pays : la
    somme dépasse donc volontairement le nombre de documents.
    """
    a = corpus.affiliations
    a = a[a["country"].notna() & (a["country"].map(str).str.strip() != "")]
    if a.empty:
        return pd.DataFrame(columns=["country", "documents", "citations"])

    pairs = a[["eid", "country"]].drop_duplicates()
    cites = corpus.documents[["eid"]].copy()
    cites["citations"] = _citations(corpus)
    pairs = pairs.merge(cites, on="eid", how="left")

    g = (pairs.groupby("country")
              .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
              .reset_index()
              .sort_values(["documents", "citations"], ascending=False, kind="stable")
              .reset_index(drop=True))
    return g.head(n) if n else g


#: Colonne de la table `affiliations` selon le niveau d'analyse demandé.
#:   « parent »    -> l'organisme mère (université, entreprise, hôpital)
#:   « subparent » -> l'unité interne (laboratoire, école, département)
#: Les deux niveaux répondent à des questions différentes : le premier situe
#: l'établissement, le second identifie l'équipe qui produit réellement.
_ORG_LEVELS = {"parent": "parent1", "subparent": "subparent"}


def org_column(level: str = "parent") -> str:
    """Colonne à analyser pour le niveau demandé."""
    try:
        return _ORG_LEVELS[level]
    except KeyError:
        raise ValueError("level must be 'parent' or 'subparent', not %r"
                         % level) from None


#: Les deux organismes mères qu'une affiliation peut citer.
PARENT_COLUMNS = ("parent1", "parent2")


def institution_rows(affiliations: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (affiliation, organisme mère), colonne ``institution``.

    Une affiliation à DOUBLE rattachement (« parent 1: Université A, parent 2:
    CNRS ») appartient aux deux organismes : elle compte pour chacun, en
    compte entier, comme un document co-signé compte pour chaque pays. Lire
    ``parent1`` seul faisait disparaître le second organisme de tous les
    classements. Un ``parent 2`` identique au ``parent 1`` ne compte qu'une
    fois.
    """
    frames = []
    for col in PARENT_COLUMNS:
        if col not in affiliations.columns:
            continue
        names = affiliations[col].astype("string").str.strip()
        kept = affiliations[names.notna() & (names != "")].copy()
        kept["institution"] = names[kept.index].map(str)
        frames.append(kept)
    if not frames:
        return affiliations.iloc[0:0].assign(institution=pd.Series(dtype=object))
    out = pd.concat(frames, ignore_index=True)
    key = [c for c in ("eid", "aff_pos") if c in out.columns] + ["institution"]
    return out.drop_duplicates(subset=key).reset_index(drop=True)


def _org_frame(corpus, level: str = "parent") -> pd.DataFrame:
    """Affiliations exploitables pour ce niveau, colonne renommée `org`.

    Au niveau « parent », une affiliation à double rattachement donne une
    ligne par organisme (voir `institution_rows`).

    Les chercheurs sans rattachement sont EXCLUS : le cleaning les marque
    « Independent researcher », ce n'est pas une organisation et sa présence
    en tête de classement n'aurait aucun sens.
    """
    from ..io.schema import INDEPENDENT_LABEL

    col = org_column(level)
    if level == "parent":
        a = institution_rows(corpus.affiliations)
        a = a[a["institution"] != INDEPENDENT_LABEL]
        if a.empty:
            return pd.DataFrame(columns=["eid", "org", "country", "parent1"])
        return pd.DataFrame({
            "eid": a["eid"].to_numpy(),
            "org": a["institution"].to_numpy(),
            "country": a["country"].to_numpy(),
            "parent1": a["institution"].to_numpy(),
        })

    a = corpus.affiliations
    a = a[a[col].notna() & (a[col].map(str).str.strip() != "")]
    a = a[a[col] != INDEPENDENT_LABEL]
    if a.empty:
        return pd.DataFrame(columns=["eid", "org", "country", "parent1"])

    # On construit colonne par colonne : au niveau « parent », `col` EST
    # « parent1 », et une sélection par liste la ferait apparaître deux fois —
    # pandas refuse ensuite de grouper sur une colonne dupliquée.
    out = pd.DataFrame({
        "eid": a["eid"].to_numpy(),
        "org": a[col].to_numpy(),
        "country": a["country"].to_numpy(),
        # `parent1` sert à rattacher une unité à son organisme ; au niveau
        # parent les deux colonnes coïncident, ce qui est correct.
        "parent1": a["parent1"].to_numpy(),
    })
    return out


def top_institutions(corpus, n: int = 20, level: str = "parent") -> pd.DataFrame:
    """Classement des organisations, au niveau demandé."""
    a = _org_frame(corpus, level)
    if a.empty:
        return pd.DataFrame(columns=["institution", "documents", "citations",
                                     "country"])

    pairs = a[["eid", "org", "country"]].drop_duplicates(subset=["eid", "org"])
    cites = corpus.documents[["eid"]].copy()
    cites["citations"] = _citations(corpus)
    pairs = pairs.merge(cites, on="eid", how="left")

    g = (pairs.groupby("org")
              .agg(documents=("eid", "nunique"),
                   citations=("citations", "sum"),
                   country=("country", lambda s: s.dropna().mode().iat[0]
                            if not s.dropna().empty else None))
              .reset_index()
              .rename(columns={"org": "institution"})
              .sort_values(["documents", "citations"], ascending=False, kind="stable")
              .reset_index(drop=True))
    return g.head(n)


def institutions_over_time(corpus, n: int = 10, cumulative: bool = True,
                           level: str = "parent") -> pd.DataFrame:
    """Production annuelle des `n` premières organisations, au niveau demandé.

    Format long : ``institution``, ``year``, ``documents``, ``cumulative``.
    Toutes les années du corpus figurent pour CHAQUE organisation, y compris à
    zéro — sinon les courbes seraient interrompues là où elle n'a rien publié,
    ce qui se lit comme une absence de donnée plutôt que comme une absence de
    production.
    """
    aff = _org_frame(corpus, level)
    empty = pd.DataFrame(columns=["institution", "year", "documents", "cumulative"])
    if aff.empty:
        return empty

    years = corpus.documents[["eid", "year"]].copy()
    years["year"] = pd.to_numeric(years["year"], errors="coerce")
    pairs = (aff[["eid", "org"]].drop_duplicates()
                .merge(years, on="eid", how="left")
                .dropna(subset=["year"]))
    if pairs.empty:
        return empty
    pairs["year"] = pairs["year"].astype(int)

    top = (pairs.groupby("org")["eid"].nunique()
                .sort_values(ascending=False, kind="stable").head(n).index)
    pairs = pairs[pairs["org"].isin(top)]

    counts = (pairs.groupby(["org", "year"])["eid"].nunique()
                   .reset_index(name="documents"))

    all_years = range(int(pairs["year"].min()), int(pairs["year"].max()) + 1)
    grid = pd.MultiIndex.from_product([list(top), list(all_years)],
                                      names=["org", "year"]).to_frame(index=False)
    out = grid.merge(counts, on=["org", "year"], how="left").fillna({"documents": 0})
    out["documents"] = out["documents"].astype(int)
    out["cumulative"] = out.groupby("org")["documents"].cumsum()
    if not cumulative:
        out = out.drop(columns=["cumulative"])
    return out.rename(columns={"org": "institution"}).reset_index(drop=True)


def institutions_by_country(corpus, n: int = 20,
                            level: str = "parent") -> pd.DataFrame:
    """Organisations groupées par pays : combien, et lesquelles dominent."""
    aff = _org_frame(corpus, level)
    aff = aff[aff["country"].notna()] if not aff.empty else aff
    empty = pd.DataFrame(columns=["country", "institutions", "documents",
                                  "top_institution"])
    if aff.empty:
        return empty

    pairs = aff[["eid", "org", "country"]].drop_duplicates()
    per_inst = pairs.groupby(["country", "org"])["eid"].nunique()

    rows = []
    for country, g in per_inst.groupby(level=0):
        g = g.droplevel(0).sort_values(ascending=False, kind="stable")
        rows.append({
            "country": country,
            "institutions": int(len(g)),
            "documents": int(pairs.loc[pairs["country"] == country, "eid"].nunique()),
            "top_institution": g.index[0],
        })
    return (pd.DataFrame(rows)
              .sort_values(["documents", "institutions"], ascending=False, kind="stable")
              .reset_index(drop=True).head(n))


def org_hierarchy(corpus, n: int = 20, min_documents: int = 1) -> pd.DataFrame:
    """Hiérarchie ORGANISME → UNITÉS, telle que le cleaning l'a établie.

    C'est le pendant analytique de la vue « organisations » du nettoyage :
    sous chaque organisme mère, les unités internes qui y sont rattachées,
    avec leur volume et leurs citations.

    Colonnes : ``parent``, ``subparent``, ``documents``, ``citations``,
    ``parent_documents``, ``share``.

    ``share`` est la part de l'unité DANS son organisme : elle dit si un
    laboratoire porte l'essentiel de la production de son université ou s'il
    n'en est qu'une composante parmi d'autres.

    Une unité à double rattachement figure sous SES DEUX organismes.
    """
    from ..io.schema import INDEPENDENT_LABEL

    base = institution_rows(corpus.affiliations)
    base = base[base["institution"] != INDEPENDENT_LABEL]
    # La suite lit « parent1 » : c'est ici l'organisme de CETTE ligne,
    # premier ou second rattachement.
    base = base.assign(parent1=base["institution"])

    # `a` ne garde que les lignes PORTANT une unité, mais le total de
    # l'organisme se calcule sur `base` : une université dont un document ne
    # mentionne aucune unité en a quand même un de plus, et l'ignorer
    # gonflerait artificiellement la part des unités.
    a = base[base["subparent"].notna()
             & (base["subparent"].map(str).str.strip() != "")]
    empty = pd.DataFrame(columns=["parent", "subparent", "documents",
                                  "citations", "parent_documents", "share"])
    if a.empty:
        return empty

    cites = corpus.documents[["eid"]].copy()
    cites["citations"] = _citations(corpus)

    pairs = a[["eid", "parent1", "subparent"]].drop_duplicates()
    pairs = pairs.merge(cites, on="eid", how="left")

    g = (pairs.groupby(["parent1", "subparent"])
              .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
              .reset_index()
              .rename(columns={"parent1": "parent"}))

    # Total de l'organisme : compté sur les documents DISTINCTS, sinon un
    # document citant deux unités du même organisme le compterait deux fois.
    parent_totals = (base[["eid", "parent1"]].drop_duplicates()
                      .groupby("parent1")["eid"].nunique()
                      .rename("parent_documents"))
    g = g.merge(parent_totals, left_on="parent", right_index=True, how="left")
    g["share"] = (100 * g["documents"] / g["parent_documents"]).round(1)

    g = g[g["documents"] >= min_documents]
    return (g.sort_values(["parent_documents", "parent", "documents"],
                          ascending=[False, True, False], kind="stable")
             .reset_index(drop=True).head(n))


def top_authors(corpus, n: int = 20) -> pd.DataFrame:
    """Classement des auteurs, avec le nombre de fois en PREMIÈRE position.

    `first_author` n'existe que parce que le cleaning indexe les auteurs :
    sans cette information, on ne peut pas distinguer un porteur de travaux
    d'un co-signataire.
    """
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    if a.empty:
        return pd.DataFrame(columns=["author", "scopus_id", "documents",
                                     "citations", "first_author"])

    cites = corpus.documents[["eid"]].copy()
    cites["citations"] = _citations(corpus)
    a = a.merge(cites, on="eid", how="left")
    a["is_first"] = pd.to_numeric(a["position"], errors="coerce").eq(1)

    # Clé de regroupement : l'identifiant Scopus s'il existe (fiable), sinon
    # le nom — deux homonymes sans identifiant restent indiscernables.
    a["key"] = a["scopus_id"].fillna("name:" + a["name"].map(str))

    g = (a.groupby("key")
           .agg(author=("name", lambda s: s.mode().iat[0] if not s.empty else None),
                scopus_id=("scopus_id", "first"),
                documents=("eid", "nunique"),
                citations=("citations", "sum"),
                first_author=("is_first", "sum"))
           .reset_index(drop=True)
           .sort_values(["documents", "citations"], ascending=False, kind="stable")
           .reset_index(drop=True))
    g["first_author"] = g["first_author"].astype(int)
    return g.head(n)


def top_sources(corpus, n: int = 20) -> pd.DataFrame:
    d = corpus.documents.copy()
    d["citations"] = _citations(corpus)
    d = d[d["source"].notna() & (d["source"].map(str).str.strip() != "")]
    if d.empty:
        return pd.DataFrame(columns=["source", "documents", "citations"])
    return (d.groupby("source")
             .agg(documents=("eid", "nunique"), citations=("citations", "sum"))
             .reset_index()
             .sort_values(["documents", "citations"], ascending=False, kind="stable")
             .reset_index(drop=True)
             .head(n))


def top_keywords(corpus, n: int = 30, kind: str = "author") -> pd.DataFrame:
    """Mots-clés les plus fréquents. `kind` : 'author', 'index' ou 'all'.

    La comparaison se fait sans tenir compte de la casse, mais la graphie
    affichée est la plus employée par les auteurs.
    """
    k = corpus.keywords
    if kind != "all":
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna()]
    if k.empty:
        return pd.DataFrame(columns=["keyword", "documents"])

    k = k.copy()
    k["norm"] = k["keyword"].map(str).str.strip().str.lower()
    g = (k.groupby("norm")
           .agg(keyword=("keyword", lambda s: s.mode().iat[0]),
                documents=("eid", "nunique"))
           .reset_index(drop=True)
           .sort_values("documents", ascending=False, kind="stable")
           .reset_index(drop=True))
    return g.head(n)


def document_types(corpus) -> pd.DataFrame:
    d = corpus.documents
    d = d[d["doc_type"].notna() & (d["doc_type"].map(str).str.strip() != "")]
    if d.empty:
        return pd.DataFrame(columns=["doc_type", "documents", "share"])
    g = (d.groupby("doc_type")
           .agg(documents=("eid", "nunique"))
           .reset_index()
           .sort_values("documents", ascending=False, kind="stable")
           .reset_index(drop=True))
    total = g["documents"].sum()
    g["share"] = (100 * g["documents"] / total).round(1) if total else 0.0
    return g
