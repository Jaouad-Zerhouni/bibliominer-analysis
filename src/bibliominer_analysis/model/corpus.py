"""`Corpus` — le point d'entrée unique du package.

Il porte les six tables tidy et expose les indicateurs. Toute la logique
bibliométrique passe par ici : l'application web, un notebook ou un script en
ligne de commande consomment exactement les mêmes fonctions.

    >>> from bibliominer_analysis import Corpus
    >>> c = Corpus.from_csv("Hosni-1107_cleaned.csv")
    >>> c.production_by_year()
    >>> c.top_institutions(n=20)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import pandas as pd

from ..io import reader as R
from ..io import schema as S
from ..metrics import access as acc
from ..metrics import aging as ag
from ..metrics import cities as cit
from ..metrics import concentration as conc
from ..metrics import clustering as clus
from ..metrics import countries as ctry
from ..metrics import evolution as evo
from ..metrics import factorial as fac
from ..metrics import impact as imp
from ..metrics import laws as lw
from ..metrics import local as loc
from ..metrics import quality as qual
from ..metrics import scimago as scim
from ..metrics import selfcitation as selfc
from ..metrics import sources as src
from ..metrics import spectroscopy as spec
from ..metrics import text as txt
from ..metrics import timeline as tl
from ..networks import analysis as netan
from ..metrics import collaboration as collab
from ..metrics import flows as fl
from ..metrics import growth as gr
from ..metrics import overview as ov
from ..metrics import themes as th
from ..metrics import production as prod
from ..networks import build as nets

PathLike = Union[str, Path]


class Corpus:
    """Un corpus chargé en mémoire, sous forme de six tables liées par `eid`."""

    __slots__ = ("documents", "authors", "affiliations",
                 "author_affiliations", "keywords", "references", "source_path")

    def __init__(self, tables: Dict[str, pd.DataFrame],
                 source_path: Optional[PathLike] = None):
        for name in R.TABLE_NAMES:
            setattr(self, name, tables[name])
        self.source_path = str(source_path) if source_path else None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_csv(cls, path: PathLike) -> "Corpus":
        """Lit un CSV nettoyé par Bibliominer (ou un export Scopus brut)."""
        df = R.read_csv(path)
        return cls(R.build_tables(df), source_path=path)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Corpus":
        return cls(R.build_tables(df))

    @classmethod
    def from_tables(cls, tables: Dict[str, pd.DataFrame]) -> "Corpus":
        """Reconstruit un corpus depuis des tables déjà calculées (Parquet).

        C'est ce que fait le backend : il ne re-parse jamais le CSV, il relit
        les six tables écrites à l'import."""
        return cls(tables)

    def tables(self) -> Dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in R.TABLE_NAMES}

    # -- description ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        y = self.year_range()
        span = "%s-%s" % y if y else "?"
        return "<Corpus %d documents, %s, %d auteurs, %d institutions>" % (
            len(self.documents), span, self.n_authors(), self.n_institutions(),
        )

    def n_institutions(self) -> int:
        """Organismes mères distincts — parent 1 ET parent 2 : un double
        rattachement nomme deux institutions."""
        from ..metrics.production import institution_rows
        rows = institution_rows(self.affiliations)
        return int(rows["institution"].nunique()) if not rows.empty else 0

    def n_authors(self) -> int:
        """Auteurs distincts — MÊME clé que `top_authors` : l'identifiant
        Scopus s'il existe, sinon le nom. Compter les noms d'un côté et les
        identifiants de l'autre donnait deux chiffres différents pour la même
        chose."""
        a = self.authors
        if a.empty:
            return 0
        key = a["scopus_id"].fillna("name:" + a["name"].map(str))
        return int(key.nunique())

    def year_range(self):
        years = pd.to_numeric(self.documents["year"], errors="coerce").dropna()
        if years.empty:
            return None
        return int(years.min()), int(years.max())

    def summary(self) -> Dict[str, Any]:
        """Chiffres clés — ce qu'affiche la page d'accueil de l'application."""
        y = self.year_range()
        return {
            "documents": len(self.documents),
            "year_min": y[0] if y else None,
            "year_max": y[1] if y else None,
            "authors": self.n_authors(),
            # Affiliations DISTINCTES telles qu'écrites : c'est le volume de
            # travail qu'a représenté la standardisation, à distinguer du
            # nombre d'institutions une fois celles-ci regroupées.
            "affiliations": int(self.affiliations["raw"].nunique(dropna=True)),
            "institutions": self.n_institutions(),
            "countries": int(self.affiliations["country"].nunique(dropna=True)),
            "sources": int(self.documents["source"].nunique(dropna=True)),
            "keywords": int(self.keywords.loc[
                self.keywords["kind"] == "author", "keyword"].nunique()),
            "references": int(len(self.references)),
            "citations": int(pd.to_numeric(
                self.documents["cited_by"], errors="coerce").fillna(0).sum()),
            "documents_without_references": int(
                len(set(self.documents["eid"]) - set(self.references["eid"]))),
        }

    # -- filtrage ------------------------------------------------------------

    def filter(self, years: Optional[Sequence[int]] = None,
               doc_types: Optional[Iterable[str]] = None,
               countries: Optional[Iterable[str]] = None,
               sources: Optional[Iterable[str]] = None) -> "Corpus":
        """Renvoie un NOUVEAU corpus restreint. L'original n'est pas modifié.

        C'est cette méthode que pilote la barre de filtres de l'interface :
        un filtre = un nouveau `Corpus`, et tous les indicateurs se recalculent
        sans qu'aucune page n'ait à connaître la logique de filtrage.

        `years` accepte un intervalle (min, max) ou une liste d'années.
        """
        docs = self.documents
        if years:
            yr = pd.to_numeric(docs["year"], errors="coerce")
            if len(years) == 2 and years[0] is not None and years[1] is not None \
                    and int(years[1]) - int(years[0]) != 1:
                docs = docs[yr.between(int(years[0]), int(years[1]))]
            else:
                docs = docs[yr.isin([int(y) for y in years])]
        if doc_types:
            docs = docs[docs["doc_type"].isin(list(doc_types))]
        if sources:
            docs = docs[docs["source"].isin(list(sources))]
        if countries:
            keep = set(self.affiliations.loc[
                self.affiliations["country"].isin(list(countries)), "eid"])
            docs = docs[docs["eid"].isin(keep)]

        eids = set(docs["eid"])
        tables = {"documents": docs.reset_index(drop=True)}
        for name in R.TABLE_NAMES[1:]:
            t = getattr(self, name)
            tables[name] = t[t["eid"].isin(eids)].reset_index(drop=True)
        return Corpus(tables, source_path=self.source_path)

    # -- indicateurs ---------------------------------------------------------
    # Chaque méthode délègue à `metrics/` et renvoie un DataFrame — jamais une
    # figure : l'interface décide comment le dessiner.

    def production_by_year(self) -> pd.DataFrame:
        return prod.by_year(self)

    def production_by_country(self, n: Optional[int] = None) -> pd.DataFrame:
        return prod.by_country(self, n)

    def top_sources(self, n: int = 20) -> pd.DataFrame:
        return prod.top_sources(self, n)

    def top_authors(self, n: int = 20) -> pd.DataFrame:
        return prod.top_authors(self, n)

    def top_institutions(self, n: int = 20,
                         level: str = "parent") -> pd.DataFrame:
        """Classement des organisations. `level` : « parent » ou « subparent »."""
        return prod.top_institutions(self, n, level)

    def institutions_over_time(self, n: int = 10,
                               level: str = "parent") -> pd.DataFrame:
        """Production annuelle des n premières organisations (format long)."""
        return prod.institutions_over_time(self, n, level=level)

    def institutions_by_country(self, n: int = 20,
                                level: str = "parent") -> pd.DataFrame:
        """Organisations groupées par pays."""
        return prod.institutions_by_country(self, n, level)

    def three_fields(self, left: str = "authors", middle: str = "keywords",
                     right: str = "sources", n: int = 10,
                     min_weight: int = 1) -> pd.DataFrame:
        """Liens d'un diagramme à trois champs (Sankey)."""
        return fl.three_fields(self, left, middle, right, n, min_weight)

    def org_hierarchy(self, n: int = 20, min_documents: int = 1) -> pd.DataFrame:
        """Hiérarchie organisme -> unités internes, avec la part de chacune."""
        return prod.org_hierarchy(self, n, min_documents)

    def top_keywords(self, n: int = 30, kind: str = "author") -> pd.DataFrame:
        return prod.top_keywords(self, n, kind)

    def document_types(self) -> pd.DataFrame:
        return prod.document_types(self)

    # -- impact --------------------------------------------------------------

    def impact(self) -> Dict[str, Any]:
        """Indices d'impact du corpus : h, g, i10, citations moyennes."""
        return imp.corpus_impact(self)

    def authors_impact(self, n: Optional[int] = 20,
                       min_documents: int = 1) -> pd.DataFrame:
        """Auteurs classés par h-index, avec g-index et premières signatures."""
        return imp.authors_impact(self, n, min_documents)

    def institutions_impact(self, n: Optional[int] = 20,
                            level: str = "parent") -> pd.DataFrame:
        """Indices h et g par organisation. `level` : « parent »/« subparent »."""
        return imp.institutions_impact(self, n, level)

    # -- lois bibliométriques ------------------------------------------------
    # Chacune renvoie l'OBSERVÉ et le THÉORIQUE : un exposant sans la courbe
    # observée ne permet pas de juger si la loi s'applique au corpus.

    def lotka(self) -> Dict[str, Any]:
        """Productivité des auteurs : combien d'auteurs ont publié x fois."""
        return lw.lotka(self)

    def bradford(self, zones: int = 3) -> Dict[str, Any]:
        """Dispersion des sources en zones de Bradford (noyau du domaine)."""
        return lw.bradford(self, zones)

    def zipf(self, n: Optional[int] = 100, kind: str = "author") -> Dict[str, Any]:
        """Fréquence des mots-clés en fonction de leur rang."""
        return lw.zipf(self, n, kind)

    # -- réseaux -------------------------------------------------------------
    # Renvoient {"nodes": [...], "edges": [...]} : la structure, pas le dessin.

    def co_citation(self, top_n: int = 50, min_weight: int = 2) -> Dict[str, Any]:
        """Fondements intellectuels : références citées ensemble."""
        return nets.co_citation(self, top_n, min_weight)

    def co_word(self, top_n: int = 50, min_weight: int = 2,
                kind: str = "author") -> Dict[str, Any]:
        """Structure thématique : mots-clés apparaissant ensemble."""
        return nets.co_word(self, top_n, min_weight, kind)

    def co_citation_authors(self, top_n: int = 50,
                            min_weight: int = 2) -> Dict[str, Any]:
        """Co-citation d'auteurs : les écoles de pensée mobilisées."""
        return nets.co_citation_authors(self, top_n, min_weight)

    def co_authorship(self, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
        """Collaborations entre AUTEURS."""
        return nets.co_authorship(self, top_n, min_weight)

    def co_institution(self, top_n: int = 50, min_weight: int = 1,
                       level: str = "parent") -> Dict[str, Any]:
        """Collaborations entre ORGANISATIONS (parent ou unité interne)."""
        return nets.co_institution(self, top_n, min_weight, level)

    def co_country(self, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
        """Collaborations entre PAYS."""
        return nets.co_country(self, top_n, min_weight)

    def bibliographic_coupling(self, top_n: int = 50,
                               min_weight: int = 2) -> Dict[str, Any]:
        """Documents partageant des références : le miroir de la co-citation."""
        return nets.bibliographic_coupling(self, top_n, min_weight)

    # -- fiche signalétique et analyse thématique -----------------------------

    def main_information(self) -> Dict[str, Any]:
        """Fiche signalétique du corpus (croissance, collaboration, âge…)."""
        return ov.main_information(self)

    def most_cited_documents(self, n: int = 20) -> pd.DataFrame:
        """Documents les plus cités, avec les citations rapportées à l'âge."""
        return ov.most_cited_documents(self, n)

    def most_cited_references(self, n: int = 20) -> pd.DataFrame:
        """Références les plus citées PAR le corpus (impact local)."""
        return ov.most_cited_references(self, n)

    def trend_topics(self, n: int = 25, min_documents: int = 2,
                     kind: str = "author") -> pd.DataFrame:
        """Sujets et leur position dans le temps (médiane et quartiles)."""
        return th.trend_topics(self, n, min_documents, kind)

    def thematic_map(self, top_n: int = 100, min_weight: int = 2,
                     kind: str = "author") -> Dict[str, Any]:
        """Carte stratégique de Callon : centralité × densité."""
        return th.thematic_map(self, top_n, min_weight, kind=kind)

    # -- croissance ----------------------------------------------------------

    def cagr(self) -> Optional[float]:
        """Croissance annuelle composée (%)."""
        return gr.cagr(self)

    def agr(self) -> pd.DataFrame:
        """Croissance annuelle simple, année par année (%)."""
        return gr.agr(self)

    def rgr_doubling_time(self) -> pd.DataFrame:
        """Taux de croissance relatif et temps de doublement, par année."""
        return gr.rgr_doubling_time(self)

    def growth_summary(self) -> Dict[str, Any]:
        """CAGR, AGR moyen, RGR moyen, temps de doublement moyen."""
        return gr.growth_summary(self)

    def trend_forecast(self, horizon: int = 5,
                       model: str = "linear") -> Dict[str, Any]:
        """Tendance par moindres carrés et projection des années à venir."""
        return gr.trend_forecast(self, horizon, model)

    def cochran_sample_size(self, confidence: float = 0.95,
                            margin: float = 0.05,
                            proportion: float = 0.5) -> Dict[str, Any]:
        """Taille d'échantillon représentatif du corpus (Cochran)."""
        return gr.cochran_sample_size(self, confidence, margin, proportion)

    # -- collaboration -------------------------------------------------------

    def collaboration_indicators(self) -> Dict[str, Any]:
        """Degré de collaboration, CI, CC, MCC, AAPP."""
        return collab.collaboration_indicators(self)

    def affiliation_profile(self) -> Dict[str, Any]:
        """Combien d'articles à une seule affiliation, à deux, à davantage ;
        combien signés par un auteur seul ; combien d'auteurs à plusieurs
        rattachements."""
        from ..metrics import affiliation_profile as prof
        return prof.affiliation_profile(self)

    def documents_by_affiliation_count(self) -> pd.DataFrame:
        """Répartition des documents par nombre d'affiliations distinctes."""
        from ..metrics import affiliation_profile as prof
        return prof.documents_by_affiliation_count(self)

    def authors_by_affiliation_count(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Auteurs rattachés à plusieurs institutions, et ceux qui en portent
        plusieurs sur un MÊME article (double rattachement déclaré)."""
        from ..metrics import affiliation_profile as prof
        return prof.authors_by_affiliation_count(self, n)

    def authorship_pattern(self) -> pd.DataFrame:
        """Répartition des documents par nombre de signataires."""
        return collab.authorship_pattern(self)

    def cai(self, block_years: int = 5) -> pd.DataFrame:
        """Indice de co-autorat par période (100 = moyenne du corpus)."""
        return collab.cai(self, block_years)

    def authorship_groups(self) -> pd.DataFrame:
        """Documents par nombre de signataires : 1, 2, 3, 4+."""
        return collab.authorship_groups(self)

    def price_law(self) -> Dict[str, Any]:
        """Loi de Price : le noyau d'auteurs produit-il la moitié ?"""
        return collab.price_law(self)

    def country_map(self) -> pd.DataFrame:
        """Production par pays + taux de collaboration internationale."""
        return nets.country_map(self)

    # -- qualite des donnees --------------------------------------------------

    def indicator_readiness(self) -> pd.DataFrame:
        """Quelle analyse est fiable sur CE corpus, et sur quelle couverture."""
        return qual.indicator_readiness(self)

    def field_completeness(self) -> pd.DataFrame:
        """Remplissage de chaque champ, et ce qu'il conditionne."""
        return qual.field_completeness(self)

    def table_completeness(self) -> pd.DataFrame:
        """Remplissage des tables liees : auteurs, affiliations, references."""
        return qual.table_completeness(self)

    def duplicates(self) -> pd.DataFrame:
        """Doublons probables, par DOI puis par titre normalise."""
        return qual.duplicates(self)

    def anomalies(self) -> pd.DataFrame:
        """Incoherences qui faussent les indicateurs sans lever d'erreur."""
        return qual.anomalies(self)

    def quality_summary(self) -> Dict[str, Any]:
        """Combien d'analyses sont pleinement exploitables."""
        return qual.quality_summary(self)

    # -- profil du corpus -----------------------------------------------------

    def price_index(self, window: int = 5) -> Dict[str, Any]:
        """Indice de Price : part des references de moins de cinq ans."""
        return ag.price_index(self, window)

    def price_index_by_year(self, window: int = 5) -> pd.DataFrame:
        """Indice de Price annee par annee."""
        return ag.price_index_by_year(self, window)

    def reference_age_distribution(self, max_age: int = 40) -> pd.DataFrame:
        """Distribution des ages de reference, avec part cumulee."""
        return ag.reference_age_distribution(self, max_age)

    def concentration(self, unit: str = "authors") -> Dict[str, Any]:
        """Gini, CR4, CR10 et courbe de Lorenz pour une dimension."""
        return conc.concentration(self, unit)

    def concentration_summary(self) -> pd.DataFrame:
        """Concentration comparee de toutes les dimensions."""
        return conc.concentration_summary(self)

    def self_citation_summary(self) -> pd.DataFrame:
        """Taux d'auto-citation aux trois niveaux."""
        return selfc.self_citation_summary(self)

    def authors_self_citation(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Auto-citation par auteur."""
        return selfc.authors_self_citation(self, n)

    # -- acces ouvert ---------------------------------------------------------

    def access_status(self) -> pd.DataFrame:
        """Documents ouverts / non signales, d'apres le fichier importe."""
        return acc.access_status(self)

    def access_routes(self) -> pd.DataFrame:
        """Voies d'acces ouvert : or, hybride, bronze, vert."""
        return acc.access_routes(self)

    def access_over_time(self) -> pd.DataFrame:
        """Evolution annuelle de la part d'acces ouvert."""
        return acc.access_over_time(self)

    def access_summary(self) -> Dict[str, Any]:
        """Les deux sources d'acces ouvert, et ce qui les separe."""
        return acc.access_summary(self)

    # -- interdisciplinarite --------------------------------------------------

    def subject_areas(self) -> pd.DataFrame:
        """Domaines scientifiques du corpus, d'apres les revues."""
        return scim.subject_areas(self)

    def subject_categories(self, n: Optional[int] = 25) -> pd.DataFrame:
        """Categories fines, avec leur meilleur quartile."""
        return scim.subject_categories(self, n)

    def interdisciplinarity(self) -> Dict[str, Any]:
        """Diversite disciplinaire : Shannon, Simpson, regularite."""
        return scim.interdisciplinarity(self)

    def journal_open_access(self) -> pd.DataFrame:
        """Documents selon l'acces de leur REVUE (propriete du support)."""
        return scim.journal_open_access(self)

    # -- SCImago --------------------------------------------------------------

    def scimago_sources(self, path: Optional[Any] = None) -> pd.DataFrame:
        """Revues du corpus enrichies par SCImago (SJR, quartile, editeur...)."""
        return scim.enrich_sources(self, path)

    def quartile_distribution(self, path: Optional[Any] = None) -> pd.DataFrame:
        """Documents repartis par quartile SCImago de leur revue."""
        return scim.quartile_distribution(self, path)

    def quartile_over_time(self, path: Optional[Any] = None) -> pd.DataFrame:
        """Evolution annuelle du profil de publication par quartile."""
        return scim.quartile_over_time(self, path)

    def scimago_coverage(self, path: Optional[Any] = None) -> Dict[str, Any]:
        """Part du corpus reellement appariee au referentiel."""
        return scim.scimago_coverage(self, path)

    # -- villes ---------------------------------------------------------------

    def top_cities(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Villes par nombre de documents."""
        return cit.top_cities(self, n)

    def cities_impact(self, n: Optional[int] = 20,
                      min_documents: int = 1) -> pd.DataFrame:
        """Indices h, g, m par ville."""
        return cit.cities_impact(self, n, min_documents)

    def collaboration_scale(self) -> pd.DataFrame:
        """Portée géographique : locale, nationale, internationale."""
        return cit.collaboration_scale(self)

    def cities_over_time(self, n: int = 8) -> pd.DataFrame:
        """Production annuelle et cumulée des principales villes."""
        return cit.cities_over_time(self, n)

    def city_hierarchy(self, n: Optional[int] = 40) -> pd.DataFrame:
        """Pays → ville → institutions."""
        return cit.city_hierarchy(self, n)

    def co_city(self, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
        """Collaboration entre VILLES, chaque lien marqué national/international."""
        return nets.co_city(self, top_n, min_weight)

    def corresponding_author_countries(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Pays du premier auteur, avec SCP / MCP."""
        return ctry.corresponding_author_countries(self, n)

    # -- citations locales ----------------------------------------------------

    def citation_pairs(self) -> pd.DataFrame:
        """Liens de citation INTERNES au corpus (citant → cité)."""
        return loc.citation_pairs(self)

    def local_citations(self) -> pd.DataFrame:
        """Citations reçues depuis l'intérieur du corpus, par document."""
        return loc.local_citations(self)

    def most_local_cited_documents(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Documents les plus cités PAR LE CORPUS lui-même."""
        return loc.most_local_cited_documents(self, n)

    def most_local_cited_authors(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Auteurs les plus cités depuis l'intérieur du corpus."""
        return loc.most_local_cited_authors(self, n)

    def most_local_cited_sources(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Revues les plus citées depuis l'intérieur du corpus."""
        return loc.most_local_cited_sources(self, n)

    def historiograph(self, n: int = 25) -> Dict[str, Any]:
        """Historiographe de Garfield : la filiation des travaux clés."""
        return loc.historiograph(self, n)

    # -- revues ---------------------------------------------------------------

    def sources_impact(self, n: Optional[int] = 20,
                       min_documents: int = 1) -> pd.DataFrame:
        """Classement des revues : h, g, m, citations."""
        return src.sources_impact(self, n, min_documents)

    def sources_over_time(self, n: int = 8) -> pd.DataFrame:
        """Production annuelle et cumulée des principales revues."""
        return src.sources_over_time(self, n)

    # -- chronologies ---------------------------------------------------------

    def authors_over_time(self, n: int = 12) -> pd.DataFrame:
        """Production annuelle des principaux auteurs."""
        return tl.authors_over_time(self, n)

    def word_dynamics(self, n: int = 10, kind: str = "author") -> pd.DataFrame:
        """Occurrences cumulées des principaux termes, année par année."""
        return tl.word_dynamics(self, n, kind)

    def average_citations_per_year(self) -> pd.DataFrame:
        """Citations moyennes par article selon l'année de publication."""
        return tl.average_citations_per_year(self)

    def reference_spectroscopy(self) -> pd.DataFrame:
        """RPYS : années de référence et écart à la médiane glissante."""
        return spec.reference_spectroscopy(self)

    # -- structure conceptuelle ----------------------------------------------

    def conceptual_structure(self, method: str = "CA", kind: str = "author",
                             top_n: int = 50, min_documents: int = 2,
                             n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Carte factorielle des mots-clés (AFC ou MDS) avec regroupement."""
        return fac.conceptual_structure(self, method, kind, top_n,
                                        min_documents, n_clusters)

    def thematic_evolution(self, cuts: Optional[Sequence[int]] = None,
                           n_periods: int = 3, top_n: int = 60,
                           min_weight: int = 2,
                           kind: str = "author") -> Dict[str, Any]:
        """Flux thématiques entre périodes successives."""
        return evo.thematic_evolution(self, cuts, n_periods, top_n,
                                      min_weight, kind=kind)

    def clustering_by_coupling(self, top_n: int = 100, min_weight: int = 3,
                               impact: str = "local") -> Dict[str, Any]:
        """Groupes de documents couplés : centralité × impact."""
        return clus.clustering_by_coupling(self, top_n, min_weight,
                                           impact=impact)

    # -- mesures de réseau ----------------------------------------------------

    def network_metrics(self, graph: Dict[str, Any],
                        overlay_unit: Optional[str] = None,
                        level: str = "parent",
                        resolution: float = 1.0) -> Dict[str, Any]:
        """Centralités, communautés et superposition temporelle d'un réseau.

        `resolution` règle la granularité de Louvain : au-dessus de 1, des
        groupes plus nombreux et plus petits.
        """
        overlay = netan.overlay_years(self, overlay_unit, level) if overlay_unit else None
        return netan.annotate(graph, overlay=overlay, resolution=resolution)

    @staticmethod
    def network_summary(graph: Dict[str, Any]) -> Dict[str, Any]:
        """Densité, transitivité, composantes, modularité d'un réseau."""
        return netan.graph_summary(graph)

    @staticmethod
    def network_layout(graph: Dict[str, Any]) -> Dict[str, list]:
        """Coordonnées 2D déterministes des nœuds (MDS sur les plus courts chemins)."""
        return netan.layout(graph)

    @staticmethod
    def attach_layout(graph: Dict[str, Any]) -> Dict[str, Any]:
        """Le réseau avec ``x``/``y`` sur chaque nœud — les positions que
        l'interface dessine, et celles de `render_network`."""
        return netan.attach_layout(graph)

    @staticmethod
    def network_density(graph: Dict[str, Any], size: int = 48) -> Dict[str, Any]:
        """Carte de densité du réseau — la vue « density » de VOSviewer."""
        return netan.density_grid(graph, netan.layout(graph), size=size)

    def citation_network(self, unit: str = "sources", top_n: int = 40,
                         min_weight: int = 1, level: str = "parent") -> Dict[str, Any]:
        """Réseau de citation DIRECTE (orienté) agrégé à l'unité demandée."""
        return loc.citation_network(self, unit, top_n, min_weight, level)

    # -- fouille de texte -----------------------------------------------------

    def top_terms(self, field: str = "abstract", ngram: int = 1,
                  n: Optional[int] = 50, min_documents: int = 2,
                  extra_stopwords: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Termes les plus fréquents des titres ou des résumés."""
        return txt.top_terms(self, field, ngram, n, min_documents, extra_stopwords)

    def text_co_occurrence(self, field: str = "abstract", ngram: int = 2,
                           top_n: int = 50, min_weight: int = 2,
                           min_documents: int = 2,
                           extra_stopwords: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Réseau de co-occurrence des termes du texte."""
        return txt.text_co_occurrence(self, field, ngram, top_n, min_weight,
                                      min_documents, extra_stopwords)

    def text_trend(self, field: str = "abstract", ngram: int = 2, n: int = 20,
                   min_documents: int = 2,
                   extra_stopwords: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Position dans le temps des termes du texte."""
        return txt.text_trend(self, field, ngram, n, min_documents, extra_stopwords)

    def topic_dendrogram(self, kind: str = "author", top_n: int = 40,
                         min_documents: int = 2,
                         max_clusters: int = 6) -> Dict[str, Any]:
        """Arbre hiérarchique des termes, coupé en groupes."""
        return fac.topic_dendrogram(self, kind, top_n, min_documents, max_clusters)

    # -- impact complémentaire -----------------------------------------------

    def countries_impact(self, n: Optional[int] = 20,
                         min_documents: int = 1) -> pd.DataFrame:
        """Indices h, g, m par pays."""
        return ctry.countries_impact(self, n, min_documents)

    def normalized_citations(self) -> pd.DataFrame:
        """Citations rapportées à la moyenne de leur année de publication."""
        return imp.normalized_citations(self)
