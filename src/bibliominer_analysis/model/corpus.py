"""`Corpus`, the single entry point of the package.

It carries the six tidy tables and exposes the indicators. All the
bibliometric logic goes through here: the web application, a notebook or
a command-line script use exactly the same functions.

    >>> from bibliominer_analysis import Corpus
    >>> c = Corpus.from_csv("corpus_cleaned.csv")
    >>> c.production_by_year()
    >>> c.top_institutions(n=20)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import pandas as pd

from ..io import reader as R
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
from .. import views as vw

PathLike = Union[str, Path]


class Corpus:
    """A corpus loaded in memory, as six tables linked by `eid`."""

    __slots__ = ("documents", "authors", "affiliations",
                 "author_affiliations", "keywords", "references", "source_path")

    def __init__(self, tables: Dict[str, pd.DataFrame],
                 source_path: Optional[PathLike] = None):
        # One spelling per author and per keyword BEFORE any analysis: see
        # `unify_spellings`. Here rather than when reading the CSV, so that corpora
        # already imported (read back from Parquet) benefit too.
        tables = R.unify_spellings(tables)
        for name in R.TABLE_NAMES:
            setattr(self, name, tables[name])
        self.source_path = str(source_path) if source_path else None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_csv(cls, path: PathLike) -> "Corpus":
        """Reads the CSV exported by the Bibliominer cleaning.

        A Scopus export that did not go through the cleaning is refused
        (`NotCleanedError`, see `reader.check_cleaned`): the analysis would draw
        wrong numbers from it without saying so.
        """
        df = R.read_csv(path)
        R.check_cleaned(df)
        return cls(R.build_tables(df), source_path=path)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Corpus":
        """A corpus from a table built in memory, WITHOUT the check of `from_csv`:
        it is the tool of tests and advanced uses, which assemble their columns
        themselves. For a file, use `from_csv`."""
        return cls(R.build_tables(df))

    @classmethod
    def from_tables(cls, tables: Dict[str, pd.DataFrame]) -> "Corpus":
        """Rebuilds a corpus from already computed tables (Parquet).

        That is what the backend does: it never parses the CSV again, it reads
        back the six tables written at import time."""
        return cls(tables)

    def tables(self) -> Dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in R.TABLE_NAMES}

    # -- description ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        y = self.year_range()
        span = "%s-%s" % y if y else "?"
        return "<Corpus %d documents, %s, %d authors, %d institutions>" % (
            len(self.documents), span, self.n_authors(), self.n_institutions(),
        )

    def n_institutions(self) -> int:
        """Distinct parent organisations, parent 1 AND parent 2: a double
        affiliation names two institutions."""
        from ..metrics.production import institution_rows
        rows = institution_rows(self.affiliations)
        return int(rows["institution"].nunique()) if not rows.empty else 0

    def n_authors(self) -> int:
        """Distinct authors, SAME key as `top_authors`: the Scopus identifier if it
        exists, otherwise the name. Counting names on one side and identifiers on
        the other gave two different figures for the same thing."""
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
        """Key figures, what the application's home page displays."""
        y = self.year_range()
        return {
            "documents": len(self.documents),
            "year_min": y[0] if y else None,
            "year_max": y[1] if y else None,
            "authors": self.n_authors(),
            # DISTINCT affiliations as written: this is the amount of work the
            # standardisation represented, to be distinguished from the number of
            # institutions once they are grouped.
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
        """Returns a NEW restricted corpus. The original is not modified.

        It is this method that the interface's filter bar drives: a filter = a
        new `Corpus`, and every indicator is recomputed without any page having
        to know the filtering logic.

        `years` accepts a range (min, max) or a list of years.
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

    # -- indicators ----------------------------------------------------------
    # Every method delegates to `metrics/` and returns a DataFrame, never a
    # figure: the interface decides how to draw it.

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
        """Ranking of organisations. `level`: "parent" or "subparent"."""
        return prod.top_institutions(self, n, level)

    def institutions_over_time(self, n: int = 10,
                               level: str = "parent") -> pd.DataFrame:
        """Yearly production of the top n organisations (long format)."""
        return prod.institutions_over_time(self, n, level=level)

    def institutions_by_country(self, n: int = 20,
                                level: str = "parent") -> pd.DataFrame:
        """Organisations grouped by country."""
        return prod.institutions_by_country(self, n, level)

    def three_fields(self, left: str = "authors", middle: str = "keywords",
                     right: str = "sources", n: int = 10,
                     min_weight: int = 1) -> pd.DataFrame:
        """Links of a three-field plot (Sankey)."""
        return fl.three_fields(self, left, middle, right, n, min_weight)

    def org_hierarchy(self, n: int = 20, min_documents: int = 1) -> pd.DataFrame:
        """Organisation -> internal units hierarchy, with the share of each."""
        return prod.org_hierarchy(self, n, min_documents)

    def top_keywords(self, n: int = 30, kind: str = "author") -> pd.DataFrame:
        return prod.top_keywords(self, n, kind)

    def document_types(self) -> pd.DataFrame:
        return prod.document_types(self)

    # -- impact --------------------------------------------------------------

    def impact(self) -> Dict[str, Any]:
        """Impact indices of the corpus: h, g, i10, mean citations."""
        return imp.corpus_impact(self)

    def authors_impact(self, n: Optional[int] = 20,
                       min_documents: int = 1) -> pd.DataFrame:
        """Authors ranked by h-index, with g-index and first authorships."""
        return imp.authors_impact(self, n, min_documents)

    def institutions_impact(self, n: Optional[int] = 20,
                            level: str = "parent") -> pd.DataFrame:
        """h and g indices per organisation. `level`: "parent"/"subparent"."""
        return imp.institutions_impact(self, n, level)

    # -- bibliometric laws ---------------------------------------------------
    # Each returns the OBSERVED and the THEORETICAL: an exponent without the
    # observed curve does not allow judging whether the law applies to the
    # corpus.

    def lotka(self) -> Dict[str, Any]:
        """Author productivity: how many authors published x times."""
        return lw.lotka(self)

    def bradford(self, zones: int = 3) -> Dict[str, Any]:
        """Scattering of sources into Bradford zones (core of the field)."""
        return lw.bradford(self, zones)

    def zipf(self, n: Optional[int] = 100, kind: str = "author") -> Dict[str, Any]:
        """Frequency of keywords as a function of their rank."""
        return lw.zipf(self, n, kind)

    # -- networks ------------------------------------------------------------
    # Return {"nodes": [...], "edges": [...]}: the structure, not the drawing.

    def co_citation(self, top_n: int = 50, min_weight: int = 2) -> Dict[str, Any]:
        """Intellectual foundations: references cited together."""
        return nets.co_citation(self, top_n, min_weight)

    def co_word(self, top_n: int = 50, min_weight: int = 2,
                kind: str = "author") -> Dict[str, Any]:
        """Thematic structure: keywords appearing together."""
        return nets.co_word(self, top_n, min_weight, kind)

    def co_citation_authors(self, top_n: int = 50,
                            min_weight: int = 2) -> Dict[str, Any]:
        """Author co-citation: the schools of thought drawn on."""
        return nets.co_citation_authors(self, top_n, min_weight)

    def co_authorship(self, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
        """Collaborations between AUTHORS."""
        return nets.co_authorship(self, top_n, min_weight)

    def co_institution(self, top_n: int = 50, min_weight: int = 1,
                       level: str = "parent") -> Dict[str, Any]:
        """Collaborations between ORGANISATIONS (parent or internal unit)."""
        return nets.co_institution(self, top_n, min_weight, level)

    def co_country(self, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
        """Collaborations between COUNTRIES."""
        return nets.co_country(self, top_n, min_weight)

    def bibliographic_coupling(self, top_n: int = 50,
                               min_weight: int = 2) -> Dict[str, Any]:
        """Documents sharing references: the mirror of co-citation."""
        return nets.bibliographic_coupling(self, top_n, min_weight)

    # -- main information and thematic analysis -------------------------------

    def main_information(self) -> Dict[str, Any]:
        """Main information about the corpus (growth, collaboration, age...)."""
        return ov.main_information(self)

    def most_cited_documents(self, n: int = 20) -> pd.DataFrame:
        """Most cited documents, with citations relative to age."""
        return ov.most_cited_documents(self, n)

    def most_cited_references(self, n: int = 20) -> pd.DataFrame:
        """References most cited BY the corpus (local impact)."""
        return ov.most_cited_references(self, n)

    def trend_topics(self, n: int = 25, min_documents: int = 2,
                     kind: str = "author") -> pd.DataFrame:
        """Topics and their position in time (median and quartiles)."""
        return th.trend_topics(self, n, min_documents, kind)

    def thematic_map(self, top_n: int = 100, min_weight: int = 2,
                     kind: str = "author") -> Dict[str, Any]:
        """Callon's strategic map: centrality × density."""
        return th.thematic_map(self, top_n, min_weight, kind=kind)

    # -- growth --------------------------------------------------------------

    def cagr(self) -> Optional[float]:
        """Compound annual growth (%)."""
        return gr.cagr(self)

    def agr(self) -> pd.DataFrame:
        """Simple annual growth, year by year (%)."""
        return gr.agr(self)

    def rgr_doubling_time(self) -> pd.DataFrame:
        """Relative growth rate and doubling time, per year."""
        return gr.rgr_doubling_time(self)

    def growth_summary(self) -> Dict[str, Any]:
        """CAGR, mean AGR, mean RGR, mean doubling time."""
        return gr.growth_summary(self)

    def trend_forecast(self, horizon: int = 5,
                       model: str = "linear") -> Dict[str, Any]:
        """Least-squares trend and projection of the coming years."""
        return gr.trend_forecast(self, horizon, model)

    def cochran_sample_size(self, confidence: float = 0.95,
                            margin: float = 0.05,
                            proportion: float = 0.5) -> Dict[str, Any]:
        """Representative sample size of the corpus (Cochran)."""
        return gr.cochran_sample_size(self, confidence, margin, proportion)

    # -- collaboration -------------------------------------------------------

    def collaboration_indicators(self) -> Dict[str, Any]:
        """Degree of collaboration, CI, CC, MCC, AAPP."""
        return collab.collaboration_indicators(self)

    def affiliation_profile(self) -> Dict[str, Any]:
        """How many articles with a single affiliation, with two, with more; how
        many signed by a single author; how many authors with several
        affiliations."""
        from ..metrics import affiliation_profile as prof
        return prof.affiliation_profile(self)

    def documents_by_affiliation_count(self) -> pd.DataFrame:
        """Distribution of documents by number of distinct affiliations."""
        from ..metrics import affiliation_profile as prof
        return prof.documents_by_affiliation_count(self)

    def authors_by_affiliation_count(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Authors affiliated with several institutions, and those carrying several
        on the SAME article (declared double affiliation)."""
        from ..metrics import affiliation_profile as prof
        return prof.authors_by_affiliation_count(self, n)

    def authorship_pattern(self) -> pd.DataFrame:
        """Distribution of documents by number of authors."""
        return collab.authorship_pattern(self)

    def cai(self, block_years: int = 5) -> pd.DataFrame:
        """Co-authorship index per period (100 = corpus average)."""
        return collab.cai(self, block_years)

    def authorship_groups(self) -> pd.DataFrame:
        """Documents by number of authors: 1, 2, 3, 4+."""
        return collab.authorship_groups(self)

    def price_law(self) -> Dict[str, Any]:
        """Price's law: does the core of authors produce half?"""
        return collab.price_law(self)

    def country_map(self) -> pd.DataFrame:
        """Production per country + international collaboration rate."""
        return nets.country_map(self)

    # -- data quality ---------------------------------------------------------

    def indicator_readiness(self) -> pd.DataFrame:
        """Which analysis is reliable on THIS corpus, and with what coverage."""
        return qual.indicator_readiness(self)

    def field_completeness(self) -> pd.DataFrame:
        """Fill rate of every field, and what it conditions."""
        return qual.field_completeness(self)

    def table_completeness(self) -> pd.DataFrame:
        """Fill rate of the linked tables: authors, affiliations, references."""
        return qual.table_completeness(self)

    def duplicates(self) -> pd.DataFrame:
        """Probable duplicates, by DOI then by normalised title."""
        return qual.duplicates(self)

    def anomalies(self) -> pd.DataFrame:
        """Inconsistencies that distort the indicators without raising an error."""
        return qual.anomalies(self)

    def quality_summary(self) -> Dict[str, Any]:
        """How many analyses are fully usable."""
        return qual.quality_summary(self)

    # -- corpus profile -------------------------------------------------------

    def price_index(self, window: int = 5) -> Dict[str, Any]:
        """Price index: share of references less than five years old."""
        return ag.price_index(self, window)

    def price_index_by_year(self, window: int = 5) -> pd.DataFrame:
        """Price index year by year."""
        return ag.price_index_by_year(self, window)

    def reference_age_distribution(self, max_age: int = 40) -> pd.DataFrame:
        """Distribution of reference ages, with cumulative share."""
        return ag.reference_age_distribution(self, max_age)

    def concentration(self, unit: str = "authors") -> Dict[str, Any]:
        """Gini, CR4, CR10 and Lorenz curve for one dimension."""
        return conc.concentration(self, unit)

    def concentration_summary(self) -> pd.DataFrame:
        """Compared concentration of all the dimensions."""
        return conc.concentration_summary(self)

    def self_citation_summary(self) -> pd.DataFrame:
        """Self-citation rate at the three levels."""
        return selfc.self_citation_summary(self)

    def authors_self_citation(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Self-citation per author."""
        return selfc.authors_self_citation(self, n)

    # -- acces ouvert ---------------------------------------------------------

    def access_status(self) -> pd.DataFrame:
        """Open / not reported documents, according to the imported file."""
        return acc.access_status(self)

    def access_routes(self) -> pd.DataFrame:
        """Open access routes: gold, hybrid, bronze, green."""
        return acc.access_routes(self)

    def access_over_time(self) -> pd.DataFrame:
        """Yearly evolution of the open access share."""
        return acc.access_over_time(self)

    def access_summary(self) -> Dict[str, Any]:
        """The two open access sources, and what separates them."""
        return acc.access_summary(self)

    # -- interdisciplinarite --------------------------------------------------

    def subject_areas(self) -> pd.DataFrame:
        """Subject areas of the corpus, according to the journals."""
        return scim.subject_areas(self)

    def subject_categories(self, n: Optional[int] = 25) -> pd.DataFrame:
        """Fine-grained categories, with their best quartile."""
        return scim.subject_categories(self, n)

    def interdisciplinarity(self) -> Dict[str, Any]:
        """Diversite disciplinaire : Shannon, Simpson, regularite."""
        return scim.interdisciplinarity(self)

    def journal_open_access(self) -> pd.DataFrame:
        """Documents by the access of their JOURNAL (a property of the venue)."""
        return scim.journal_open_access(self)

    # -- SCImago --------------------------------------------------------------

    def scimago_sources(self, path: Optional[Any] = None) -> pd.DataFrame:
        """Journals of the corpus enriched by SCImago (SJR, quartile, publisher...)."""
        return scim.enrich_sources(self, path)

    def quartile_distribution(self, path: Optional[Any] = None) -> pd.DataFrame:
        """Documents distributed by the SCImago quartile of their journal."""
        return scim.quartile_distribution(self, path)

    def quartile_over_time(self, path: Optional[Any] = None) -> pd.DataFrame:
        """Yearly evolution of the publication profile by quartile."""
        return scim.quartile_over_time(self, path)

    def scimago_coverage(self, path: Optional[Any] = None) -> Dict[str, Any]:
        """Share of the corpus actually matched to the reference table."""
        return scim.scimago_coverage(self, path)

    # -- cities ---------------------------------------------------------------

    def top_cities(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Cities by number of documents."""
        return cit.top_cities(self, n)

    def cities_impact(self, n: Optional[int] = 20,
                      min_documents: int = 1) -> pd.DataFrame:
        """h, g, m indices per city."""
        return cit.cities_impact(self, n, min_documents)

    def collaboration_scale(self) -> pd.DataFrame:
        """Geographic scope: local, national, international."""
        return cit.collaboration_scale(self)

    def cities_over_time(self, n: int = 8) -> pd.DataFrame:
        """Yearly and cumulative production of the main cities."""
        return cit.cities_over_time(self, n)

    def city_hierarchy(self, n: Optional[int] = 40) -> pd.DataFrame:
        """Country → city → institutions."""
        return cit.city_hierarchy(self, n)

    def co_city(self, top_n: int = 50, min_weight: int = 1) -> Dict[str, Any]:
        """Collaboration between CITIES, each link marked national/international."""
        return nets.co_city(self, top_n, min_weight)

    def first_author_countries(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Documents by country of the FIRST author, with SCP / MCP.

        Not the corresponding author: Scopus does not always export the
        correspondence address, and a count depending on it would change from one
        export to another. The first author is present everywhere."""
        return ctry.corresponding_author_countries(self, n)

    def corresponding_author_countries(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Old name of `first_author_countries`, kept for existing scripts: the
        computation is indeed about the first author."""
        return self.first_author_countries(n)

    # -- citations locales ----------------------------------------------------

    def citation_pairs(self) -> pd.DataFrame:
        """Citation links INTERNAL to the corpus (citing -> cited)."""
        return loc.citation_pairs(self)

    def local_citations(self) -> pd.DataFrame:
        """Citations received from within the corpus, per document."""
        return loc.local_citations(self)

    def most_local_cited_documents(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Documents most cited BY THE CORPUS itself."""
        return loc.most_local_cited_documents(self, n)

    def most_local_cited_authors(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Authors most cited from within the corpus."""
        return loc.most_local_cited_authors(self, n)

    def most_local_cited_sources(self, n: Optional[int] = 20) -> pd.DataFrame:
        """Journals most cited from within the corpus."""
        return loc.most_local_cited_sources(self, n)

    def historiograph(self, n: int = 25) -> Dict[str, Any]:
        """Garfield's historiograph: the lineage of the key works."""
        return loc.historiograph(self, n)

    # -- journals -------------------------------------------------------------

    def sources_impact(self, n: Optional[int] = 20,
                       min_documents: int = 1) -> pd.DataFrame:
        """Ranking of journals: h, g, m, citations."""
        return src.sources_impact(self, n, min_documents)

    def sources_over_time(self, n: int = 8) -> pd.DataFrame:
        """Yearly and cumulative production of the main journals."""
        return src.sources_over_time(self, n)

    # -- chronologies ---------------------------------------------------------

    def authors_over_time(self, n: int = 12) -> pd.DataFrame:
        """Yearly production of the main authors."""
        return tl.authors_over_time(self, n)

    def word_dynamics(self, n: int = 10, kind: str = "author") -> pd.DataFrame:
        """Cumulative occurrences of the main terms, year by year."""
        return tl.word_dynamics(self, n, kind)

    def average_citations_per_year(self) -> pd.DataFrame:
        """Mean citations per article by publication year."""
        return tl.average_citations_per_year(self)

    def reference_spectroscopy(self) -> pd.DataFrame:
        """RPYS: reference years and deviation from the rolling median."""
        return spec.reference_spectroscopy(self)

    # -- structure conceptuelle ----------------------------------------------

    def conceptual_structure(self, method: str = "CA", kind: str = "author",
                             top_n: int = 50, min_documents: int = 2,
                             n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Factorial map of the keywords (CA or MDS) with clustering."""
        return fac.conceptual_structure(self, method, kind, top_n,
                                        min_documents, n_clusters)

    def thematic_evolution(self, cuts: Optional[Sequence[int]] = None,
                           n_periods: int = 3, top_n: int = 60,
                           min_weight: int = 2,
                           kind: str = "author") -> Dict[str, Any]:
        """Thematic flows between successive periods."""
        return evo.thematic_evolution(self, cuts, n_periods, top_n,
                                      min_weight, kind=kind)

    def clustering_by_coupling(self, top_n: int = 100, min_weight: int = 3,
                               impact: str = "local") -> Dict[str, Any]:
        """Clusters of coupled documents: centrality × impact."""
        return clus.clustering_by_coupling(self, top_n, min_weight,
                                           impact=impact)

    # -- network measures -----------------------------------------------------

    def network_metrics(self, graph: Dict[str, Any],
                        overlay_unit: Optional[str] = None,
                        level: str = "parent",
                        resolution: float = 1.0) -> Dict[str, Any]:
        """Centralities, communities and time overlay of a network.

        `resolution` sets the Louvain granularity: above 1, more and smaller
        groups.
        """
        overlay = netan.overlay_years(self, overlay_unit, level) if overlay_unit else None
        return netan.annotate(graph, overlay=overlay, resolution=resolution)

    @staticmethod
    def network_summary(graph: Dict[str, Any]) -> Dict[str, Any]:
        """Density, transitivity, components, modularity of a network."""
        return netan.graph_summary(graph)

    @staticmethod
    def network_layout(graph: Dict[str, Any]) -> Dict[str, list]:
        """Deterministic 2D coordinates of the nodes (MDS on the shortest paths)."""
        return netan.layout(graph)

    @staticmethod
    def attach_layout(graph: Dict[str, Any]) -> Dict[str, Any]:
        """The network with ``x``/``y`` on each node: the positions the interface
        draws, and those of `render_network`."""
        return netan.attach_layout(graph)

    @staticmethod
    def network_density(graph: Dict[str, Any], size: int = 48) -> Dict[str, Any]:
        """Density map of the network."""
        return netan.density_grid(graph, netan.layout(graph), size=size)

    def citation_network(self, unit: str = "sources", top_n: int = 40,
                         min_weight: int = 1, level: str = "parent") -> Dict[str, Any]:
        """DIRECT citation network (directed) aggregated to the requested unit."""
        return loc.citation_network(self, unit, top_n, min_weight, level)

    # -- text mining ---------------------------------------------------------

    def top_terms(self, field: str = "abstract", ngram: int = 1,
                  n: Optional[int] = 50, min_documents: int = 2,
                  extra_stopwords: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Most frequent terms of titles or abstracts."""
        return txt.top_terms(self, field, ngram, n, min_documents, extra_stopwords)

    def text_co_occurrence(self, field: str = "abstract", ngram: int = 2,
                           top_n: int = 50, min_weight: int = 2,
                           min_documents: int = 2,
                           extra_stopwords: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Co-occurrence network of the text terms."""
        return txt.text_co_occurrence(self, field, ngram, top_n, min_weight,
                                      min_documents, extra_stopwords)

    def text_trend(self, field: str = "abstract", ngram: int = 2, n: int = 20,
                   min_documents: int = 2,
                   extra_stopwords: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Position in time of the text terms."""
        return txt.text_trend(self, field, ngram, n, min_documents, extra_stopwords)

    def topic_dendrogram(self, kind: str = "author", top_n: int = 40,
                         min_documents: int = 2,
                         max_clusters: int = 6) -> Dict[str, Any]:
        """Hierarchical tree of the terms, cut into groups."""
        return fac.topic_dendrogram(self, kind, top_n, min_documents, max_clusters)

    # -- additional impact ----------------------------------------------------

    def countries_impact(self, n: Optional[int] = 20,
                         min_documents: int = 1) -> pd.DataFrame:
        """h, g, m indices per country."""
        return ctry.countries_impact(self, n, min_documents)

    def normalized_citations(self) -> pd.DataFrame:
        """Citations relative to the mean of their publication year."""
        return imp.normalized_citations(self)

    # -- what the interface displays, assembled -------------------------------
    # Every method IS what a card of the interface shows, for the same options
    # (see `views.py`): the web API only calls them.

    def document_list(self, n: Optional[int] = 100, sort: str = "citations") -> pd.DataFrame:
        """The documents: title, first author, year, source, type, global and local
        citations, DOI. `sort`: citations, year, title."""
        return vw.document_list(self, n, sort)

    def most_normalized_documents(self, n: Optional[int] = 25) -> pd.DataFrame:
        """Documents ranked by citations normalised by year."""
        return vw.most_normalized_documents(self, n)

    def network(self, unit: str = "authors", top_n: int = 50, min_weight: int = 1,
                normalization: str = "none", overlay: bool = False,
                kind: str = "author", level: str = "parent",
                resolution: float = 1.0) -> Dict[str, Any]:
        """The "Network lab" network, measured: `unit` among authors, keywords,
        institutions, countries, references, cited-authors, coupling;
        `normalization` among none, association, jaccard, salton, inclusion,
        equivalence."""
        return vw.network(self, unit, top_n, min_weight, normalization, overlay,
                          kind, level, resolution)

    def density_map(self, unit: str = "keywords", top_n: int = 50, min_weight: int = 1,
                    size: int = 48, kind: str = "author",
                    level: str = "parent") -> Dict[str, Any]:
        """The density map of a "Network lab" network."""
        return vw.density_map(self, unit, top_n, min_weight, size, kind, level)

    def term_network(self, field: str = "abstract", ngram: int = 2, top_n: int = 40,
                     min_weight: int = 2, min_documents: int = 2,
                     normalization: str = "none",
                     stopwords: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """The network of the text terms, measured (Text mining page)."""
        return vw.term_network(self, field, ngram, top_n, min_weight, min_documents,
                               normalization, stopwords)

    def citation_graph(self, unit: str = "sources", top_n: int = 40, min_weight: int = 1,
                       level: str = "parent") -> Dict[str, Any]:
        """The direct citation network, measured (Citation network page)."""
        return vw.citation_graph(self, unit, top_n, min_weight, level)

    def citation_balance(self, unit: str = "sources", top_n: int = 40,
                         min_weight: int = 1, level: str = "parent") -> pd.DataFrame:
        """Citations received minus given, per entity of the citation network."""
        return vw.citation_balance(self, unit, top_n, min_weight, level)

    # -- the interface's figures ---------------------------------------------

    @staticmethod
    def figure_catalog() -> pd.DataFrame:
        """All the figures of the interface: name, title, page, section, options
        (with their on-screen value) and the method that carries the data."""
        from ..figures.catalog import catalog_table
        return catalog_table()

    def figure_spec(self, name: str, **options: Any):
        """The description (`FigureSpec`) of a simple figure of the interface (bars,
        lines, scatter), to be modified before `render_figure`."""
        from ..figures.catalog import figure_spec
        return figure_spec(self, name, **options)

    def figure(self, name: str, fmt: Optional[str] = None, path: Optional[Any] = None,
               dpi: int = 300, style: Optional[Dict[str, Any]] = None,
               **options: Any) -> bytes:
        """A figure of the interface, as an image (PNG, SVG, PDF or JPG).

        ``name``: the one from `figure_catalog()` (``top-authors``,
        ``collaboration-map``...). The default options are those of the screen;
        ``n=10`` for a top 10, and `filter` for a period:

            >>> corpus.filter(years=(2023, 2025)).figure(
            ...     "top-authors", n=10, path="top_authors.svg")

        With ``path``, the image is also written; its extension chooses the
        format. PNG at 300 dpi by default, vector SVG. ``style`` dresses the
        figure like the interface's export dialog:

            >>> corpus.figure("documents-per-year", path="production.png",
            ...               style={"kind": "pie", "show_title": True,
            ...                      "subtitle": "Years 2010-2013"})
        """
        from ..figures.catalog import figure_bytes
        return figure_bytes(self, name, fmt=fmt, path=path, dpi=dpi, style=style,
                            **options)
