"""Le package SEUL, de bout en bout, sur de vrais fichiers nettoyés.

Aucune interface, aucun serveur : exactement ce que fait quelqu'un qui installe
`bibliominer-analysis` et ouvre un notebook. Chaque méthode publique de
`Corpus` est appelée, et le test ÉCHOUE si une méthode publique n'est pas dans
la liste : une fonction ajoutée plus tard ne peut pas échapper au test.

Au-delà du « ça ne plante pas », on vérifie les invariants qu'un résultat
bibliométrique doit respecter quel que soit le corpus : des parts qui somment à
100 %, h ≤ g ≤ documents, des zones de Bradford qui somment au total, des liens
de réseau qui pointent vers des nœuds existants, une disposition déterministe.

Les fichiers réels ne sont pas versionnés (ce sont les corpus de
l'utilisateur) : le test lit leurs chemins dans ``BIBLIOMINER_CLEANED_CSV``
(séparés par ``os.pathsep``) et se met en attente sans eux.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import time

import numpy as np
import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.figures import render_figure
from bibliominer_analysis.figures.network import render_network
from bibliominer_analysis.figures.render import FigureSpec, Series

PATHS = [p for p in os.environ.get("BIBLIOMINER_CLEANED_CSV", "").split(os.pathsep) if p]

pytestmark = pytest.mark.skipif(
    not PATHS, reason="BIBLIOMINER_CLEANED_CSV ne pointe vers aucun fichier nettoyé")

#: Au-delà, un appel est signalé comme trop lent pour une interface.
SLOW_SECONDS = 20.0

#: Méthode -> arguments. TOUTES les méthodes publiques doivent y figurer.
CALLS = {
    "access_over_time": {}, "access_routes": {}, "access_status": {},
    "access_summary": {}, "affiliation_profile": {}, "agr": {},
    "anomalies": {}, "authors_by_affiliation_count": {},
    "documents_by_affiliation_count": {},
    "authors_impact": {"n": 20}, "authors_over_time": {"n": 12},
    "authors_self_citation": {"n": 20}, "authorship_groups": {},
    "authorship_pattern": {}, "average_citations_per_year": {},
    "bibliographic_coupling": {"top_n": 50, "min_weight": 2},
    "bradford": {"zones": 3}, "cagr": {}, "cai": {"block_years": 5},
    "citation_network": {"unit": "sources"}, "citation_pairs": {},
    "cities_impact": {}, "cities_over_time": {}, "city_hierarchy": {},
    "clustering_by_coupling": {}, "co_authorship": {}, "co_citation": {},
    "co_citation_authors": {}, "co_city": {}, "co_country": {},
    "co_institution": {}, "co_word": {"top_n": 50, "min_weight": 1},
    "cochran_sample_size": {}, "collaboration_indicators": {},
    "collaboration_scale": {}, "concentration": {"unit": "authors"},
    "concentration_summary": {}, "conceptual_structure": {"method": "CA"},
    "corresponding_author_countries": {}, "first_author_countries": {},
    "countries_impact": {},
    "country_map": {}, "document_types": {}, "duplicates": {},
    "field_completeness": {}, "growth_summary": {}, "historiograph": {},
    "impact": {}, "indicator_readiness": {}, "institutions_by_country": {},
    "institutions_impact": {}, "institutions_over_time": {},
    "interdisciplinarity": {}, "journal_open_access": {}, "local_citations": {},
    "lotka": {}, "main_information": {}, "most_cited_documents": {},
    "most_cited_references": {}, "most_local_cited_authors": {},
    "most_local_cited_documents": {}, "most_local_cited_sources": {},
    "n_authors": {}, "n_institutions": {}, "normalized_citations": {},
    "org_hierarchy": {},
    "price_index": {}, "price_index_by_year": {}, "price_law": {},
    "production_by_country": {}, "production_by_year": {},
    "quality_summary": {}, "quartile_distribution": {},
    "quartile_over_time": {}, "reference_age_distribution": {},
    "reference_spectroscopy": {}, "rgr_doubling_time": {},
    "scimago_coverage": {}, "scimago_sources": {},
    "self_citation_summary": {}, "sources_impact": {},
    "sources_over_time": {}, "subject_areas": {}, "subject_categories": {},
    "summary": {}, "table_completeness": {}, "tables": {},
    "text_co_occurrence": {"field": "abstract"},
    "text_trend": {"field": "abstract"}, "thematic_evolution": {},
    "thematic_map": {}, "three_fields": {}, "top_authors": {},
    "top_cities": {}, "top_institutions": {}, "top_keywords": {},
    "top_sources": {}, "top_terms": {"field": "abstract"},
    "topic_dendrogram": {}, "trend_forecast": {}, "trend_topics": {},
    "word_dynamics": {}, "year_range": {}, "zipf": {},
    # ce que l'interface montre, assemblé par le paquet (views.py)
    "citation_balance": {}, "citation_graph": {}, "density_map": {},
    "document_list": {}, "most_normalized_documents": {},
    "network": {"unit": "keywords", "normalization": "association",
                "overlay": True},
    "term_network": {},
}

#: Appelées à part, parce qu'elles prennent un graphe ou construisent le corpus.
SEPARATE = {"from_csv", "from_dataframe", "from_tables", "filter",
            "network_density", "network_layout", "network_metrics",
            "network_summary", "attach_layout",
            # le catalogue des figures : `test_every_catalog_figure_renders`
            "figure", "figure_spec", "figure_catalog"}

NETWORKS = ("co_word", "co_authorship", "co_citation", "co_citation_authors",
            "co_country", "co_institution", "co_city", "bibliographic_coupling",
            "citation_network")


def test_every_public_method_is_exercised():
    public = {n for n, m in inspect.getmembers(Corpus)
              if not n.startswith("_") and callable(m)}
    missing = public - set(CALLS) - SEPARATE
    assert not missing, "méthodes publiques non testées : %s" % sorted(missing)


@pytest.fixture(scope="module", params=PATHS, ids=lambda p: os.path.basename(p))
def run(request):
    """Charge le fichier puis appelle chaque méthode une fois, en chronométrant."""
    corpus = Corpus.from_csv(request.param)
    results, errors, timings = {}, {}, {}
    for name, kwargs in CALLS.items():
        started = time.perf_counter()
        try:
            results[name] = getattr(corpus, name)(**kwargs)
        except Exception as exc:                      # noqa: BLE001
            errors[name] = "%s: %s" % (type(exc).__name__, exc)
        timings[name] = time.perf_counter() - started
    return corpus, results, errors, timings


def test_no_method_raises(run):
    _, _, errors, _ = run
    assert not errors, json.dumps(errors, indent=1)


def test_no_method_is_too_slow(run):
    _, _, _, timings = run
    slow = {k: round(v, 1) for k, v in timings.items() if v > SLOW_SECONDS}
    assert not slow, "trop lent pour une interface (s) : %s" % slow


def _finite(value):
    if isinstance(value, float):
        return not math.isinf(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    if isinstance(value, pd.DataFrame):
        numeric = value.select_dtypes(include=[np.number])
        return not np.isinf(numeric.to_numpy(dtype=float)).any() if not numeric.empty else True
    return True


def test_no_infinite_value_reaches_the_output(run):
    """Un infini casse la sérialisation JSON de l'API et les graphiques."""
    _, results, _, _ = run
    bad = [k for k, v in results.items() if not _finite(v)]
    assert not bad, bad


def test_citation_balance_received_equals_emitted(run):
    """Chaque citation interne est reçue par l'un et émise par l'autre."""
    _, r, _, _ = run
    b = r["citation_balance"]
    assert int(b["received"].sum()) == int(b["emitted"].sum())
    assert list(b["balance"]) == sorted(b["balance"], reverse=True)


def test_document_list_is_the_corpus(run):
    corpus, r, _, _ = run
    docs = r["document_list"]
    assert len(docs) == min(100, len(corpus.documents))
    assert list(docs["citations"]) == sorted(docs["citations"], reverse=True)


def test_every_catalog_figure_renders(run):
    """Chaque figure de l'interface, produite par le paquet sur ce fichier :
    une image, ou un refus EXPLIQUÉ quand la donnée manque (SCImago…)."""
    corpus, _, _, _ = run
    errors = {}
    for name in Corpus.figure_catalog()["name"]:
        try:
            assert corpus.figure(name, dpi=40)[:4] == b"\x89PNG", name
        except ValueError as exc:
            if not any(word in str(exc) for word in ("no data", "empty", "without")):
                errors[name] = str(exc)
        except Exception as exc:                      # noqa: BLE001
            errors[name] = "%s: %s" % (type(exc).__name__, exc)
    assert not errors, json.dumps(errors, indent=1)


def test_main_information_is_consistent(run):
    corpus, r, _, _ = run
    info = r["main_information"]
    assert info["documents"] == len(corpus.documents)
    years = pd.to_numeric(corpus.documents["year"], errors="coerce").dropna()
    assert int(r["production_by_year"]["documents"].sum()) == len(years)


def test_impact_indices_are_ordered(run):
    corpus, r, _, _ = run
    imp = r["impact"]
    assert 0 <= imp["h_index"] <= imp["g_index"] <= imp["documents"]
    authors = r["authors_impact"]
    if not authors.empty:
        assert (authors["h_index"] <= authors["g_index"]).all()
        assert (authors["g_index"] <= authors["documents"]).all()


def test_shares_sum_to_one_hundred(run):
    _, r, _, _ = run
    for name in ("authorship_pattern", "authorship_groups"):
        df = r[name]
        if not df.empty and df["documents"].sum() > 0:
            assert df["share"].sum() == pytest.approx(100.0, abs=0.6), name


def test_lotka_observed_shares_form_a_distribution(run):
    _, r, _, _ = run
    table = r["lotka"]["table"]
    if not table.empty:
        assert table["share_observed"].sum() == pytest.approx(1.0, abs=0.01)
        assert int(table["n_authors"].sum()) == r["lotka"]["total_authors"]


def test_bradford_zones_cover_every_document(run):
    _, r, _, _ = run
    b = r["bradford"]
    if not b["table"].empty:
        assert int(b["zones"]["documents"].sum()) == b["total_documents"]
        assert list(b["table"]["zone"]) == sorted(b["table"]["zone"])


def test_collaboration_indicators_are_in_range(run):
    _, r, _, _ = run
    c = r["collaboration_indicators"]
    if c["documents"]:
        assert 0 <= c["degree_of_collaboration"] <= 1
        assert 0 <= c["collaborative_coefficient"] <= 1
        assert 0 <= c["modified_collaborative_coefficient"] <= 1
        assert c["collaboration_index"] >= 1


@pytest.mark.parametrize("name", NETWORKS)
def test_networks_are_well_formed(run, name):
    corpus, r, _, _ = run
    g = r[name]
    ids = {n["id"] for n in g["nodes"]}
    assert g["n_nodes"] == len(g["nodes"])
    assert g["n_edges"] == len(g["edges"])
    for e in g["edges"]:
        assert e["source"] in ids, name
        assert e["target"] in ids, name
        assert e["weight"] >= 1
    if g["n_edges"]:
        annotated = corpus.network_metrics(g)
        pr = sum(n["pagerank"] for n in annotated["nodes"])
        assert pr == pytest.approx(1.0, abs=0.01), name
        summary = Corpus.network_summary(annotated)
        assert summary["nodes"] == g["n_nodes"]


def test_layout_is_deterministic_and_complete(run):
    _, r, _, _ = run
    g = r["co_word"]
    if not g["n_nodes"]:
        pytest.skip("pas de réseau de co-mots sur ce fichier")
    first, second = Corpus.network_layout(g), Corpus.network_layout(g)
    assert first == second
    assert set(first) == {n["id"] for n in g["nodes"]}
    assert all(all(math.isfinite(v) for v in xy) for xy in first.values())
    density = Corpus.network_density(g)
    assert density["cells"]
    # Les positions posées sur les nœuds (celles que dessine l'interface)
    # sont exactement celles de la disposition.
    placed = Corpus.attach_layout(g)
    assert {n["id"]: [n["x"], n["y"]] for n in placed["nodes"]} == first


def test_filter_returns_a_consistent_subcorpus(run):
    corpus, _, _, _ = run
    lo, hi = corpus.year_range()
    if lo is None:
        pytest.skip("pas d'années")
    sub = corpus.filter(years=(hi, hi))
    years = pd.to_numeric(sub.documents["year"], errors="coerce").dropna()
    assert (years == hi).all()
    assert set(sub.documents["eid"]) <= set(corpus.documents["eid"])


def test_figures_render_without_an_interface(run):
    corpus, r, _, _ = run
    per_year = r["production_by_year"]
    spec = FigureSpec(kind="bar", title="Documents per year",
                      categories=[str(y) for y in per_year["year"]],
                      series=[Series(name="Documents",
                                     values=[float(v) for v in per_year["documents"]])])
    assert render_figure(spec, fmt="png").startswith(b"\x89PNG")
    g = r["co_word"]
    if g["n_nodes"]:
        png = render_network(corpus.network_metrics(g), size_by="pagerank")
        assert png.startswith(b"\x89PNG")
