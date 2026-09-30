"""Corrections issues de l'audit des formules, septembre 2026.

Chaque test porte une valeur calculée À LA MAIN depuis la définition publiée,
et chacun échouait sur le code d'avant l'audit : c'est ce qui prouve que le
défaut est corrigé, et pas seulement que le code tourne.
"""

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.io import parsers as P
from bibliominer_analysis.metrics import growth as gr
from bibliominer_analysis.metrics import laws as lw
from bibliominer_analysis.metrics import selfcitation as sc
from bibliominer_analysis.metrics import themes as th
from bibliominer_analysis.networks import analysis as na


def _doc(i, year=2020, authors="1:A.", source="J1", kws="", refs="",
         title=None, doi="", ids="", full=""):
    return {
        "Title": title or "Document number %d with a long enough title" % i,
        "Year": str(year), "Cited by": "0", "EID": "eid-%d" % i,
        "Source title": source, "Document Type": "Article",
        "Authors": authors, "Author(s) ID": ids, "Author full names": full,
        "Author Keywords": kws, "References": refs, "DOI": doi,
        "Affiliations": "parent 1: U, city: X, country: Morocco",
    }


def _corpus(rows):
    return Corpus.from_dataframe(pd.DataFrame(rows))


# --- Lotka : test de Kolmogorov-Smirnov (Pao, 1985) -----------------------

def test_ks_valeur_critique_sur_le_nombre_d_auteurs():
    """1,36/√N avec N = 1 000 auteurs, pas N = 4 niveaux de productivité.

    Avec l'ancien N = 4, la valeur critique valait 0,68 et D = 0,1 passait.
    """
    obs = np.array([0.70, 0.15, 0.10, 0.05])
    theo = np.array([0.60, 0.20, 0.12, 0.08])
    r = lw.kolmogorov_smirnov(obs, theo, sample_size=1000)
    assert r["critical_5pct"] == pytest.approx(1.36 / math.sqrt(1000), abs=1e-4)
    assert r["d"] == pytest.approx(0.10, abs=1e-4)
    assert r["follows_lotka"] is False


def test_lotka_ks_couvre_les_niveaux_sans_auteur():
    """A a 3 documents, B, C, D, E en ont 1 ; personne n'en a 2.

    Parts observées sur 1..3 : 0,8 / 0 / 0,2. Lotka strict (1/x²)/ζ(2) :
    0,6079 / 0,1520 / 0,0675. Répartitions : 0,8 / 0,8 / 1,0 contre
    0,6079 / 0,7599 / 0,8275 -> D = 0,1921. En sautant le niveau 2, l'ancien
    calcul comparait 1,0 à 0,6755 et trouvait D = 0,3245.
    """
    c = _corpus([_doc(1, authors="1:A.; 2:B."), _doc(2, authors="1:A.; 2:C."),
                 _doc(3, authors="1:A.; 2:D.; 3:E.")])
    ks = c.lotka()["ks_test_strict"]
    assert ks["d"] == pytest.approx(0.1921, abs=1e-4)
    assert ks["critical_5pct"] == pytest.approx(1.36 / math.sqrt(5), abs=1e-4)


# --- MCC (Savanur & Srikanth, 2010) ----------------------------------------

def test_mcc_a_est_le_nombre_total_d_auteurs():
    """3 articles, chacun à 2 auteurs tous différents : 6 auteurs.

    CC = 1 − (3 × 1/2)/3 = 0,5. MCC = 0,5 × 6/5 = 0,6. Avec A = maximum par
    article (2), l'ancien calcul donnait 1,0 : « collaboration maximale ».
    """
    c = _corpus([_doc(1, authors="1:A.; 2:B."), _doc(2, authors="1:C.; 2:D."),
                 _doc(3, authors="1:E.; 2:F.")])
    r = c.collaboration_indicators()
    assert r["collaborative_coefficient"] == pytest.approx(0.5)
    assert r["modified_collaborative_coefficient"] == pytest.approx(0.6)


# --- Bradford ---------------------------------------------------------------

def test_bradford_une_revue_qui_remplit_deux_zones():
    """J1 : 8 documents sur 10 remplit les zones 1 ET 2 (seuils 3,33 et 6,67).

    Les deux revues suivantes sont donc en zone 3. L'ancien `if` n'avançait que
    d'une zone, et mettait J2 en zone 2.
    """
    rows = [_doc(i, source="J1") for i in range(1, 9)]
    rows += [_doc(9, source="J2"), _doc(10, source="J3")]
    table = _corpus(rows).bradford()["table"]
    zones = dict(zip(table["source"], table["zone"]))
    assert zones == {"J1": 1, "J2": 3, "J3": 3}


# --- Centralité de vecteur propre -------------------------------------------

def test_vecteur_propre_d_une_etoile():
    """Étoile à 3 branches : λ = √3, vecteur (√3, 1, 1, 1)/√6.

    Centre = 0,7071, feuilles = 0,4082. Graphe biparti : l'ancienne itération
    oscillait sans converger et s'arrêtait sur 0,5 pour les quatre nœuds,
    centre et feuilles indiscernables.
    """
    A = np.zeros((4, 4))
    for leaf in (1, 2, 3):
        A[0, leaf] = A[leaf, 0] = 1.0
    x = na._eigenvector(A)
    assert x[0] == pytest.approx(math.sqrt(3) / math.sqrt(6), abs=1e-6)
    assert x[1] == pytest.approx(1 / math.sqrt(6), abs=1e-6)


def test_modularite_du_resume_suit_les_communautes_affichees():
    graph = {
        "nodes": [{"id": "a", "community": 1}, {"id": "b", "community": 1},
                  {"id": "c", "community": 2}, {"id": "d", "community": 2}],
        "edges": [{"source": "a", "target": "b", "weight": 3},
                  {"source": "c", "target": "d", "weight": 3},
                  {"source": "b", "target": "c", "weight": 1}],
    }
    G = nx.Graph()
    for e in graph["edges"]:
        G.add_edge(e["source"], e["target"], weight=e["weight"])
    expected = nx.community.modularity(G, [{"a", "b"}, {"c", "d"}], weight="weight")
    assert na.graph_summary(graph)["modularity"] == pytest.approx(round(expected, 3))


# --- Carte thématique de Callon ---------------------------------------------

def test_densite_de_callon_sur_l_indice_d_equivalence():
    """{a, b} co-occurrent dans 2 documents : e = 2²/(2·2) = 1, densité = 100·1/2 = 50.
    {c, d} dans 3 documents : e = 3²/(3·3) = 1, densité = 50 aussi.

    Sur les comptages bruts, l'ancien calcul donnait 100 et 150 : le thème le
    plus fréquent paraissait plus « développé » à structure identique.
    """
    rows = [_doc(i, kws="a;b") for i in (1, 2)] + [_doc(i, kws="c;d") for i in (3, 4, 5)]
    clusters = th.thematic_map(_corpus(rows), min_weight=2)["clusters"]
    assert sorted(clusters["density"]) == [50.0, 50.0]


# --- Croissance ------------------------------------------------------------

def test_quantile_pour_une_confiance_non_tabulee():
    """0,975 bilatéral -> z = 2,2414 ; l'ancien arrondi à 0,97 donnait 2,1705."""
    assert gr._z_score(0.975) == pytest.approx(2.2414, abs=5e-4)


def test_temps_de_doublement_sur_le_rgr_exact():
    """1 500 documents puis 1 : RGR = ln(1501/1500) = 0,000666.

    Dt = ln 2 / RGR = 1 040,1 ans. Calculé sur le RGR arrondi à 0,0007,
    l'ancien résultat était 990,2.
    """
    rows = [_doc(i, year=2020) for i in range(1, 1501)] + [_doc(1501, year=2021)]
    table = gr.rgr_doubling_time(_corpus(rows))
    dt = float(table.loc[table["year"] == 2021, "doubling_time"].iat[0])
    assert dt == pytest.approx(math.log(2) / math.log(1501 / 1500), abs=0.01)


def test_derniere_annee_partielle_signalee():
    from datetime import date
    this_year = date.today().year
    rows = [_doc(i, year=y) for i, y in enumerate(
        [this_year - 2, this_year - 1, this_year - 1, this_year], start=1)]
    fit = gr.trend_forecast(_corpus(rows))["fit"]
    assert fit["last_year_partial"] is True


# --- Appariement des colonnes d'auteurs -------------------------------------

def test_un_identifiant_manquant_ne_decale_pas_les_suivants():
    """Roe n'a pas d'identifiant : la colonne en compte deux pour trois auteurs.

    Par rang, l'ancien code donnait l'identifiant de Poe à Roe, et aucun à Poe.
    """
    a = P.parse_authors("Doe J.; Roe A.; Poe B.", "", "111111; 333333")
    assert [x["scopus_id"] for x in a] == [None, None, None]


def test_le_nom_complet_rattache_par_patronyme():
    a = P.parse_authors("Doe J.; Roe A.; Poe B.",
                        "Doe, John (111111); Poe, Bob (333333)", "")
    assert [x["scopus_id"] for x in a] == ["111111", None, "333333"]


def test_l_identifiant_entre_parentheses_prime_sur_la_colonne():
    a = P.parse_authors("Doe J.; Roe A.", "Doe, John (111111); Roe, Ann (222222)",
                        "222222; 111111")
    assert [x["scopus_id"] for x in a] == ["111111", "222222"]


# --- Auto-citation ----------------------------------------------------------

def test_deux_homonymes_ne_s_auto_citent_pas():
    """Deux « Smith J. » d'identifiants différents : l'un cite l'autre.

    Ce n'est PAS une auto-citation. Par le nom, l'ancien calcul la comptait.
    """
    cited = _doc(1, authors="1:Smith J.", ids="1:111111", doi="10.1000/found",
                 title="The foundational paper that everybody cites here")
    citing = _doc(2, authors="1:Smith J.", ids="1:222222",
                  refs="ref1 | 10.1000/found | 2019 | Smith J. | "
                       "The foundational paper that everybody cites here")
    out = sc.authors_self_citation(_corpus([cited, citing]), min_citations=1)
    assert len(out) == 1
    assert int(out["self_citations"].iat[0]) == 0
