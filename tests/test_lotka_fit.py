"""Lotka generalise : y = C/n^a, exposant par maximum de vraisemblance."""

import numpy as np
import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics import laws as lw


def _corpus(productivities):
    """Un corpus ou chaque auteur signe le nombre de documents demande."""
    rows, eid = [], 0
    for i, k in enumerate(productivities):
        for _ in range(k):
            eid += 1
            rows.append({
                "Title": "Document %d with a reasonably long title" % eid,
                "Year": "2020", "Cited by": "1", "EID": "eid-%d" % eid,
                "Source title": "J1", "Document Type": "Article",
                "Authors": "1:A%d." % i, "Author Keywords": "alpha",
                "Affiliations": "parent 1: U, city: X, country: Morocco",
            })
    return Corpus.from_dataframe(pd.DataFrame(rows))


# ------------------------------------------------------------------ zeta ----

def test_zeta_valeurs_connues():
    assert lw._zeta(2.0) == pytest.approx(np.pi ** 2 / 6, abs=1e-4)
    assert lw._zeta(4.0) == pytest.approx(np.pi ** 4 / 90, abs=1e-4)


def test_zeta_indefinie_en_dessous_de_un():
    """La serie diverge : aucune constante ne normalise la loi."""
    assert np.isnan(lw._zeta(1.0))
    assert np.isnan(lw._zeta(0.755))


# ------------------------------------------------------------------- MLE ----

def test_mle_retrouve_un_exposant_connu():
    """Echantillon tire d'une loi zeta d'exposant 2,5 : on doit le retrouver."""
    rng = np.random.default_rng(7)
    a = 2.5
    n = np.arange(1, 200, dtype=float)
    p = n ** (-a)
    p /= p.sum()
    sample = rng.choice(n, size=6000, p=p)
    assert lw._lotka_mle(sample) == pytest.approx(a, abs=0.08)


def test_mle_resiste_a_une_queue_qui_trompe_les_moindres_carres():
    """Le defaut corrige : une queue de niveaux a un seul auteur.

    Les moindres carres sur log-log donnent le meme poids a chaque NIVEAU de
    productivite. Quand la queue en compte beaucoup, tous a un seul auteur, la
    pente s'aplatit jusqu'a passer sous 1 — et la loi n'a alors meme plus de
    constante de normalisation.
    """
    productivities = [1] * 18 + [2] * 10 + [3, 3, 4, 4, 6, 7, 7, 7, 8, 10, 11, 12, 25, 50]
    c = _corpus(productivities)
    fit = c.lotka()["fit"]
    assert fit["exponent_ols"] < 1.0, "le biais des moindres carres doit etre visible"
    assert fit["exponent"] > 1.0, "la vraisemblance doit rester dans le domaine valide"
    assert np.isfinite(fit["constant"])


# ------------------------------------------------------------- la formule ----

def test_la_loi_ajustee_suit_bien_y_egale_c_sur_n_puissance_a():
    c = _corpus([1] * 12 + [2] * 6 + [3] * 3 + [5, 8])
    r = c.lotka()
    a, const = r["fit"]["exponent"], r["fit"]["constant"]
    for row in r["table"].itertuples():
        attendu = const * row.documents_written ** (-a)
        assert row.share_fitted == pytest.approx(attendu, abs=1e-4)


def test_constante_normalise_la_loi():
    """C = 1/ζ(a) : la loi somme a 1 sur TOUS les entiers, pas seulement ceux vus."""
    c = _corpus([1] * 12 + [2] * 6 + [3] * 3 + [5, 8])
    fit = c.lotka()["fit"]
    a, const = fit["exponent"], fit["constant"]
    n = np.arange(1, 4000, dtype=float)
    assert float(np.sum(const * n ** (-a))) == pytest.approx(1.0, abs=0.01)


def test_les_trois_colonnes_sont_presentes():
    c = _corpus([1] * 8 + [2] * 4 + [3, 5])
    cols = set(c.lotka()["table"].columns)
    assert {"share_observed", "share_fitted", "share_lotka"} <= cols


def test_lotka_strict_reste_l_exposant_deux():
    """La colonne de reference ne doit PAS bouger avec l'ajustement."""
    c = _corpus([1] * 8 + [2] * 4 + [3, 5])
    t = c.lotka()["table"].set_index("documents_written")
    zeta2 = np.pi ** 2 / 6
    for n in t.index:
        assert t.loc[n, "share_lotka"] == pytest.approx((1 / n ** 2) / zeta2, abs=1e-4)


def test_deux_tests_ks_distincts():
    """Suivre une loi de puissance et suivre CELLE de Lotka sont deux questions."""
    c = _corpus([1] * 18 + [2] * 10 + [3, 3, 4, 4, 7, 7, 7, 25, 50])
    r = c.lotka()
    assert set(r) >= {"ks_test", "ks_test_strict"}
    assert r["ks_test"]["d"] is not None
    assert r["ks_test_strict"]["d"] is not None


def test_ajustement_plus_proche_que_lotka_strict():
    """C'est tout l'objet de la correction : l'ecart doit diminuer."""
    c = _corpus([1] * 18 + [2] * 10 + [3, 3, 4, 4, 6, 7, 7, 7, 8, 10, 11, 12, 25, 50])
    r = c.lotka()
    assert r["ks_test"]["d"] <= r["ks_test_strict"]["d"]
