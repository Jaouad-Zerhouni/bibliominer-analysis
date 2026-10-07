"""Generalised Lotka: y = C/n^a, exponent by maximum likelihood."""

import numpy as np
import pandas as pd
import pytest

from bibliominer_analysis import Corpus
from bibliominer_analysis.metrics import laws as lw


def _corpus(productivities):
    """A corpus where each author signs the requested number of documents."""
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

def test_zeta_known_values():
    assert lw._zeta(2.0) == pytest.approx(np.pi ** 2 / 6, abs=1e-4)
    assert lw._zeta(4.0) == pytest.approx(np.pi ** 4 / 90, abs=1e-4)


def test_zeta_undefined_below_one():
    """The series diverges: no constant normalises the law."""
    assert np.isnan(lw._zeta(1.0))
    assert np.isnan(lw._zeta(0.755))


# ------------------------------------------------------------------- MLE ----

def test_mle_finds_a_known_exponent():
    """A sample drawn from a zeta law with exponent 2.5: the fit must recover it."""
    rng = np.random.default_rng(7)
    a = 2.5
    n = np.arange(1, 200, dtype=float)
    p = n ** (-a)
    p /= p.sum()
    sample = rng.choice(n, size=6000, p=p)
    assert lw._lotka_mle(sample) == pytest.approx(a, abs=0.08)


def test_mle_resists_a_tail_that_fools_least_squares():
    """The fixed defect: a tail of levels with a single author each.

    Least squares on log-log give the same weight to every productivity
    LEVEL. When the tail has many of them, all with a single author, the slope
    flattens until it drops below 1, and the law then no longer even has a
    normalising constant.
    """
    productivities = [1] * 18 + [2] * 10 + [3, 3, 4, 4, 6, 7, 7, 7, 8, 10, 11, 12, 25, 50]
    c = _corpus(productivities)
    fit = c.lotka()["fit"]
    assert fit["exponent_ols"] < 1.0, "the least-squares bias must be visible"
    assert fit["exponent"] > 1.0, "the likelihood must stay in the valid domain"
    assert np.isfinite(fit["constant"])


# --------------------------------------------------------------- the formula ----

def test_the_fitted_law_follows_y_equals_c_over_n_power_a():
    c = _corpus([1] * 12 + [2] * 6 + [3] * 3 + [5, 8])
    r = c.lotka()
    a, const = r["fit"]["exponent"], r["fit"]["constant"]
    for row in r["table"].itertuples():
        expected = const * row.documents_written ** (-a)
        assert row.share_fitted == pytest.approx(expected, abs=1e-4)


def test_constant_normalises_the_law():
    """C = 1/ζ(a): the law sums to 1 over ALL integers, not only the observed ones."""
    c = _corpus([1] * 12 + [2] * 6 + [3] * 3 + [5, 8])
    fit = c.lotka()["fit"]
    a, const = fit["exponent"], fit["constant"]
    n = np.arange(1, 4000, dtype=float)
    assert float(np.sum(const * n ** (-a))) == pytest.approx(1.0, abs=0.01)


def test_the_three_columns_are_present():
    c = _corpus([1] * 8 + [2] * 4 + [3, 5])
    cols = set(c.lotka()["table"].columns)
    assert {"share_observed", "share_fitted", "share_lotka"} <= cols


def test_strict_lotka_stays_exponent_two():
    """The reference column must NOT move with the fit."""
    c = _corpus([1] * 8 + [2] * 4 + [3, 5])
    t = c.lotka()["table"].set_index("documents_written")
    zeta2 = np.pi ** 2 / 6
    for n in t.index:
        assert t.loc[n, "share_lotka"] == pytest.approx((1 / n ** 2) / zeta2, abs=1e-4)


def test_two_distinct_ks_tests():
    """Following a power law and following LOTKA's law are two questions."""
    c = _corpus([1] * 18 + [2] * 10 + [3, 3, 4, 4, 7, 7, 7, 25, 50])
    r = c.lotka()
    assert set(r) >= {"ks_test", "ks_test_strict"}
    assert r["ks_test"]["d"] is not None
    assert r["ks_test_strict"]["d"] is not None


def test_fit_closer_than_strict_lotka():
    """That is the whole point of the fix: the gap must shrink."""
    c = _corpus([1] * 18 + [2] * 10 + [3, 3, 4, 4, 6, 7, 7, 7, 8, 10, 11, 12, 25, 50])
    r = c.lotka()
    assert r["ks_test"]["d"] <= r["ks_test_strict"]["d"]
