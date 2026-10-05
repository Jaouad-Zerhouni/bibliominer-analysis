"""Bibliometric laws: Lotka, Bradford, Zipf.

These three laws describe very unequal concentrations, each on a
different object:

  - **Lotka (1926)**: the productivity of AUTHORS. The number of authors
    who published *x* articles decreases as 1/x^n. Many publish once, very
    few publish a lot.

  - **Bradford (1934)**: the scattering of SOURCES. Ranking journals by
    decreasing productivity and forming three zones each holding a third
    of the articles, the number of journals grows geometrically from one
    zone to the next. The first zone is the "core" of the field.

  - **Zipf (1949)**: the frequency of WORDS. The frequency of a term is
    inversely proportional to its rank, raised to a power *s*.

Every function returns both the OBSERVED data and the THEORETICAL fit:
without both, one cannot judge whether the law applies to the corpus. An
exponent alone means nothing.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fits y = C · x^(-b) by least squares on the logarithms.

    Returns the exponent, the constant and the R²; the latter is essential: it
    says whether the law really DESCRIBES the corpus. An exponent without an
    R² is a number without a guarantee.
    """
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return {"exponent": float("nan"), "constant": float("nan"), "r2": float("nan")}

    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)

    predicted = slope * lx + intercept
    ss_res = float(np.sum((ly - predicted) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"exponent": float(-slope), "constant": float(10 ** intercept),
            "r2": float(r2)}


def _zeta(a: float, terms: int = 2000) -> float:
    """ζ(a) = Σ 1/n^a, the constant that makes C/n^a a REAL distribution.

    Without it, "C" is only an intercept in counting units: it cannot be
    compared with an observed share, which is a probability. Dividing by ζ(a)
    guarantees that the theoretical shares sum to 1 over all integers,
    exactly like the observed shares.

    The sum is truncated then completed by the Euler-Maclaurin term: the tail
    of a 1/n^a series decreases slowly when `a` is close to 1, and ignoring
    it would overestimate C by several percent.
    """
    if not np.isfinite(a) or a <= 1.0:
        return float("nan")
    n = np.arange(1, terms + 1, dtype=float)
    head = float(np.sum(n ** (-a)))
    tail = terms ** (1.0 - a) / (a - 1.0) - 0.5 * terms ** (-a)
    return head + tail


#: Terms of the series for the likelihood equation. Near a = 1 the tail of
#: Σ ln n · n^-a decreases very slowly: 4,000 terms measurably biased the
#: expectation there, and therefore the exponent.
_MLE_TERMS = 200_000
_MLE_GRID = np.arange(1, _MLE_TERMS + 1, dtype=float)
_MLE_LOG_GRID = np.log(_MLE_GRID)


def _zeta_log_ratio(a: float) -> float:
    """Σ(ln n · n^-a) / ζ(a), the expectation of ln(x) under the power law.

    It is the quantity to set equal to the observed mean of ln(x) to solve
    the likelihood equation.
    """
    weights = _MLE_GRID ** (-a)
    return float(np.sum(_MLE_LOG_GRID * weights) / np.sum(weights))


def _lotka_mle(values: np.ndarray, lo: float = 1.01, hi: float = 6.0,
               iterations: int = 60) -> float:
    """Lotka exponent by maximum likelihood (discrete zeta distribution).

    **Why not least squares on log-log.** It is the method found everywhere,
    and it is a bad one here, as Clauset, Shalizi & Newman (2009) showed. It
    gives the same weight to every productivity LEVEL: on a real corpus, the
    tail has many levels containing a single author, and these isolated points
    dominate the regression to the point of flattening the slope. Measured on
    the test corpus: 0.755 by least squares against 1.754 by likelihood. The
    first is below 1, so the law does not even have a normalising constant
    there: the fit was unusable.

    Maximum likelihood weights by AUTHOR, which is the right unit of
    observation. The equation E[ln x] = observed mean of ln x is solved by
    bisection; the function is monotonically decreasing in `a`, so the
    bisection surely converges.
    """
    v = np.asarray([x for x in values if x is not None and x >= 1], dtype=float)
    if v.size == 0:
        return float("nan")
    target = float(np.mean(np.log(v)))
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _zeta_log_ratio(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Lotka, author productivity
# ---------------------------------------------------------------------------

def lotka(corpus) -> Dict[str, object]:
    """Distribution of the number of authors by number of publications.

    Returns ``{"table": DataFrame, "fit": {...}}``.

    A row reads: "``n_authors`` AUTHORS published ``documents_written``
    documents EACH". It is not a count of single-author articles; the reverse
    wording, which the table lends itself to, is the classic misreading of
    this law.

    For each number of documents *x*, the table gives three columns:

      - ``share_observed``: the real share of authors;
      - ``share_fitted``: the FITTED law ``y = C/x^a``, with the exponent
        estimated by maximum likelihood and ``C = 1/ζ(a)``;
      - ``share_lotka``: STRICT Lotka, exponent 2, i.e. ``(1/x²)/ζ(2)``.

    The last two answer distinct questions: does production follow *a* power
    law, and does it follow *Lotka's*? The second is far more demanding, and
    confusing it with the first leads to concluding there is a gap where there
    is none.
    """
    a = corpus.authors
    a = a[a["name"].notna() & (a["name"].map(str).str.strip() != "")]
    empty = pd.DataFrame(columns=["documents_written", "n_authors",
                                  "share_observed", "share_lotka"])
    if a.empty:
        return {"table": empty, "fit": _fit_power_law(np.array([]), np.array([]))}

    key = a["scopus_id"].fillna("name:" + a["name"].map(str))
    per_author = a.assign(key=key).drop_duplicates(
        subset=["key", "eid"]).groupby("key").size()

    dist = per_author.value_counts().sort_index()
    total_authors = int(dist.sum())

    x = dist.index.to_numpy(dtype=float)
    y = dist.to_numpy(dtype=float)

    observed = y / total_authors

    # STRICT Lotka: the exponent is 2 by hypothesis. It is the historical
    # reference, the one compared against, not a fit.
    zeta2 = np.pi ** 2 / 6.0
    share_lotka = (1.0 / x ** 2) / zeta2

    # GENERALISED Lotka: y_n = C / n^a, with the exponent estimated on the
    # corpus and C = 1/ζ(a) making it a probability distribution. This curve
    # says whether production follows A power law; the strict version only says
    # whether it follows LOTKA's, a different and far more demanding question.
    ols = _fit_power_law(x, y)
    exponent = _lotka_mle(per_author.to_numpy(dtype=float))
    zeta_a = _zeta(exponent)
    if np.isfinite(zeta_a) and zeta_a > 0:
        constant = 1.0 / zeta_a
        share_fitted = constant * x ** (-exponent)
    else:
        # An exponent ≤ 1 makes the series diverge: no constant normalises the law.
        # It is said, rather than returning a number that would make no sense.
        constant = float("nan")
        share_fitted = np.full_like(x, np.nan, dtype=float)

    table = pd.DataFrame({
        "documents_written": dist.index.astype(int),
        "n_authors": dist.to_numpy(dtype=int),
        "share_observed": observed.round(4),
        "share_fitted": np.round(share_fitted, 4),
        "share_lotka": share_lotka.round(4),
    }).reset_index(drop=True)

    fit = {
        "exponent": float(exponent),
        "constant": float(constant),
        # R² of the log-log regression: a DESCRIPTIVE measure of how well the points
        # align, not the quality of the estimate. It is kept because it is read
        # everywhere, with the least-squares exponent next to it so that the gap
        # between the two methods is visible.
        "r2": float(ols["r2"]),
        "exponent_ols": float(ols["exponent"]),
    }

    # The test compares the distributions over ALL levels 1..x_max, not only
    # the observed ones. A level without any author (nobody with 4 documents,
    # but someone with 5) has a zero observed share and a non-zero theoretical
    # share: omitting it dropped that mass from the theoretical distribution,
    # and underestimated the gap D.
    grid = np.arange(1, int(x.max()) + 1, dtype=float)
    observed_grid = pd.Series(observed, index=x).reindex(grid, fill_value=0.0).to_numpy()
    fitted_grid = (constant * grid ** (-exponent)) if np.isfinite(constant) \
        else np.full_like(grid, np.nan)
    lotka_grid = (1.0 / grid ** 2) / zeta2

    return {"table": table, "fit": fit,
            "total_authors": total_authors,
            # Two tests, two questions. The first: does production follow the FITTED
            # law? The second: does it follow strict Lotka?
            "ks_test": kolmogorov_smirnov(observed_grid, fitted_grid,
                                          sample_size=total_authors),
            "ks_test_strict": kolmogorov_smirnov(observed_grid, lotka_grid,
                                                 sample_size=total_authors)}


def kolmogorov_smirnov(observed: np.ndarray, theoretical: np.ndarray,
                       sample_size: Optional[int] = None) -> Dict[str, object]:
    """Kolmogorov-Smirnov test between the observed distribution and Lotka.

    A fitted exponent and an R² say that the law *resembles* the data. They
    do NOT say whether the remaining gap is compatible with chance: that is
    the question this test answers, and it is the one a reviewer asks.

    The cumulative distribution functions are compared: ``D`` is their
    maximum gap. The critical value at 5 % is ``1.36/√N`` (Pao, 1985), where
    **N is the number of OBSERVATIONS, the authors**, not the number of
    productivity levels. `observed` and `theoretical` are shares per level;
    `sample_size` carries N. Without it, the number of values passed is used,
    which only makes sense if each value is an observation.

    Fixed defect: N was the number of levels (often fewer than ten); the
    critical value then exceeded 0.4 and almost any corpus "followed Lotka".

    Caveat: the exponent of the FITTED law is estimated on the same data. The
    test is then conservative (it rejects less often than it should); the
    strict version, with the exponent fixed at 2 a priori, does not have this
    bias.

    The p-value comes from the Kolmogorov series, computed here directly: it
    avoids a dependency on scipy for a sum that converges in a few terms.
    """
    obs = np.asarray(observed, dtype=float)
    theo = np.asarray(theoretical, dtype=float)
    if obs.size == 0 or not np.isfinite(theo).all():
        return {"d": None, "p_value": None, "critical_5pct": None, "follows_lotka": None}
    n = int(sample_size) if sample_size else obs.size

    cdf_obs = np.cumsum(obs)
    cdf_theo = np.cumsum(theo)
    d = float(np.max(np.abs(cdf_obs - cdf_theo)))
    critical = 1.36 / np.sqrt(n)

    # Kolmogorov series: Q(t) = 2 Σ (-1)^{k-1} exp(-2k²t²).
    t = np.sqrt(n) * d
    if t <= 0:
        p = 1.0
    else:
        terms = [((-1) ** (k - 1)) * np.exp(-2.0 * (k ** 2) * (t ** 2))
                 for k in range(1, 101)]
        p = float(min(1.0, max(0.0, 2.0 * sum(terms))))

    return {
        "d": round(d, 4),
        "p_value": round(p, 4),
        "critical_5pct": round(float(critical), 4),
        # "Follows Lotka" = the gap is NOT significant. A test that does not reject
        # is not proof that the law holds; it is only the absence of proof to the
        # contrary, and it must be said that way.
        "follows_lotka": bool(d <= critical),
    }


# ---------------------------------------------------------------------------
# Bradford, scattering of sources
# ---------------------------------------------------------------------------

def bradford(corpus, zones: int = 3) -> Dict[str, object]:
    """Sources ranked by productivity, split into Bradford zones.

    Returns ``{"table": DataFrame, "zones": DataFrame, "multiplier": float}``.

    Each zone contains approximately the same NUMBER OF ARTICLES; it is the
    number of SOURCES per zone that grows, in principle geometrically. The
    "multiplier" is the mean ratio between the numbers of sources of two
    successive zones: it is what says whether the law holds.
    """
    d = corpus.documents
    d = d[d["source"].notna() & (d["source"].map(str).str.strip() != "")]
    empty = pd.DataFrame(columns=["rank", "source", "documents",
                                  "cumulative", "zone"])
    if d.empty:
        return {"table": empty,
                "zones": pd.DataFrame(columns=["zone", "sources", "documents"]),
                "multiplier": float("nan")}

    counts = (d.groupby("source")["eid"].nunique()
                .sort_values(ascending=False, kind="stable").reset_index(name="documents"))
    counts["rank"] = np.arange(1, len(counts) + 1)
    counts["cumulative"] = counts["documents"].cumsum()

    total = int(counts["documents"].sum())

    # Zone assignment. Subtlety: a source belongs to the zone it is FILLING,
    # not to the one it overflows into. Otherwise a very productive journal,
    # exceeding the first third on its own, would end up in zone 2, while it
    # IS the core.
    target = total / zones
    zone, acc, assigned = 1, 0, []
    for docs in counts["documents"].tolist():
        assigned.append(zone)
        acc += docs
        # `while`, not `if`: a journal productive enough to fill TWO zones at once
        # must move the counter forward by two. With `if`, the next journal fell
        # into a zone that was already full.
        while zone < zones and acc >= target * zone:
            zone += 1
    counts["zone"] = assigned

    summary = (counts.groupby("zone")
                     .agg(sources=("source", "count"),
                          documents=("documents", "sum"))
                     .reset_index())

    # Multiplier: mean ratio of the number of sources between successive zones.
    # Theoretically constant if the law applies.
    ratios = []
    s = summary["sources"].tolist()
    for i in range(1, len(s)):
        if s[i - 1] > 0:
            ratios.append(s[i] / s[i - 1])
    multiplier = float(np.mean(ratios)) if ratios else float("nan")

    table = counts[["rank", "source", "documents", "cumulative", "zone"]]
    return {"table": table, "zones": summary, "multiplier": round(multiplier, 2),
            "total_documents": total, "total_sources": int(len(counts))}


# ---------------------------------------------------------------------------
# Zipf, word frequency
# ---------------------------------------------------------------------------

def zipf(corpus, n: Optional[int] = 100, kind: str = "author") -> Dict[str, object]:
    """Frequency of keywords as a function of their rank.

    Returns ``{"table": DataFrame, "fit": {...}}``. The table carries the
    rank, the term, its observed frequency and the frequency **predicted** by
    the fit, so that the two curves can be overlaid and judged visually.
    """
    k = corpus.keywords
    if kind != "all":
        k = k[k["kind"] == kind]
    k = k[k["keyword"].notna()]
    empty = pd.DataFrame(columns=["rank", "keyword", "frequency", "predicted"])
    if k.empty:
        return {"table": empty, "fit": _fit_power_law(np.array([]), np.array([]))}

    k = k.copy()
    k["norm"] = k["keyword"].map(str).str.strip().str.lower()
    counts = (k.groupby("norm")
                .agg(keyword=("keyword", lambda s: s.mode().iat[0]),
                     frequency=("eid", "nunique"))
                .reset_index(drop=True)
                .sort_values("frequency", ascending=False, kind="stable")
                .reset_index(drop=True))
    counts["rank"] = np.arange(1, len(counts) + 1)

    # The fit covers the WHOLE distribution; the truncation at n only concerns
    # the display. Fitting on an extract would distort the exponent.
    fit = _fit_power_law(counts["rank"].to_numpy(dtype=float),
                         counts["frequency"].to_numpy(dtype=float))

    out = counts.head(n) if n else counts
    out = out.copy()
    if not np.isnan(fit["exponent"]):
        out["predicted"] = (fit["constant"]
                            * out["rank"] ** (-fit["exponent"])).round(2)
    else:
        out["predicted"] = np.nan

    return {"table": out[["rank", "keyword", "frequency", "predicted"]],
            "fit": fit, "total_keywords": int(len(counts))}
