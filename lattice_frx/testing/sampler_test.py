"""Tests for `lattice_frx.sampler`: the tiered Gaussian samplers and the
byte-stream samplers (`uniform_from_bytes`, `fixed_weight_ternary`).

`sampler_for`'s tier gate is exercised against a *real* concrete
parameter set rather than round numbers — the six σ and the draw count
below come from the Jindo lattice PCS (ePrint 2026/044) at its
`n=1024, batch=1` point, inlined as constants so this package keeps no
dependency on a consumer's parameter search.

They are worth pinning because they span the whole gate: two σ are small
enough for exact explicit support, and the other four sit far below the
σ≈2**50 regime where the rounded tier's Theorem-9 budget could clear
`2**-64`, so they fall through to exact rejection sampling. A parameter
set landing in the rounded tier is the case not covered here, and
`test_rounded_tier_is_selected_in_the_large_sigma_regime` covers it
synthetically.
"""
import math

import numpy as np
import pytest
from scipy import stats

from lattice_frx import sampler

# Jindo (ePrint 2026/044) at n=1024, batch=1 -- see the module docstring.
SMALL_SIGMA = {
    "ecd_std_dev": 4.787466224214409,
    "mlwe_std_dev": 6.770275002573077,
}
WIDE_SIGMA = {
    "ecd_blind_std_dev": 1154200.6570634034,
    "mask_std_dev": 1733.2479139039056,
    "mask_blind_std_dev": 417865273.06726474,
    "mask_mlwe_std_dev": 2451.1013707864035,
}
# The conservative per-proof scalar-draw count that parameter set derives.
SAMPLE_COUNT = 1089792


def _bin_counts(samples: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, int]:
    """Histogram `samples` over the closed integer range `[lo, hi]`.
    Returns `(counts, n_outside)` — `counts[x - lo]` for `x` in range, and
    the count of samples that fell outside it (should be ~0 whenever
    `[lo, hi]` is a wide-enough tail cut)."""
    vals, cnts = np.unique(samples, return_counts=True)
    counts = np.zeros(hi - lo + 1, dtype=np.float64)
    in_range = (vals >= lo) & (vals <= hi)
    counts[(vals[in_range] - lo).astype(np.int64)] = cnts[in_range]
    return counts, int(cnts[~in_range].sum())


def _chi_square_vs_true_gaussian(samples: np.ndarray, center: float, sigma: float, tail_cut: float = 8.0):
    """χ² goodness-of-fit of `samples` against the (tail-truncated) true
    discrete Gaussian density at `(center, sigma)`. `tail_cut=8` truncates
    at a window wide enough (~1e-14 tail mass) that no real sample should
    ever land outside it; bins with expected count < 5 are merged away
    (scipy's own rule of thumb) so the statistic isn't dominated by noise
    in bins nobody was ever going to land in."""
    n = samples.shape[0]
    k = math.ceil(tail_cut * sigma) + 1
    lo, hi = int(round(center)) - k, int(round(center)) + k

    xs = np.arange(lo, hi + 1)
    weights = np.exp(-((xs - center) ** 2) / (2.0 * sigma * sigma))
    expected = weights / weights.sum() * n

    observed, outside = _bin_counts(samples, lo, hi)
    assert outside == 0, f"{outside} sample(s) fell outside the {tail_cut}-sigma reference window"

    keep = expected >= 5.0
    observed, expected = observed[keep], expected[keep]
    expected *= observed.sum() / expected.sum()  # renormalize after trimming
    return stats.chisquare(observed, expected)


def test_discrete_gaussian_window_matches_brief_formula():
    # "round(c) ± ceil(tail_cut*sigma)+1" (task-10-brief.md) => a window of
    # 2*(ceil(tail_cut*sigma)+1)+1 candidate integers per center.
    rng = np.random.default_rng(0)
    sigma, tail_cut = 5.0, 5.0
    k = math.ceil(tail_cut * sigma) + 1
    samples = sampler.discrete_gaussian(rng, np.array([0.0]), sigma, tail_cut=tail_cut)
    assert samples.shape == (1,)
    # A single draw always lands inside the explicit window.
    assert abs(int(samples[0])) <= k


@pytest.mark.parametrize("center", [0.0, 0.37])
def test_discrete_gaussian_chi_square(center):
    rng = np.random.default_rng(1234)
    sigma, n = 5.0, 200_000
    samples = sampler.discrete_gaussian(rng, np.full(n, center), sigma)
    chi2 = _chi_square_vs_true_gaussian(samples, center, sigma)
    assert chi2.pvalue > 1e-6


@pytest.mark.parametrize("sigma", [30.0, 200.0])
@pytest.mark.parametrize("center", [0.0, 0.37])
def test_rejection_gaussian_chi_square(sigma, center):
    rng = np.random.default_rng(int(sigma) * 1000 + 7)
    n = 200_000
    samples = sampler.rejection_gaussian(rng, np.full(n, center), sigma)
    chi2 = _chi_square_vs_true_gaussian(samples, center, sigma)
    assert chi2.pvalue > 1e-6


@pytest.mark.parametrize("center", [0.0, 0.37])
def test_rejection_gaussian_cross_checks_against_discrete_gaussian(center):
    # Two independently-implemented exact samplers (finite-support inversion
    # vs. Laplace-proposal rejection) agreeing on their sample distribution
    # is the strongest fidelity check available without a closed-form
    # reference -- run at sigma=30, where both tiers are actually usable
    # (discrete_gaussian's window is 2*151+1=303, well under the 4096 cap).
    #
    # This compares two *samples* to each other, not a sample against a
    # fixed/exact model (unlike `_chi_square_vs_true_gaussian`), so
    # `stats.chisquare` is the wrong tool: it treats `f_exp` as exact,
    # zero-variance probabilities, but a same-size random sample used as
    # `f_exp` carries its own sampling variance, understating the null's
    # true variance and biasing toward spuriously tiny p-values.
    # `chi2_contingency`'s homogeneity test on a 2xB contingency table
    # accounts for both samples' variance, the correct form for "do these
    # two independent samples come from the same distribution".
    sigma, n = 30.0, 200_000
    explicit = sampler.discrete_gaussian(np.random.default_rng(21), np.full(n, center), sigma)
    rejection = sampler.rejection_gaussian(np.random.default_rng(22), np.full(n, center), sigma)

    tail_cut = 8.0
    k = math.ceil(tail_cut * sigma) + 1
    lo, hi = int(round(center)) - k, int(round(center)) + k
    exp_counts, exp_outside = _bin_counts(explicit, lo, hi)
    rej_counts, rej_outside = _bin_counts(rejection, lo, hi)
    assert exp_outside == 0 and rej_outside == 0

    keep = exp_counts >= 5.0
    table = np.vstack([exp_counts[keep], rej_counts[keep]])
    result = stats.chi2_contingency(table)
    assert result.pvalue > 1e-6


def test_rejection_gaussian_mean_and_variance_sanity_at_large_sigma():
    rng = np.random.default_rng(43)
    sigma, n, center = 4.18e8, 100_000, 12345.6789
    samples = sampler.rejection_gaussian(rng, np.full(n, center), sigma)
    assert samples.shape == (n,)

    mean = samples.mean()
    var = samples.astype(np.float64).var()
    assert abs(mean - center) < 5 * sigma / math.sqrt(n)  # standard error of the mean
    assert abs(var - sigma * sigma) / (sigma * sigma) < 0.05


@pytest.mark.parametrize("sigma", [30.0, 200.0, 4.18e8])
def test_rejection_gaussian_acceptance_rate(sigma):
    # A single (non-retried) round of the propose/accept step, so this
    # measures the same per-round acceptance rate the K-bound derivation
    # in rejection_gaussian's docstring claims is comfortably >= ~0.5 for
    # these sigma (this is the module-private seam mentioned in encoder.py
    # -- direct access is deliberate here, to inspect the mechanism the
    # public retry loop hides).
    rng = np.random.default_rng(int(sigma) % 997 + 1)
    centers = rng.uniform(-1.0, 1.0, 50_000)
    x0 = np.rint(centers)
    k_bound = sampler._laplace_rejection_bound(sigma)
    _, accept = sampler._laplace_rejection_round(rng, centers, x0, sigma, k_bound)
    assert accept.mean() >= 0.4, accept.mean()


def test_rounded_gaussian_mean_and_variance_sanity():
    rng = np.random.default_rng(42)
    sigma, n = 2.0**20, 200_000
    samples = sampler.rounded_gaussian(rng, np.zeros(n), sigma)
    assert samples.shape == (n,)

    mean = samples.mean()
    var = samples.astype(np.float64).var()
    # Rounding to the nearest integer perturbs mean/variance by O(1),
    # utterly negligible next to sigma=2**20 ~ 1e6.
    assert abs(mean) < 5 * sigma / math.sqrt(n)  # standard error of the mean
    assert abs(var - sigma * sigma) / (sigma * sigma) < 0.05


def test_sampler_for_small_sigma_tiers_use_explicit_support():
    for name, sigma in SMALL_SIGMA.items():
        fn = sampler.sampler_for(sigma, SAMPLE_COUNT)
        assert fn is sampler.discrete_gaussian, name


def test_sampler_for_window_gate_matches_discrete_gaussian_window():
    # The gate's window estimate and discrete_gaussian's actual allocated
    # window must now agree exactly (fix round 1, item 3) -- not just be
    # "close" -- so this pins the formula on both sides together instead
    # of just re-deriving sampler_for's copy in isolation.
    tail_cut = 5.0
    for sigma in (4.79, 6.77, 100.0, 4095.0):
        k = math.ceil(tail_cut * sigma) + 1
        gate_window = 2 * (math.ceil(tail_cut * sigma) + 1) + 1
        actual_window = 2 * k + 1
        assert gate_window == actual_window


def test_sampler_for_resolves_every_concrete_sigma_without_raising():
    expected_tier = {
        **{name: sampler.discrete_gaussian for name in SMALL_SIGMA},
        **{name: sampler.rejection_gaussian for name in WIDE_SIGMA},
    }
    for name, sigma in {**SMALL_SIGMA, **WIDE_SIGMA}.items():
        got = sampler.sampler_for(sigma, SAMPLE_COUNT)
        assert got is expected_tier[name], (name, sigma, got)


def test_rounded_tier_is_selected_in_the_large_sigma_regime():
    """The tier no concrete parameter set above reaches: sigma wide enough
    that the per-sample Theorem-9 distance, unioned over every draw, is
    still below 2**-64."""
    sigma = 2.0**50
    assert sampler.rounded_gaussian_distance(sigma) * SAMPLE_COUNT < 2**-64
    assert sampler.sampler_for(sigma, SAMPLE_COUNT) is sampler.rounded_gaussian


def test_a_large_enough_draw_count_pushes_the_rounded_tier_to_rejection():
    """The union bound is what the count feeds, so raising it alone must be
    able to disqualify a sigma the smaller count admitted."""
    sigma = 2.0**50
    per_sample = sampler.rounded_gaussian_distance(sigma)
    too_many = int(2**-64 / per_sample) + 1
    assert sampler.sampler_for(sigma, too_many) is sampler.rejection_gaussian


@pytest.mark.parametrize("bad_count", [0, -1])
def test_sampler_for_rejects_a_non_positive_sample_count(bad_count):
    with pytest.raises(ValueError, match="sample_count"):
        sampler.sampler_for(100.0, bad_count)


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_sampler_for_raises_only_for_degenerate_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        sampler.sampler_for(sigma, SAMPLE_COUNT)


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_rejection_gaussian_raises_for_degenerate_sigma(sigma):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="rejection_gaussian"):
        sampler.rejection_gaussian(rng, np.array([0.0]), sigma)


# --- byte-stream samplers ---------------------------------------------------
#
# The tests below stand in for a consumer's XOF with deterministic
# pseudo-random bytes: the byte-stream samplers' contract is exactly that the
# *source* of the bytes is not their business.


def _stream(n_bytes: int, seed: int) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n_bytes, dtype=np.uint8).tobytes()


def test_uniform_from_bytes_is_a_deterministic_function_of_the_bytes():
    q, count = (1 << 50) - 27, 1000
    data = _stream(sampler.uniform_bytes_needed(q, count), seed=7)
    a = sampler.uniform_from_bytes(data, q, count)
    b = sampler.uniform_from_bytes(data, q, count)
    assert a.dtype == np.uint64 and a.shape == (count,)
    np.testing.assert_array_equal(a, b)
    assert (a < q).all()
    # A uint8 ndarray carrying the same bytes is the same stream.
    as_array = np.frombuffer(data, dtype=np.uint8)
    np.testing.assert_array_equal(sampler.uniform_from_bytes(as_array, q, count), a)


def test_uniform_from_bytes_chi_square_at_a_small_modulus():
    q, n = 17, 200_000
    data = _stream(sampler.uniform_bytes_needed(q, n), seed=11)
    samples = sampler.uniform_from_bytes(data, q, n)
    counts = np.bincount(samples.astype(np.int64), minlength=q)
    assert stats.chisquare(counts).pvalue > 1e-6


def test_uniform_from_bytes_ks_where_rejection_actually_bites():
    # A modulus just above 2**63 pushes the rejection probability to ~1/2
    # (the largest multiple of q below 2**64 is q itself), so this exercises
    # the accept/reject path and its budget, not just the modular map.
    q, n = (1 << 63) + 11, 100_000
    needed = sampler.uniform_bytes_needed(q, n)
    assert needed > 8 * n  # rejection visibly inflated the budget
    samples = sampler.uniform_from_bytes(_stream(needed, 13), q, n)
    ks = stats.kstest(samples / q, "uniform")
    assert ks.pvalue > 1e-6


def test_uniform_bytes_needed_budget_is_minimal_for_the_stated_fail_prob():
    # Independent cross-check of the budget computation: scipy's binomial
    # survival function, against the module's own tail evaluation.
    q, count, fail_prob = (1 << 63) + 11, 1000, 2.0**-128
    needed = sampler.uniform_bytes_needed(q, count, fail_prob)
    assert needed % 8 == 0
    attempts = needed // 8
    p_rej = 1.0 - ((1 << 64) // q) * q / 2.0**64
    assert stats.binom.sf(attempts - count, attempts, p_rej) <= fail_prob
    assert stats.binom.sf(attempts - 1 - count, attempts - 1, p_rej) > fail_prob


def test_uniform_bytes_needed_has_no_slack_when_the_modulus_divides_2_64():
    for q in (1, 1 << 32, 1 << 50):
        assert sampler.uniform_bytes_needed(q, 64) == 8 * 64


def test_uniform_from_bytes_chunks_are_little_endian_uint64():
    # With a power-of-two modulus nothing is ever rejected, so the output is
    # exactly chunk % q chunk-by-chunk — which pins the `<u8` layout.
    data = _stream(8 * 64, 5)
    chunks = np.frombuffer(data, dtype="<u8")
    np.testing.assert_array_equal(
        sampler.uniform_from_bytes(data, 1 << 32, 64),
        chunks % np.uint64(1 << 32),
    )
    assert (sampler.uniform_from_bytes(data, 1, 64) == 0).all()


def test_uniform_from_bytes_requires_the_exact_byte_count():
    q, count = 17, 100
    needed = sampler.uniform_bytes_needed(q, count)
    for n_bytes in (needed - 8, needed + 8):
        with pytest.raises(ValueError, match="bytes"):
            sampler.uniform_from_bytes(_stream(n_bytes, 3), q, count)


def test_uniform_from_bytes_raises_when_every_chunk_is_rejected():
    # All-0xff chunks equal 2**64 - 1, which lies in the rejection region of
    # any modulus that does not divide 2**64 — so the whole budget drains and
    # the sampler must report it (probability <= fail_prob on honest bytes).
    q, count = (1 << 63) + 11, 16
    needed = sampler.uniform_bytes_needed(q, count)
    with pytest.raises(RuntimeError, match="uniform_from_bytes"):
        sampler.uniform_from_bytes(b"\xff" * needed, q, count)


@pytest.mark.parametrize("bad_modulus", [0, 1 << 64])
def test_uniform_bytes_needed_rejects_a_bad_modulus(bad_modulus):
    with pytest.raises(ValueError, match="modulus"):
        sampler.uniform_bytes_needed(bad_modulus, 4)


@pytest.mark.parametrize("bad_count", [0, -1])
def test_uniform_bytes_needed_rejects_a_non_positive_count(bad_count):
    with pytest.raises(ValueError, match="count"):
        sampler.uniform_bytes_needed(17, bad_count)


def test_uniform_from_bytes_rejects_a_non_byte_buffer():
    q, count = 17, 4
    needed = sampler.uniform_bytes_needed(q, count)
    with pytest.raises(TypeError, match="uint8"):
        sampler.uniform_from_bytes(np.zeros(needed, dtype=np.uint32), q, count)


def test_fixed_weight_ternary_weight_support_and_determinism():
    weight, degree = 39, 64
    data = _stream(sampler.fixed_weight_ternary_bytes_needed(weight, degree), seed=17)
    c = sampler.fixed_weight_ternary(data, weight, degree)
    assert c.dtype == np.int64 and c.shape == (degree,)
    assert int(np.count_nonzero(c)) == weight
    assert set(np.unique(c)).issubset({-1, 0, 1})
    np.testing.assert_array_equal(sampler.fixed_weight_ternary(data, weight, degree), c)


def test_fixed_weight_ternary_position_marginals_chi_square():
    # Each draw's support is a weight-subset of the positions, so per-position
    # counts over n draws have mean n*weight/degree with *negative* cross-
    # position correlation — Pearson's statistic against the multinomial
    # reference is conservative here (its true variance is smaller), which is
    # the safe direction for a p > 1e-6 gate.
    weight, degree, n = 13, 32, 20_000
    needed = sampler.fixed_weight_ternary_bytes_needed(weight, degree)
    counts = np.zeros(degree, dtype=np.int64)
    for k in range(n):
        counts += sampler.fixed_weight_ternary(_stream(needed, 100_000 + k), weight, degree) != 0
    assert counts.sum() == n * weight
    assert stats.chisquare(counts).pvalue > 1e-6


def test_fixed_weight_ternary_sign_balance_and_position_independence():
    weight, degree, n = 13, 32, 20_000
    needed = sampler.fixed_weight_ternary_bytes_needed(weight, degree)
    plus = np.zeros(degree, dtype=np.int64)
    minus = np.zeros(degree, dtype=np.int64)
    for k in range(n):
        c = sampler.fixed_weight_ternary(_stream(needed, 200_000 + k), weight, degree)
        plus += c == 1
        minus += c == -1
    assert stats.binomtest(int(plus.sum()), int(plus.sum() + minus.sum()), 0.5).pvalue > 1e-6
    # Sign must not depend on where the coefficient landed.
    assert stats.chi2_contingency(np.vstack([plus, minus])).pvalue > 1e-6


def test_fixed_weight_ternary_bytes_needed_budget_is_minimal():
    # Independent recomputation of the per-position rejection probabilities
    # and the union bound the round budget is derived from.
    from fractions import Fraction

    weight, degree, fail_prob = 39, 64, 2.0**-128

    def union_bound(rounds: int) -> Fraction:
        total = Fraction(0)
        for m in range(degree - weight + 1, degree + 1):
            largest_multiple = (1 << 64) // m * m
            total += Fraction((1 << 64) - largest_multiple, 1 << 64) ** rounds
        return total

    needed = sampler.fixed_weight_ternary_bytes_needed(weight, degree, fail_prob)
    sign_bytes = (weight + 7) // 8
    assert (needed - sign_bytes) % (8 * weight) == 0
    rounds = (needed - sign_bytes) // (8 * weight)
    assert union_bound(rounds) <= Fraction(fail_prob)
    assert rounds == 1 or union_bound(rounds - 1) > Fraction(fail_prob)


def test_fixed_weight_ternary_needs_one_round_when_no_position_can_reject():
    # degree=8, weight=1 puts the single Fisher-Yates draw at modulus 8,
    # which divides 2**64 — no rejection region, so exactly one chunk plus
    # one sign byte.
    assert sampler.fixed_weight_ternary_bytes_needed(1, 8) == 8 + 1


def test_fixed_weight_ternary_signs_live_in_the_trailing_bytes():
    # Flipping only the sign bytes must preserve the support and flip signs —
    # this pins the stream layout (position chunks first, sign bits last).
    weight, degree = 5, 16
    needed = sampler.fixed_weight_ternary_bytes_needed(weight, degree)
    sign_bytes = (weight + 7) // 8
    data = _stream(needed, 23)
    flipped = data[:-sign_bytes] + bytes(b ^ 0xFF for b in data[-sign_bytes:])
    a = sampler.fixed_weight_ternary(data, weight, degree)
    b = sampler.fixed_weight_ternary(flipped, weight, degree)
    np.testing.assert_array_equal(a != 0, b != 0)
    np.testing.assert_array_equal(a[a != 0], -b[b != 0])


def test_fixed_weight_ternary_requires_the_exact_byte_count():
    weight, degree = 5, 16
    needed = sampler.fixed_weight_ternary_bytes_needed(weight, degree)
    for n_bytes in (needed - 1, needed + 1):
        with pytest.raises(ValueError, match="bytes"):
            sampler.fixed_weight_ternary(_stream(n_bytes, 3), weight, degree)


def test_fixed_weight_ternary_raises_when_a_position_rejects_its_whole_budget():
    # weight=2, degree=10 puts the draws at moduli 9 and 10, neither of which
    # divides 2**64 — so all-0xff chunks land in every rejection region.
    weight, degree = 2, 10
    needed = sampler.fixed_weight_ternary_bytes_needed(weight, degree)
    with pytest.raises(RuntimeError, match="fixed_weight_ternary"):
        sampler.fixed_weight_ternary(b"\xff" * needed, weight, degree)


@pytest.mark.parametrize("weight,degree", [(0, 8), (-1, 8), (9, 8)])
def test_fixed_weight_ternary_rejects_a_bad_weight(weight, degree):
    with pytest.raises(ValueError, match="weight"):
        sampler.fixed_weight_ternary_bytes_needed(weight, degree)
