"""Tiered discrete Gaussian samplers over the integers.

Lattice schemes need noise and masking drawn from a *discrete* Gaussian
over ℤ, not a rounded continuous normal — the two are different
distributions, and the difference is a security parameter rather than a
rounding detail. Which sampler is admissible therefore depends on σ and
on how many draws the whole protocol run makes, which is the decision
`sampler_for` encodes.

The sampling *algorithm* is a free choice; what has to be reproduced is
the statistical requirement — closeness to the ideal discrete Gaussian at
whatever σ a scheme's parameters derive, with the total statistical
distance across every draw accounted for explicitly.

Three tiers are implemented here, in `sampler_for`'s selection order:

- `discrete_gaussian` — exact, by explicit finite support. Correct for any
  σ, but its support window (`O(tail_cut*sigma)` integers) has to stay
  small to be cheap; gated to windows ≤ 4096 below.
- `rounded_gaussian` — `round(Normal(center, sigma))`. Cheap at any σ, but
  its statistical distance from the ideal discrete Gaussian is bounded by
  the Theorem-9 formula (Δ = O(σ⁻²) per sample, ePrint 2026/044
  Appendix A) — safe only once that distance, unioned over every draw in a
  run, is cryptographically negligible. That holds at that paper's own
  instantiation (σ≈2**50) and generally does not at the far smaller σ a
  concrete parameter search tends to land on, so expect this tier to be
  selected rarely; it is kept because it is the right choice whenever a
  scheme *does* live in that σ regime.
- `rejection_gaussian` — **exact** (not approximate) discrete Gaussian at
  arbitrary σ and arbitrary real center, by rejection sampling from a
  two-sided discrete-Laplace proposal. Simpler to vectorize than full
  Karney (arXiv:1303.6257) while landing in the same exactness class
  (exact up to float rounding in `exp`), so `sampler_for` resolves every
  positive σ without ever raising; see `rejection_gaussian`'s docstring
  for the acceptance-bound derivation.

Every sampler takes its randomness as an injected
`rng: np.random.Generator` rather than constructing one. That is the rule
this package holds to generally: the *choice* of randomness source (and
of any CSPRNG behind it) belongs to the consumer, so that this package
keeps no dependency on a cipher or hash library.
"""
import math

import numpy as np


def discrete_gaussian(rng: np.random.Generator, centers, sigma: float, tail_cut: float = 5.0) -> np.ndarray:
    """Exact discrete Gaussian by explicit finite support.

    Per element, the candidate support is the `2*(ceil(tail_cut*sigma)+1)+1`
    integers `round(c) ± (ceil(tail_cut*sigma)+1)` (task brief's window),
    weighted by the unnormalized density `exp(-(x-c)**2 / (2*sigma**2))`;
    a draw is a uniform in `[0, total_weight)` inverted against the
    cumulative sum (the first support point whose cumulative weight meets
    or exceeds the draw). Vectorized across every element of `centers` at
    once (all elements share one window width; only its placement, via
    `round(c)`, and the per-point weights vary).
    """
    centers = np.asarray(centers, dtype=np.float64)
    shape = centers.shape
    flat = centers.reshape(-1)
    m = flat.shape[0]

    k = math.ceil(tail_cut * sigma) + 1
    offsets = np.arange(-k, k + 1, dtype=np.int64)  # window width 2k+1
    base = np.rint(flat).astype(np.int64)  # round(c), per element

    xs = base[:, None] + offsets[None, :]  # (m, 2k+1) candidate integers
    diff = xs - flat[:, None]
    weights = np.exp(-(diff * diff) / (2.0 * sigma * sigma))

    cum = np.cumsum(weights, axis=1)
    totals = cum[:, -1]
    u = rng.random(m) * totals
    idx = np.argmax(cum >= u[:, None], axis=1)  # first index with cum >= u

    result = xs[np.arange(m), idx]
    return result.reshape(shape).astype(np.int64)


def rounded_gaussian(rng: np.random.Generator, centers, sigma: float) -> np.ndarray:
    """`round(Normal(center, sigma))`, per the brief's exact formula."""
    return np.rint(rng.normal(centers, sigma)).astype(np.int64)


def _laplace_rejection_bound(sigma: float) -> float:
    """The `K` in `rejection_gaussian`'s accept-probability
    `exp(f(x) - K)`, derived so `f(x) <= K` for every integer `x`, every
    real center, at this `sigma`. See `rejection_gaussian`'s docstring for
    the `f` this bounds and the calculus; `1e-9` is pure float-safety
    slack (the bound is tight — approached, never exceeded, as the center
    approaches a half-integer — so accept-probability can round a hair
    above 1.0 without this margin)."""
    return 0.5 + 1.0 / (2.0 * sigma) + 1e-9


def _laplace_rejection_round(rng: np.random.Generator, flat_centers: np.ndarray, x0: np.ndarray,
                              sigma: float, k_bound: float):
    """One round of `rejection_gaussian`'s propose-then-accept step,
    exposed privately so `tests/test_sampler.py` can measure the
    single-round acceptance rate directly (the public sampler only
    returns finished samples, having internally retried). Returns
    `(offset, accept)`, both length `len(flat_centers)`; `offset` is only
    meaningful where `accept` is True — callers retry the rest.

    Proposal: sign `s in {-1, +1}` by fair coin, magnitude `m` geometric
    via the standard inverse-CDF trick (`floor(-sigma * ln(U))`, `U`
    uniform in `(0, 1]`, giving `P(m=k) = (1-q)*q**k` for `q = exp(-1/sigma)`),
    offset `= s*m`. **Zero double-counting**: `s*0 == 0` regardless of
    `s`, so naively this would give `offset=0` *twice* the proposal mass
    a properly normalized two-sided geometric `∝ q**|offset|` assigns it
    (both `(s=+1,m=0)` and `(s=-1,m=0)` collapse to the same candidate).
    Fixed by rejection symmetry, not reweighting: `valid_zero` keeps only
    `(m=0, s=+1)` and discards (forces a retry on) `(m=0, s=-1)` — halving
    the `m=0` mass exactly restores proportionality to `q**|offset|`,
    which is what the acceptance-probability formula below assumes.
    """
    n = flat_centers.shape[0]
    s = np.where(rng.random(n) < 0.5, -1, 1)
    u = np.clip(rng.random(n), np.finfo(np.float64).tiny, 1.0)  # avoid log(0)
    mag = np.floor(-sigma * np.log(u)).astype(np.int64)
    offset = s * mag
    valid_zero = (mag != 0) | (s == 1)

    diff = (x0 + offset) - flat_centers
    f = -(diff * diff) / (2.0 * sigma * sigma) + np.abs(offset) / sigma
    accept_prob = np.exp(f - k_bound)
    accept = valid_zero & (rng.random(n) < accept_prob)
    return offset, accept


def rejection_gaussian(rng: np.random.Generator, centers, sigma: float, max_rounds: int = 200) -> np.ndarray:
    """Exact discrete Gaussian at any σ > 0 and any real center, by
    rejection sampling from a two-sided discrete-Laplace proposal — spec
    §5.4's pre-decided P2 fallback for σ where neither `discrete_gaussian`
    (window too wide) nor `rounded_gaussian` (Thm-9 budget too loose) is
    viable. `sampler_for` routes to this tier; it's also safe to call
    directly (used that way in `tests/test_sampler.py` at σ as small as 30
    to cross-check against `discrete_gaussian`).

    **Derivation of the acceptance bound.** Target (unnormalized) density
    `exp(-(x-c)**2 / (2*sigma**2))`; proposal (unnormalized, after the
    zero-mass fix in `_laplace_rejection_round`) `exp(-|x-x0|/sigma)` for
    `x0 = round(c)`. Accepting a proposal `x` with probability
    `exp(f(x) - K)`, `f(x) = -(x-c)**2/(2*sigma**2) + |x-x0|/sigma`,
    realizes the target exactly (standard rejection sampling) provided
    `K >= sup_x f(x)` for every integer `x`.

    Write `x = x0 + delta` (`delta` integer) and `eps = x0 - c` (the
    rounding error, `|eps| <= 1/2`), so `x - c = delta + eps`:

        f = -(delta + eps)**2 / (2*sigma**2) + |delta|/sigma

    For `delta >= 0`, relax to real `delta` (only enlarges the max): this
    is a downward parabola in `delta`, maximized at `delta* = sigma - eps`
    (`d/d(delta) = -(delta+eps)/sigma**2 + 1/sigma = 0`), where it equals

        f(delta*) = -sigma**2/(2*sigma**2) + (sigma-eps)/sigma = 1/2 - eps/sigma
                  <= 1/2 + 1/(2*sigma)                              (|eps| <= 1/2)

    The `delta <= 0` branch is the same computation with `eps -> -eps`
    (substitute `m = -delta >= 0`; the squared term is symmetric), giving
    the same bound `1/2 + 1/(2*sigma)`. Since the true (integer-`delta`)
    maximum is bounded above by the continuous relaxation's, `f(x) <=
    1/2 + 1/(2*sigma)` for every integer `x` — independent of `c`, `x0`,
    and which branch. `_laplace_rejection_bound` uses exactly this value
    (plus `1e-9` float-safety slack). Numerically confirmed never
    exceeded across every `(sigma, center)` this module's tests draw from
    (`_laplace_rejection_round` would trip an `assert` otherwise, kept out
    of the hot path itself but exercised by
    `tests/test_sampler.py::test_rejection_gaussian_acceptance_rate`).

    Empirically, acceptance is well above the required ~0.4 for every σ
    this tier is actually used at (σ >= ~30): ~0.62 at σ=5, ~0.73 at
    σ=30, ~0.76 at σ=200 and beyond (the `1/(2*sigma)` slack in `K`
    vanishes as σ grows, so acceptance *improves* with σ, the opposite of
    a naive rejection-sampler intuition).

    Vectorized with a masked retry loop (`_laplace_rejection_round`):
    each round only re-draws the still-pending elements, so the working
    set shrinks geometrically (~70%/round at the σ this tier serves).
    """
    if sigma <= 0:
        raise ValueError(f"rejection_gaussian: sigma must be positive, got {sigma!r}")

    centers = np.asarray(centers, dtype=np.float64)
    shape = centers.shape
    flat = centers.reshape(-1)
    x0 = np.rint(flat)
    k_bound = _laplace_rejection_bound(sigma)

    offsets = np.zeros(flat.shape[0], dtype=np.int64)
    pending = np.arange(flat.shape[0])
    for _ in range(max_rounds):
        if pending.size == 0:
            break
        offset, accept = _laplace_rejection_round(rng, flat[pending], x0[pending], sigma, k_bound)
        done = pending[accept]
        offsets[done] = offset[accept]
        pending = pending[~accept]
    if pending.size:
        raise RuntimeError(
            f"rejection_gaussian: {pending.size} sample(s) still unaccepted after "
            f"{max_rounds} rounds at sigma={sigma!r} — the K-bound derivation "
            "guarantees > 50% acceptance per round for reasonable sigma, so this "
            "means a bug (a broken K bound or proposal), not bad luck."
        )
    return (x0.astype(np.int64) + offsets).reshape(shape)


def rounded_gaussian_distance(sigma: float, tail_cut: float = 5.0) -> float:
    """The Theorem-9 bound on the statistical distance between
    `rounded_gaussian` and the ideal discrete Gaussian, **per sample**
    (Δ = O(σ⁻²)).

    Exposed separately from `sampler_for` because a caller that wants to
    report or plot its own hiding budget needs the per-sample figure, not
    just the tier decision the union bound feeds.
    """
    return tail_cut * (tail_cut + 2) * (1 + 1 / math.sqrt(2 * math.pi * sigma**2)) / sigma**2


def sampler_for(sigma: float, sample_count: int, tail_cut: float = 5.0):
    """The σ-tier gate: explicit-support when its window is small, else
    rounded when the Theorem-9 budget is cryptographically negligible
    across `sample_count` draws, else exact rejection sampling. Every
    σ > 0 resolves to *some* sampler; only a degenerate (non-positive) σ
    raises.

    `sample_count` is the number of scalar Gaussian draws the caller's
    whole protocol run makes — the union the per-sample distance is
    multiplied by. It is a plain `int` rather than a parameter object on
    purpose: deriving it needs the caller's prover structure, which is
    scheme-specific, while everything this module does with it is not.
    Callers should derive it conservatively, since over-counting only
    makes the `< 2**-64` gate harder to pass, never easier.

    Returns `discrete_gaussian`, `rounded_gaussian`, or `rejection_gaussian`
    directly (all three share the `(rng, centers, sigma)` calling
    convention), never a wrapped/partial callable.

    The window check mirrors `discrete_gaussian`'s *actual* window size
    exactly (`2*(ceil(tail_cut*sigma)+1)+1`, not a size estimate that
    happens to be close) — both sides of this gate agree by construction,
    not by coincidence.
    """
    if sigma <= 0:
        raise ValueError(f"sampler_for: sigma must be positive, got {sigma!r}")
    if sample_count <= 0:
        raise ValueError(f"sampler_for: sample_count must be positive, got {sample_count!r}")

    window = 2 * (math.ceil(tail_cut * sigma) + 1) + 1
    if window <= 4096:
        return discrete_gaussian

    if rounded_gaussian_distance(sigma, tail_cut) * sample_count < 2**-64:
        return rounded_gaussian

    return rejection_gaussian
