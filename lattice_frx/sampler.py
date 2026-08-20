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
keeps no dependency on a cipher or hash library. (frx and
zk-dtypes are the array layer, not primitives — see CLAUDE.md.)

**The byte-stream contract.** Fiat-Shamir consumers need samplers that
are a *deterministic function of transcript bytes*, which a `Generator`
cannot express. For those, this module's second family
(`uniform_from_bytes`, `fixed_weight_ternary`) takes the randomness as
an injected byte buffer instead — `bytes`, `bytearray`, or a `uint8`
ndarray — under one contract:

- The stream is consumed as consecutive **little-endian `uint64`
  chunks** (plus, where a sampler needs them, trailing raw bit/byte
  fields its docstring lays out).
- Consumption is **fixed ahead of time**: every rejection loop runs a
  budget computed from a stated failure probability rather than
  retrying open-endedly, so the total byte count is a function of the
  parameters alone. Each sampler's companion `*_bytes_needed` function
  is that count, and the sampler requires **exactly** that many bytes —
  a mismatch raises `ValueError` rather than silently reading a prefix,
  so two callers sharing one squeezed block cannot mis-slice quietly.
- Identical bytes yield identical output. Where the bytes come from — a
  XOF over a transcript, a CSPRNG, a test vector — is the consumer's
  choice, exactly as `rng` is for the Gaussian tier.
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


_TWO_64 = 1 << 64


def _require_byte_stream(data, needed: int, caller: str) -> np.ndarray:
    """Validate and normalize an injected byte stream to a `uint8` array of
    exactly `needed` bytes. The two failure modes stay distinct on purpose
    (mirroring `canonical.py`): a wrong *kind* of buffer is a `TypeError`,
    a wrong *length* is a `ValueError` — they are different caller bugs."""
    if isinstance(data, (bytes, bytearray)):
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
    elif isinstance(data, np.ndarray):
        if data.dtype != np.uint8:
            raise TypeError(f"{caller}: byte stream array must be uint8, got {data.dtype}")
        buf = np.ascontiguousarray(data)
    else:
        raise TypeError(
            f"{caller}: byte stream must be bytes, bytearray, or a uint8 ndarray, "
            f"got {type(data).__name__}"
        )
    if buf.size != needed:
        raise ValueError(
            f"{caller}: expected exactly {needed} bytes for these parameters "
            f"(the *_bytes_needed count), got {buf.size}"
        )
    return buf


def _log_binom_sf(attempts: int, allowed: int, p_rej: float) -> float:
    """`log P(Binomial(attempts, p_rej) > allowed)` — the probability that a
    fixed-budget rejection pass fails, i.e. that more than `allowed` of the
    `attempts` draws land in the rejection region.

    Evaluated as the tail sum of exact binomial terms in log space
    (`lgamma`; ~1e-12 relative precision, against decision thresholds like
    2**-128 where the neighboring budgets differ by orders of magnitude —
    the independent scipy cross-check lives in `sampler_test.py`). Two
    shortcuts keep the search cheap without giving up soundness:

    - `allowed + 1 <= floor(attempts * p_rej)` returns `log(1/2)` outright:
      a binomial's median is at least `floor(mean)`, so the true tail is
      >= 1/2 — far above any cryptographic failure target, which is the
      only question the caller asks in that regime.
    - The sum stops once past the mode with the running term 60 nats below
      the accumulated total; the remaining terms are decreasing, so the
      truncation error is negligible at the thresholds compared against.
    """
    if allowed >= attempts:
        return -math.inf
    if allowed + 1 <= math.floor(attempts * p_rej):
        return math.log(0.5)
    log_p, log_1p = math.log(p_rej), math.log1p(-p_rej)
    lg_n = math.lgamma(attempts + 1)
    acc = -math.inf
    for k in range(allowed + 1, attempts + 1):
        term = (lg_n - math.lgamma(k + 1) - math.lgamma(attempts - k + 1)
                + k * log_p + (attempts - k) * log_1p)
        acc = float(np.logaddexp(acc, term))
        if k > attempts * p_rej and term < acc - 60.0:
            break
    return acc


def _rejection_slack(count: int, p_rej: float, fail_prob: float) -> int:
    """The minimal `slack` such that `count + slack` draws, each rejected
    independently with probability `p_rej`, yield at least `count` accepted
    ones except with probability <= `fail_prob` — i.e. the smallest budget
    with `P(Bin(count + slack, p_rej) > slack) <= fail_prob`. That tail is
    non-increasing in `slack` (adding a draw adds at most one rejection but
    also one more allowance), so binary search applies."""
    log_eps = math.log(fail_prob)

    def ok(slack: int) -> bool:
        return _log_binom_sf(count + slack, slack, p_rej) <= log_eps

    if ok(0):
        return 0
    lo, hi = 0, 1
    while not ok(hi):
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _validate_uniform_params(count: int, modulus: int, fail_prob: float) -> None:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count!r}")
    if not 1 <= modulus < _TWO_64:
        raise ValueError(f"modulus must be in [1, 2**64), got {modulus!r}")
    if not 0.0 < fail_prob < 1.0:
        raise ValueError(f"fail_prob must be in (0, 1), got {fail_prob!r}")


def uniform_bytes_needed(count: int, modulus: int, fail_prob: float = 2.0**-128) -> int:
    """The exact byte count `uniform_from_bytes` consumes for these
    parameters — `8 * (count + slack)`, where the slack is the minimal
    rejection budget at failure probability `fail_prob` (see
    `_rejection_slack`). Zero whenever `modulus` divides `2**64`, since no
    chunk is ever rejected then."""
    _validate_uniform_params(count, modulus, fail_prob)
    largest_multiple = _TWO_64 // modulus * modulus
    if largest_multiple == _TWO_64:
        return 8 * count
    p_rej = (_TWO_64 - largest_multiple) / _TWO_64
    return 8 * (count + _rejection_slack(count, p_rej, fail_prob))


def uniform_from_bytes(data, modulus: int, count: int, fail_prob: float = 2.0**-128) -> np.ndarray:
    """`count` independent uniform draws from `[0, modulus)` as a
    deterministic function of the injected byte stream (see the module
    docstring for the stream contract).

    Each little-endian `uint64` chunk `c` is **accepted** iff it lies below
    the largest multiple of `modulus` not exceeding `2**64`, and an
    accepted chunk contributes `c % modulus` — exactly uniform, with no
    modulo bias (the accepted region is a whole number of congruence
    classes). The first `count` accepted chunks, in stream order, are the
    result. The stream length is fixed at `uniform_bytes_needed(...)`:
    rejection is paid for with a precomputed budget rather than an
    open-ended retry, so on an honestly random stream the sampler fails
    (raises) with probability at most `fail_prob`, and never reads a
    data-dependent number of bytes."""
    needed = uniform_bytes_needed(count, modulus, fail_prob)  # validates params
    buf = _require_byte_stream(data, needed, "uniform_from_bytes")
    chunks = buf.view(np.dtype("<u8"))
    largest_multiple = _TWO_64 // modulus * modulus
    if largest_multiple == _TWO_64:
        accepted = chunks
    else:
        accepted = chunks[chunks < np.uint64(largest_multiple)]
    if accepted.size < count:
        raise RuntimeError(
            f"uniform_from_bytes: only {accepted.size} of the required {count} "
            f"draws were accepted from the stream — on honestly random bytes "
            f"this has probability <= {fail_prob!r}, so suspect the stream "
            f"(or a caller's slicing), not bad luck."
        )
    return accepted[:count] % np.uint64(modulus)
