"""The host-side integer structure both rings need identically.

A primitive root of `q`, the bit-reversal its twiddle table is indexed by,
and the index maps of the ring's own symmetries (`galois_map`,
`slot_exponents`) — the host ring fills lattigo's tables and loops with
them, the traced one derives the opcode's generator, its ordering adapter,
and its gather tables from them. None of it is about a limb or an array:
host-side integer functions, computed once per parameterisation, living
below both rings rather than inside either.

`primitive_root` is lattigo's `PrimitiveRoot` (subring.go): the smallest `g`
with `g^((q-1)/f) != 1 mod q` for every prime factor `f` of `q-1`. Its
factorization is reimplemented rather than ported — only the resulting *set* of
unique prime factors feeds the search, so any correct factorization agrees on
it.
"""

import functools
import math
import random

import numpy as np

from lattice_frx.primes import is_prime as _is_prime


def _pollard_rho_factor(n: int, rng: random.Random) -> int:
    """A nontrivial factor of composite n (Pollard's rho, Floyd's cycle
    detection, retried with a fresh polynomial constant on failure — the
    same retry-on-failure shape as lattigo's `GetFactorPollardRho`, which
    notes "will miss some small prime factors" on any single attempt)."""
    if n % 2 == 0:
        return 2
    while True:
        c = rng.randrange(1, n - 1)
        x = y = rng.randrange(2, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def prime_factors(n: int) -> set[int]:
    """The set of unique prime factors of n: trial division for small
    factors, then Pollard's rho (recursively, until every remaining
    cofactor is prime) for the rest. Only the resulting *set* matters to
    `primitive_root` (see module docstring)."""
    rng = random.Random(n)  # deterministic across calls for the same n
    factors: set[int] = set()

    m = n
    p = 2
    while p * p <= m and p < 1 << 20:
        while m % p == 0:
            factors.add(p)
            m //= p
        p = 3 if p == 2 else p + 2

    stack = [m] if m > 1 else []
    while stack:
        k = stack.pop()
        if k == 1:
            continue
        if _is_prime(k):
            factors.add(k)
            continue
        f = _pollard_rho_factor(k, rng)
        stack.append(f)
        stack.append(k // f)

    return factors


def primitive_root(q: int, factors: set[int]) -> int:
    """Port of lattigo `PrimitiveRoot` (subring.go): the smallest g such
    that, for every prime factor f of q-1, g^((q-1)/f) != 1 mod q. `g`
    starts at 2 but is pre-incremented before the first test, so 3 is the
    first candidate tried."""
    g = 2
    while True:
        g += 1
        if all(pow(g, (q - 1) // f, q) != 1 for f in factors):
            return g


def bit_reverse(x: int, bits: int) -> int:
    """Port of lattigo `utils.BitReverse64`: reverse the low `bits` bits
    of x."""
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def normalize_galois_k(d: int, k: int) -> int:
    """`k` reduced mod `2d`, required odd. `gcd(k, 2d) = 1` is what makes
    `σ_k` an automorphism, and `2d` is a power of two, so odd is the whole
    condition; the even case is a projection, not a ring map, and raising
    beats silently folding it."""
    k = int(k) % (2 * d)
    if k % 2 == 0:
        raise ValueError(f"galois: k must be odd mod {2 * d}, got {k}")
    return k


def galois_map(d: int, k: int) -> tuple[list[int], list[bool]]:
    """The coefficient-index action of `σ_k : X ↦ X^k` on `Z[X]/(X^d + 1)`.

    For source index `i`, coefficient `a_i` lands at `i·k mod 2d` folded
    into `[0, d)`, negated (`negate[i]`) when the fold crossed `d` —
    `X^d = -1`. Host-side integers like everything else here: both rings
    build their own gather/sign structures from this one map.
    """
    k = normalize_galois_k(d, k)
    dest, negate = [], []
    for i in range(d):
        j = (i * k) % (2 * d)
        dest.append(j - d if j >= d else j)
        negate.append(j >= d)
    return dest, negate


@functools.lru_cache(maxsize=None)
def galois_gather_table(d: int, k: int):
    """`σ_k` as a gather: output index `m` reads source `src[m]`, negated
    where `negate[m]`.

    `galois_map`'s forward action inverted by two scatters. Both traced rings
    apply `σ_k` this way — one `take` and one sign `where` per limb — so the
    inversion lives here, beside the map it inverts, rather than once per
    ring. One bool mask serves every limb; the sign is a `where`, not per-limb
    sign arrays.

    `k` is not normalised here: `galois_map` does it on entry, so a second
    call could not change the answer — and it is what raises on an even `k`,
    which is a projection rather than a ring map.

    Cached because it is a pure function of `(d, k)` and every call rebuilds
    the same two arrays. Two `k` that differ by `2d` are separate cache keys
    holding equal tables, which costs a little memory and keeps the key the
    caller's own value. Host arrays only — a traced value must never be
    cached, since whichever call fills the cache decides what lands in it.
    """
    dest, negate = galois_map(d, k)
    src = np.empty(d, dtype=np.int32)
    src[dest] = np.arange(d, dtype=np.int32)
    negate_at_dest = np.zeros(d, dtype=bool)
    negate_at_dest[dest] = negate
    src.flags.writeable = False
    negate_at_dest.flags.writeable = False
    return src, negate_at_dest


def slot_exponents(d: int) -> list[int]:
    """`e(j)` for every NTT slot of the contract's (lattigo's) order: slot
    `j` of a forward transform holds the evaluation at `ψ^{e(j)}`, with
    `e(j) = 2·brv(j) + 1`.

    The formula is the bit-reversed enumeration of the odd exponents —
    exactly the walk the twiddle table takes — and it is not trusted on
    inspection: `ring_test` re-derives the table numerically (the monomial
    `X` through the host ring, discrete logs base `ψ`) and requires
    equality. Modulus-free on purpose: the order is index-structural, which
    is why one table serves every limb.
    """
    bits = (d - 1).bit_length()
    return [2 * bit_reverse(j, bits) + 1 for j in range(d)]


def split_root(q: int) -> int:
    """The split constant of a partial-split modulus: the canonical (smaller)
    square root of `-1` modulo q, so `X^d + 1 ≡ (X^{d/2} - r)(X^{d/2} + r)`.

    The partial-split sibling of `primitive_root`: a per-modulus ring
    constant computed once, below both the traced and host split rings
    rather than inside either. Only defined at primes
    `q ≡ 5 (mod 8)`. The two failure modes carry distinct messages on
    purpose: a non-prime is a caller bug anywhere, while `q ≡ 1 (mod 8)`
    usually means an NTT-friendly limb strayed into the partial-split
    mode — the two ring modes must never be mixed, since the mismatch
    surfaces downstream as a soundness gap, not an error.

    `r = 2^((q-1)/4) mod q` works because 2 is a quadratic non-residue
    exactly when q ≡ ±3 (mod 8), so its `(q-1)/4` power squares to the
    Legendre symbol `-1`. Of the pair `{r, q-r}` the smaller is returned,
    as a deterministic pin.
    """
    if not _is_prime(q):
        raise ValueError(f"split_root: modulus must be prime, got {q!r}")
    if q % 8 != 5:
        raise ValueError(
            f"split_root: q must be ≡ 5 (mod 8), got {q} ≡ {q % 8} (mod 8) — "
            "an NTT-friendly limb (≡ 1 mod 2d, hence ≡ 1 mod 8) belongs to the "
            "NTT ring mode, not the partial-split one"
        )
    r = pow(2, (q - 1) // 4, q)
    return min(r, q - r)
