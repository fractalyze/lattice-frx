"""The constants an NTT over a modulus needs, before any ring exists.

A primitive root of `q` and the bit-reversal its twiddle table is indexed by
are needed identically by both rings here — the host one fills lattigo's tables
with them, the traced one derives the opcode's generator and its ordering
adapter from them — and neither is about a ring, a limb, or an array. They are
host-side integer functions over a modulus, computed once per parameterisation,
so they live below both rather than inside either.

`primitive_root` is lattigo's `PrimitiveRoot` (subring.go): the smallest `g`
with `g^((q-1)/f) != 1 mod q` for every prime factor `f` of `q-1`. Its
factorization is reimplemented rather than ported — only the resulting *set* of
unique prime factors feeds the search, so any correct factorization agrees on
it.
"""

import math
import random

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
