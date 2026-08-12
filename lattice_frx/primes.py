"""Scheme-agnostic primality and NTT-friendly prime search.

Two layers share this module:

- `is_prime` — deterministic Miller-Rabin, the substrate's single public
  primality test (previously a private copy inside `ring.py`).
- `next_prime` / `prev_prime` / `find_nearest_ntt_primes` — the ringo-snark
  crt-stack prime walk (`num.NextPrime` / `num.PrevPrime`,
  math/num/prime.go; `crt.FindNearestNTTPrimes`, math/crt/primes.go — pin
  b31f20a), the prime-chain generator the v2 protocol's parameter search
  uses. Distinct from lattigo's own NTT-prime walk: different start
  anchor, different stepping split, so the two produce different chains
  at the same (rank, bits) — a consumer must pick the one its reference
  uses rather than assuming they agree.

`MAX_MODULUS` (2**50) is part of the ported contract — the crt stack caps
every modulus at `num.MaxModulus` (math/num/mod.go), and the walk's
overflow/underflow errors key off it.
"""

import math

# First 12 prime bases: deterministic Miller-Rabin against these is exact
# for every n < 3,317,044,064,679,887,385,961,981 (Sorenson & Webster),
# comfortably past the 2**64 range callers' moduli live in.
_MR_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

# num.MaxModulusBits / num.MaxModulus (math/num/mod.go).
MAX_MODULUS_BITS = 50
MAX_MODULUS = 1 << MAX_MODULUS_BITS


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin, exact for n < 2**64."""
    if n < 2:
        return False
    for w in _MR_WITNESSES:
        if n % w == 0:
            return n == w
    d, r = n - 1, 0
    while d % 2 == 0:
        d, r = d // 2, r + 1
    for a in _MR_WITNESSES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(x: int, skip: int) -> int:
    """Port of `num.NextPrime`: walk up from x in steps of skip, capped at
    `MAX_MODULUS`."""
    t = x + skip
    while True:
        if t > MAX_MODULUS:
            raise OverflowError("next_prime: overflow")
        if is_prime(t):
            return t
        t += skip


def prev_prime(x: int, skip: int) -> int:
    """Port of `num.PrevPrime`: walk down from x in steps of skip.

    Go's uint64 wrap-around on underflow trips its `> MaxModulus` guard; the
    signed walk here reaches `t <= 1` instead — same error, different route.
    """
    t = x - skip
    while True:
        if t > MAX_MODULUS or t <= 1:
            raise OverflowError("prev_prime: underflow")
        if is_prime(t):
            return t
        t -= skip


def find_nearest_ntt_primes(rank: int, bits: float, cnt: int) -> list[int]:
    """Port of `crt.FindNearestNTTPrimes` (math/crt/primes.go).

    NTT-friendly means `≡ 1 (mod 2*rank)`. The start point anchors at the
    largest multiple of `2*rank` below `2**bits`, plus one; `cnt` splits as
    `next = cnt >> 1` above and `prev = cnt - next` below, each side falling
    back to the other when it runs out of primes under the `MAX_MODULUS`
    cap. Result is sorted ascending (`num.CmpModulus`).
    """
    gap = rank << 1
    # Go: (uint64(math.Floor(math.Exp2(bits)) / float64(gap))) * gap + 1 —
    # float division then truncation; exact here because bits <= 50 < 53.
    start = int(math.floor(math.exp2(bits)) / gap) * gap + 1
    next_cnt = cnt >> 1
    prev_cnt = cnt - next_cnt

    next_primes: list[int] = []
    t = start
    for _ in range(next_cnt):
        try:
            t = next_prime(t, gap)
        except OverflowError:
            break
        next_primes.append(t)

    prev_primes: list[int] = []
    t = start
    for _ in range(prev_cnt):
        try:
            t = prev_prime(t, gap)
        except OverflowError:
            break
        prev_primes.append(t)

    if len(next_primes) < next_cnt and len(prev_primes) < prev_cnt:
        raise ValueError("find_nearest_ntt_primes: not enough primes found")
    elif len(next_primes) < next_cnt:
        t = prev_primes[prev_cnt - 1]
        for _ in range(cnt - (len(next_primes) + len(prev_primes))):
            t = prev_prime(t, gap)
            prev_primes.append(t)
    elif len(prev_primes) < prev_cnt:
        t = next_primes[next_cnt - 1]
        for _ in range(cnt - (len(next_primes) + len(prev_primes))):
            t = next_prime(t, gap)
            next_primes.append(t)

    return sorted(prev_primes + next_primes)
