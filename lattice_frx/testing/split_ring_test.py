"""Tests for `lattice_frx.split_ring`'s partial-split host ring.

The multiplication cross-check is deliberately two-sided: the ring's own
`mul` is a coefficient-domain negacyclic schoolbook convolution, and this
file re-derives the product independently through the two-factor CRT
(`to_split` → per-half twisted convolution → `from_split`) — two different
algorithms agreeing on random inputs is the correctness gate, mirroring how
`ring_test.py` checks the traced ring against `HostRnsRing`.

The invertibility tests are the property the LNP consumer buys (eprint
2022/284, Lemma 2.6): nonzero σ₋₁-invariant small elements are invertible at
a partial-split modulus. Invertibility is cross-checked against a test-side
extended-Euclid over each half field.
"""
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from lattice_frx import primes
from lattice_frx import roots
from lattice_frx.split_ring import HostSplitRing

_D = 16
_Q_MODULI = tuple(primes.find_nearest_split_primes(30.0, 2))


def _ring() -> HostSplitRing:
    return HostSplitRing(_Q_MODULI, _D)


def _random_canonical(rng, ring) -> np.ndarray:
    rows = [rng.integers(0, q, ring.d, dtype=np.uint64) for q in ring.q_moduli]
    return np.stack(rows)


def _twisted_mul(u: list[int], v: list[int], s: int, q: int) -> list[int]:
    """Independent product in `F_q[X]/(X^{d/2} - s)`: full convolution, then
    fold the top half back with the twist `X^{d/2} ≡ s`."""
    n = len(u)
    conv = [0] * (2 * n)
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            conv[i + j] += ui * vj
    return [(conv[k] + s * conv[k + n]) % q for k in range(n)]


def _half_inverse(u: list[int], s: int, q: int):
    """Test-side extended Euclid over `F_q[X]` modulo `X^{d/2} - s`; returns
    an inverse of `u` or None when `gcd` is not a unit."""
    n = len(u)
    modulus = [(-s) % q] + [0] * (n - 1) + [1]  # X^n - s, ascending coeffs

    def deg(p):
        for i in range(len(p) - 1, -1, -1):
            if p[i] % q:
                return i
        return -1

    # Euclid with Bezout tracking on the first argument.
    r0, r1 = modulus, [x % q for x in u]
    t0, t1 = [0], [1]
    while deg(r1) > 0:
        d0, d1 = deg(r0), deg(r1)
        if d0 < d1:
            r0, r1, t0, t1 = r1, r0, t1, t0
            continue
        f = r0[d0] * pow(r1[d1], -1, q) % q
        shift = d0 - d1
        r0 = [(c - f * (r1[i - shift] if 0 <= i - shift <= d1 else 0)) % q for i, c in enumerate(r0)]
        t0 = [((t0[i] if i < len(t0) else 0) - f * (t1[i - shift] if 0 <= i - shift < len(t1) else 0)) % q
              for i in range(max(len(t0), len(t1) + shift))]
    if deg(r1) < 0:
        return None
    c_inv = pow(r1[0], -1, q)
    out = [(x * c_inv) % q for x in t1]
    return (out + [0] * n)[:n]


class SplitRootTest(absltest.TestCase):

    def test_split_root_squares_to_minus_one(self):
        for q in _Q_MODULI:
            r = roots.split_root(q)
            self.assertEqual(r * r % q, q - 1)
            # Deterministic canonical pick: the smaller of the two roots.
            self.assertEqual(r, min(r, q - r))

    def test_split_root_rejects_an_ntt_friendly_modulus(self):
        # An NTT-friendly limb is ≡ 1 (mod 2d) hence ≡ 1 (mod 8) — the other
        # ring mode. The error must name the mode confusion, since silently
        # mixing the two is a soundness bug in the consumer, not a crash.
        [q] = primes.find_nearest_ntt_primes(_D, 30.0, 1)
        with self.assertRaisesRegex(ValueError, "NTT"):
            roots.split_root(q)

    def test_split_root_rejects_a_non_prime(self):
        for bad in (0, 1, 21, 4):  # 21 ≡ 5 (mod 8), still rejected
            with self.assertRaisesRegex(ValueError, "prime"):
                roots.split_root(bad)


class SplitRingConstructionTest(parameterized.TestCase):

    def test_rejects_an_ntt_friendly_modulus_with_the_mode_message(self):
        [q] = primes.find_nearest_ntt_primes(_D, 30.0, 1)
        with self.assertRaisesRegex(ValueError, "NTT"):
            HostSplitRing((q,), _D)

    def test_rejects_a_non_prime_modulus(self):
        with self.assertRaisesRegex(ValueError, "prime"):
            HostSplitRing((21,), _D)  # 21 ≡ 5 (mod 8), not prime

    @parameterized.parameters(2, 3, 12)
    def test_rejects_a_bad_degree(self, d):
        with self.assertRaisesRegex(ValueError, "degree"):
            HostSplitRing(_Q_MODULI, d)


class SplitRingOpsTest(parameterized.TestCase):

    def test_contract_failure_modes_carry_the_split_prefix(self):
        ring = _ring()
        good = _random_canonical(np.random.default_rng(0), ring)
        with self.assertRaisesRegex(TypeError, r"^SplitRing\.mul"):
            ring.mul(good.astype(np.int64), good)
        bad = good.copy()
        bad[0, 0] = np.uint64(ring.q_moduli[0])
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.mul"):
            ring.mul(bad, good)

    def test_add_neg_roundtrip_and_no_mutation(self):
        ring = _ring()
        rng = np.random.default_rng(1)
        a = _random_canonical(rng, ring)
        b = _random_canonical(rng, ring)
        a0, b0 = a.copy(), b.copy()
        np.testing.assert_array_equal(ring.sub(ring.add(a, b), b), a)
        np.testing.assert_array_equal(ring.add(a, ring.neg(a)), np.zeros_like(a))
        np.testing.assert_array_equal(a, a0)
        np.testing.assert_array_equal(b, b0)

    def test_mul_matches_the_independent_twisted_crt_path(self):
        ring = _ring()
        rng = np.random.default_rng(2)
        for _ in range(8):
            a = _random_canonical(rng, ring)
            b = _random_canonical(rng, ring)
            got = ring.mul(a, b)
            sa, sb = ring.to_split(a), ring.to_split(b)
            prod = np.zeros_like(sa)
            for l, q in enumerate(ring.q_moduli):
                for h, sgn in enumerate((1, -1)):
                    s = sgn * ring.split_roots[l] % q
                    prod[l, h] = _twisted_mul(
                        [int(x) for x in sa[l, h]], [int(x) for x in sb[l, h]], s, q)
            np.testing.assert_array_equal(ring.from_split(prod), got)

    def test_mul_ring_axioms(self):
        ring = _ring()
        rng = np.random.default_rng(3)
        a = _random_canonical(rng, ring)
        b = _random_canonical(rng, ring)
        c = _random_canonical(rng, ring)
        np.testing.assert_array_equal(ring.mul(a, b), ring.mul(b, a))
        np.testing.assert_array_equal(
            ring.mul(a, ring.add(b, c)), ring.add(ring.mul(a, b), ring.mul(a, c)))
        one = ring.from_signed([1] + [0] * (ring.d - 1))
        np.testing.assert_array_equal(ring.mul(a, one), a)

    def test_mul_wraps_negacyclically(self):
        # X^{d-1} * X = X^d = -1.
        ring = _ring()
        x_last = ring.from_signed([0] * (ring.d - 1) + [1])
        x_one = ring.from_signed([0, 1] + [0] * (ring.d - 2))
        minus_one = ring.from_signed([-1] + [0] * (ring.d - 1))
        np.testing.assert_array_equal(ring.mul(x_last, x_one), minus_one)

    def test_split_roundtrip(self):
        ring = _ring()
        rng = np.random.default_rng(4)
        a = _random_canonical(rng, ring)
        np.testing.assert_array_equal(ring.from_split(ring.to_split(a)), a)

    def test_galois_sigma_minus_one_is_an_involution(self):
        ring = _ring()
        rng = np.random.default_rng(5)
        a = _random_canonical(rng, ring)
        k = 2 * ring.d - 1  # σ₋₁ : X ↦ X^{-1}
        np.testing.assert_array_equal(ring.galois(ring.galois(a, k), k), a)


class SplitRingInvertibilityTest(absltest.TestCase):

    def test_nonzero_sigma_invariant_small_elements_are_invertible(self):
        # Lemma 2.6 (eprint 2022/284): at q ≡ 5 (mod 8), σ₋₁-invariant c ≠ 0
        # with small coefficients is invertible — the LNP challenge-space
        # property this ring exists to give its consumer.
        ring = _ring()
        rng = np.random.default_rng(6)
        k = 2 * ring.d - 1
        for _ in range(50):
            free = rng.integers(-2, 3, ring.d // 2)
            if not free.any():
                continue
            coeffs = [0] * ring.d
            coeffs[0] = int(free[0])
            for i in range(1, ring.d // 2):
                coeffs[i] = int(free[i])
                coeffs[ring.d - i] = -int(free[i])  # σ₋₁-invariant embedding
            c = ring.from_signed(coeffs)
            np.testing.assert_array_equal(ring.galois(c, k), c)  # sanity: fixed by σ₋₁
            self.assertTrue(ring.is_invertible(c))

    def test_zero_and_half_zero_elements_are_not_invertible(self):
        ring = _ring()
        zero = ring.from_signed([0] * ring.d)
        self.assertFalse(ring.is_invertible(zero))
        # An element that is a multiple of one CRT half: zero in half 0.
        rng = np.random.default_rng(7)
        sp = ring.to_split(_random_canonical(rng, ring))
        sp[:, 0, :] = 0
        self.assertFalse(ring.is_invertible(ring.from_split(sp)))

    def test_is_invertible_agrees_with_extended_euclid(self):
        ring = _ring()
        rng = np.random.default_rng(8)
        for _ in range(10):
            a = _random_canonical(rng, ring)
            sp = ring.to_split(a)
            expected = True
            for l, q in enumerate(ring.q_moduli):
                for h, sgn in enumerate((1, -1)):
                    s = sgn * ring.split_roots[l] % q
                    inv = _half_inverse([int(x) for x in sp[l, h]], s, q)
                    if inv is None:
                        expected = False
                    else:
                        one = _twisted_mul([int(x) for x in sp[l, h]], inv, s, q)
                        self.assertEqual(one, [1] + [0] * (ring.d // 2 - 1))
            self.assertEqual(ring.is_invertible(a), expected)


if __name__ == "__main__":
    absltest.main()
