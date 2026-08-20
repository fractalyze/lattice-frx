"""Tests for `lattice_frx.primes`' partial-split prime search.

The NTT-friendly walk (`find_nearest_ntt_primes`) predates this file and is
pinned transitively by the ring suites; what is tested here is the second
prime family — partial-split moduli `q ≡ 5 (mod 8)`, where `X^d + 1` factors
into exactly two irreducible halves and LNP-style challenge differences stay
invertible (eprint 2022/284, Lemmas 2.5/2.6).
"""
from absl.testing import absltest
from absl.testing import parameterized

from lattice_frx import primes


class SplitPrimesTest(parameterized.TestCase):

    @parameterized.parameters(20.0, 32.0, 49.0)
    def test_find_nearest_split_primes_shape_and_residue(self, bits):
        got = primes.find_nearest_split_primes(bits, 4)
        self.assertLen(got, 4)
        self.assertEqual(got, sorted(got))
        for q in got:
            self.assertTrue(primes.is_prime(q))
            self.assertEqual(q % 8, 5)
            self.assertLessEqual(q, primes.MAX_MODULUS)
            # Near the requested size: within a generous walk distance.
            self.assertLess(abs(q - 2**bits), 2**bits * 0.01)

    def test_split_root_squares_to_minus_one(self):
        for q in primes.find_nearest_split_primes(32.0, 4):
            r = primes.split_root(q)
            self.assertEqual(r * r % q, q - 1)
            # Deterministic canonical pick: the smaller of the two roots.
            self.assertEqual(r, min(r, q - r))

    def test_split_root_rejects_an_ntt_friendly_modulus(self):
        # An NTT-friendly limb is ≡ 1 (mod 2d) hence ≡ 1 (mod 8) — the other
        # ring mode. The error must name the mode confusion, since silently
        # mixing the two is a soundness bug in the consumer, not a crash.
        [q] = primes.find_nearest_ntt_primes(128, 32.0, 1)
        with self.assertRaisesRegex(ValueError, "NTT"):
            primes.split_root(q)

    @parameterized.parameters(0, 1, 21, 4)  # non-primes (21 ≡ 5 mod 8)
    def test_split_root_rejects_a_non_prime(self, bad):
        with self.assertRaisesRegex(ValueError, "prime"):
            primes.split_root(bad)


if __name__ == "__main__":
    absltest.main()
