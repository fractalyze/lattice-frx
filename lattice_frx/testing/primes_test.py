"""Tests for `lattice_frx.primes`' partial-split prime search.

The NTT-friendly walk (`find_nearest_ntt_primes`) predates this file and is
pinned transitively by the ring suites; what is tested here is the second
prime family's walk — partial-split moduli `q ≡ 5 (mod 8)`, where `X^d + 1`
factors into exactly two irreducible halves and LNP-style challenge
differences stay invertible (eprint 2022/284, Lemmas 2.5/2.6). The family's
ring constant (`roots.split_root`) is covered in `split_ring_test.py`,
beside its consumer.
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


if __name__ == "__main__":
    absltest.main()
