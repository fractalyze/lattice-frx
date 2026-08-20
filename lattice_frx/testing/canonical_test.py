"""Tests for `lattice_frx.canonical` — the array contract every
`_lattice` module enforces, exercised directly rather than through a ring
or reconstruction op.

`tests/test_ring.py` and `tests/test_rns.py` already cover the contract as
their callers see it. This file covers the predicates themselves, because
they are what leaves for lattice-frx: a consumer that never builds an
`RnsRing` still has to be able to ask "does this array satisfy the
contract?" and get the same answer, and the two failure modes have to stay
distinguishable by exception type.
"""
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from lattice_frx import canonical

Q_MODULI = (97, 89)


def _canonical_array() -> np.ndarray:
    return np.array([[0, 1, 96], [0, 1, 88]], dtype=np.uint64)


class CanonicalTest(parameterized.TestCase):

    def test_canonical_array_accepted_by_both_predicates(self):
        a = _canonical_array()
        self.assertTrue(canonical.is_canonical(a, Q_MODULI))
        canonical.require_canonical(a, Q_MODULI, "ctx")  # must not raise

    def test_residue_equal_to_its_modulus_is_not_canonical(self):
        """The range is `[0, q_l)`, so `q_l` itself is the first bad value —
        the boundary a `<=`/`<` slip would let through."""
        a = _canonical_array()
        a[0, 0] = Q_MODULI[0]
        self.assertFalse(canonical.is_canonical(a, Q_MODULI))
        with self.assertRaisesRegex(ValueError, "canonical"):
            canonical.require_canonical(a, Q_MODULI, "ctx")

    def test_range_is_checked_per_limb_not_against_one_modulus(self):
        """A residue can be canonical for limb 0 and not for limb 1. Checking
        every limb against a single modulus would accept this array."""
        a = _canonical_array()
        a[1, 0] = 90  # < 97 (limb 0's modulus) but >= 89 (limb 1's own)
        self.assertFalse(canonical.is_canonical(a, Q_MODULI))

    @parameterized.parameters(np.int64, np.uint32, np.float64, object)
    def test_non_uint64_dtype_is_not_canonical(self, bad_dtype):
        a = np.zeros((2, 3), dtype=bad_dtype)
        self.assertFalse(canonical.is_canonical(a, Q_MODULI))

    @parameterized.parameters(np.int64, np.uint32, np.float64, object)
    def test_require_canonical_raises_type_error_for_dtype(self, bad_dtype):
        """The dtype failure raises `TypeError` while the range failure raises
        `ValueError` (above). The split is the contract: one means the caller
        never embedded its values, the other means it skipped a reduction."""
        a = np.zeros((2, 3), dtype=bad_dtype)
        with self.assertRaisesRegex(TypeError, "uint64"):
            canonical.require_canonical(a, Q_MODULI, "ctx")

    def test_context_is_prefixed_to_the_message(self):
        """Callers pass their own operation name so the raise points at the
        call the caller made, not at this module."""
        a = np.zeros((2, 3), dtype=np.int64)
        with self.assertRaisesRegex(TypeError, r"^RnsRing\.ntt: "):
            canonical.require_canonical(a, Q_MODULI, "RnsRing.ntt")

    def test_extra_leading_axes_are_checked(self):
        """`v2/entities.py` validates stacks whose last two axes are
        `(limbs, d)` with the reference's own nesting in front, so the modulus
        column has to broadcast across those leading axes rather than only
        matching a bare 2-D array."""
        a = np.zeros((2, 3, 2, 4), dtype=np.uint64)
        self.assertTrue(canonical.is_canonical(a, Q_MODULI))
        a[1, 2, 1, 3] = Q_MODULI[1]
        self.assertFalse(canonical.is_canonical(a, Q_MODULI))

    def test_moduli_accepted_as_any_int_sequence(self):
        """Callers pass tuples, lists, and numpy arrays of moduli; the cached
        comparison column keys off the values, not the container."""
        a = _canonical_array()
        self.assertTrue(canonical.is_canonical(a, list(Q_MODULI)))
        self.assertTrue(canonical.is_canonical(a, np.array(Q_MODULI, dtype=np.uint64)))


if __name__ == "__main__":
    absltest.main()
