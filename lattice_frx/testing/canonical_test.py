"""Tests for `lattice_frx.canonical` — the host array contract, exercised
directly rather than through a ring or reconstruction op.

`host_ring_conformance_test.py` and `rns_test.py` already cover the
contract as their callers see it, and `ring_test.py` covers the one seam
where the traced ring takes a host array. This file covers the predicates
themselves, because they are what a consumer gets on their own: one that
never builds a ring still has to be able to ask "does this array satisfy
the contract?" and get the same answer, and the two failure modes have to
stay distinguishable by exception type.
"""
import numpy as np
import pytest

from lattice_frx import canonical

Q_MODULI = (97, 89)


def _canonical_array() -> np.ndarray:
    return np.array([[0, 1, 96], [0, 1, 88]], dtype=np.uint64)


def test_canonical_array_accepted_by_both_predicates():
    a = _canonical_array()
    assert canonical.is_canonical(a, Q_MODULI)
    canonical.require_canonical(a, Q_MODULI, "ctx")  # must not raise


def test_residue_equal_to_its_modulus_is_not_canonical():
    """The range is `[0, q_l)`, so `q_l` itself is the first bad value —
    the boundary a `<=`/`<` slip would let through."""
    a = _canonical_array()
    a[0, 0] = Q_MODULI[0]
    assert not canonical.is_canonical(a, Q_MODULI)
    with pytest.raises(ValueError, match="canonical"):
        canonical.require_canonical(a, Q_MODULI, "ctx")


def test_range_is_checked_per_limb_not_against_one_modulus():
    """A residue can be canonical for limb 0 and not for limb 1. Checking
    every limb against a single modulus would accept this array."""
    a = _canonical_array()
    a[1, 0] = 90  # < 97 (limb 0's modulus) but >= 89 (limb 1's own)
    assert not canonical.is_canonical(a, Q_MODULI)


@pytest.mark.parametrize("bad_dtype", [np.int64, np.uint32, np.float64, object])
def test_non_uint64_dtype_is_not_canonical(bad_dtype):
    a = np.zeros((2, 3), dtype=bad_dtype)
    assert not canonical.is_canonical(a, Q_MODULI)


@pytest.mark.parametrize("bad_dtype", [np.int64, np.uint32, np.float64, object])
def test_require_canonical_raises_type_error_for_dtype(bad_dtype):
    """The dtype failure raises `TypeError` while the range failure raises
    `ValueError` (above). The split is the contract: one means the caller
    never embedded its values, the other means it skipped a reduction."""
    a = np.zeros((2, 3), dtype=bad_dtype)
    with pytest.raises(TypeError, match="uint64"):
        canonical.require_canonical(a, Q_MODULI, "ctx")


def test_context_is_prefixed_to_the_message():
    """Callers pass their own operation name so the raise points at the
    call the caller made, not at this module."""
    a = np.zeros((2, 3), dtype=np.int64)
    with pytest.raises(TypeError, match=r"^HostRnsRing\.ntt: "):
        canonical.require_canonical(a, Q_MODULI, "HostRnsRing.ntt")


def test_extra_leading_axes_are_checked():
    """`v2/entities.py` validates stacks whose last two axes are
    `(limbs, d)` with the reference's own nesting in front, so the modulus
    column has to broadcast across those leading axes rather than only
    matching a bare 2-D array."""
    a = np.zeros((2, 3, 2, 4), dtype=np.uint64)
    assert canonical.is_canonical(a, Q_MODULI)
    a[1, 2, 1, 3] = Q_MODULI[1]
    assert not canonical.is_canonical(a, Q_MODULI)


def test_moduli_accepted_as_any_int_sequence():
    """Callers pass tuples, lists, and numpy arrays of moduli; the cached
    comparison column keys off the values, not the container."""
    a = _canonical_array()
    assert canonical.is_canonical(a, list(Q_MODULI))
    assert canonical.is_canonical(a, np.array(Q_MODULI, dtype=np.uint64))
