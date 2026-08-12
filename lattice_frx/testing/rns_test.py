"""Property tests for `lattice_frx.rns` — exact RNS reconstruction
and centered lifts, ported from `jindo/jindo/rns.go`. No golden fixtures:
these are pure-math properties (CRT roundtrip, centering boundary,
fast/slow path agreement, floor-shift) checked with independent
computation, per the task brief.
"""
import random

import numpy as np
import pytest

from lattice_frx import rns


def test_crt_roundtrip_including_negatives():
    q_moduli = (97, 89)
    Q = 97 * 89
    bound = Q // 2  # |v| <= bound < Q/2 (Q is odd)
    rnd = random.Random(42)
    vals = [rnd.randrange(-bound, bound + 1) for _ in range(50)]

    coeffs = rns.set_big_coeffs(vals, q_moduli)
    got = rns.reconstruct_centered(coeffs, q_moduli)

    assert got == vals


def test_crt_roundtrip_three_limbs():
    q_moduli = (97, 89, 83)
    Q = 97 * 89 * 83
    bound = Q // 2
    rnd = random.Random(7)
    vals = [rnd.randrange(-bound, bound + 1) for _ in range(50)]

    coeffs = rns.set_big_coeffs(vals, q_moduli)
    got = rns.reconstruct_centered(coeffs, q_moduli)

    assert got == vals


def test_centering_boundary_at_exactly_q_half_maps_to_negative():
    """Go's `reconstructTo` (https://github.com/SNUCP/jindo/blob/68ae757d789d4423fb27eb462719c8c993d5277b/jindo/rns.go#L100) uses the non-strict `acc >= qHalf` comparison: an
    accumulator that lands exactly on Q>>1 *does* get centered to
    `acc - Q`, not left positive. q_moduli=(97, 89) gives Q=8633,
    Q>>1=4316; the balanced forms of 4316 mod 97 (=48, not >48) and
    4316 mod 89 (=44, not >44) disagree, so this also forces the slow
    (CRT) path rather than being caught by the fast path."""
    q_moduli = (97, 89)
    Q = 97 * 89
    q_half = Q >> 1
    assert q_half == 4316

    coeffs = rns.set_big_coeffs([q_half], q_moduli)
    got = rns.reconstruct_centered(coeffs, q_moduli)

    assert got == [q_half - Q]


def test_fast_path_agrees_for_small_values():
    """Values small enough that every limb's balanced form is the same
    int take `reconstructTo`'s fast path (rns.go) — must still reproduce the
    original values exactly."""
    q_moduli = (97, 89, 83)
    vals = [5, -5, 0, 40, -40]

    coeffs = rns.set_big_coeffs(vals, q_moduli)
    got = rns.reconstruct_centered(coeffs, q_moduli)

    assert got == vals


def test_slow_path_forced_for_large_values():
    """Values large enough that limbs disagree on the balanced form must
    fall through to `reconstructTo`'s full gadget-sum CRT (rns.go) and still
    reproduce the original values exactly."""
    q_moduli = (97, 89, 83)
    Q = 97 * 89 * 83
    vals = [300000, -300000, 12345]
    assert all(abs(v) < Q // 2 for v in vals)

    coeffs = rns.set_big_coeffs(vals, q_moduli)
    got = rns.reconstruct_centered(coeffs, q_moduli)

    assert got == vals


def test_rsh_floor_on_negatives():
    assert rns.rsh_floor([-5], 1) == [-3]
    assert rns.rsh_floor([-4], 1) == [-2]
    assert rns.rsh_floor([-1], 1) == [-1]
    assert rns.rsh_floor([5], 1) == [2]
    assert rns.rsh_floor([0], 3) == [0]


def test_set_big_coeffs_returns_uint64():
    q_moduli = (97, 89, 83)
    coeffs = rns.set_big_coeffs([5, -5, 40], q_moduli)
    assert coeffs.dtype == np.uint64


def test_reconstruct_centered_accepts_uint64_and_returns_python_ints():
    """`reconstruct_centered` takes the public uint64 contract and
    returns plain Python ints (the host boundary -- see module
    docstring): no fixed-width dtype could hold an arbitrary-precision
    centered reconstruction, so the return stays a `list[int]`, not an
    array."""
    q_moduli = (97, 89, 83)
    vals = [5, -5, 0, 40, -40]
    coeffs = rns.set_big_coeffs(vals, q_moduli)
    assert coeffs.dtype == np.uint64

    got = rns.reconstruct_centered(coeffs, q_moduli)
    assert got == vals
    assert all(isinstance(v, int) for v in got)


def test_lift_centered_returns_uint64():
    from_moduli = (97, 89)
    to_moduli = (67, 71, 73)
    coeffs = rns.set_big_coeffs([1, -1, 100], from_moduli)
    lifted = rns.lift_centered(coeffs, from_moduli, to_moduli)
    assert lifted.dtype == np.uint64


def test_noncanonical_uint64_input_rejected():
    """A residue `== q_0` (out of the contract's `[0, q_l)` range) must
    fail `_coerce`'s canonicality check, not silently compute a wrong
    reconstruction."""
    q_moduli = (97, 89)
    bad = np.zeros((2, 3), dtype=np.uint64)
    bad[0, 0] = q_moduli[0]  # == q0, not canonical
    with pytest.raises(ValueError, match="canonical"):
        rns.reconstruct_centered(bad, q_moduli)


@pytest.mark.parametrize("bad_dtype", [np.float64, object])
def test_non_uint64_dtype_rejected(bad_dtype):
    """Only `uint64` (the contract) is accepted; anything else -- float,
    or the retired `object` migration shim -- fails loud and names the
    contract."""
    q_moduli = (97, 89)
    bad = np.zeros((2, 3), dtype=bad_dtype)
    with pytest.raises(TypeError, match="uint64"):
        rns.reconstruct_centered(bad, q_moduli)


def test_rescale_floor_matches_three_step_composition():
    """`rescale_floor` must equal the inline three-call composition
    `prover.py`'s `_commit_col`/`_outer_commit` currently run (module
    docstring), on random norm-bounded inputs -- property test against
    the existing functions, not a fixed golden."""
    from_moduli = (97, 89)
    to_moduli = (67, 71, 73)
    Q = 97 * 89
    bound = Q // 2 - 1
    shift_bits = 3

    rnd = random.Random(2026)
    for _ in range(20):
        vals = [rnd.randrange(-bound, bound + 1) for _ in range(50)]
        coeffs = rns.set_big_coeffs(vals, from_moduli)

        got = rns.rescale_floor(coeffs, from_moduli, shift_bits, to_moduli)

        reconstructed = rns.reconstruct_centered(coeffs, from_moduli)
        shifted = rns.rsh_floor(reconstructed, shift_bits)
        want = rns.set_big_coeffs(shifted, to_moduli)

        assert got.dtype == np.uint64
        assert (got == want).all()


def test_lift_centered_round_trip():
    """`lift_centered` from a Q-basis to a QOut-basis, then reconstructing
    back under the Q-basis (via `reconstruct_centered` on the lifted
    QOut-basis coefficients), must recover the original centered ints —
    the norm-bounded-input property the docstring documents."""
    from_moduli = (97, 89)
    to_moduli = (67, 71, 73)
    Q = 97 * 89
    bound = Q // 2 - 1
    vals = [0, 1, -1, 100, -100, bound, -bound]

    coeffs = rns.set_big_coeffs(vals, from_moduli)
    lifted = rns.lift_centered(coeffs, from_moduli, to_moduli)
    got = rns.reconstruct_centered(lifted, to_moduli)

    assert got == vals
