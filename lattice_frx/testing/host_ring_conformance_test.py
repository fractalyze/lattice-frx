"""Backend-conformance suite for the `(limbs, d)` uint64 ring/RNS contract.

Parameterized over backend name (today: `"reference"` only). `ring.RnsRing`
is deliberately NOT in `BACKENDS`: it does not implement this contract and
cannot, since a field dtype is per-modulus and its element is therefore one
array per limb rather than one `(limbs, d)` uint64 array. Its own conformance
is `ring_test.py`, which pins it against this one op for op. What is parameterized here is any future backend that does speak
the host contract. For each such backend this module asserts the properties
ANY correct backend implementing the public contract (`_lattice/ring.py`'s module docstring: `(limbs, d)`
`dtype=np.uint64` arrays of canonical, `< q_l`-per-limb, standard-form
residues) must satisfy:

(a) dtype/shape per the documented contract (including the two
    documented deviations -- `to_balanced_limb0` returns `int64`,
    `rns.reconstruct_centered` returns `list[int]` -- not blanket
    "everything is uint64").
(b) purity -- no op mutates an array it was handed.
(c) canonicality -- every uint64-array output has every residue `< q_l`
    for its limb, not just "some dtype that happens to be uint64".
(d) algebraic sanity that any correct backend must reproduce, with the
    expected values derived independently of the backend under test
    (plain per-limb Python-int modular arithmetic for add/sub/neg/
    mul_add/mul_scalar/mul_scalar_then_sub, or a known mathematical
    identity of the ring itself for mul/ntt/intt -- never "run the op
    under test and compare to itself").
(e) the `TypeError` contract: float and the retired `object`
    migration-shim dtype both fail loud, across the whole op surface.

This module intentionally does NOT re-litigate golden-vector fidelity
(`test_ring.py`'s `test_ntt_matches_lattigo_order`) or RNS-specific
CRT/centering edge cases (`test_rns.py`) -- those pin the reference
backend's own internals against lattigo. This module pins the
backend-agnostic *contract* every backend must share, kept lean: one
test per property above, looping internally over every public `RnsRing`
op and the `rns` boundary ops, rather than exploding into a
`pytest.mark.parametrize` matrix of op x backend x property.
"""
import random

import numpy as np
import pytest

from lattice_frx import host_ring as host_mod
from lattice_frx import rns

# Backend registry: name -> `RnsRing`-alike constructor `(q_moduli, d)`.
# Every value here must satisfy this module's whole property set; a
# Another host-contract implementation joins by adding one entry. `RnsRing`
# is not one and cannot be: its element is per-limb field arrays, not a
# `(limbs, d)` uint64 array, and `ring_test` is its conformance instead.
BACKENDS = {
    "reference": host_mod.HostRnsRing,
}

Q_MODULI = (34359753217, 34359754753)
Q_OUT_MODULI = (67, 71, 73)
D = 256


@pytest.fixture(params=sorted(BACKENDS))
def ring(request):
    return BACKENDS[request.param](Q_MODULI, D)


def _rand_vec(seed: int, lo: int = 0, hi: int = 1000) -> list[int]:
    rnd = random.Random(seed)
    return [rnd.randrange(lo, hi) for _ in range(D)]


def _assert_no_mutation(op_name, fn, *args):
    """Snapshot every ndarray argument, call `fn`, then assert none of
    them changed -- the shared purity check every op-under-test below
    reuses instead of hand-rolling a copy/compare per call site."""
    snapshots = [(a, a.copy()) for a in args if isinstance(a, np.ndarray)]
    fn(*args)
    for arr, before in snapshots:
        assert np.array_equal(arr, before), f"{op_name} mutated an input array in place"


# --- (a) dtype/shape per the documented contract ------------------------


def test_ops_respect_documented_dtype_and_shape_contract(ring):
    limbs = len(Q_MODULI)
    a = ring.from_signed(_rand_vec(1))
    b = ring.from_signed(_rand_vec(2))
    acc = ring.from_signed(_rand_vec(3))
    assert a.dtype == np.uint64 and a.shape == (limbs, D)

    a_ntt, b_ntt, acc_ntt = ring.ntt(a), ring.ntt(b), ring.ntt(acc)
    outputs = {
        "ntt": a_ntt,
        "intt": ring.intt(a_ntt),
        "add": ring.add(a, b),
        "sub": ring.sub(a, b),
        "neg": ring.neg(a),
        "mul": ring.mul(a_ntt, b_ntt),
        "mul_add": ring.mul_add(a_ntt, b_ntt, acc_ntt),
        "mul_scalar": ring.mul_scalar(a, 7),
        "mul_scalar_then_sub": ring.mul_scalar_then_sub(a, 7, acc),
    }
    for name, out in outputs.items():
        assert out.dtype == np.uint64, f"{name}: expected uint64, got {out.dtype}"
        assert out.shape == (limbs, D), f"{name}: expected shape {(limbs, D)}, got {out.shape}"

    # Documented deviation: a balanced value is signed by definition, so
    # this one op returns int64, not the public uint64 contract.
    balanced = ring.to_balanced_limb0(a)
    assert balanced.dtype == np.int64
    assert balanced.shape == (D,)

    # rns boundary ops: reconstruct_centered is the host boundary itself
    # (list[int], no fixed-width dtype fits an arbitrary-precision
    # centered value); set_big_coeffs/lift_centered/rescale_floor return
    # to the public uint64 contract.
    coeffs = ring.from_signed(_rand_vec(4))
    reconstructed = rns.reconstruct_centered(coeffs, Q_MODULI)
    assert isinstance(reconstructed, list) and len(reconstructed) == D
    assert all(isinstance(v, int) for v in reconstructed)

    set_back = rns.set_big_coeffs(reconstructed, Q_MODULI)
    assert set_back.dtype == np.uint64 and set_back.shape == (limbs, D)

    lifted = rns.lift_centered(coeffs, Q_MODULI, Q_OUT_MODULI)
    assert lifted.dtype == np.uint64 and lifted.shape == (len(Q_OUT_MODULI), D)

    rescaled = rns.rescale_floor(coeffs, Q_MODULI, 3, Q_OUT_MODULI)
    assert rescaled.dtype == np.uint64 and rescaled.shape == (len(Q_OUT_MODULI), D)


# --- (b) purity -- no op mutates its inputs ------------------------------


def test_ops_do_not_mutate_their_inputs(ring):
    a = ring.from_signed(_rand_vec(11))
    b = ring.from_signed(_rand_vec(12))
    acc = ring.from_signed(_rand_vec(13))
    a_ntt, b_ntt, acc_ntt = ring.ntt(a), ring.ntt(b), ring.ntt(acc)

    _assert_no_mutation("ntt", ring.ntt, a)
    _assert_no_mutation("intt", ring.intt, a_ntt)
    _assert_no_mutation("add", ring.add, a, b)
    _assert_no_mutation("sub", ring.sub, a, b)
    _assert_no_mutation("neg", ring.neg, a)
    _assert_no_mutation("mul", ring.mul, a_ntt, b_ntt)
    _assert_no_mutation("mul_add", ring.mul_add, a_ntt, b_ntt, acc_ntt)
    _assert_no_mutation("mul_scalar", ring.mul_scalar, a, 7)
    _assert_no_mutation("mul_scalar_then_sub", ring.mul_scalar_then_sub, a, 7, acc)
    _assert_no_mutation("to_balanced_limb0", ring.to_balanced_limb0, a)

    coeffs = ring.from_signed(_rand_vec(14))
    _assert_no_mutation("rns.reconstruct_centered", rns.reconstruct_centered, coeffs, Q_MODULI)
    _assert_no_mutation("rns.lift_centered", rns.lift_centered, coeffs, Q_MODULI, Q_OUT_MODULI)
    _assert_no_mutation("rns.rescale_floor", rns.rescale_floor, coeffs, Q_MODULI, 3, Q_OUT_MODULI)


# --- (c) canonicality -- every residue < q_l for its limb ---------------


def test_uint64_outputs_are_canonical_per_limb(ring):
    a = ring.from_signed(_rand_vec(21))
    b = ring.from_signed(_rand_vec(22))
    acc = ring.from_signed(_rand_vec(23))
    a_ntt, b_ntt, acc_ntt = ring.ntt(a), ring.ntt(b), ring.ntt(acc)

    q_col = np.array(Q_MODULI, dtype=np.uint64)[:, None]
    outputs = [
        ring.ntt(a), ring.intt(a_ntt), ring.add(a, b), ring.sub(a, b), ring.neg(a),
        ring.mul(a_ntt, b_ntt), ring.mul_add(a_ntt, b_ntt, acc_ntt),
        ring.mul_scalar(a, 7), ring.mul_scalar_then_sub(a, 7, acc),
    ]
    for out in outputs:
        assert (out < q_col).all()

    coeffs = ring.from_signed(_rand_vec(24))
    reconstructed = rns.reconstruct_centered(coeffs, Q_MODULI)
    assert (rns.set_big_coeffs(reconstructed, Q_MODULI) < q_col).all()

    q_out_col = np.array(Q_OUT_MODULI, dtype=np.uint64)[:, None]
    assert (rns.lift_centered(coeffs, Q_MODULI, Q_OUT_MODULI) < q_out_col).all()
    assert (rns.rescale_floor(coeffs, Q_MODULI, 3, Q_OUT_MODULI) < q_out_col).all()


# --- (d) algebraic sanity, expectations derived independently -----------


def test_algebraic_sanity_derived_independently_of_backend(ring):
    # add/sub/neg must match plain per-limb modular arithmetic computed
    # with vanilla Python-int `%` -- never by calling the op under test
    # to produce its own "expected" value. `from_signed` is not itself
    # under test here: it's the trivial broadcast-and-reduce primitive
    # every correct backend must implement identically (the public
    # contract fixes its meaning byte-for-byte), used only to get known
    # values onto the ring.
    rnd = random.Random(31)
    a_vals = [rnd.randrange(0, min(Q_MODULI)) for _ in range(D)]
    b_vals = [rnd.randrange(0, min(Q_MODULI)) for _ in range(D)]
    a = ring.from_signed(a_vals)
    b = ring.from_signed(b_vals)

    a_obj = np.array([a_vals] * len(Q_MODULI), dtype=object)
    b_obj = np.array([b_vals] * len(Q_MODULI), dtype=object)
    q_col = np.array(Q_MODULI, dtype=object)[:, None]

    assert (ring.add(a, b).astype(object) == (a_obj + b_obj) % q_col).all()
    assert (ring.sub(a, b).astype(object) == (a_obj - b_obj) % q_col).all()
    assert (ring.neg(a).astype(object) == (-a_obj) % q_col).all()

    # Roundtrip identities that follow from the above, restated directly
    # against the ring's own ops (any correct backend must be
    # self-consistent, not just individually correct per op):
    assert np.array_equal(ring.sub(ring.add(a, b), b), a)
    assert np.array_equal(ring.add(a, ring.neg(a)), ring.from_signed([0] * D))

    # mul via ntt of known small polys: X * X^(d-1) = X^d = -1 mod
    # (X^d + 1) -- the negacyclic ring's defining relation, a
    # mathematical fact independent of any implementation. Any correct
    # ntt/mul/intt must reproduce it.
    x = ring.from_signed([0, 1] + [0] * (D - 2))
    x_pow_dm1 = ring.from_signed([0] * (D - 1) + [1])
    prod = ring.intt(ring.mul(ring.ntt(x), ring.ntt(x_pow_dm1)))
    assert np.array_equal(prod, ring.from_signed([-1] + [0] * (D - 1)))

    # intt . ntt is the identity on arbitrary canonical input.
    for _ in range(5):
        vals = [rnd.randrange(0, min(Q_MODULI)) for _ in range(D)]
        vec = ring.from_signed(vals)
        assert np.array_equal(ring.intt(ring.ntt(vec)), vec)

    # mul_add/mul_scalar/mul_scalar_then_sub -- the workhorse elementwise
    # ops the commit/evaluate/verify hot path calls the most -- checked
    # against the same independently-computed object-dtype `%` formula
    # `test_ring.py` uses, not against the op under test. Dtype/shape/
    # purity/canonicality checks elsewhere in this module say nothing
    # about *values*, so a backend that gets one of these three
    # elementwise formulas wrong would otherwise pass the whole suite.
    acc_vals = [rnd.randrange(0, min(Q_MODULI)) for _ in range(D)]
    acc = ring.from_signed(acc_vals)
    acc_obj = np.array([acc_vals] * len(Q_MODULI), dtype=object)
    s = 7

    want_mul_add = (acc_obj + a_obj * b_obj) % q_col
    assert (ring.mul_add(a, b, acc).astype(object) == want_mul_add).all()

    want_mul_scalar = (a_obj * s) % q_col
    assert (ring.mul_scalar(a, s).astype(object) == want_mul_scalar).all()

    want_mul_scalar_then_sub = (acc_obj - a_obj * s) % q_col
    assert (ring.mul_scalar_then_sub(a, s, acc).astype(object) == want_mul_scalar_then_sub).all()


# --- (e) the TypeError contract for non-uint64 dtypes --------------------


@pytest.mark.parametrize("bad_dtype", [np.float64, object])
def test_non_uint64_dtype_raises_typeerror_across_the_op_surface(ring, bad_dtype):
    """float and the retired `object` migration-shim dtype must both fail
    loud and name the contract -- checked across every public op that
    takes an array (not just one representative call, per `test_ring.py`/
    `test_rns.py`'s narrower per-module versions of this check)."""
    limbs = len(Q_MODULI)
    bad = np.zeros((limbs, D), dtype=bad_dtype)
    good = ring.from_signed(_rand_vec(41))

    for op in (
        ring.ntt, ring.intt, ring.to_balanced_limb0,
        lambda x: ring.add(x, good),
        lambda x: ring.sub(x, good),
        ring.neg,
        lambda x: ring.mul(x, good),
        lambda x: ring.mul_add(x, good, good),
        lambda x: ring.mul_scalar(x, 7),
        lambda x: ring.mul_scalar_then_sub(x, 7, good),
    ):
        with pytest.raises(TypeError, match="uint64"):
            op(bad)

    for op in (
        lambda x: rns.reconstruct_centered(x, Q_MODULI),
        lambda x: rns.lift_centered(x, Q_MODULI, Q_OUT_MODULI),
        lambda x: rns.rescale_floor(x, Q_MODULI, 3, Q_OUT_MODULI),
    ):
        with pytest.raises(TypeError, match="uint64"):
            op(bad)
