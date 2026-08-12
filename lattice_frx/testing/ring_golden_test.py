"""The lattigo golden: `RnsRing.ntt` must reproduce lattigo's output *order*.

`ring.py` names this the interop-riskiest property it has — an NTT that
computes the right values in the wrong order is silently wrong for every
consumer that mixes this ring's output with lattigo wire bytes, and it stays
silent through a round trip and through the convolution property, because a
pointwise product does not care what order its slots are in. So the only thing
that can catch it is a vector produced by lattigo itself.

`ring_conformance_test.py` deliberately does not cover this — it pins the
backend-agnostic contract and says so — which leaves the order pinned by this
module alone. Measured rather than asserted: permuting `ntt`'s output and
`intt`'s input inversely (so the two stay mutually consistent) leaves
`canonical_test`, `ring_conformance_test`, `rns_test` and `sampler_test` all
green, and fails only here. A one-sided permutation is caught more widely,
which is the misleading case — it suggests the property is covered when the
consistent version, the one a real backend swap would produce, is not.

That is what this gate exists to make safe: swapping the hand-walked CT/GS
cores for an accelerated backend computes the same values in whatever order
that backend natively emits, and the permutation adapter which then has to
present lattigo's order is exactly the thing no other test here would notice
the absence of.

`ntt_vectors.json` was produced by lattigo's own `MForm`+`NTT`. Its `NttMont`
is therefore in Montgomery form against lattigo's `2^64` radix, and the
standard-form value this ring speaks is `m * 2^-64 mod q` — the un-Montgomery
fold is applied below rather than baked into the fixture, so the fixture stays
exactly what lattigo emitted.
"""

import json
import pathlib
import random

import numpy as np
import pytest

from lattice_frx import ring

_TESTDATA = pathlib.Path(__file__).resolve().parent / "testdata"

# The golden's own moduli. 36-bit limbs, which is the width that makes this
# fixture worth keeping: it is past a 32-bit lane, so a backend that narrows
# its residues fails here rather than at some consumer's wire boundary.
_GOLDEN_MODULI = (34359753217, 34359754753)
_D = 256


def _golden() -> dict:
    return json.loads((_TESTDATA / "ntt_vectors.json").read_text())


def _golden_input(g: dict) -> np.ndarray:
    """`VecIn` as the public contract: one canonical residue row per limb.

    The fixture shares one unreduced integer vector across both limbs, so the
    reduction happens here in Python-int precision — at 36-bit moduli it cannot
    be done in `uint64` after the fact without overflowing.
    """
    unreduced = np.array([g["VecIn"]] * len(g["QModuli"]), dtype=object)
    return (unreduced % np.array(g["QModuli"], dtype=object)[:, None]).astype(np.uint64)


def _standard_form(g: dict) -> np.ndarray:
    """`NttMont` un-Montgomeried: `m * 2^-64 mod q`, per limb."""
    return np.array(
        [
            [(value * pow(2, -64, q)) % q for value in limb]
            for limb, q in zip(g["NttMont"], g["QModuli"])
        ],
        dtype=object,
    )


def test_ntt_reproduces_lattigo_order() -> None:
    """The gate: values *and* the slots they land in, against lattigo's own."""
    g = _golden()
    got = ring.RnsRing(tuple(g["QModuli"]), d=_D).ntt(_golden_input(g))
    assert (got == _standard_form(g)).all()


def test_intt_inverts_ntt_on_the_golden_input() -> None:
    g = _golden()
    r = ring.RnsRing(tuple(g["QModuli"]), d=_D)
    vec = _golden_input(g)
    assert (r.intt(r.ntt(vec)) == vec).all()


@pytest.mark.parametrize("seed", [0xC0FFEE, 0x5EED, 1])
def test_intt_inverts_ntt_on_random_vectors(seed: int) -> None:
    rnd = random.Random(seed)
    r = ring.RnsRing(_GOLDEN_MODULI, _D)
    values = [rnd.randrange(0, min(_GOLDEN_MODULI)) for _ in range(_D)]
    vec = r.from_signed(values)
    assert (r.intt(r.ntt(vec)) == vec).all()


def test_the_ring_is_negacyclic() -> None:
    """`X * X^(d-1) = X^d = -1`, the relation that makes the transform the
    negacyclic one rather than the cyclic one, checked through the ring's own
    NTT and pointwise multiply."""
    r = ring.RnsRing(_GOLDEN_MODULI, _D)
    x = r.from_signed([0, 1] + [0] * (_D - 2))
    x_pow_dm1 = r.from_signed([0] * (_D - 1) + [1])

    product = r.intt(r.mul(r.ntt(x), r.ntt(x_pow_dm1)))
    assert (product == r.from_signed([-1] + [0] * (_D - 1))).all()
