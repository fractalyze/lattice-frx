"""The ring against the host implementation, which is the only oracle it needs.

`host_ring.HostRnsRing` computes over exact Python integers, one coefficient at
a time, reproducing lattigo's tables and loop structure — slow and obviously
right. `ring.RnsRing` computes the same ring through `frx.lax.ntt` and field
dtypes so a tracer can enter it. Every case here is the same question: do the
two agree.

**That question also settles the output order**, which is the property this
package says no property test can catch. It cannot be caught *within* one
implementation — a consistently permuted transform round-trips and convolves
like the correct one — but it is caught *between* two, because the host side's
order is lattigo's natively and a permutation would have to be present in both
to hide. The chain is: host == lattigo (the consumer's golden, at its own moduli
chain), and ring == host (here). What that does not cover is the two drifting
together, which is what the consumer's cross-verification on a pin bump is for.

The moduli are 36-bit on purpose. Past a 32-bit lane, so a ring that narrows a residue fails here rather than at a consumer's wire boundary — and at
that width the host side cannot use `uint64` arithmetic either, which is why
the reference carries exact Python integers and the comparison below goes
through `to_host`.
"""

import random

import frx
import numpy as np
import pytest

from lattice_frx import host_ring as host_mod
from lattice_frx import ring as ring_mod

# NTT-friendly, 36-bit, and `1 mod 2d` at d=256.
_Q_MODULI = (34359753217, 34359754753)
_D = 256


@pytest.fixture(scope="module")
def rings():
    return host_mod.HostRnsRing(_Q_MODULI, _D), ring_mod.RnsRing(_Q_MODULI, _D)


def _random_host(seed: int) -> np.ndarray:
    rnd = random.Random(seed)
    return np.array(
        [[rnd.randrange(0, q) for _ in range(_D)] for q in _Q_MODULI], dtype=np.uint64
    )


def test_the_field_carries_the_full_limb_width(rings) -> None:
    """The reason this backend exists: 36-bit residues survive the round trip.

    An integer lane would not carry them — frx runs without x64, so a `uint64`
    array narrows to `uint32` and truncates. A field's storage follows its
    modulus instead.
    """
    _, dev = rings
    host = _random_host(1)
    assert np.array_equal(dev.to_host(dev.from_host(host)), host)
    assert max(int(v) for v in host[0]) > (1 << 32)


def test_ntt_agrees_with_the_reference(rings) -> None:
    """The order gate, transitively: the reference's order is lattigo's."""
    ref, dev = rings
    host = _random_host(2)
    assert np.array_equal(dev.to_host(dev.ntt(dev.from_host(host))), ref.ntt(host))


def test_intt_agrees_with_the_reference(rings) -> None:
    ref, dev = rings
    host = _random_host(3)
    assert np.array_equal(dev.to_host(dev.intt(dev.from_host(host))), ref.intt(host))


def test_intt_inverts_ntt(rings) -> None:
    _, dev = rings
    host = _random_host(4)
    element = dev.from_host(host)
    assert np.array_equal(dev.to_host(dev.intt(dev.ntt(element))), host)


@pytest.mark.parametrize(
    "op", ["add", "sub", "mul", "mul_add", "mul_scalar", "mul_scalar_then_sub", "neg"]
)
def test_arithmetic_agrees_with_the_reference(rings, op: str) -> None:
    ref, dev = rings
    a_host, b_host, c_host = _random_host(5), _random_host(6), _random_host(7)
    a, b, c = dev.from_host(a_host), dev.from_host(b_host), dev.from_host(c_host)
    scalar = 123456789

    if op == "neg":
        got, want = dev.neg(a), ref.neg(a_host)
    elif op in ("add", "sub", "mul"):
        got = getattr(dev, op)(a, b)
        want = getattr(ref, op)(a_host, b_host)
    elif op == "mul_add":
        got, want = dev.mul_add(a, b, c), ref.mul_add(a_host, b_host, c_host)
    elif op == "mul_scalar":
        got, want = dev.mul_scalar(a, scalar), ref.mul_scalar(a_host, scalar)
    else:
        got = dev.mul_scalar_then_sub(a, scalar, c)
        want = ref.mul_scalar_then_sub(a_host, scalar, c_host)

    assert np.array_equal(dev.to_host(got), want)


def test_from_signed_agrees_with_the_reference(rings) -> None:
    ref, dev = rings
    rnd = random.Random(8)
    values = [rnd.randrange(-(1 << 20), 1 << 20) for _ in range(_D)]
    assert np.array_equal(dev.to_host(dev.from_signed(values)), ref.from_signed(values))


def test_to_balanced_limb0_agrees_with_the_reference(rings) -> None:
    ref, dev = rings
    host = _random_host(9)
    got = dev.to_balanced_limb0(dev.from_host(host))
    want = ref.to_balanced_limb0(host)
    assert got.dtype == np.int64
    assert np.array_equal(got, want)


def test_the_ring_is_negacyclic(rings) -> None:
    """`X * X^(d-1) = -1`, through the ring's own transform."""
    _, dev = rings
    x = dev.from_signed([0, 1] + [0] * (_D - 2))
    x_pow = dev.from_signed([0] * (_D - 1) + [1])
    product = dev.intt(dev.mul(dev.ntt(x), dev.ntt(x_pow)))
    want = dev.from_signed([-1] + [0] * (_D - 1))
    assert np.array_equal(dev.to_host(product), dev.to_host(want))


def test_the_transform_traces_as_one_computation(rings) -> None:
    """The point of the backend: `ntt` composes into a `jit` zone.

    The element is a tuple of per-limb field arrays, which is a pytree already,
    so it crosses the boundary without a registered type of its own.
    """
    ref, dev = rings
    host = _random_host(10)
    compiled = frx.jit(dev.ntt)
    assert np.array_equal(dev.to_host(compiled(dev.from_host(host))), ref.ntt(host))


def test_an_element_survives_a_vmapped_batch(rings) -> None:
    """A batch axis over elements, which is what a consumer's hot path adds."""
    ref, dev = rings
    hosts = [_random_host(11), _random_host(12), _random_host(13)]
    stacked = tuple(
        frx.numpy.stack([dev.from_host(h)[i] for h in hosts])
        for i in range(len(_Q_MODULI))
    )
    batched = frx.jit(frx.vmap(dev.ntt))(stacked)
    for index, host in enumerate(hosts):
        got = dev.to_host(tuple(limb[index] for limb in batched))
        assert np.array_equal(got, ref.ntt(host))
