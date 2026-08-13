"""The RNS negacyclic ring `Z[X]/(X^d + 1)`: one field array per limb, traced.

This is the ring — the one a consumer computes with, and the one that composes
into a `jit` zone. `host_ring.HostRnsRing` is the same ring written the slow
obvious way, over exact Python integers a coefficient at a time, reproducing
lattigo's tables and loop structure natively; it is what this is checked
against, and it is a host implementation rather than a lesser one, since a
reconstruction or a balanced lift has nowhere to happen but the host.

The two are named the way round they are used: the unqualified name is the
default, and `host_` marks the one you reach for deliberately.

## Why the array contract is different from the host one

`canonical.py`'s `(limbs, d)` `uint64` is a host contract, and the reason it
cannot be this one is not style. frx runs without x64, so `uint64`
narrows to `uint32` and every residue above `2**32` truncates without raising —
at `primes.MAX_MODULUS_BITS = 50` that is every limb this package targets. The
width has to be carried by a **field dtype**, whose storage follows its modulus
(`zk_dtypes.prime_field` mints a 64-bit field at a 50-bit `q`) rather than by an
integer lane, and which reduces internally so no modular arithmetic is written
here at all.

A field dtype is per-modulus and an array has one dtype, so an RNS element with
a different `q_l` per limb cannot be one array. It is a **tuple of `[d]` field
arrays, one per limb** — which is already a pytree, so it threads `jit` and
`vmap` without a registered type of its own. The moduli live on the ring, which
is static.

## Why the transform is one opcode and one gather

`frx.lax.ntt`'s `NEGACYCLIC_NTT` is this ring's transform. It takes a generator
of the multiplicative group and derives the root itself as `g^((q-1)/2d)` —
which is *exactly* `generateNTTConstants`' `psi = g^((q-1)/NthRoot)` with
lattigo's `PrimitiveRoot` as `g`. So the root needs no search here: the
reference's own `_primitive_root` feeds the opcode and the two land on the same
psi by construction. (FIPS 204 is the harder case — it names the root rather
than the generator, so a scheme against it has to search for the preimage.)

What the opcode does not match is the order. It emits natural order where
lattigo's bit-reversed table order is what this package's contract promises, and
the two differ by exactly one bit-reversal — measured against lattigo's own
vector before that fixture went back to its consumer. The host side applies no such
permutation and reserved this adapter for whatever computes in another
order; `_bit_reverse` below is it, applied
at this module's boundary and nowhere else. It is an involution, so the same
index vector serves both directions.

## What is not here, and why

`to_balanced_limb0` and everything in `rns.py` that reconstructs are **host
boundary** operations, not traced ones: a balanced lift of a 50-bit residue does
not fit a signed lane, and a CRT reconstruction spans `50 * limbs` bits, past
every lane and every field. `rns.reconstruct_centered` already returns
`list[int]` for that reason. `to_balanced_limb0` below therefore materialises to
the host and says so, rather than pretending to be traceable and truncating.
"""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import lax

from lattice_frx.roots import bit_reverse, prime_factors, primitive_root

# An RNS element: one `[d]` field array per limb, in `q_moduli` order. A tuple
# is a pytree already, which is the whole reason it is not a class.
RnsElement = tuple[Any, ...]


class RnsRing:
    """`Z[X]/(X^d + 1)` over an RNS chain, traced.

    Same op surface and same output *values* as `host_ring.HostRnsRing`,
    against a per-limb field-array element instead of a `(limbs, d)` uint64
    array. `ring_test` requires the two to agree, which is also what pins this
    ring's output order: the host side reproduces lattigo's order natively, so
    agreeing with it is agreeing with lattigo.
    """

    def __init__(self, q_moduli, d: int) -> None:
        self.q_moduli = tuple(int(q) for q in q_moduli)
        self.d = d
        if d & (d - 1):
            raise ValueError(f"d ({d}) must be a power of 2")
        for q in self.q_moduli:
            if (q - 1) % (2 * d):
                raise ValueError(f"{q} is not 1 mod NthRoot={2 * d}")
        self.fields = tuple(zk_dtypes.prime_field(q) for q in self.q_moduli)
        # lattigo's `PrimitiveRoot` *is* what `lax.ntt` calls `generator`.
        self._generators = tuple(
            primitive_root(q, prime_factors(q - 1)) for q in self.q_moduli
        )
        self._bit_reverse = np.array(
            [bit_reverse(i, (d - 1).bit_length()) for i in range(d)], dtype=np.int32
        )

    def from_host(self, arr: np.ndarray) -> RnsElement:
        """The host contract's `(limbs, d)` uint64 array as an element.

        The reduction happens in Python-int precision before the cast, because
        a residue at these widths cannot be reduced after landing in a lane.
        """
        return tuple(
            fnp.asarray(
                np.array([int(v) % q for v in arr[i]], dtype=np.uint64).astype(field)
            )
            for i, (q, field) in enumerate(zip(self.q_moduli, self.fields))
        )

    def to_host(self, a: RnsElement) -> np.ndarray:
        """An element back as the host contract — the boundary, not a traced op."""
        return np.array(
            [[int(v) for v in np.asarray(limb).astype(object)] for limb in a],
            dtype=np.uint64,
        )

    def from_signed(self, values) -> RnsElement:
        """A signed integer coefficient vector, embedded per limb."""
        row = [int(v) for v in values]
        if len(row) != self.d:
            raise ValueError(f"from_signed: expected {self.d} values, got {len(row)}")
        return tuple(
            fnp.asarray(
                np.array([v % q for v in row], dtype=np.uint64).astype(field)
            )
            for q, field in zip(self.q_moduli, self.fields)
        )

    def ntt(self, a: RnsElement) -> RnsElement:
        """Forward transform, in lattigo's output order.

        The opcode emits natural order; the gather is the adapter that presents
        the order this package's contract promises.
        """
        return tuple(
            fnp.take(
                lax.ntt(limb, ntt_type=lax.NttType.NEGACYCLIC_NTT, generator=g),
                self._bit_reverse,
                axis=-1,
            )
            for limb, g in zip(a, self._generators)
        )

    def intt(self, a: RnsElement) -> RnsElement:
        """Inverse transform, undoing `ntt` including its trailing `1/d`.

        The adapter runs first here — the input is in lattigo's order and the
        opcode wants its own — and `NEGACYCLIC_INTT` applies the `1/d` scaling
        that lattigo folds into the tail of its GS network.
        """
        return tuple(
            lax.ntt(
                fnp.take(limb, self._bit_reverse, axis=-1),
                ntt_type=lax.NttType.NEGACYCLIC_INTT,
                generator=g,
            )
            for limb, g in zip(a, self._generators)
        )

    def add(self, a: RnsElement, b: RnsElement) -> RnsElement:
        return tuple(x + y for x, y in zip(a, b))

    def sub(self, a: RnsElement, b: RnsElement) -> RnsElement:
        return tuple(x - y for x, y in zip(a, b))

    def neg(self, a: RnsElement) -> RnsElement:
        return tuple(-x for x in a)

    def mul(self, a: RnsElement, b: RnsElement) -> RnsElement:
        """Pointwise — the NTT domain's multiplication, as in the reference."""
        return tuple(x * y for x, y in zip(a, b))

    def mul_add(self, a: RnsElement, b: RnsElement, acc: RnsElement) -> RnsElement:
        return tuple(x * y + z for x, y, z in zip(a, b, acc))

    def _scalar(self, s: int, index: int) -> Any:
        """A host integer as a field element of limb `index`."""
        q, field = self.q_moduli[index], self.fields[index]
        return fnp.asarray(np.array([int(s) % q], dtype=np.uint64).astype(field))[0]

    def mul_scalar(self, a: RnsElement, s: int) -> RnsElement:
        return tuple(limb * self._scalar(s, i) for i, limb in enumerate(a))

    def mul_scalar_then_sub(
        self, a: RnsElement, s: int, acc: RnsElement
    ) -> RnsElement:
        return tuple(
            z - limb * self._scalar(s, i) for i, (limb, z) in enumerate(zip(a, acc))
        )

    def to_balanced_limb0(self, a: RnsElement) -> np.ndarray:
        """Limb 0's coefficients as signed integers in `(-q0/2, q0/2]`.

        A **host boundary op**, like `rns.reconstruct_centered`: the balanced
        lift of a residue this wide does not fit a signed lane, so this
        materialises to the host and returns `int64` rather than pretending to
        be traceable. Calling it under `jit` fails, which is the honest
        outcome — the traced path has no signed representation to give.
        """
        q0 = self.q_moduli[0]
        row = [int(v) for v in np.asarray(a[0]).astype(object)]
        half = q0 >> 1
        return np.array([v - q0 if v > half else v for v in row], dtype=np.int64)
