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

## Why the element is two types

Which domain a value is in is part of what it is: pointwise `mul` is the ring's
multiplication only in the NTT domain, and a balanced lift only means anything
in the coefficient domain — same storage, different object. lattigo carries
this as a runtime `IsNTT` flag per polynomial; here the information is static
at trace time, so it is carried by the container type instead: `Coeff` and
`Eval`, two one-field `NamedTuple`s over the same limbs tuple. A flag would
have been a trace-splitting branch for something the graph already knows; a
type makes `mul(coeff, coeff)` a `TypeError` at trace time and costs the
compiled graph nothing. The host contract carries no domain — a `(limbs, d)`
array is just bytes — so the two embedding constructors are named for the
domain the caller is asserting, `coeff_from_host` and `eval_from_host`.

Limb arrays may carry leading batch axes: every op here is pointwise per limb
or transforms `axis=-1`, so an element whose limbs are `[..., d]` is a batch
of elements, and a module vector (MLWE's `s`, a commitment's opening) is the
`[k, d]` case of the same convention. `stack` assembles those batches, and
`matvec` is the one op that reads the axes rather than mapping over them.

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

from collections.abc import Sequence
from typing import Any, NamedTuple, TypeVar

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import lax

from lattice_frx.roots import (
    bit_reverse,
    galois_map,
    normalize_galois_k,
    prime_factors,
    primitive_root,
    slot_exponents,
)


class Coeff(NamedTuple):
    """A coefficient-domain element: one `[..., d]` field array per limb."""

    limbs: tuple[Any, ...]


class Eval(NamedTuple):
    """An NTT-domain element, in the contract's (lattigo's) bit-reversed order."""

    limbs: tuple[Any, ...]


# Domain-generic ops take and return one domain, the same one; `mul` and
# `matvec` are not among them, which is the point of having the two types.
_E = TypeVar("_E", Coeff, Eval)


def _require_domain(op: str, domain: type, *values) -> None:
    """Every operand in the one domain `op` is defined in, or a `TypeError`.

    Variadic so an op is one call covering all operands — a per-operand
    guard that forgets one would readmit, at that operand, the bug class
    the types exist to kill.
    """
    for value in values:
        if type(value) is not domain:
            raise TypeError(
                f"{op}: expected {domain.__name__}, got {type(value).__name__}"
            )


def _same_domain(op: str, *values) -> type:
    """The shared domain of `values`, or a `TypeError` naming the mismatch."""
    first = type(values[0])
    if first not in (Coeff, Eval) or any(type(v) is not first for v in values[1:]):
        raise TypeError(
            f"{op}: operands must share a domain, got "
            + ", ".join(type(v).__name__ for v in values)
        )
    return first


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
        # Per-`k` gather/sign and permutation tables for the two `galois`
        # forms — trace-time constants, built lazily because most rings
        # never rotate.
        self._galois_tables: dict[int, tuple[Any, Any]] = {}
        self._galois_eval_perms: dict[int, Any] = {}

    def _limbs_from_host(self, arr: np.ndarray) -> tuple[Any, ...]:
        """The host contract's `(limbs, d)` uint64 array as per-limb field arrays.

        The reduction happens in Python-int precision before the cast, because
        a residue at these widths cannot be reduced after landing in a lane.
        """
        return tuple(
            fnp.asarray(
                np.array([int(v) % q for v in arr[i]], dtype=np.uint64).astype(field)
            )
            for i, (q, field) in enumerate(zip(self.q_moduli, self.fields))
        )

    def coeff_from_host(self, arr: np.ndarray) -> Coeff:
        """A host array as a coefficient-domain element.

        The host contract carries no domain — the caller is asserting one by
        choosing this constructor over `eval_from_host`.
        """
        return Coeff(self._limbs_from_host(arr))

    def eval_from_host(self, arr: np.ndarray) -> Eval:
        """A host array as an NTT-domain element, asserted to be in the
        contract's (lattigo's) bit-reversed order."""
        return Eval(self._limbs_from_host(arr))

    def to_host(self, a: Coeff | Eval) -> np.ndarray:
        """An element back as the host contract — the boundary, not a traced op.

        The domain does not survive the trip; the host array is just bytes.
        """
        return np.array(
            [[int(v) for v in np.asarray(limb).astype(object)] for limb in a.limbs],
            dtype=np.uint64,
        )

    def from_signed(self, values) -> Coeff:
        """A signed integer coefficient vector, embedded per limb.

        `Coeff` by construction: "small signed coefficients" is a
        coefficient-domain notion.
        """
        row = [int(v) for v in values]
        if len(row) != self.d:
            raise ValueError(f"from_signed: expected {self.d} values, got {len(row)}")
        return Coeff(
            tuple(
                fnp.asarray(
                    np.array([v % q for v in row], dtype=np.uint64).astype(field)
                )
                for q, field in zip(self.q_moduli, self.fields)
            )
        )

    def ntt(self, a: Coeff) -> Eval:
        """Forward transform, in lattigo's output order.

        The opcode emits natural order; the gather is the adapter that presents
        the order this package's contract promises.
        """
        _require_domain("ntt", Coeff, a)
        return Eval(
            tuple(
                fnp.take(
                    lax.ntt(limb, ntt_type=lax.NttType.NEGACYCLIC_NTT, generator=g),
                    self._bit_reverse,
                    axis=-1,
                )
                for limb, g in zip(a.limbs, self._generators)
            )
        )

    def intt(self, a: Eval) -> Coeff:
        """Inverse transform, undoing `ntt` including its trailing `1/d`.

        The adapter runs first here — the input is in lattigo's order and the
        opcode wants its own — and `NEGACYCLIC_INTT` applies the `1/d` scaling
        that lattigo folds into the tail of its GS network.
        """
        _require_domain("intt", Eval, a)
        return Coeff(
            tuple(
                lax.ntt(
                    fnp.take(limb, self._bit_reverse, axis=-1),
                    ntt_type=lax.NttType.NEGACYCLIC_INTT,
                    generator=g,
                )
                for limb, g in zip(a.limbs, self._generators)
            )
        )

    def add(self, a: _E, b: _E) -> _E:
        domain = _same_domain("add", a, b)
        return domain(tuple(x + y for x, y in zip(a.limbs, b.limbs)))

    def sub(self, a: _E, b: _E) -> _E:
        domain = _same_domain("sub", a, b)
        return domain(tuple(x - y for x, y in zip(a.limbs, b.limbs)))

    def neg(self, a: _E) -> _E:
        domain = _same_domain("neg", a)
        return domain(tuple(-x for x in a.limbs))

    def galois(self, a: Coeff, k: int) -> Coeff:
        """`σ_k : X ↦ X^k` on coefficients — one gather and one sign-select
        per limb, from `roots.galois_map`'s action inverted into a `take`
        table. Conformance against the host oracle is what pins it."""
        _require_domain("galois", Coeff, a)
        src, negate = self._galois_table(k)
        gathered = tuple(fnp.take(limb, src, axis=-1) for limb in a.limbs)
        return Coeff(tuple(fnp.where(negate, -g, g) for g in gathered))

    def galois_eval(self, a: Eval, k: int) -> Eval:
        """`σ_k` without leaving the NTT domain: a pure slot permutation.

        Slot `j` of the contract's order holds the evaluation at `ψ^{e(j)}`
        (`roots.slot_exponents`), so output slot `j` reads the input slot
        holding `ψ^{k·e(j)}`. Which `2d`-th root plays `ψ` cancels out of
        the permutation (any two differ by an odd unit `c`, and `e ↦ c·e`
        commutes with `e ↦ k·e`); that the closed-form exponent table really
        is the pinned order's is re-derived from the host oracle in
        `ring_test`, not trusted."""
        _require_domain("galois_eval", Eval, a)
        perm = self._galois_eval_perm(k)
        return Eval(tuple(fnp.take(limb, perm, axis=-1) for limb in a.limbs))

    def _galois_table(self, k: int):
        """`(src, negate)` for `σ_k` as a gather: output index `m` reads
        source `src[m]`, negated where `negate[m]` — `galois_map`'s forward
        action inverted by two numpy scatters. One bool mask serves every
        limb; the sign is a `where`, not per-limb sign arrays."""
        key = normalize_galois_k(self.d, k)
        if key not in self._galois_tables:
            dest, negate = galois_map(self.d, key)
            src = np.empty(self.d, dtype=np.int32)
            src[dest] = np.arange(self.d, dtype=np.int32)
            negate_at_dest = np.zeros(self.d, dtype=bool)
            negate_at_dest[dest] = negate
            self._galois_tables[key] = (src, negate_at_dest)
        return self._galois_tables[key]

    def _galois_eval_perm(self, k: int):
        """Both directions of the slot↔exponent map are closed-form
        (`e = 2·brv + 1` and its inverse `t ↦ brv((t-1)/2)`), so the
        permutation is pure index arithmetic — no table beyond the cached
        result."""
        key = normalize_galois_k(self.d, k)
        if key not in self._galois_eval_perms:
            bits = (self.d - 1).bit_length()
            self._galois_eval_perms[key] = np.array(
                [
                    bit_reverse((((key * e) % (2 * self.d)) - 1) >> 1, bits)
                    for e in slot_exponents(self.d)
                ],
                dtype=np.int32,
            )
        return self._galois_eval_perms[key]

    def mul(self, a: Eval, b: Eval) -> Eval:
        """Pointwise — the ring's multiplication in the NTT domain *only*,
        which is what requiring `Eval` operands proves."""
        _require_domain("mul", Eval, a, b)
        return Eval(tuple(x * y for x, y in zip(a.limbs, b.limbs)))

    def mul_add(self, a: Eval, b: Eval, acc: Eval) -> Eval:
        _require_domain("mul_add", Eval, a, b, acc)
        return Eval(tuple(x * y + z for x, y, z in zip(a.limbs, b.limbs, acc.limbs)))

    def stack(self, elements: Sequence[_E]) -> _E:
        """A batch element from same-domain elements: limbs gain a leading axis.

        The module convention's constructor — `stack` of `k` elements is the
        `[k, d]` vector `matvec` consumes, and stacking stacks nests to
        matrices. Consumers assemble batches through this rather than
        re-deriving the limb transpose by hand.
        """
        domain = _same_domain("stack", *elements)
        return domain(
            tuple(fnp.stack(list(limbs)) for limbs in zip(*(e.limbs for e in elements)))
        )

    def matvec(self, mat: Eval, vec: Eval) -> Eval:
        """`A·s` over the ring: per limb `[..., m, k, d] × [..., k, d] → [..., m, d]`.

        The module layer (MLWE's `A·s`, a commitment key's application) is this
        shape convention, not a type of its own: a vector of ring elements is
        an element whose limbs carry a leading axis. The contraction multiplies
        pointwise, so it exists only in the NTT domain — which the operand
        types prove.
        """
        _require_domain("matvec", Eval, mat, vec)
        # Limbs of one element share their leading axes, so limb 0 speaks for
        # all — and the guard states the contract itself (the contracted `k`
        # extent), not just the rank relation, because a size-1 axis would
        # broadcast straight past a rank check into well-shaped wrong values.
        m_limb, v_limb = mat.limbs[0], vec.limbs[0]
        if m_limb.ndim != v_limb.ndim + 1 or m_limb.shape[-2] != v_limb.shape[-2]:
            raise ValueError(
                f"matvec: mat limbs {tuple(m_limb.shape)} do not contract with "
                f"vec limbs {tuple(v_limb.shape)}; want mat = vec's shape plus "
                "one leading `m` axis, sharing the `k` extent"
            )
        return Eval(
            tuple(
                (m * v[..., None, :, :]).sum(axis=-2)
                for m, v in zip(mat.limbs, vec.limbs)
            )
        )

    def _scalar(self, s: int, index: int) -> Any:
        """A host integer as a field element of limb `index`."""
        q, field = self.q_moduli[index], self.fields[index]
        return fnp.asarray(np.array([int(s) % q], dtype=np.uint64).astype(field))[0]

    def mul_scalar(self, a: _E, s: int) -> _E:
        """Domain-generic: a scalar acts coefficient-wise in either domain."""
        domain = _same_domain("mul_scalar", a)
        return domain(
            tuple(limb * self._scalar(s, i) for i, limb in enumerate(a.limbs))
        )

    def mul_scalar_then_sub(self, a: _E, s: int, acc: _E) -> _E:
        domain = _same_domain("mul_scalar_then_sub", a, acc)
        return domain(
            tuple(
                z - limb * self._scalar(s, i)
                for i, (limb, z) in enumerate(zip(a.limbs, acc.limbs))
            )
        )

    def to_balanced_limb0(self, a: Coeff) -> np.ndarray:
        """Limb 0's coefficients as signed integers in `(-q0/2, q0/2]`.

        A **host boundary op**, like `rns.reconstruct_centered`: the balanced
        lift of a residue this wide does not fit a signed lane, so this
        materialises to the host and returns `int64` rather than pretending to
        be traceable. Calling it under `jit` fails, which is the honest
        outcome — the traced path has no signed representation to give.

        `Coeff` only: "small signed coefficients" is a coefficient-domain
        notion, and a balanced lift of NTT-domain values is a bug wearing a
        plausible shape.
        """
        _require_domain("to_balanced_limb0", Coeff, a)
        q0 = self.q_moduli[0]
        row = [int(v) for v in np.asarray(a.limbs[0]).astype(object)]
        half = q0 >> 1
        return np.array([v - q0 if v > half else v for v in row], dtype=np.int64)
