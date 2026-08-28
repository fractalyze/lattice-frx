"""The partial-split ring — `Z_q[X]/(X^d + 1)` at moduli `q ≡ 5 (mod 8)`.

The second modulus shape this substrate carries, and deliberately the
opposite of `host_ring.py`'s: an NTT-friendly limb (`q ≡ 1 mod 2d`) splits
`X^d + 1` into linear factors, which is what makes the NTT work and what
makes small challenge differences non-invertible; a partial-split limb
(`q ≡ 5 mod 8`) splits it into exactly two irreducible degree-`d/2` halves

    X^d + 1 ≡ (X^{d/2} - r)(X^{d/2} + r)  (mod q),   r² ≡ -1 (mod q),

which kills the NTT but buys the property LNP-style proofs (eprint
2022/284) stand on: every nonzero σ₋₁-invariant element with coefficients
below `q/2` is invertible (Lemma 2.6), so soundness extraction can divide
by challenge differences. One ring cannot have both — `q ≡ 1 (mod 2d)`
forces `q ≡ 1 (mod 8)` for `d ≥ 4` — so this is a mode, not a parameter,
and `roots.split_root` rejects a stray NTT-friendly limb loudly.

Two implementations, in the pairing `ring.py` and `host_ring.py` already
have: **`SplitRing`** is the traced default a consumer computes with, and
**`HostSplitRing`** is the slow obvious reference it is checked against.
The host one is the better oracle here than usual, because its product is a
schoolbook convolution over exact Python ints and shares no step with the
CRT halves the traced one contracts.

`HostSplitRing`'s representation and contract are `host_ring.py`'s, shared
through `_HostRingBase`: `(limbs, d)` `dtype=np.uint64` canonical arrays
outside, exact Python ints inside. What it adds over the base:

- `mul` — coefficient-domain negacyclic **schoolbook convolution**. There
  is deliberately no CRT shortcut in it: the schoolbook product is the
  modulus-shape-agnostic oracle that both the split path below and
  `SplitRing`'s traced twisted convolution are checked against — the
  same oracle-vs-accelerated structure the NTT ring has with `ring.py`.
- `to_split` / `from_split` — the two-factor CRT view, an element as its
  residues modulo `X^{d/2} ∓ r`, shape `(limbs, 2, d/2)`. Half `0` is the
  `X^{d/2} ≡ +r` residue, half `1` the `-r` one. This is also the traced
  ring's `Split` domain, and the wire between the two: each half's product
  is an `r`-twisted convolution, i.e. a structured `d/2 × d/2` matvec.
- `matmul` — the batched product, and the one path here that does not go
  through `mul`. It derives the same `X^d ≡ -1` fold as an anticirculant
  matrix rather than a convolution loop, shares no code with the CRT split,
  and is checked against `mul` through `matvec` — so the oracle stance above
  survives it. Exactness is gated on operand magnitude; see the method.
- `is_invertible` — a unit test of the ring in the literal sense: each
  half is a field, so an element is invertible iff every `(limb, half)`
  residue is nonzero. Cheap, host-side, and exactly the predicate the LNP
  challenge space needs pinned.

Neither ring has an `ntt`/`intt`, and neither has an `Eval` domain — the
moduli that make this ring's challenge differences invertible are exactly
the ones with no `2d`-th root of unity. A consumer that wants device-shaped
products reaches `SplitRing` below; it does not get to borrow the NTT
ring's transform, because the two modulus families are mutually exclusive.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from numpy.lib.stride_tricks import as_strided

from lattice_frx.canonical import require_canonical
from lattice_frx.domains import Coeff, Split, require_domain, same_domain
from lattice_frx.host_ring import _HostRingBase
from lattice_frx.roots import split_root


class HostSplitRing(_HostRingBase):
    """The partial-split ring over `prod(q_moduli)`, every limb ≡ 5 (mod 8).

    `split_roots[l]` is `roots.split_root(q_l)` — the canonical square
    root of `-1` naming limb `l`'s two CRT halves.
    """

    _op_prefix = "SplitRing"

    def __init__(self, q_moduli, d: int):
        if d < 4:
            raise ValueError(f"HostSplitRing: degree must be >= 4 to split, got {d!r}")
        super().__init__(q_moduli, d)  # power-of-two degree checked there
        self.split_roots: tuple[int, ...] = tuple(split_root(q) for q in self.q_moduli)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Coefficient-domain negacyclic product, schoolbook: full
        convolution, then `X^d ≡ -1` folds the top half in with a sign.
        O(d²) per limb and proud of it — see the module docstring for why
        the oracle path must not share the CRT split it certifies."""
        a = self._coerce(a, "mul")
        b = self._coerce(b, "mul")
        rows = []
        for l, q in enumerate(self.q_moduli):
            row_b = b[l]
            conv = [0] * (2 * self.d)
            for i, ai in enumerate(a[l]):
                if ai:
                    for j, bj in enumerate(row_b):
                        conv[i + j] += ai * bj
            rows.append([(conv[k] - conv[k + self.d]) % q for k in range(self.d)])
        return np.array(rows, dtype=np.uint64)

    def _negacyclic(self, balanced: np.ndarray) -> np.ndarray:
        """`Neg(a)` for every element of a balanced stack, as a **read-only
        view**: `(..., d) → (..., d, d)` with `Neg(a)[k][j] = ±a[k−j mod d]`.

        `Neg` is Toeplitz — constant down each diagonal — so all `d²` of its
        entries are already present in the `2d−1` values `[−a[d−1] … −a[1],
        a[0] … a[d−1]]`, and the matrix is that buffer read with a row stride
        of `+1` and a column stride of `−1`. Materialising it instead costs
        `rows·cols·limbs·d²` int64 twice over — once for the gather, once
        for the sign multiply. Measured at the LNP shape and `d = 128`,
        that is a 201 MB peak transient against 14 MB here, and it is most
        of why the op is 3x what the materialised form was.

        The negated half is what carries `X^d ≡ −1`: reading left of the
        diagonal walks into it, which is exactly the fold `mul` spells as
        `conv[k] − conv[k + d]`.

        `writeable=False` is load-bearing, not caution. Every value in the
        buffer is aliased `d` times over, so a single write would silently
        change a whole diagonal.
        """
        d = self.d
        buffer = np.concatenate([-balanced[..., 1:], balanced], axis=-1)
        stride = buffer.strides[-1]
        return as_strided(
            buffer[..., d - 1 :],
            shape=buffer.shape[:-1] + (d, d),
            strides=buffer.strides[:-1] + (stride, -stride),
            writeable=False,
        )

    def _fits_int64(self, matrix: np.ndarray, other: np.ndarray) -> bool:
        """Whether `matmul`'s batched path is exact for these operands.

        Its widest partial sum is `cols · d · ‖matrix‖∞ · ‖other‖∞` over
        balanced lifts, and every product and accumulation on that path is
        `int64`. This is the bound computed exactly, not a measured
        crossover: the answer is a property of the arithmetic, so a wrong
        constant here is a wrong *value*, which a test can catch.
        """
        cols = matrix.shape[1]
        widest = int(np.abs(self._balanced(matrix)).max()) * int(
            np.abs(self._balanced(other)).max()
        )
        return cols * self.d * widest < 2**63

    def matmul(self, matrix: np.ndarray, other: np.ndarray) -> np.ndarray:
        """`matrix @ other` over module stacks: `(rows, cols, limbs, d) ×
        (cols, width, limbs, d) → (rows, width, limbs, d)`.

        `matvec`'s two-sided sibling, and the only op here that does not
        reach `mul` per entry. A negacyclic product is a matrix-vector
        product — `a·b = Neg(a) @ b` for the anticirculant `Neg(a)[k][j] =
        ±a[k−j mod d]` — so the whole contraction is one `int64` sum over
        `(cols, d)`, with `Neg` built once over `matrix` and reused down
        every column of `other`, and `Neg` itself is a view rather than an
        array (`_negacyclic`). That reuse is the win: against the
        `matvec`-per-column spelling a consumer writes today, **112x** at
        d = 64 and **116x** at d = 128, measured at `(256, 3) × (3, 16)`.

        `mul` is untouched and stays the schoolbook oracle — this path is
        checked against it, through `matvec`, and shares none of it.

        **Exactness.** The batched path holds only while `cols · d ·
        ‖matrix‖∞ · ‖other‖∞` fits `int64` (`_fits_int64`); past that the
        op falls back to composing `matvec` per column. So `matmul` is
        correct for any operands and never slower than the spelling it
        replaces — what varies is only whether it is faster. The consumer
        this exists for always fits: a `Bin_1` ternary challenge matrix
        against a ~32-bit modulus at d = 128 leaves over 23 bits of
        headroom.

        Building `Neg` needs the coefficient layout, which is why the op
        lives here and not in the consumer: this module's convention hands
        consumers `(limbs, d)` canonical arrays, not the position of a
        coefficient within one.

        For the vector case, pass the column: `matmul(A, v[:, None])`.
        `matvec` stays on the schoolbook composition on purpose — base
        `matvec` is the oracle the future traced twisted-convolution
        matvec is checked against, so it must not borrow this path.
        """
        limbs_d = (len(self.q_moduli), self.d)
        mshape = getattr(matrix, "shape", None)
        oshape = getattr(other, "shape", None)
        if (
            mshape is None
            or oshape is None
            or len(mshape) != 4
            or mshape[1] == 0
            or mshape[2:] != limbs_d
            or oshape[2:] != limbs_d
            or oshape[0] != mshape[1]
        ):
            raise ValueError(
                f"{self._op_prefix}.matmul: expected (rows, cols, limbs, d) × "
                f"(cols, width, limbs, d) with cols >= 1 and (limbs, d) = "
                f"{limbs_d}, got {mshape!r} × {oshape!r}"
            )
        # `matvec`'s empty case, same reason and same contract note.
        self._require(matrix, "matmul", batched=True)
        self._require(other, "matmul", batched=True)
        if mshape[0] == 0 or oshape[1] == 0:
            return self.zeros(mshape[0], oshape[1])
        if not self._fits_int64(matrix, other):
            return np.stack(
                [self.matvec(matrix, other[:, w]) for w in range(oshape[1])],
                axis=1,
            )
        neg = self._negacyclic(self._balanced(matrix))
        # No `optimize=`: the contraction is `int64`, so no path it could
        # pick reaches BLAS, and the intermediates it materialises to get
        # there cost 2.3x the straight sum — measured at both degrees.
        product = np.einsum("rclkj,cwlj->rwlk", neg, self._balanced(other))
        # `_reduce`'s modulus column is object dtype, so it would run this
        # `%` as per-element Python-int arithmetic — measured 10.6x slower,
        # which is the cost this path exists to avoid. Same values.
        return (product % self._q_int64).astype(np.uint64)

    def to_split(self, a: np.ndarray) -> np.ndarray:
        """The two-factor CRT view, `(limbs, 2, d/2)`: half `h` holds the
        residue modulo `X^{d/2} - s` for `s = +r` (h=0) / `-r` (h=1),
        i.e. `low[i] + s·high[i]` after substituting `X^{d/2} ≡ s`."""
        a = self._coerce(a, "to_split")
        half = self.d // 2
        out = np.empty((len(self.q_moduli), 2, half), dtype=np.uint64)
        for l, q in enumerate(self.q_moduli):
            r = self.split_roots[l]
            low, high = a[l][:half], a[l][half:]
            out[l, 0] = [(lo + r * hi) % q for lo, hi in zip(low, high)]
            out[l, 1] = [(lo - r * hi) % q for lo, hi in zip(low, high)]
        return out

    def from_split(self, sp: np.ndarray) -> np.ndarray:
        """Inverse of `to_split`: `low = (u + v)/2`, `high = (u - v)/(2r)`
        per limb, both divisions exact modulo the odd prime q."""
        half = self.d // 2
        limbs = len(self.q_moduli)
        if not isinstance(sp, np.ndarray) or sp.shape != (limbs, 2, half):
            raise ValueError(
                f"SplitRing.from_split: expected shape {(limbs, 2, half)}, "
                f"got {getattr(sp, 'shape', None)!r}"
            )
        require_canonical(sp.reshape(limbs, self.d), self.q_moduli, "SplitRing.from_split")
        out_rows = []
        for l, q in enumerate(self.q_moduli):
            inv2 = pow(2, -1, q)
            inv2r = pow(2 * self.split_roots[l], -1, q)
            # int() is load-bearing here: `sp` never passes _coerce, so its
            # elements are np.uint64 scalars whose products would wrap.
            u = [int(x) for x in sp[l, 0]]
            v = [int(x) for x in sp[l, 1]]
            low = [(ui + vi) * inv2 % q for ui, vi in zip(u, v)]
            high = [(ui - vi) * inv2r % q for ui, vi in zip(u, v)]
            out_rows.append(low + high)
        return np.array(out_rows, dtype=np.uint64)

    def is_invertible(self, a: np.ndarray) -> bool:
        """Each CRT half is a field, so `a` is a unit iff no `(limb, half)`
        residue is identically zero — the Lemma-2.6 predicate the LNP
        challenge space consumes."""
        sp = self.to_split(a)
        return bool(sp.any(axis=2).all())


# ---------------------------------------------------------------------------
# The array contract changes here. Above is the **host** one — `(limbs, d)`
# `np.uint64`, exact Python ints inside. Below, an element is a tuple of
# per-limb `frx` field arrays, one per modulus, and `np.uint64` survives in
# exactly two roles: the staging dtype an integer passes through on its way
# to `.astype(field)`, and what `to_host` hands back. It is never what a
# traced op computes on. CLAUDE.md calls that seam the most confusable thing
# in the package — `np.uint64` narrows to `uint32` under frx and truncates
# silently — so keeping the two roles above straight is the point of naming
# the boundary methods (`coeff_from_host`, `split_from_host`, `to_host`)
# rather than letting the conversion happen wherever it is convenient.
# ---------------------------------------------------------------------------

_DOMAINS = (Coeff, Split)

# Domain-generic ops take and return one domain, the same one; `mul` is not
# among them, which is the point of having the two types.
_S = TypeVar("_S", Coeff, Split)


class SplitRing:
    """The partial-split ring `Z_q[X]/(X^d + 1)` at `q ≡ 5 (mod 8)`, traced.

    `HostSplitRing`'s traced counterpart, and the same pairing `ring.py` has
    with `host_ring.py`: the unqualified name is the one a consumer computes
    with and the one that composes into a `jit` zone, while the `host_` one is
    the slow obvious reference it is checked against. The host ring earns that
    role here twice over — it multiplies by schoolbook convolution over exact
    Python ints, which shares no step with the CRT halves below.

    ## Why the element is two types, and why neither is `Eval`

    Same discipline as `ring.py`, one domain different. `Coeff` is shared —
    coefficients mean the same thing in both rings — and this ring's second
    domain is `Split`, the two-factor CRT view: an element as its residues
    modulo `X^{d/2} ∓ r`, one `[..., 2, d/2]` field array per limb.

    `Split` is emphatically *not* `Eval`. A fully-splitting modulus reduces
    the ring to `d` copies of `F_q`, which is what makes the NTT ring's `mul`
    pointwise. A partial-split modulus reduces it to two copies of
    `F_q[X]/(X^{d/2} ∓ r)`, which are fields but not `F_q` — so the product
    of two halves is still a convolution, just a twisted one of half the
    degree. The domain buys structure and a 2x smaller product, not a free
    one. That is the price of the invertibility LNP soundness extracts with
    (eprint 2022/284, Lemma 2.6). It also means crossing between the domains
    costs almost nothing — `to_split` is `d/2` multiplies and `d` adds per
    limb, an `O(d)` change of basis rather than an `O(d log d)` transform —
    so a consumer crosses per operation instead of planning a domain
    schedule around it, which is the opposite of how the NTT ring is used.

    ## Why `mul` is a gather rather than a matrix

    In `F_q[X]/(X^{d/2} - s)` the product is `T_s(u)·v` for the `s`-circulant
    `T_s(u)[k][j] = u[(k-j) mod n]`, scaled by `s` where the index wraps. All
    `n²` entries of `T_s(u)` are already present in the `2n-1` values
    `[s·u[1] … s·u[n-1], u[0] … u[n-1]]`, so the matrix is one `take` from
    that buffer and the twist costs `n-1` multiplies rather than `n²`. This
    is the same Toeplitz observation `HostSplitRing._negacyclic` makes about
    the anticirculant `Neg`, at `s = -1` and half the degree — but expressed
    as a gather rather than a stride trick, because a traced backend has no
    `as_strided` and a gather is what it would lower to anyway.

    Both halves ride the same gather: the twists differ per half (`+r`, `-r`)
    and enter through the buffer, so `mul` is one `take` and one contraction
    over a `[..., 2, n, n]` operand, with every leading batch axis carried
    through untouched.

    ## What is not here

    No `ntt`/`intt` — the moduli that make this ring invertible are exactly
    the ones that have no `2d`-th root of unity.

    No `to_balanced_limb0`, which `RnsRing` does carry. There it is a host
    boundary op that materialises because a 50-bit balanced lift fits no
    lane; here the same lift is already one `to_host` away from
    `HostSplitRing`'s, so a second spelling would be a copy of a boundary
    rather than a capability. Norms stay there for the same reason.

    No module layer yet — `matvec`/`matmul`/`combine`/`galois` are still
    `HostSplitRing`-only. The domain and the product are what a consumer
    needs pinned first, since the module ops contract *through* `mul` and a
    wrong twist would be inherited by all of them.

    The **constructors** are here, though, and they come before that layer
    on purpose. Counted across the only consumer, `zeros`, `one`,
    `uniform_stack` and `from_signed_stack` are reached an order of
    magnitude more often than `mul` is, and without them a consumer builds
    every operand through `HostSplitRing` and crosses `coeff_from_host` —
    so `SplitRing` could not close a `jit` zone even for a `mul`-only
    workload. A ring that passes every boundary check and is still unusable
    by its consumer is the failure this package has hit before.
    """

    def __init__(self, q_moduli, d: int) -> None:
        if d < 4 or d & (d - 1):
            raise ValueError(
                f"SplitRing: degree must be a power of two >= 4 to split, got {d!r}"
            )
        self.q_moduli: tuple[int, ...] = tuple(int(q) for q in q_moduli)
        self.d = d
        self.half = d // 2
        # Rejects an NTT-friendly limb by naming the other mode — the two
        # families are mutually exclusive and a silent mix is a soundness gap.
        self.split_roots: tuple[int, ...] = tuple(split_root(q) for q in self.q_moduli)
        self.fields = tuple(zk_dtypes.prime_field(q) for q in self.q_moduli)
        # `from_split`'s two exact divisions, per limb: by `2` and by `2r`.
        # Host ints, computed once beside the roots they come from.
        self._inv_two = tuple(pow(2, -1, q) for q in self.q_moduli)
        self._inv_two_root = tuple(
            pow(2 * r, -1, q) for q, r in zip(self.q_moduli, self.split_roots)
        )
        # `T_s(u)[k][j] = w[n-1+k-j]` over the twisted buffer `w` — a
        # trace-time constant, shared by both halves and every limb.
        rows = np.arange(self.half, dtype=np.int32)
        self._gather = self.half - 1 + rows[:, None] - rows[None, :]

    def _scalar(self, s: int, index: int) -> Any:
        """A host integer as a field element of limb `index`."""
        q, field = self.q_moduli[index], self.fields[index]
        return fnp.asarray(np.array([int(s) % q], dtype=np.uint64).astype(field))[0]

    def _twists(self, index: int) -> Any:
        """Limb `index`'s two half-twists `(+r, -r)` as a `(2, 1)` column, so
        one multiply twists both halves of a `[..., 2, n-1]` slice at once.

        Rebuilt per call, deliberately. Caching would only save anything if it
        held the **converted** array, and that is a tracer leak waiting to
        happen: whichever call populates the cache first decides what is in
        it, so a ring whose first `mul` runs inside a `jit` would store that
        trace's tracer and hand it to every later call, eager ones included.
        Two elements fold to a constant under `jit`, so there is nothing to
        win — which is why `ring.py` caches numpy tables and never traced
        values either.
        """
        q, r = self.q_moduli[index], self.split_roots[index]
        # `split_root` returns `min(r, q - r)`, so `r` is already canonical.
        column = np.array([[r], [(-r) % q]], dtype=np.uint64)
        return fnp.asarray(column.astype(self.fields[index]))

    def _tail(self, domain: type) -> tuple[int, ...]:
        """The host-side trailing extents an element of `domain` occupies."""
        return (2, self.half) if domain is Split else (self.d,)

    def _limbs_from_host(self, arr: np.ndarray, op: str, domain: type):
        """A host array as per-limb field arrays, batch axes moved inside.

        The host convention puts a module stack's leading axes *in front* of
        the limb axis (`(k, limbs, d)`), while a traced element carries them
        inside each limb (`limbs × [k, d]`) — one array per limb is forced by
        `prime_field` being per-modulus. So the boundary is a `moveaxis` — here
        and inverted in `to_host`, and nowhere else.

        The reduction runs in Python-int precision before the cast, because a
        residue at these widths cannot be reduced after landing in a lane.
        """
        arr = np.asarray(arr)
        tail = self._tail(domain)
        limbs = len(self.q_moduli)
        if arr.ndim < len(tail) + 1 or arr.shape[-len(tail) :] != tail:
            raise ValueError(
                f"SplitRing.{op}: expected a host array ending in "
                f"(limbs, {', '.join(str(t) for t in tail)}) with limbs={limbs}, "
                f"got shape {arr.shape!r}"
            )
        limb_axis = arr.ndim - 1 - len(tail)
        if arr.shape[limb_axis] != limbs:
            raise ValueError(
                f"SplitRing.{op}: expected {limbs} limbs at axis {limb_axis}, "
                f"got {arr.shape[limb_axis]}"
            )
        moved = np.moveaxis(arr, limb_axis, 0)
        return tuple(
            fnp.asarray(
                np.array([int(v) % q for v in row.reshape(-1)], dtype=np.uint64)
                .reshape(row.shape)
                .astype(field)
            )
            for row, q, field in zip(moved, self.q_moduli, self.fields)
        )

    def coeff_from_host(self, arr: np.ndarray) -> Coeff:
        """A host array as a coefficient-domain element.

        The host contract carries no domain — a `(limbs, d)` array is just
        bytes — so the caller asserts one by choosing this constructor.
        """
        return Coeff(self._limbs_from_host(arr, "coeff_from_host", Coeff))

    def split_from_host(self, arr: np.ndarray) -> Split:
        """A `HostSplitRing.to_split` array `(…, limbs, 2, d/2)` as a
        CRT-domain element — the two rings' shared wire for this domain."""
        return Split(self._limbs_from_host(arr, "split_from_host", Split))

    def to_host(self, a: Coeff | Split) -> np.ndarray:
        """An element back as the host contract — the boundary, not a traced op.

        The domain does not survive the trip; what it decides is the shape,
        `(…, limbs, d)` for `Coeff` and `(…, limbs, 2, d/2)` for `Split`,
        which is what `HostSplitRing` and its `to_split` take respectively.
        """
        domain = same_domain("to_host", _DOMAINS, a)
        # A field array converts one element at a time — `astype(np.uint64)`
        # on the whole thing asks the dtype to infer a field from a scalar.
        rows = []
        for limb in a.limbs:
            obj = np.asarray(limb).astype(object)
            rows.append(
                np.array([int(v) for v in obj.reshape(-1)], dtype=np.uint64).reshape(obj.shape)
            )
        stacked = np.stack(rows)
        return np.moveaxis(stacked, 0, stacked.ndim - 1 - len(self._tail(domain)))

    def from_signed(self, values) -> Coeff:
        """A signed integer coefficient vector, embedded per limb.

        `Coeff` by construction: "small signed coefficients" is a
        coefficient-domain notion.
        """
        row = [int(v) for v in values]
        if len(row) != self.d:
            raise ValueError(f"SplitRing.from_signed: expected {self.d} values, got {len(row)}")
        return Coeff(
            tuple(
                fnp.asarray(np.array([v % q for v in row], dtype=np.uint64).astype(field))
                for q, field in zip(self.q_moduli, self.fields)
            )
        )

    def from_signed_stack(self, rows) -> Coeff:
        """`from_signed` per row: a `(k, d)` array — or a sequence of
        length-`d` rows — as a `Coeff` whose limbs carry one leading axis.

        The module convention's constructor and `_HostRingBase`'s
        counterpart, deliberately not `stack([from_signed(r) for r in rows])`:
        that reduces and converts `k` times and then asks the array layer to
        join the results, where one reduction over the whole block and one
        conversion per limb say the same thing.

        The reduction runs in object dtype so numpy drives the loop over
        exact Python ints, which is what lets raw signed values through — a
        `-1` witness coefficient reduces to `q-1` here rather than
        truncating in a lane on the way in.
        """
        def bad(detail: str) -> ValueError:
            return ValueError(
                f"SplitRing.from_signed_stack: expected a non-empty "
                f"(k, {self.d}) stack of signed values, got {detail}"
            )

        try:
            block = [[int(v) for v in row] for row in rows]
        except TypeError as exc:
            # A rank-one input — one element's coefficients, where a stack of
            # them was wanted. It has to land as a shape `ValueError` and not
            # as the `TypeError` the comprehension would otherwise raise: this
            # package keeps the two failure modes apart because they are
            # different caller bugs, and "not iterable" names neither.
            raise bad("a rank-one input") from exc
        if not block or any(len(row) != self.d for row in block):
            raise bad(
                f"{len(block)} rows of lengths "
                f"{sorted({len(row) for row in block})}"
            )
        values = np.array(block, dtype=object)
        return Coeff(
            tuple(
                fnp.asarray((values % q).astype(np.uint64).astype(field))
                for q, field in zip(self.q_moduli, self.fields)
            )
        )

    def zeros(self, domain: type, *lead: int) -> Coeff | Split:
        """The additive identity of `domain`, as a `lead`-shaped stack.

        Unlike `one`, zero is the *same* element in both domains — `to_split`
        is linear, so it carries `0` to `0` — which is exactly why the domain
        has to be said here rather than inferred from the value. `zeros(D, 0)`
        is the empty stack a contraction over nothing hands back, which
        `_HostRingBase.matvec` documents as a real statement rather than an
        error.

        The domain arrives as a **type, not a flag**: it is fixed at trace
        time, it names the return type instead of branching on it inside the
        graph, and it is the same thing `_tail` and `_limbs_from_host`
        already take. The runtime `IsNTT`-style flag CLAUDE.md rejects is a
        different animal — that one splits a trace for information the graph
        already has.
        """
        if domain not in _DOMAINS:
            raise TypeError(
                f"SplitRing.zeros: expected one of "
                f"{', '.join(d.__name__ for d in _DOMAINS)}, got "
                f"{getattr(domain, '__name__', domain)!r}"
            )
        shape = (*lead, *self._tail(domain))
        return domain(tuple(fnp.zeros(shape, dtype=field) for field in self.fields))

    def one(self) -> Coeff:
        """The multiplicative identity, **in the coefficient domain**.

        Named for the ring element rather than for whichever domain this
        ring's product happens to live in, matching `_HostRingBase.one` for
        the reason it gives: returning the `Split` view here would make `one`
        two different elements across the two rings under a single name. A
        consumer multiplying by it crosses, and that `to_split` folds to a
        constant under `jit`.
        """
        return self.from_signed([1] + [0] * (self.d - 1))

    def uniform_stack(self, rng: np.random.Generator, *lead: int) -> Coeff:
        """A `lead`-shaped stack of independent uniform elements, in `Coeff`.

        Uniform per limb is uniform over `R_q` by CRT, so here the domain is
        a choice rather than a constraint; `Coeff` keeps one answer and
        matches the array `to_host` hands `HostSplitRing`.

        A **boundary** constructor: it holds a host generator, so it cannot
        run under `jit`. That is the honest outcome rather than a traced RNG
        this package does not own — the same stance `to_balanced_limb0`
        takes — and the generator is injected like every other randomness
        consumer here.

        Drawn limb by limb in `q_moduli` order, which is `_HostRingBase`'s
        own order, so the two rings are byte-identical at one seed. A
        consumer replaying these draws depends on that order and not merely
        on the distribution.
        """
        return Coeff(
            tuple(
                fnp.asarray(
                    rng.integers(0, q, size=(*lead, self.d), dtype=np.uint64).astype(field)
                )
                for q, field in zip(self.q_moduli, self.fields)
            )
        )

    def constant_coeff(self, a: Coeff) -> tuple[Any, ...]:
        """Every element's constant coefficient, one `Z_q` array per limb.

        `Coeff`-only, and the domain type is what proves it: a CRT half holds
        a residue modulo `X^{d/2} ∓ r` and has no constant term to read, so
        asking one for it is a bug in a plausible shape.

        Deliberately **not** a domain type on the way out. What comes back is
        a `Z_q` value per limb, not a ring element, and dressing it as `Coeff`
        would put a ring-element type on something no ring op can take. A
        bare tuple is already a pytree, so it threads `jit`/`vmap` without a
        registration of its own — and unlike the host's, it needs no
        defensive copy, because a traced array cannot be written through.

        Named rather than left to the caller as `limb[..., 0]` for the reason
        `_HostRingBase.constant_coeff` gives: that index is this backend's
        array layout, not the ring's interface.
        """
        require_domain("constant_coeff", Coeff, a)
        return tuple(limb[..., 0] for limb in a.limbs)

    def to_split(self, a: Coeff) -> Split:
        """The two-factor CRT view: half `h` is `low + s·high` for `s = +r`
        (h=0) / `-r` (h=1), which is `X^{d/2} ≡ s` substituted into the
        element. `d/2` multiplies and `d` adds per limb — cheap enough that a
        consumer can cross domains per operation rather than staying in one.

        The `(2, 1)` twist column `mul` uses is exactly `(+r, -r)`, so both
        halves come out of one broadcast multiply-add against it rather than
        a scalar `r`, a negate and a `stack`.
        """
        require_domain("to_split", Coeff, a)
        halves = []
        for i, limb in enumerate(a.limbs):
            low, high = limb[..., : self.half], limb[..., self.half :]
            halves.append(low[..., None, :] + high[..., None, :] * self._twists(i))
        return Split(tuple(halves))

    def from_split(self, a: Split) -> Coeff:
        """Inverse of `to_split`: `low = (u + v)/2`, `high = (u - v)/(2r)`,
        both divisions exact modulo an odd prime."""
        require_domain("from_split", Split, a)
        rows = []
        for i, limb in enumerate(a.limbs):
            inv2 = self._scalar(self._inv_two[i], i)
            inv2r = self._scalar(self._inv_two_root[i], i)
            u, v = limb[..., 0, :], limb[..., 1, :]
            rows.append(fnp.concatenate([(u + v) * inv2, (u - v) * inv2r], axis=-1))
        return Coeff(tuple(rows))

    def mul(self, a: Split, b: Split) -> Split:
        """The ring's multiplication, `Split`-only — which is what requiring
        the domain proves.

        Per limb: build the twisted buffer `[s·u[1:], u]`, read `T_s(u)` out
        of it with one gather, and contract against `v`. Both halves ride the
        same gather because the twist enters through the buffer, so this is
        one `take` and one `[..., 2, n, n]` contraction — `d²/2` products
        against the coefficient domain's `d²`, and every leading batch axis
        rides through untouched.

        **A module layer must not be a loop of this.** `T_s(u)` is built from
        the left operand, so contracting a matrix against `w` columns by
        calling `mul` per column rebuilds the same buffer `w` times — the
        waste `HostSplitRing.matmul` avoids by building `Neg` once and reusing
        it down every column, worth 112-116x there. The traced module layer
        wants that same shape: gather the left operand once, contract wide.
        """
        require_domain("mul", Split, a, b)
        halves = []
        for i, (x, y) in enumerate(zip(a.limbs, b.limbs)):
            buffer = fnp.concatenate([x[..., 1:] * self._twists(i), x], axis=-1)
            twisted = fnp.take(buffer, self._gather, axis=-1)
            halves.append((twisted * y[..., None, :]).sum(axis=-1))
        return Split(tuple(halves))

    def add(self, a: _S, b: _S) -> _S:
        domain = same_domain("add", _DOMAINS, a, b)
        return domain(tuple(x + y for x, y in zip(a.limbs, b.limbs)))

    def sub(self, a: _S, b: _S) -> _S:
        domain = same_domain("sub", _DOMAINS, a, b)
        return domain(tuple(x - y for x, y in zip(a.limbs, b.limbs)))

    def neg(self, a: _S) -> _S:
        domain = same_domain("neg", _DOMAINS, a)
        return domain(tuple(-x for x in a.limbs))

    def mul_scalar(self, a: _S, s: int) -> _S:
        """Domain-generic: a `Z_q` scalar acts entrywise in either domain —
        `to_split` is linear, so scaling commutes with it."""
        domain = same_domain("mul_scalar", _DOMAINS, a)
        return domain(tuple(limb * self._scalar(s, i) for i, limb in enumerate(a.limbs)))

    def stack(self, elements: Sequence[_S]) -> _S:
        """A batch element from same-domain elements: limbs gain a leading axis.

        The module convention's constructor, as in `ring.py` — stacking nests,
        so a stack of stacks is a matrix.
        """
        domain = same_domain("stack", _DOMAINS, *elements)
        return domain(
            tuple(fnp.stack(list(limbs)) for limbs in zip(*(e.limbs for e in elements)))
        )
