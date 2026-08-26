"""The partial-split host ring — `Z_q[X]/(X^d + 1)` at moduli `q ≡ 5 (mod 8)`.

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

Representation and contract are `host_ring.py`'s, shared through
`_HostRingBase`: `(limbs, d)` `dtype=np.uint64` canonical arrays outside,
exact Python ints inside. What this class adds:

- `mul` — coefficient-domain negacyclic **schoolbook convolution**. There
  is deliberately no CRT shortcut in it: the schoolbook product is the
  modulus-shape-agnostic oracle that the split path below (and, later, the
  traced twisted-convolution matvec) is checked against, the same
  oracle-vs-accelerated structure the NTT ring has with `ring.py`.
- `to_split` / `from_split` — the two-factor CRT view, an element as its
  residues modulo `X^{d/2} ∓ r`, shape `(limbs, 2, d/2)`. Half `0` is the
  `X^{d/2} ≡ +r` residue, half `1` the `-r` one. This is the future traced
  representation: each half's product is an `r`-twisted convolution, i.e.
  a structured `d/2 × d/2` matvec.
- `matmul` — the batched product, and the one path here that does not go
  through `mul`. It derives the same `X^d ≡ -1` fold as an anticirculant
  matrix rather than a convolution loop, shares no code with the CRT split,
  and is checked against `mul` through `matvec` — so the oracle stance above
  survives it. Exactness is gated on operand magnitude; see the method.
- `is_invertible` — a unit test of the ring in the literal sense: each
  half is a field, so an element is invertible iff every `(limb, half)`
  residue is nonzero. Cheap, host-side, and exactly the predicate the LNP
  challenge space needs pinned.

There is no `ntt`/`intt` here on purpose, and no Eval domain: every
operation is coefficient-domain. A consumer that wants device-shaped
products waits for the split-domain traced ring; it does not get to borrow
the NTT ring's, because the moduli are mutually exclusive.
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

from lattice_frx.canonical import require_canonical
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
