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
and `primes.split_root` rejects a stray NTT-friendly limb loudly.

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

from lattice_frx.canonical import require_canonical
from lattice_frx.host_ring import _HostRingBase
from lattice_frx.primes import split_root


class HostSplitRing(_HostRingBase):
    """The partial-split ring over `prod(q_moduli)`, every limb ≡ 5 (mod 8).

    `split_roots[l]` is `primes.split_root(q_l)` — the canonical square
    root of `-1` naming limb `l`'s two CRT halves.
    """

    backend = "reference"
    _op_prefix = "SplitRing"

    def __init__(self, q_moduli, d: int):
        if d < 4 or d & (d - 1):
            raise ValueError(
                f"HostSplitRing: degree must be a power of two >= 4, got {d!r}"
            )
        super().__init__(q_moduli, d)
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
            ra = [int(x) for x in a[l]]
            rb = [int(x) for x in b[l]]
            conv = [0] * (2 * self.d)
            for i, ai in enumerate(ra):
                if ai:
                    for j, bj in enumerate(rb):
                        conv[i + j] += ai * bj
            rows.append([(conv[k] - conv[k + self.d]) % q for k in range(self.d)])
        return np.array(rows, dtype=np.uint64)

    def to_split(self, a: np.ndarray) -> np.ndarray:
        """The two-factor CRT view, `(limbs, 2, d/2)`: half `h` holds the
        residue modulo `X^{d/2} - s` for `s = +r` (h=0) / `-r` (h=1),
        i.e. `low[i] + s·high[i]` after substituting `X^{d/2} ≡ s`."""
        a = self._coerce(a, "to_split")
        half = self.d // 2
        out = np.empty((len(self.q_moduli), 2, half), dtype=np.uint64)
        for l, q in enumerate(self.q_moduli):
            r = self.split_roots[l]
            low = [int(x) for x in a[l][:half]]
            high = [int(x) for x in a[l][half:]]
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
        require_canonical(
            np.ascontiguousarray(sp).reshape(limbs, self.d),
            self.q_moduli,
            "SplitRing.from_split",
        )
        out_rows = []
        for l, q in enumerate(self.q_moduli):
            inv2 = pow(2, -1, q)
            inv2r = pow(2 * self.split_roots[l], -1, q)
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
