"""The same ring on the host, over exact Python integers, ported from lattigo v6
`ring/subring.go` (NTT table generation) and `ring/ntt.go` (the CT
forward / GS inverse core loops), specialized to the Standard NTT
(`NumberTheoreticTransformerStandard`, `NthRoot = 2*d`).

This is the interop-riskiest module here: an NTT that computes the
*right values in the wrong order* is silently wrong for every downstream
consumer that mixes reference-ring output with lattigo wire bytes. That
ordering was pinned against vectors produced by lattigo's actual
`MForm`+`NTT`, so `ntt()`/`intt()` below reproduce lattigo's own
bit-reversed table order and CT/GS loop structure natively —
**no output permutation is applied anywhere in this module.** (A
permutation *adapter* belongs to whatever computes the NTT in a different
natural order and has to present lattigo's at its boundary — `ring.py` is
that, and it applies exactly one bit-reversal for exactly this reason.)

The second reference this class covers (ringo-snark `math/crt`, pin
`b31f20a`) needs no separate ring class: its `NewTransformer` fills the same bit-reversed twiddle table
(`tw[bitrev(j)] = psi^j`), runs the same CT forward / GS inverse cores
with the same trailing `1/rank` scaling, and derives psi from the same
smallest-primitive-root walk. The one textual difference — crt's walk
tests g=2 where lattigo's starts at 3 — never matters for an NTT-friendly
limb: q ≡ 1 (mod 2·d) with d ≥ 4 forces q ≡ 1 (mod 8), which makes 2 a
quadratic residue and hence never a primitive root, so both walks land on
the same root. That equivalence is what lets one class serve both
references; a consumer holding goldens from either can pin it per limb
(precondition, psi, and full transform digests).

Montgomery form and lazy (over-range) reduction are dropped throughout:
values are exact Python ints internally and every reduction is a plain
`%`. Only the loop structure, the table-generation order, and the
resulting table/output *values* need to match lattigo — the Montgomery
representation is an implementation detail of lattigo's uint64
arithmetic that has no effect on the mathematical values once
un-Montgomeried (see `serde.py`'s docstring for that fold).

**Public contract:** every `HostRnsRing` op below takes and returns
`(limbs, d)` `dtype=np.uint64` numpy arrays of canonical (`< q_l` per
limb) standard-form residues, defined and enforced in
`lattice_frx/canonical.py`. Internally, each op converts to an
object-dtype array of exact Python ints (the lattigo-order NTT and every
other computation below is unchanged — d=256 makes the round-trip
conversion free) and converts back on the way out; `_coerce` is the
single entry point for that conversion plus the contract check.

That contract is a *host* contract, and it is worth being explicit that
it is not a device-shaped one, because it looks like it should be. frx
runs without x64, so `fnp.asarray` on a `uint64` array yields `uint32` and
truncates every limb above `2**32` without raising; with
`MAX_MODULUS_BITS = 50` that is every limb this ring is built for. The
width a traced backend can actually carry comes from a field dtype, whose
storage follows its modulus (`zk_dtypes.prime_field(q)` mints a 64-bit
field at a 50-bit `q`), not from an integer lane. See issue #1 for the
measurement and for what the traced contract has to become.

Lattigo functions ported here:

- `PrimitiveRoot` (subring.go) — smallest primitive root mod q. `g` starts
  at 2 but is pre-incremented before the first candidate test, so 3 is the
  first g actually tried.
- `SubRing.generateNTTConstants` (subring.go) — psi/psi_inv from the
  primitive root, `NInv`, and the bit-reversed-index table fill
  (`RootsForward`/`RootsBackward`).
- `nttLazy` (ntt.go) — CT decimation-in-time forward core.
- `inttLazy` (ntt.go) plus `INTTStandard`'s trailing `NInv` scaling
  (ntt.go) — GS decimation-in-frequency inverse core, folded with the
  final `1/d` scaling lattigo applies after the GS network.

(lattigo also has SIMD-unrolled variants, `nttUnrolled16Lazy` /
`inttLazyUnrolled16`, used at these ring degrees in practice — they compute
the identical butterfly network as the `*Lazy` cores above, just 8-wide
unrolled with periodic lazy-range reduction for uint64 overflow safety.
Nothing here needs that: we port the non-unrolled cores, which are the
same math without the SIMD/overflow bookkeeping.)

`PrimitiveRoot`'s factorization of `q - 1` is reimplemented locally (trial
division for small factors, Pollard's rho for the rest) rather than porting
lattigo's `utils/factorization` package: only the resulting *set* of unique
prime factors feeds `PrimitiveRoot`'s search, so any correct factorization
algorithm agrees with lattigo's on that set.
"""
import numpy as np

# The array contract and the NTT's constants live in this package's shared
# modules; `roots` holds the ones the traced ring needs identically.
from lattice_frx.canonical import require_canonical
from lattice_frx.roots import bit_reverse, galois_map, primitive_root, prime_factors


def _generate_ntt_table(q: int, d: int) -> tuple[list[int], list[int], int]:
    """Port of `SubRing.generateNTTConstants` (subring.go), specialized to
    the Standard NTT (`NthRoot = 2*d`) and with the Montgomery folds
    dropped (psi/psi_inv/NInv/table entries are plain residues, not
    `MForm`-ed).

    Returns `(roots_forward, roots_backward, n_inv)`, each a length-`d`
    list of ints (`roots_forward[i]`/`roots_backward[i]` hold
    `psi^j`/`psi_inv^j` at the bit-reversed index `i = bitrev(j)` — the
    same order lattigo's own table ends up in after its incremental
    bit-reversed-index-write loop below).
    """
    nth_root = 2 * d
    if (q - 1) % nth_root != 0:
        raise ValueError(f"_generate_ntt_table: {q} is not 1 mod NthRoot={nth_root}")

    factors = prime_factors(q - 1)
    g = primitive_root(q, factors)

    log_nth_root = (nth_root >> 1).bit_length() - 1  # == log2(d)

    # subring.go: s.NInv = MForm(ModExp(NthRoot>>1, Modulus-2, Modulus), ...)
    # i.e. (NthRoot>>1)^-1 mod q via Fermat; `pow(base, -1, q)` is the same
    # value, computed directly rather than via q-2 exponentiation.
    n_inv = pow(nth_root >> 1, -1, q)

    psi = pow(g, (q - 1) // nth_root, q)
    psi_inv = pow(g, q - (q - 1) // nth_root - 1, q)

    half = nth_root >> 1  # == d
    roots_forward = [0] * half
    roots_backward = [0] * half
    roots_forward[0] = 1
    roots_backward[0] = 1

    # subring.go's loop: for j in [1, NthRoot>>1), write
    # RootsForward[bitrev(j)] = RootsForward[bitrev(j-1)] * Psi (and the
    # PsiInv analogue for RootsBackward) — filling the tables in
    # bit-reversed index order, one natural-order power of psi at a time.
    for j in range(1, half):
        idx_prev = bit_reverse(j - 1, log_nth_root)
        idx_next = bit_reverse(j, log_nth_root)
        roots_forward[idx_next] = (roots_forward[idx_prev] * psi) % q
        roots_backward[idx_next] = (roots_backward[idx_prev] * psi_inv) % q

    return roots_forward, roots_backward, n_inv


def _ntt_forward(vec: list[int], q: int, roots: list[int]) -> list[int]:
    """Port of lattigo `nttLazy` (ntt.go): the CT decimation-in-time
    forward core, with the lazy [0, 4q) output range and Montgomery
    butterfly multiply dropped in favor of plain `%`."""
    n = len(vec)
    p2 = [0] * n

    t = n >> 1
    f = roots[1]
    for jx in range(t):
        jy = jx + t
        u, v = vec[jx], vec[jy]
        v = (v * f) % q
        p2[jx] = (u + v) % q
        p2[jy] = (u - v) % q

    m = 2
    while m < n:
        t >>= 1
        for i in range(m):
            j1 = (i * t) << 1
            f = roots[m + i]
            for jx in range(j1, j1 + t):
                jy = jx + t
                u, v = p2[jx], p2[jy]
                v = (v * f) % q
                p2[jx] = (u + v) % q
                p2[jy] = (u - v) % q
        m <<= 1

    return p2


def _ntt_backward(vec: list[int], q: int, roots: list[int], n_inv: int) -> list[int]:
    """Port of lattigo `inttLazy` (ntt.go): the GS decimation-in-frequency
    inverse core, plus the trailing `1/d` scaling `INTTStandard` applies
    after the GS network (folded in here rather than as a separate
    `MulScalar` pass — same mathematical effect)."""
    n = len(vec)
    p2 = [0] * n

    t = 1
    h = n >> 1
    for i in range(h):
        j1 = i * 2 * t
        f = roots[h + i]
        for jx in range(j1, j1 + t):
            jy = jx + t
            u, v = vec[jx], vec[jy]
            p2[jx] = (u + v) % q
            p2[jy] = ((u - v) * f) % q
    t <<= 1

    m = n >> 1
    while m > 1:
        h = m >> 1
        for i in range(h):
            j1 = i * 2 * t
            f = roots[h + i]
            for jx in range(j1, j1 + t):
                jy = jx + t
                u, v = p2[jx], p2[jy]
                p2[jx] = (u + v) % q
                p2[jy] = ((u - v) * f) % q
        t <<= 1
        m >>= 1

    return [(x * n_inv) % q for x in p2]


class _HostRingBase:
    """The modulus-shape-agnostic half of a host negacyclic ring: the
    `(limbs, d)` uint64 public contract, exact-Python-int internals, and
    every operation that only needs coefficient arithmetic mod each limb
    (add/sub/neg, Galois, signed embedding, balanced lift). Subclasses add
    the multiplication story their modulus shape allows — `HostRnsRing`
    via the NTT (fully-splitting limbs), `HostSplitRing`
    (`lattice_frx/split_ring.py`) via coefficient-domain convolution
    (partial-split limbs). `_op_prefix` names the subclass in contract
    errors, which tests pin."""

    _op_prefix = "HostRing"

    def __init__(self, q_moduli, d: int):
        self.q_moduli: tuple[int, ...] = tuple(int(q) for q in q_moduli)
        self.d = d
        self._q_col = np.array(self.q_moduli, dtype=object)[:, None]

    def _coerce(self, a: np.ndarray, op: str) -> np.ndarray:
        """Boundary for the public `(limbs, d)` uint64 contract (module
        docstring): checked by `canonical.require_canonical`, then
        converted into an object-dtype array of exact Python ints for the
        internal math.
        """
        require_canonical(a, self.q_moduli, f"{self._op_prefix}.{op}")
        return a.astype(object)

    def _reduce(self, arr: np.ndarray) -> np.ndarray:
        """Elementwise mod `q_l` per limb, converting the exact-Python-int
        internal representation back to the public uint64 contract.
        `arr` holds Python ints (object dtype) by the time every caller
        below reaches this — `.astype(np.uint64)` after the `%` is
        lossless by construction (moduli fit in uint64; see module
        docstring)."""
        return (arr % self._q_col).astype(np.uint64)

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = self._coerce(a, "add")
        b = self._coerce(b, "add")
        return self._reduce(a + b)

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = self._coerce(a, "sub")
        b = self._coerce(b, "sub")
        return self._reduce(a - b)

    def neg(self, a: np.ndarray) -> np.ndarray:
        a = self._coerce(a, "neg")
        return self._reduce(-a)

    def galois(self, a: np.ndarray, k: int) -> np.ndarray:
        """`roots.galois_map`'s action, one coefficient at a time.

        Coefficient-domain by definition; the NTT-domain action lives on the
        traced ring as `galois_eval`, checked against this oracle."""
        a = self._coerce(a, "galois")
        dest, negate = galois_map(self.d, k)
        out = np.zeros_like(a)
        for i, (j, neg) in enumerate(zip(dest, negate)):
            out[:, j] = -a[:, i] if neg else a[:, i]
        return self._reduce(out)

    def from_signed(self, vals) -> np.ndarray:
        """Broadcast a length-`d` sequence of (possibly negative,
        possibly unreduced) ints to every limb, reducing each mod that
        limb's modulus. Returns the public `dtype=np.uint64` contract;
        `vals` itself is the host-boundary raw-int input (frequently
        negative, per `from_signed([-1, ...])` callers), so it is not
        itself part of the canonical-residue contract `_coerce` checks.

        `vals` is frequently a numpy `int64` array (every sampler in
        `lattice_frx/sampler.py` returns one) rather than a list of plain
        `int`s -- `list(vals)` alone would leave `numpy.int64` *scalar*
        objects sitting inside this (nominally exact-Python-int,
        object-dtype) array, since `dtype=object` only controls the
        array's own dtype, not what a fixed-width numpy scalar's `%`/`*`
        does when later combined with a plain Python `int` (`add`/`mul`/
        `mul_add`/... below all silently stay in `numpy.int64` land in
        that case). That's invisible at small moduli, where the values
        never leave the `int64` range -- but this module's whole premise
        (see the module docstring: "every reduction is a plain `%`" on
        "exact Python ints") breaks silently once a modulus is large
        enough that a downstream product overflows 64 bits: confirmed via
        an actual `RuntimeWarning: overflow encountered in scalar
        multiply` in `mul_scalar_then_sub`, hit by a consumer's commit path
        at `q_moduli` ~2**54 (vs ~2**35 at its smaller parameter point) --
        a latent gap in this function that only a wide-enough modulus
        exercises, which is why it is guarded unconditionally. `int(v)`
        forces every element to a genuine arbitrary-precision Python int
        before it ever enters the object array, restoring the documented
        invariant unconditionally (a no-op when `vals` already held plain
        ints).
        """
        limbs = len(self.q_moduli)
        row = [int(v) for v in vals]
        return self._reduce(np.array([row] * limbs, dtype=object))

    def to_balanced_limb0(self, a: np.ndarray) -> np.ndarray:
        """Limb 0 of `a`, remapped from `[0, q0)` to the balanced range
        `(-q0/2, q0/2]`. Only meaningful when the true (CRT-reconstructed)
        coefficients are already known to fit within limb 0's range alone
        — e.g. small test/challenge polynomials — since this reads limb 0
        in isolation, without reconstructing across limbs.

        Returns `dtype=np.int64`, not the public uint64 contract: a
        balanced value is signed by definition (can't be a canonical
        `[0, q0)` residue), the same host-boundary-signed-output stance
        `lattice_frx/rns.py`'s `reconstruct_centered` documents for the
        analogous whole-ring case. `int64` (not `object`) because the
        result is a single limb's worth of already-small values (`|v| <
        q0/2`, comfortably under `2**63`) — no arbitrary-precision need,
        so there's no reason to leave the fixed-width contract here.
        """
        a = self._coerce(a, "to_balanced_limb0")
        q0 = self.q_moduli[0]
        half = q0 // 2
        row = a[0]
        return np.array([v - q0 if v > half else v for v in row], dtype=np.int64)


class HostRnsRing(_HostRingBase):
    """The RNS negacyclic ring `Z[X]/(X^d + 1)` over `prod(q_moduli)`, in
    lattigo's own NTT table/output order (Standard NTT, `NthRoot = 2*d`).

    Public arrays are `(limbs, d)` `dtype=np.uint64` numpy arrays,
    standard (not Montgomery) form, every residue canonical (`< q_l` per
    limb) — see the module docstring's "Public contract" section.
    `ntt`/`intt` expect/produce coefficients already reduced mod each
    limb's modulus (as `from_signed` and the other operations below
    always leave them); `_coerce` additionally reduces input defensively
    once it's converted to the internal object-dtype representation.
    """

    backend = "reference"
    _op_prefix = "RnsRing"

    def __init__(self, q_moduli, d: int):
        super().__init__(q_moduli, d)
        self._tables = [_generate_ntt_table(q, d) for q in self.q_moduli]

    def ntt(self, a: np.ndarray) -> np.ndarray:
        a = self._coerce(a, "ntt")
        rows = []
        for i, q in enumerate(self.q_moduli):
            roots_forward, _, _ = self._tables[i]
            vec = [int(x) % q for x in a[i]]
            rows.append(_ntt_forward(vec, q, roots_forward))
        return np.array(rows, dtype=np.uint64)

    def intt(self, a: np.ndarray) -> np.ndarray:
        a = self._coerce(a, "intt")
        rows = []
        for i, q in enumerate(self.q_moduli):
            _, roots_backward, n_inv = self._tables[i]
            vec = [int(x) % q for x in a[i]]
            rows.append(_ntt_backward(vec, q, roots_backward, n_inv))
        return np.array(rows, dtype=np.uint64)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Elementwise NTT-domain product (`MulCoeffsMontgomery`
        semantics, Montgomery fold dropped)."""
        a = self._coerce(a, "mul")
        b = self._coerce(b, "mul")
        return self._reduce(a * b)

    def mul_add(self, a: np.ndarray, b: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """`acc + a*b`, elementwise NTT-domain (`MulCoeffsMontgomeryThenAdd`)."""
        a = self._coerce(a, "mul_add")
        b = self._coerce(b, "mul_add")
        acc = self._coerce(acc, "mul_add")
        return self._reduce(acc + a * b)

    def mul_scalar(self, a: np.ndarray, s: int) -> np.ndarray:
        """`a*s` (lattigo `MulRNSScalarMontgomery`, Montgomery fold
        dropped): `s` is a single scalar shared across every limb, reduced
        mod each limb's own modulus by the `%` below -- the same
        shared-scalar convention as `mul_scalar_then_sub` below, just
        without the accumulate-and-subtract."""
        a = self._coerce(a, "mul_scalar")
        return self._reduce(a * s)

    def mul_scalar_then_sub(self, a: np.ndarray, s: int, acc: np.ndarray) -> np.ndarray:
        """`acc - a*s` (lattigo `MulScalarThenSub`, operations.go):
        `s` is a single scalar shared across every limb, reduced mod each
        limb's own modulus by the `%` below."""
        a = self._coerce(a, "mul_scalar_then_sub")
        acc = self._coerce(acc, "mul_scalar_then_sub")
        return self._reduce(acc - a * s)
