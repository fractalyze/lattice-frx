"""Exact RNS reconstruction, centered lifts, and basis conversion.

Two references share this module, and they disagree on one comparison —
see `reconstruct_signed_mixed_radix` for the full statement of it:

- the v1 reference (`jindo/jindo/rns.go`'s `RNSReconstructor`:
  `reconstructTo` / `setBigCoeffTo`) — `reconstruct_centered`,
  `lift_centered`, `rescale_floor`;
- the v2 reference (ringo-snark `math/crt`'s `Embedder` / `Pow2Cutter`,
  pin `b31f20a`) — `reconstruct_signed_mixed_radix`, `embed`, `pow2_cut`.

The crt half lives here rather than in `ring.py` because it is basis
conversion, not ring algebra: coefficient-domain only, no NTT, no
polynomial structure — the same host boundary `reconstruct_centered`
already owns.

All arithmetic here is exact Python-int math — no Montgomery form, no
`big.Int` buffer reuse (that was `rns.go`'s own uint64-width performance
concern; this module doesn't need it). The fidelity requirement is
matching lattigo's *comparisons* exactly, and there are two asymmetric
ones baked into `rns.go` that are easy to get backwards:

- `toBalanced(x, q)` (rns.go) maps `x` into the balanced range using
  a **strict** `x > q>>1` test — so `x == q>>1` (achievable when `q` is
  even, or via the CRT sum landing exactly there) stays *positive*.
- The final centering in `reconstructTo`
  (https://github.com/SNUCP/jindo/blob/68ae757d789d4423fb27eb462719c8c993d5277b/jindo/rns.go#L100) uses `acc >= qHalf`
  — **non-strict** — so `acc == Q>>1` exactly *does* get centered to
  `acc - Q`, landing on the negative side.

Both ports below reproduce these comparisons verbatim; `reconstruct_centered`
first tries the fast path (every limb's balanced form agrees on the same
small int — the common case for norm-bounded polynomials, letting most
coefficients skip the CRT sum entirely) before falling back to the full
gadget-sum CRT.

`lift_centered` is the exact drop-in for lattigo's `BasisExtender.ModUpQtoP`
on norm-bounded inputs: reconstruct the true (small, centered) integer from
one RNS basis, then re-reduce it into another. lattigo's actual ModUpQtoP
is a *different, faster* algorithm (an approximate RNS basis-extension via
floating-point rounding of gadget sums, not exact bignum CRT) that can
diverge from this exact reconstruction off the "honest" (tightly
norm-bounded) path. We accept that known divergence here rather than
porting lattigo's approximate algorithm: every value `lift_centered` is
used on here is expected to be norm-bounded by construction, and the resulting
proof-level accept/reject equality against the Go reference is asserted
end-to-end in Tasks 14/16, not by matching ModUpQtoP's internals bit for
bit.

**Public contract:** wherever this module interfaces with ring data it
uses the same `(limbs, d)` `dtype=np.uint64` contract as
`lattice_frx/host_ring.py`, defined in `lattice_frx/canonical.py` --
`reconstruct_centered`'s `coeffs` and `set_big_coeffs`'s/`lift_centered`'s
return. `reconstruct_centered`'s own return stays `list[int]`: the whole
point of this function is being the host boundary where RNS residues
become one true (possibly huge, possibly negative) integer, so there is
no fixed-width dtype to put it in. `rescale_floor` is the named
boundary op: the reconstruct -> floor-shift -> re-embed cutoff dance
`prover.py` runs at each MSIS commitment step, collapsed into one
function so callers use it directly instead of re-deriving the
composition.
"""
import numpy as np

from lattice_frx.canonical import require_canonical
from lattice_frx.primes import MAX_MODULUS, MAX_MODULUS_BITS


def _coerce(coeffs: np.ndarray, q_moduli: tuple[int, ...], name: str) -> np.ndarray:
    """Boundary for this module's free-standing functions, mirroring
    `HostRnsRing._coerce` (host_ring.py): the array contract is checked by
    `canonical.require_canonical` and the array returned unchanged --
    every element read below already goes through `int(coeffs[...])`, so
    this module's arithmetic is dtype-agnostic once past this gate.
    """
    require_canonical(coeffs, q_moduli, f"rns.{name}")
    return coeffs


def _to_balanced(x: int, q: int) -> int:
    """Port of `toBalanced` (rns.go): `x` mapped from `[0, q)` to the
    balanced range, with the **strict** `x > q>>1` comparison rns.go uses
    (so `x == q>>1` stays positive, not centered)."""
    half = q >> 1
    return x - q if x > half else x


def _gadgets(q_moduli: tuple[int, ...], Q: int) -> list[int]:
    """Port of the gadget precomputation in `newRNSReconstructor`
    (rns.go): `gad[i] = (Q/q_i) * ((Q/q_i)^-1 mod q_i) mod Q`."""
    gad = []
    for q in q_moduli:
        q_div = Q // q
        q_inv = pow(q_div, -1, q)
        gad.append((q_div * q_inv) % Q)
    return gad


def _reconstruct(coeffs: np.ndarray, q_moduli: tuple[int, ...], center_from: int) -> list[int]:
    """Shared CRT body of the two reconstructions below: exact gadget-sum
    CRT into `[0, Q)`, then `acc - Q` for every `acc >= center_from`.
    The two callers differ **only** in that threshold (see each one's
    docstring); everything else -- the fast path, the gadget sum, the
    contract check -- is identical, and the fast path in particular is
    threshold-independent, since a coefficient small enough to agree
    across every limb's balanced form is nowhere near either threshold.
    """
    limbs = len(q_moduli)
    d = coeffs.shape[1]
    Q = 1
    for q in q_moduli:
        Q *= q
    gad = _gadgets(q_moduli, Q)

    out = [0] * d
    for i in range(d):
        c_signed = _to_balanced(int(coeffs[0, i]), q_moduli[0])
        is_small = True
        for j in range(1, limbs):
            if c_signed != _to_balanced(int(coeffs[j, i]), q_moduli[j]):
                is_small = False
                break

        if is_small:
            out[i] = c_signed
            continue

        acc = 0
        for j in range(limbs):
            acc += int(coeffs[j, i]) * gad[j]
        acc %= Q

        if acc >= center_from:
            acc -= Q
        out[i] = acc

    return out


def reconstruct_centered(coeffs: np.ndarray, q_moduli) -> list[int]:
    """Port of `reconstructTo` (rns.go). `coeffs` is a `(limbs, d)`
    array of per-limb residues (public `dtype=np.uint64` contract, see
    module docstring), each already reduced to `[0, q_i)` (the
    `RnsRing` convention); `q_moduli` is a plain tuple the same length as
    `coeffs`'s first axis. Returns the length-`d` list of centered
    (balanced) Python ints in `(-Q/2, Q/2]`, `Q = prod(q_moduli)` -- the
    host boundary's point: there is no fixed-width dtype for an integer
    this large, so the return stays exact Python ints, not an array.

    Fast path: if every limb's balanced form (`_to_balanced`) agrees on
    the same value, that value *is* the reconstruction (a coefficient
    with |v| small enough to fit inside every limb's own balanced range
    can't be anything else mod Q). Otherwise falls back to the full
    gadget-sum CRT, reduced mod Q and centered with the **non-strict**
    `acc >= Q>>1` test from Go's `reconstructTo`
    (https://github.com/SNUCP/jindo/blob/68ae757d789d4423fb27eb462719c8c993d5277b/jindo/rns.go#L100; see module docstring for why this
    one is non-strict while `_to_balanced`'s is strict).
    """
    q_moduli = tuple(int(q) for q in q_moduli)
    coeffs = _coerce(coeffs, q_moduli, "reconstruct_centered")
    Q = 1
    for q in q_moduli:
        Q *= q
    return _reconstruct(coeffs, q_moduli, Q >> 1)


def reconstruct_signed_mixed_radix(coeffs: np.ndarray, q_moduli) -> list[int]:
    """The v2 reference's reconstruction: `crt.Embedder`'s signed
    mixed-radix reading (`isMixedRadixNegative`, math/crt/embed_utils.go,
    pin b31f20a). Same `(limbs, d)` uint64 contract in, same exact-int
    list out as `reconstruct_centered` above.

    Go computes it in mixed-radix (Garner) form -- convert to digits
    `d_j`, then call the value negative when the digit vector exceeds
    `(q_j >> 1)_j` lexicographically from the most significant digit
    down. That digit vector reconstructs to exactly `(Q-1)/2`: summing
    `(q_j-1)/2 * prod_{i<j} q_i` telescopes to `(Q-1)/2` for odd moduli
    (every NTT prime here is odd). So the test is exactly `x > (Q-1)/2`,
    which is why this module can share the plain CRT sum above instead of
    carrying a second, digit-based algorithm -- the mixed-radix form is
    Go's route to the answer, not part of the answer.

    **This is not `reconstruct_centered`.** For odd `Q` the two agree
    everywhere except `x == Q>>1 == (Q-1)/2`: v1's rns.go centers it
    (non-strict `acc >= Q>>1`), crt leaves it positive (`x <= Q>>1` is
    non-negative, matching its own single-limb `embedToModOut` path).
    One coefficient value in `Q`, but a deterministic interop surface --
    the gate would disagree with the Go reference on exactly that
    coefficient. Keep them separate.
    """
    q_moduli = tuple(int(q) for q in q_moduli)
    coeffs = _coerce(coeffs, q_moduli, "reconstruct_signed_mixed_radix")
    # The telescoping above needs `q_j >> 1 == (q_j - 1)/2`, so an even
    # limb silently breaks the threshold (and with it the equivalence to
    # Go's digit comparison). Every NTT prime is odd, so this only fires
    # on misuse -- but it fires loudly instead of returning a wrong sign.
    # A single limb is exempt: there are no digits to compare, and the
    # threshold reduces to `x <= q>>1`, which is what Go's own one-limb
    # path (`embedToModOut`) tests regardless of parity.
    if len(q_moduli) > 1 and any(q % 2 == 0 for q in q_moduli):
        raise ValueError(
            "reconstruct_signed_mixed_radix: every modulus in a multi-limb "
            "chain must be odd; got " + repr(q_moduli)
        )
    Q = 1
    for q in q_moduli:
        Q *= q
    return _reconstruct(coeffs, q_moduli, (Q >> 1) + 1)


def set_big_coeffs(vals: list[int], q_moduli) -> np.ndarray:
    """Port of `setBigCoeffTo` (rns.go): reduce each (possibly
    negative, possibly unreduced) int in `vals` mod every limb's modulus.
    Python's `%` already gives the same non-negative result as Go's
    `big.Int.Mod` for a negative dividend, so no sign adjustment is
    needed. Returns the public `(limbs, d)` `dtype=np.uint64` contract
    (module docstring) -- lossless by construction, since every `v % q`
    below is already `< q` and every modulus fits in uint64."""
    q_moduli = tuple(int(q) for q in q_moduli)
    rows = [[v % q for v in vals] for q in q_moduli]
    return np.array(rows, dtype=np.uint64)


def rsh_floor(vals: list[int], k: int) -> list[int]:
    """Elementwise arithmetic (floor) right-shift matching Go's
    `big.Int.Rsh` on Python ints: `>>` already floors toward -infinity for
    Python ints (e.g. `-5 >> 1 == -3`), the same two's-complement-style
    semantics `big.Int.Rsh` implements."""
    return [v >> k for v in vals]


def lift_centered(coeffs: np.ndarray, from_moduli, to_moduli) -> np.ndarray:
    """Exact replacement for lattigo `BasisExtender.ModUpQtoP` on
    norm-bounded inputs: reconstruct the true centered integer from
    `from_moduli`, then re-reduce it into `to_moduli`. See the module
    docstring for the known, accepted divergence from lattigo's actual
    (approximate) ModUpQtoP off the norm-bounded path. `coeffs` in,
    return out are both the public `(limbs, d)` `dtype=np.uint64`
    contract (`reconstruct_centered`/`set_big_coeffs` above already
    enforce/produce it)."""
    return set_big_coeffs(reconstruct_centered(coeffs, from_moduli), to_moduli)


def embed(coeffs: np.ndarray, from_moduli, to_moduli) -> np.ndarray:
    """Port of `crt.Embedder.EmbedTo` (math/crt/embed.go, pin b31f20a):
    reconstruct the signed mixed-radix value from `from_moduli`, then
    re-reduce it into `to_moduli`. Both arrays are the public
    `(limbs, d)` uint64 contract.

    The two instantiations the v2 protocol builds are Q -> Q_amb (the
    prover's ambient lift, `embQToQAmb` in prover.go) and Q_out -> Q (the
    verifier's inner-commitment re-embed, `embQOutToQ` in verifier.go).

    Coefficient domain only -- Go panics on an NTT-form input, since
    reduction mod a *different* modulus is meaningless on evaluations at
    this modulus's roots of unity. There is no `IsNTT` flag on the array
    contract here, so that stays the caller's invariant.

    Go's overlap shortcut (copy the limb straight across when
    `modOut[i] == modIn[j]`) is an optimization, not a semantic branch:
    `q_out | Q` makes `x` and `x - Q` agree mod `q_out`, so reducing the
    reconstruction reproduces the copied residue. Not ported.
    """
    return set_big_coeffs(reconstruct_signed_mixed_radix(coeffs, from_moduli), to_moduli)


def pow2_cut(coeffs: np.ndarray, from_moduli, log_cut: int, to_moduli) -> np.ndarray:
    """Port of `crt.Pow2Cutter.CutTo` (math/crt/embed.go, pin b31f20a):
    `[p]_from -> [p / 2^log_cut]_to`, the v2 commitment cutoff.

    Go runs it as three embeds -- `x` into the single modulus `2^log_cut`
    (giving the balanced remainder `r`), `r` back into `to_moduli`, and
    `x` into `to_moduli` -- then `(x - r) * (2^log_cut)^-1`. Since
    `x - r` is divisible by `2^log_cut` exactly, that inverse multiply is
    exact division, so this module computes the division directly on the
    reconstructed integer. Same values, no modular inverse needed.

    **This rounds; it does not floor.** `r` is the *balanced* remainder
    (`[-2^(log_cut-1), 2^(log_cut-1)]`, ties positive via crt's
    non-negative `x <= halfQIn` test), so `(x - r) / 2^log_cut` is
    round-half-down, not the `>>` of v1's `rescale_floor` below. The two
    differ by one on every input whose low `log_cut` bits exceed half --
    roughly half of all inputs, not an edge case.

    Guards mirror Go's: `NewPow2Cutter` panics outside `[1, 63]`, and its
    `num.NewModulus(1 << log_cut)` additionally caps the cut modulus at
    `num.MaxModulus`.
    """
    if not 1 <= log_cut <= 63:
        raise ValueError(f"pow2_cut: log_cut must be in [1, 63]; got {log_cut}")
    cut = 1 << log_cut
    if cut >= MAX_MODULUS:
        raise ValueError(
            f"pow2_cut: the cut modulus 2**{log_cut} exceeds the crt stack's "
            f"MaxModulus (2**{MAX_MODULUS_BITS})"
        )

    half = cut >> 1
    out = []
    for x in reconstruct_signed_mixed_radix(coeffs, from_moduli):
        rem = x % cut
        if rem > half:
            rem -= cut
        out.append((x - rem) // cut)
    return set_big_coeffs(out, to_moduli)


def rescale_floor(coeffs: np.ndarray, from_moduli, shift_bits: int, to_moduli) -> np.ndarray:
    """The reconstruct -> floor-shift -> re-embed cutoff dance
    `commitColTo`/`outerCommitTo` (prover.go) run at each MSIS commitment
    step -- named here as one boundary op so `prover.py`'s `_commit_col`
    (in-commitment cutoff) and `_outer_commit` (out-commitment cutoff)
    call it directly instead of re-deriving the composition
    (`verifier.py`'s `_outer`/`_inner` checks run the different
    `mul_scalar`-then-cutoff comparison direction, not this dance).
    Behaviorally exactly:

        set_big_coeffs(rsh_floor(reconstruct_centered(coeffs, from_moduli), shift_bits), to_moduli)

    `coeffs` is `(len(from_moduli), d)`; the return is `(len(to_moduli),
    d)` -- both the public `dtype=np.uint64` contract (module
    docstring). `reconstruct_centered` and `set_big_coeffs` already
    validate/produce that contract at the two array boundaries; nothing
    additional to check here.
    """
    reconstructed = reconstruct_centered(coeffs, from_moduli)
    shifted = rsh_floor(reconstructed, shift_bits)
    return set_big_coeffs(shifted, to_moduli)
