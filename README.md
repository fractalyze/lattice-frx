# lattice-frx

The scheme-agnostic substrate that lattice-based cryptography is built on:
the negacyclic polynomial ring and its NTT, RNS/CRT reconstruction and
basis conversion, and exact discrete Gaussian samplers.

It holds the machinery a lattice scheme needs *before* it becomes a
particular scheme — nothing about commitments, encapsulation, or
signatures lives here.

## What the substrate actually is

Lattice cryptography does not work over scalars. Its objects are
polynomials in `Z_q[X]/(X^d + 1)` — degree below `d`, coefficients mod
`q`, and `X^d = -1` (negacyclic). Two reasons: the hardness assumptions
(Ring-LWE, Module-LWE, MSIS) are stated over that ring, and multiplication
in it costs `O(d log d)` rather than `O(d²)`.

From that follows the asymmetry that most of this package exists to
manage: **values must be small signed integers to mean anything** —
security is expressed as a bound on their norm — **while their storage is
residues mod q**. Moving between those two views, exactly and without
sign or tie-breaking errors, is most of the work.

### `ring.py` and `host_ring.py` — the ring and its NTT

Two implementations of one ring. `ring.py` is the default: per-limb field
arrays, `frx.lax.ntt`, composes into a `jit` zone. `host_ring.py` is the same
ring over exact Python integers — slower, obviously right, and the only place a
reconstruction or a balanced lift can happen, since neither fits a lane.

The Number Theoretic Transform is the FFT over a finite field: transform
both operands, multiply pointwise, transform back. The negacyclic ring
needs a `2d`-th root of unity, which exists only when `q ≡ 1 (mod 2d)` —
that condition is what "NTT-friendly prime" means.

And one ring in a second **mode**: `split_ring.py` carries the same
quotient at partial-split moduli `q ≡ 5 (mod 8)`, where `X^d + 1` factors
into exactly two irreducible halves instead of `d` linear ones. That
kills the NTT but buys what LNP-style proofs (eprint 2022/284) stand on:
small σ₋₁-invariant challenge differences stay invertible, which full
splitting can never give. The two prime families are mutually exclusive
(`q ≡ 1 (mod 2d)` forces `q ≡ 1 (mod 8)`), so the mode is chosen by the
modulus and every constructor rejects the other family's primes loudly.
It comes in the same traced/host pair the NTT ring does — `SplitRing`
over per-limb field arrays and `HostSplitRing` over exact Python ints —
and the traced one multiplies in a **`Split` domain**, the two-factor CRT
view, where each half is a twisted convolution gathered out of a `2n-1`
buffer. That is `d²/2` products against the coefficient domain's `d²`,
and it composes into a `jit` zone; the module layer over it (`matvec`,
`matmul`) is still host-only.

The subtlety worth knowing before you use this: **the order an NTT emits
its outputs in is implementation-defined** (bit-reversed vs natural). Two
libraries computing "the same" NTT agree on the values and can disagree on
their order, which is silently wrong for anything that mixes one's output
with the other's bytes. `HostRnsRing` reproduces lattigo's own bit-reversed
table order natively and applies no output permutation anywhere. A
permutation *adapter* belongs to an accelerated backend that computes in a
different natural order and has to present lattigo's at its boundary.

The same class covers ringo-snark's `math/crt` transformer, which fills
the same twiddle table and runs the same butterfly cores — see the module
docstring for why the two references' primitive-root walks provably land
on the same root for any NTT-friendly limb.

The traced element comes in two container types, `Coeff` and `Eval`,
because which domain a value is in is part of what it is — pointwise
`mul` is the ring's multiplication only in the NTT domain — and the
domain is static at trace time, so a type carries it rather than a
lattigo-style runtime flag. Limb arrays may carry leading batch axes
(`[..., d]`); `stack` assembles such batches and `matvec` reads them as
the module layer (MLWE's `A·s`). The Galois automorphisms come in both
domains — `galois` on coefficients, `galois_eval` as a pure slot
permutation whose table is derived through the pinned order contract, so
rotations never leave the NTT domain. The full rationale, including the
deliberately rejected ring-element-as-dtype alternative, is in
[docs/ring-representation.md](docs/ring-representation.md).

### `rns.py` — CRT reconstruction and basis conversion

A useful `Q` runs to 50–200+ bits, past a machine word. So a value is
carried as its residues against a chain of small primes `q_0, q_1, …`;
CRT makes that a faithful representation of `Z_Q` and every operation
becomes per-limb and parallel. What you then need is the way back out:
reconstruction, centered (balanced) lifts, basis extension between chains,
and rescaling.

The tie-breaking here is not a detail. The two reference implementations
this module tracks **disagree at exactly `Q/2`** — one centers it, the
other leaves it positive — and `pow2_cut` *rounds* where `rescale_floor`
*floors*, which differ by one on roughly half of all inputs. Both
behaviors are ported deliberately and documented at their definitions,
which is the kind of thing a shared library exists to pin so that
consumers stop re-deriving it.

### `sampler.py` — discrete Gaussians, and byte-stream challenge draws

Noise and masking must come from a *discrete* Gaussian over `ℤ`. A
rounded continuous normal is a different distribution, and the difference
is a security parameter. Three tiers: exact by explicit support (small σ),
rounded (cheap, admissible only when its Theorem-9 statistical distance,
unioned over every draw a run makes, is negligible), and exact rejection
sampling from a discrete-Laplace proposal (any σ, any real center).

`sampler_for(sigma, sample_count)` picks the tier. The count is a plain
`int` because deriving it needs the caller's prover structure, which is
scheme-specific, while nothing this package does with it is.

A second family takes its randomness as an injected **byte stream**
rather than a `Generator`, for draws that must be a deterministic
function of a Fiat-Shamir transcript: `uniform_from_bytes` (uniform
`Z_q`, no modulo bias) and `fixed_weight_ternary` (the SampleInBall-
shaped challenge set). Rejection runs on budgets computed from a stated
failure probability, so a draw consumes exactly the byte count its
`*_bytes_needed` companion quotes; the stream contract is pinned in the
module docstring, and the XOF or CSPRNG producing the bytes stays the
consumer's, as ever.

### `gadget.py` — digit decomposition

The digit view against the gadget vector `(1, 2^w, 2^{2w}, …)` that FHE
key switching, digit-decomposed proofs, and trapdoor constructions all
share. Balanced digits in `[-B/2, B/2)` — whose representable interval
is asymmetric on purpose, pinned and tested at its exact endpoints —
plus the textbook unsigned form, over an already-reconstructed lift,
exact integers only.

### `norms.py` — the measurement security bounds are stated in

ℓ∞ and squared ℓ2 over an already-reconstructed balanced lift, exact
integers only. They take the lift rather than residues so the caller's
choice between the two reconstructions — which disagree at exactly
`Q/2` — stays visible at the call site.

### `domains.py`, `primes.py`, `canonical.py`

Which domain a traced element is in, carried as its type — `Coeff` shared by
both rings, `Eval` the NTT ring's and `Split` the partial-split one's — plus
the two guards every traced op opens with. A ring passes its own pair, so the
other ring's domain is refused at the guard rather than producing well-shaped
nonsense.

Prime search for both modulus families — NTT-friendly (`≡ 1 mod 2d`) and
partial-split (`≡ 5 mod 8`), one shared walk — and the array contract
every module above enforces: `dtype=np.uint64`, every residue below its
own limb's modulus, defined once with a `bool` form for callers deciding
something and a raising form for callers about to assume it. Per-modulus
ring constants (`primitive_root`, `split_root`) live in `roots.py`, below
both rings.

## Where this sits

The other `*-frx` repos are split by **cryptographic function** — hash,
encrypt, sign — with `hash-frx` at the bottom because the other two hash.
This one splits on a different axis, **mathematical substrate**, and the
org's two existing lattice schemes land on opposite sides of the function
split: ML-KEM in `enc-frx`, ML-DSA in `sig-frx`. So this is a layer
*below* both, beside `hash-frx`:

```
   hash-frx          lattice-frx        <- substrates, no frx deps
    /      \          /        \
enc-frx   sig-frx ---'          `--- consumers (jindo-zorch, ...)
(ML-KEM)  (ML-DSA)
```

**That is why this package has no frx dependency and must not grow one.**
A dependency on `enc-frx` would make `enc-frx`'s own adoption of this
substrate circular. Two things that consequently stay out:

- **The CSPRNG behind a uniform sampler** (expanding a CRS seed into
  uniform public matrices — every MSIS/MLWE commitment needs one). The
  algorithm choice, typically AES-CTR or an XOF, belongs to the consumer
  and is injected. The precedent is already here: the Gaussian samplers
  take `rng: np.random.Generator`, and the byte-stream samplers take the
  squeezed bytes themselves — the bytes→`Z_q` mapping lives here, the
  byte *producer* does not.
- **Hashes and Fiat-Shamir transcripts.** Those are `hash-frx`'s and
  `zorch`'s.

Whether ML-KEM and ML-DSA actually adopt this is open and separate. Their
NTTs are hand-written against fixed lane widths (int16/int32), which is a
real argument for keeping them where they are; this ring is runtime-
parametric over 64-bit moduli. The layering exists to keep that edge
*possible*, not to assert it.

## What does not belong here

Encoders, parameter searches, Fiat-Shamir chain composition,
prover/verifier logic, and proof container formats are all scheme-specific.

One rule is worth stating because it is easy to violate while passing every
import check: **a function here must not take a consumer's parameter
object.** Signatures take primitives, containers, and this package's own
types. A tier gate that reads five fields off some scheme's `Parameters`
imports nothing and is still unusable by the next consumer — which is
exactly how the first version of `sampler_for` came to be bypassed rather
than reused.

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m unittest discover -s lattice_frx/testing -t . -p '*_test.py'
```

Or hermetically via Bazel — provisions its own Python 3.11 toolchain and
the locked wheels (this is what CI runs):

```bash
bazel test //...
```

## Consuming it

As a bzlmod module:

```python
bazel_dep(name = "lattice_frx", version = "0.0.1")
git_override(
    module_name = "lattice_frx",
    commit = "<sha>",
    remote = "https://github.com/fractalyze/lattice-frx.git",
)
```

To build against a local checkout instead of the pin:

```bash
echo 'common --override_module=lattice_frx=/abs/path/to/lattice-frx' >> .bazelrc.user
```

## Testing

The suite is property-based: CRT round-trips including negatives, the
centering boundary, fast-path/slow-path agreement, NTT round-trips,
no-mutation-of-inputs across every public op, the contract's two failure
modes, and distribution checks for each sampler tier.

It is written on `absl.testing` (`absltest` + `parameterized`); every
`*_test.py` carries its own `absltest.main()` guard, so a file runs
directly, under Bazel's plain `py_test`, or through stdlib `unittest`
discovery — the three commands above are the same suite.

Golden-vector conformance against a *specific* reference at a *specific*
moduli chain belongs to the consumer that derives those moduli — it is a
test of "my dependency behaves correctly at my parameters", and
jindo-zorch carries one.
