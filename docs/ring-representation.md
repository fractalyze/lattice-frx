# How a ring element is represented, and why

The design decisions behind `ring.py`'s element representation, recorded so
the next consumer (an FHE evaluation layer, a commitment scheme, a folding
prover) extends the pattern instead of re-litigating it — and so the one
alternative we deliberately rejected stays rejected until its revisit signal
actually fires.

## The representation

A traced RNS ring element is a **pair of `NamedTuple` containers over the same
shape** — `Coeff` and `Eval`, each holding one `[..., d]` field array per limb:

```
Coeff(limbs=(field_q0[..., d], field_q1[..., d], ...))   # coefficient domain
Eval (limbs=(field_q0[..., d], field_q1[..., d], ...))   # NTT domain, contract order
```

Three separate decisions are packed in there, each forced by a constraint:

1. **Width is carried by the field dtype, not an integer lane.** frx runs
   without x64, so `uint64` narrows to `uint32` and truncates 50-bit residues
   without raising. `zk_dtypes.prime_field(q)` sizes storage from its modulus
   and reduces internally — the dtype is the contract. (The `(limbs, d)`
   `uint64` array in `canonical.py` is the *host* contract; the two meet only
   at `coeff_from_host` / `eval_from_host` / `to_host`.)
2. **An element is a tuple of per-limb arrays, not one array.** A field dtype
   is per-modulus and an array has one dtype, so an RNS element with a
   different `q_l` per limb cannot be one array. A tuple is already a pytree,
   so it threads `jit` and `vmap` with no registered type.
3. **The domain is carried by the container type, not a flag.** Pointwise
   `mul` is the ring's multiplication only in the NTT domain; a balanced lift
   only means anything in the coefficient domain. lattigo tracks this with a
   runtime `IsNTT` flag per polynomial — in a traced stack that information is
   static at trace time, so a type does the job with no runtime cost:
   `mul(coeff, coeff)` is a `TypeError` when the graph is built, and a flag
   branch would have split the trace for information the graph already had.
   The NTT-order contract (lattigo's bit-reversed table order) rides on
   `Eval`'s definition.

## The module layer is a shape convention

MLWE's `A·s + e` — and every Ajtai-style commitment — is a matrix of ring
elements times a vector of ring elements. That is **not a new type**: a vector
of elements is an element whose limb arrays carry a leading axis.

```
element:  limbs of [d]
vector:   limbs of [k, d]      — stack of k elements
matrix:   limbs of [m, k, d]   — stack of m vectors
A·s:      matvec — per limb, [.., m, k, d] × [.., k, d] → [.., m, d]  (Eval only)
```

Every other op is pointwise per limb or transforms `axis=-1`, so batching is
what the ops already do. `stack` is the convention's constructor — consumers
assemble batches through it rather than re-deriving the limb transpose by
hand — and `matvec` is the one op that reads the leading axes rather than
mapping over them.

## Rejected: a ring element as a zk_dtypes scalar

The tempting extension of the EC-point precedent — where a point's coordinates
pack into one scalar dtype — is a `negacyclic_ring(q, d)` dtype whose scalars
*are* ring elements. Rejected, for reasons that don't age out on their own:

1. **Size inversion.** An EC point is O(1) field elements; a ring element is
   `d = 2^10..2^17` of them. An 8KB–1MB "scalar" inverts the array/element
   size relationship the whole array stack's layout, fusion, and
   vectorization assumptions sit on, and a scalar op becomes a d-sized kernel
   — at which point the dtype abstraction is providing nothing.
2. **RNS multiplies the dtype space.** Folding limbs into the dtype mints one
   dtype per (moduli-chain, d) pair, and rescaling — which drops a limb —
   changes an element's dtype mid-computation. Keeping limbs outside the
   dtype collapses back to the tuple-of-arrays this repo already has, with an
   extra layer.
3. **It doesn't solve the domain problem.** Coefficient and NTT domain share
   a storage shape, so one dtype still can't tell them apart — and two dtypes
   per ring doubles the mint.

**Revisit signal:** a *second* consumer whose ring-generic code demonstrably
cannot be expressed over "typed tuples of field arrays". One consumer that
merely finds a dtype more ergonomic is not the signal — that was the
`sampler_for` lesson: abstractions minted for one consumer get bypassed by the
next.

## What stays out of this layer

Cancelling `intt ∘ ntt` round-trips, keeping long-lived operands (keys, CRS)
resident in the NTT domain, hoisting transforms out of loops — those are
compiler rewrites over the traced graph, which is exactly why the transform is
an opcode (`frx.lax.ntt`) rather than an opaque call: the backend can see it.
This package's job ends at making the graph honest; scheduling it is the
backend's.
