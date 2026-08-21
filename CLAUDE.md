# Project context for Claude Code

Read [README.md](README.md) first — it explains what this package is, why
each module is here, and where the repo sits relative to the other
`*-frx` repos. The rules below are the short imperative form of it.

- **No dependency on a scheme repo. Ever.** This package sits *below*
  enc-frx and sig-frx, which already carry ML-KEM and ML-DSA; depending on
  either would make their own adoption of this substrate circular. Same for
  hash-frx and for any crypto library. CI's `no-scheme-deps` job fails the
  build if `MODULE.bazel`, `lattice_frx/BUILD.bazel`, or `requirements.in`
  names one. Do not "temporarily" add one.
- **The array layer is not that direction.** `frx` and `zk-dtypes` are
  *below* enc-frx and sig-frx too — both already sit on them — so depending
  on them closes no loop, and they are what a traced backend here is made
  of. The circularity rule is about schemes, not about arrays; reading it as
  "no frx of any kind" is what would push the traced ring into each consumer
  and have it written three times.
- **Inject primitives, don't depend on them.** A CSPRNG behind a uniform
  sampler, a hash behind a Fiat-Shamir transcript — those are the
  consumer's choice and arrive as arguments. The samplers' existing
  `rng: np.random.Generator` parameter is the pattern to copy. The runtime
  dependencies are numpy, frx and zk-dtypes and nothing else — the array
  layer, never a primitive.
- **Never take a consumer's parameter object.** Signatures take
  primitives, containers, and this package's own types. `sampler_for`
  takes the draw count as an `int`, not a `Parameters` — the first
  version took the object, imported nothing from the consumer, passed
  every boundary check, and was still unusable by the second consumer,
  which silently reimplemented it instead. That failure mode is invisible
  to an import-graph check, so it has to be a review habit.
- **The domain is a type, not a flag.** Traced ring ops take `Coeff` or
  `Eval`, never bare limb tuples; `mul`/`mul_add`/`matvec` are `Eval`-only,
  `to_balanced_limb0` is `Coeff`-only, and the embedding constructors are
  named for the domain the caller asserts (`coeff_from_host` /
  `eval_from_host`). Do not add a runtime `IsNTT`-style flag (lattigo's
  move) — the domain is static at trace time and a flag branch splits the
  trace for information the graph already has.
- **No ring-element dtype, and the module layer is a shape convention.**
  The dtype boundary sits at the field (`zk_dtypes.prime_field`); a ring
  element is a typed tuple of per-limb `[..., d]` arrays, a module vector is
  the `[k, d]` case of the same limbs, and `A·s` is `matvec`, not a new
  type. The rejection rationale and its revisit signal (a *second* consumer
  that provably cannot live on typed tuples) are in
  [docs/ring-representation.md](docs/ring-representation.md) — read it
  before proposing either a `negacyclic_ring` dtype or a `Module` class.
- **The array contract has one definition**, in `canonical.py`
  (`is_canonical` to ask, `require_canonical` to enforce): `dtype=uint64`,
  every residue below its own limb's modulus. Call it; never re-hand-write
  the comparison. Keep the two failure modes distinct — `TypeError` for
  dtype, `ValueError` for range — they are different caller bugs.
- **The `uint64` contract is a HOST contract, not a device-shaped one**,
  however much it looks like one. frx runs without x64, so `fnp.asarray`
  narrows `uint64` to `uint32` and truncates every limb above `2**32`
  *without raising* — at `MAX_MODULUS_BITS = 50` that is every limb this
  package targets. A traced backend carries width in a field dtype
  (`zk_dtypes.prime_field(q)`), which is per-modulus, so the traced shape
  is one array per limb rather than one `(limbs, d)` array — which is what
  `ring.py` carries and `host_ring.py` does not. Do not write "device-shaped"
  about the `uint64` contract, and do not assume `np` → `fnp` is the
  migration.
- **Tests are property-based and absltest-based**
  (`lattice_frx/testing/*_test.py`, on `absl.testing`'s `absltest` /
  `parameterized` — not pytest; every file ends in an `absltest.main()`
  guard so a plain `py_test` runs it). A golden
  keyed to a *specific* reference at a *specific* moduli chain belongs to
  the consumer that derives those moduli — it tests "my dependency behaves
  correctly at my parameters". Don't pull one in here.
- **So the NTT-order gate is the consumer's, and it has to actually run.**
  The rule above and the NTT-order rule below would otherwise cancel out:
  order is declared part of the contract, no property test can catch a
  *consistent* permutation, and goldens are not kept here — which would
  leave the property with no gate anywhere. Measured, with `ntt`'s output
  and `intt`'s input permuted inversely, every suite in this repo stays
  green. What closes it is the last rule on this page: a change to the
  transform is a breaking change, and the consumer's pin bump plus its
  cross-verification gates are the gate. Sequence it that way rather than
  merging a transform change and bumping later.
- **The NTT's output order is part of its contract.** `HostRnsRing`
  reproduces lattigo's bit-reversed table order natively and applies no
  output permutation; `RnsRing` presents the same order with exactly one
  bit-reversal at its boundary, because the opcode emits natural order. An NTT that computes the right values in the wrong
  order is silently wrong for anything mixing this output with another
  implementation's bytes, and no property test catches it — a permutation
  belongs in an accelerated backend's boundary adapter, not here.
- **Two ring modes, mutually exclusive moduli.** The NTT rings
  (`ring.py`/`host_ring.py`) need limbs `≡ 1 (mod 2d)`; the partial-split
  ring (`split_ring.py`, the LNP line's mode) needs limbs `≡ 5 (mod 8)`,
  and no prime is both. Never feed one family's primes to the other's
  ring — constructors guard it, and the guard's error names the other
  mode because a silent mix would surface as a consumer's soundness gap.
- **Two reference implementations disagree on purpose.** `rns.py` ports
  both, and the differences are real: reconstruction tie-breaking at
  exactly `Q/2`, and `pow2_cut` rounding where `rescale_floor` floors
  (differing by one on ~half of all inputs). They are documented at their
  definitions. Do not "unify" them.
- Consumers pin this repo by commit (bzlmod `git_override` and/or a
  pyproject git URL). A breaking change here needs the consumer's pin
  bumped and its suite re-run — for jindo-zorch that means both
  cross-verification gates, not just unit tests.
