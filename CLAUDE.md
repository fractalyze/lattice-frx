# Project context for Claude Code

Read [README.md](README.md) first — it explains what this package is, why
each module is here, and where the repo sits relative to the other
`*-frx` repos. The rules below are the short imperative form of it.

- **No frx dependency. Ever.** This package sits *below* enc-frx and
  sig-frx, which already carry ML-KEM and ML-DSA; depending on either
  would make their own adoption of this substrate circular. CI's
  `no-frx-deps` job fails the build if `MODULE.bazel`,
  `lattice_frx/BUILD.bazel`, or `requirements.in` names an frx module or
  a crypto library. Do not "temporarily" add one.
- **Inject primitives, don't depend on them.** A CSPRNG behind a uniform
  sampler, a hash behind a Fiat-Shamir transcript — those are the
  consumer's choice and arrive as arguments. The samplers' existing
  `rng: np.random.Generator` parameter is the pattern to copy. numpy is
  the only runtime dependency.
- **Never take a consumer's parameter object.** Signatures take
  primitives, containers, and this package's own types. `sampler_for`
  takes the draw count as an `int`, not a `Parameters` — the first
  version took the object, imported nothing from the consumer, passed
  every boundary check, and was still unusable by the second consumer,
  which silently reimplemented it instead. That failure mode is invisible
  to an import-graph check, so it has to be a review habit.
- **The array contract has one definition**, in `canonical.py`
  (`is_canonical` to ask, `require_canonical` to enforce): `dtype=uint64`,
  every residue below its own limb's modulus. Call it; never re-hand-write
  the comparison. Keep the two failure modes distinct — `TypeError` for
  dtype, `ValueError` for range — they are different caller bugs.
- **Tests are property-based** (`lattice_frx/testing/*_test.py`). A golden
  keyed to a *specific* reference at a *specific* moduli chain belongs to
  the consumer that derives those moduli — it tests "my dependency behaves
  correctly at my parameters". Don't pull one in here.
- **The NTT's output order is part of its contract.** `RnsRing`
  reproduces lattigo's bit-reversed table order natively and applies no
  output permutation. An NTT that computes the right values in the wrong
  order is silently wrong for anything mixing this output with another
  implementation's bytes, and no property test catches it — a permutation
  belongs in an accelerated backend's boundary adapter, not here.
- **Two reference implementations disagree on purpose.** `rns.py` ports
  both, and the differences are real: reconstruction tie-breaking at
  exactly `Q/2`, and `pow2_cut` rounding where `rescale_floor` floors
  (differing by one on ~half of all inputs). They are documented at their
  definitions. Do not "unify" them.
- Consumers pin this repo by commit (bzlmod `git_override` and/or a
  pyproject git URL). A breaking change here needs the consumer's pin
  bumped and its suite re-run — for jindo-zorch that means both
  cross-verification gates, not just unit tests.
