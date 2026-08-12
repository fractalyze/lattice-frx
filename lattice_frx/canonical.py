"""The array contract: `dtype=np.uint64`, every residue `< q_l`.

This is the one place that says what "canonical" means for the
`(limbs, d)` arrays every module here takes and returns. `ring.py`'s
module docstring states the contract in prose ("Public contract: ...
`dtype=np.uint64` numpy arrays of canonical (`< q_l` per limb)
standard-form residues"); the predicates below are that sentence in
code, and every enforcement site calls one of them rather than
re-deriving the comparison.

Two shapes, because callers need two different answers:

- `is_canonical` — a plain `bool`, for a caller deciding something (a
  verifier asking "is this proof well-formed?", where an out-of-range
  residue means *False*, not an exception).
- `require_canonical` — the raising boundary, for a caller about to do
  ring arithmetic on the array. It splits the two failures by type:
  `TypeError` for the wrong dtype, `ValueError` for an out-of-range
  residue. The split matters because they are different bugs. A wrong
  dtype (`object`, or a signed array) is a caller that never embedded
  its host-side Python ints, or one that would wrap on the first
  operation; an out-of-range residue is a value that skipped a
  reduction. Collapsing them into one exception type loses that.

This is a host contract, and `uint64` is not the device-shaped one it
resembles: frx runs without x64, so `fnp.asarray` narrows it to `uint32` and
truncates any residue above `2**32` silently, which at
`primes.MAX_MODULUS_BITS = 50` is every limb the package targets. A
traced contract carries its width in a field dtype instead
(`zk_dtypes.prime_field(q)`), and since that dtype is per-modulus it
cannot be one array across limbs of different `q_l` — so the traced form
of this contract is per-limb, not `(limbs, d)`. Issue #1 carries the
measurement and the migration.

The dtype rule is deliberately strict — no "close enough" integer dtype
is accepted. A signed array carrying the same values wraps on the first
operation here, and an `object` array of host-side Python ints silently
opts out of the fixed-width arithmetic the contract exists to promise,
so accepting either would move the failure somewhere far less legible
than this boundary.

The moduli column is cached per `q_moduli` tuple: a `Prover`/`Verifier`
runs these checks many times per commit/evaluate/verify against the same
one or two tuples, so rebuilding the tiny comparison array on every call
is pure waste. `q_moduli` tuples are few and fixed per run, which makes
an unbounded cache safe.
"""
import functools

import numpy as np

# The contract's dtype, named once so the two predicates below agree by
# construction rather than by both spelling out `np.uint64`.
_CONTRACT_DTYPE = np.dtype(np.uint64)


@functools.lru_cache(maxsize=None)
def _modulus_column(q_moduli: tuple[int, ...]) -> np.ndarray:
    """The per-limb modulus as a `(limbs, 1)` uint64 column, ready to
    broadcast against a `(limbs, d)` array.

    `uint64` rather than object dtype: this column is only ever compared
    with `<` against values already known to fit in uint64, so it runs as
    a vectorized machine-word compare instead of per-element Python
    `__lt__`.
    """
    return np.array(q_moduli, dtype=np.uint64)[:, None]


def _residues_in_range(arr: np.ndarray, q_moduli: tuple[int, ...]) -> bool:
    """Whether every residue is below its own limb's modulus.

    `arr` must broadcast against the `(limbs, 1)` modulus column —
    validating the *shape* is the caller's job, since what a wrong shape
    means differs per caller (a wire-format mismatch for a proof reader,
    a programming error for a ring op).
    """
    return bool((arr < _modulus_column(q_moduli)).all())


def is_canonical(arr: np.ndarray, q_moduli) -> bool:
    """Whether `arr` satisfies the contract: uint64, every residue `< q_l`."""
    return arr.dtype == _CONTRACT_DTYPE and _residues_in_range(
        arr, tuple(int(q) for q in q_moduli)
    )


def require_canonical(arr: np.ndarray, q_moduli, context: str) -> None:
    """`is_canonical`, raising instead of returning — the boundary check
    at the head of an operation that is about to assume the contract.

    `context` names the failing operation (`"RnsRing.ntt"`,
    `"rns.reconstruct_centered"`) and is prefixed to the message, so the
    raise points at the call the caller made rather than at this module.
    """
    if arr.dtype != _CONTRACT_DTYPE:
        raise TypeError(
            f"{context}: expected dtype=np.uint64 (the host ring "
            f"contract); got dtype={arr.dtype!r}"
        )
    if not _residues_in_range(arr, tuple(int(q) for q in q_moduli)):
        raise ValueError(
            f"{context}: input has a residue >= its limb's modulus — "
            "the public contract requires canonical standard-form "
            "residues (< q_l per limb)"
        )
