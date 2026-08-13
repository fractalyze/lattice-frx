"""The **host** array contract: `dtype=np.uint64`, every residue `< q_l`.

This is the one place that says what "canonical" means for the
`(limbs, d)` arrays the host side takes and returns. `host_ring.py`'s
module docstring states the contract in prose ("Public contract: ...
`dtype=np.uint64` numpy arrays of canonical (`< q_l` per limb)
standard-form residues"); the predicates below are that sentence in
code, and every enforcement site calls one of them rather than
re-deriving the comparison.

There are two rings and only one of them has this contract. `ring.py`'s
element is a tuple of per-limb field arrays and is described there, not
here — so this module is not the single definition of "the array
contract" any more, it is the single definition of the *host* one.

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

`uint64` is not the device-shaped contract it resembles: frx runs without
x64, so `fnp.asarray` narrows it to `uint32` and truncates any residue
above `2**32` silently, which at `primes.MAX_MODULUS_BITS = 50` is every
limb the package targets. A traced contract carries its width in a field
dtype instead (`zk_dtypes.prime_field(q)`), and since that dtype is
per-modulus it cannot be one array across limbs of different `q_l` — so
the traced form is per-limb, not `(limbs, d)`. That is `ring.py`, which
holds the measurement.

**There is no traced counterpart to these predicates, deliberately.**
Neither half of the contract survives the crossing as something worth
asking: a `prime_field(q)` array reduces internally, so an out-of-range
residue is unrepresentable rather than invalid, and the dtype half is
static metadata the opcode already rejects. What is left is the boundary
where a host array becomes an element, and the array there is still a
host array — so `RnsRing.from_host` calls `require_canonical` and the
traced ring adds nothing of its own. Keeping the check at that seam is
what makes the two rings refuse the same inputs and not merely agree on
the values they accept.

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

    `context` names the failing operation (`"HostRnsRing.ntt"`,
    `"RnsRing.from_host"`, `"rns.reconstruct_centered"`) and is prefixed
    to the message. Both rings reach this predicate, so the context is
    the only thing saying which one refused — name the class, so the
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
