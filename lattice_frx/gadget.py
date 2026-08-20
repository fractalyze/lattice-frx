"""Base-`2^w` digit decomposition — the digit view three consumer families share.

FHE key switching decomposes to keep noise growth additive, lattice range and
norm proofs decompose a witness into small digits, and Micciancio–Peikert
trapdoors are built on the gadget vector `(1, 2^w, 2^{2w}, …)` itself. What
they share is exactly this module: representing a bounded integer as digits
against that vector, exactly, with the edge conventions pinned once.

Two digit conventions, both with the recomposition identity
`Σ dᵢ·2^{iw} == value` as the invariant their tests hold:

- **Balanced** (`decompose`): digits in `[-B/2, B/2)`, `B = 2^w` — the form
  proofs and trapdoors want, since the digit *norm* is what the security
  statement bounds. The excluded `+B/2` endpoint makes the representable
  interval asymmetric: exactly `[-(B/2)·S, (B/2-1)·S]` with
  `S = (B^t - 1)/(B - 1)` over `t` digits — not the symmetric
  `[-B^t/2, B^t/2)` intuition suggests. A value outside it raises
  `ValueError` rather than wrapping.
- **Unsigned** (`decompose_unsigned`): textbook digits in `[0, B)`, values in
  `[0, B^t)`.

Everything is exact Python integers over an already-reconstructed balanced
lift (`list[int]`, as the `rns` reconstructions return), the same host
boundary and for the same reason as `norms.py`: which reconstruction produced
the lift is the caller's pinned choice, and no lane holds a full-`Q` value. A
traced (per-limb, on-device) decomposition is a separate step gated on a
consumer that needs it; this module fixes the semantics that one must match.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def _require_shape(log_base: int, num_digits: int, op: str) -> None:
    if log_base < 1:
        raise ValueError(f"{op}: log_base must be >= 1, got {log_base}")
    if num_digits < 0:
        raise ValueError(f"{op}: num_digits must be >= 0, got {num_digits}")


def decompose(value: int, log_base: int, num_digits: int) -> list[int]:
    """Balanced digits of `value`, least significant first.

    Each digit lies in `[-B/2, B/2)`; the remainder-`B/2` tie goes to `-B/2`,
    which is what excludes `+B/2` and makes the interval asymmetric (see the
    module docstring for the exact endpoints).
    """
    _require_shape(log_base, num_digits, "decompose")
    base = 1 << log_base
    half = base >> 1
    x = int(value)
    digits = []
    for _ in range(num_digits):
        digit = ((x + half) % base) - half
        digits.append(digit)
        x = (x - digit) >> log_base  # exact: x - digit is divisible by base
    if x:
        raise ValueError(
            f"decompose: {int(value)} needs more than {num_digits} balanced "
            f"base-2^{log_base} digits (residual {x})"
        )
    return digits


def decompose_unsigned(value: int, log_base: int, num_digits: int) -> list[int]:
    """Textbook digits in `[0, B)`, least significant first; `value` in `[0, B^t)`."""
    _require_shape(log_base, num_digits, "decompose_unsigned")
    x = int(value)
    if x < 0:
        raise ValueError(f"decompose_unsigned: value must be >= 0, got {x}")
    mask = (1 << log_base) - 1
    digits = []
    for _ in range(num_digits):
        digits.append(x & mask)
        x >>= log_base
    if x:
        raise ValueError(
            f"decompose_unsigned: {int(value)} needs more than {num_digits} "
            f"base-2^{log_base} digits"
        )
    return digits


def recompose(digits: Iterable[int], log_base: int) -> int:
    """`Σ dᵢ·2^{iw}` — the shared inverse of both conventions."""
    _require_shape(log_base, 0, "recompose")
    return sum(int(d) << (log_base * i) for i, d in enumerate(digits))


def decompose_vector(
    values: Iterable[int], log_base: int, num_digits: int
) -> list[list[int]]:
    """Balanced digits of a coefficient vector, **digit-major**: row `i` holds
    every value's `i`-th digit, so each row is one future ring element — the
    orientation a gadget consumer feeds key switching or a digit-decomposed
    witness."""
    per_value = [decompose(v, log_base, num_digits) for v in values]
    if not per_value:
        return [[] for _ in range(num_digits)]
    return [list(row) for row in zip(*per_value)]


def recompose_vector(rows: Sequence[Sequence[int]], log_base: int) -> list[int]:
    """The inverse of `decompose_vector`: recompose each column back to its value."""
    return [recompose(column, log_base) for column in zip(*rows)]
