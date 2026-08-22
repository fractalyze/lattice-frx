"""Norms of balanced lifts — the measurement every security statement bounds.

The README's framing is that values must be small signed integers to mean
anything, with security expressed as a bound on their norm; this module is
that measurement's one definition, and deliberately nothing more.

Both functions take an **already-reconstructed balanced lift** — a sequence of
signed integers, as `rns.reconstruct_centered`,
`rns.reconstruct_signed_mixed_radix`, and `ring.to_balanced_limb0` return —
rather than residues. That is not laziness but the contract: the two ported
reconstructions disagree at exactly `Q/2` (see `rns.py`'s module docstring),
so the norm of the same residues differs by one on that coefficient depending
on the reading, and a norm that took residues would have to crown one reading
as "the" norm. Composing explicitly — `linf(reconstruct_centered(coeffs, q))`
— keeps that pinned choice visible at the call site.

Host boundary, like the reconstructions themselves: a lift of a 50-bit
residue has no signed lane and a full-`Q` lift spans `50 · limbs` bits, so
everything here is exact Python integers. The `int(...)` coercions are
load-bearing — a numpy `int64` square wraps silently at values every real
modulus chain produces.

`l2_squared` rather than `l2`: exact integers end at the square, a square
root would leave the exact world for a float, and every bound a scheme states
can be compared squared.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _exact_flat(values: Iterable[int]) -> np.ndarray:
    """`values` flattened into an object array of genuine Python ints.

    `dtype=object` is not by itself what makes the arithmetic exact: an
    object array can hold `numpy.int64` *scalars*, whose `*` wraps at 64
    bits exactly like the fixed-width array they came from, so the
    coercion has to reach the elements. An integer-dtype ndarray gets
    there through `astype(object)`, which converts elementwise in C and
    yields real Python ints — which is why that case is not the loop.
    """
    if isinstance(values, np.ndarray):
        flat = values.reshape(-1)
        if flat.dtype != object:
            return flat.astype(object)
        return np.array([int(v) for v in flat], dtype=object)
    # A one-shot iterable has no shape and can only be walked once.
    return np.array([int(v) for v in values], dtype=object)


def linf(values: Iterable[int]) -> int:
    """`max |v|` over exact integers; 0 on empty input."""
    return max((abs(int(v)) for v in values), default=0)


def l2_squared(values: Iterable[int]) -> int:
    """`Σ v²` over exact integers; 0 on empty input.

    The self-dot rather than a `sum` of squares: identical exact
    arithmetic, and ~2.2x faster measured over n = 256..65536. Input of
    any rank is flattened first — `@` on a rank-2 input would otherwise
    quietly return a matrix product instead of this scalar.
    """
    flat = _exact_flat(values)
    return int(flat @ flat) if flat.size else 0
