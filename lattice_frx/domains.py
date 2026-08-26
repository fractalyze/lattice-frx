"""Which domain a traced element is in, carried as its type.

The rings here compute in more than one representation of the same element,
and which one a value is in is part of *what it is*: pointwise `mul` is the
ring's multiplication only in the NTT domain, a balanced lift only means
anything on coefficients, and a two-factor CRT half is neither. lattigo
carries this as a runtime `IsNTT` flag per polynomial. Here the information
is static at trace time, so a container type carries it instead — a flag
would be a trace-splitting branch for something the graph already knows,
while a type makes `mul(coeff, coeff)` a `TypeError` at trace time and costs
the compiled graph nothing.

The three types are one-field `NamedTuple`s over the same per-limb tuple, so
they are already pytrees and thread `jit`/`vmap` without a registration of
their own. They differ only in what they assert, which is the whole point.

`Coeff` is shared: both rings have a coefficient domain and it means the same
thing in each. `Eval` belongs to the NTT ring (`ring.py`) and `Split` to the
partial-split one (`split_ring.py`) — the two modulus families are mutually
exclusive, so no element is ever legally in both, and a ring passes its own
pair to `same_domain` rather than accepting all three.

The guards live here rather than in either ring because duplicating them is
how one copy comes to forget an operand.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class Coeff(NamedTuple):
    """A coefficient-domain element: one `[..., d]` field array per limb."""

    limbs: tuple[Any, ...]


class Eval(NamedTuple):
    """An NTT-domain element, in the contract's (lattigo's) bit-reversed order."""

    limbs: tuple[Any, ...]


class Split(NamedTuple):
    """A two-factor CRT element: one `[..., 2, d/2]` field array per limb.

    Half `0` is the residue modulo `X^{d/2} - r`, half `1` modulo
    `X^{d/2} + r`. Unlike `Eval` this domain does *not* make multiplication
    pointwise — each half is a twisted convolution — so it buys structure
    (and the halved operand count), not a free product.
    """

    limbs: tuple[Any, ...]


def require_domain(op: str, domain: type, *values) -> None:
    """Every operand in the one domain `op` is defined in, or a `TypeError`.

    Variadic so an op is one call covering all operands — a per-operand
    guard that forgets one would readmit, at that operand, the bug class
    the types exist to kill.
    """
    for value in values:
        if type(value) is not domain:
            raise TypeError(
                f"{op}: expected {domain.__name__}, got {type(value).__name__}"
            )


def same_domain(op: str, allowed: tuple[type, ...], *values) -> type:
    """The shared domain of `values`, or a `TypeError` naming the mismatch.

    `allowed` is the calling ring's own pair, so a domain that belongs to the
    other ring is rejected here rather than sailing through a generic op and
    surfacing as a shape error several calls later.
    """
    first = type(values[0])
    if first not in allowed or any(type(v) is not first for v in values[1:]):
        raise TypeError(
            f"{op}: operands must share a domain, got "
            + ", ".join(type(v).__name__ for v in values)
        )
    return first
