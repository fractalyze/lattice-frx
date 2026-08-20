"""What the two-line norms actually pin: exactness, conventions, and the tie.

Exact integers all the way (a numpy `int64` square wraps where the exact path
cannot), the empty-input convention, and the `Q/2` one-off between the two
reconstructions; the lift-not-residues rationale itself lives in `norms.py`.
"""

import random

import numpy as np
import pytest

from lattice_frx import norms, rns


def test_linf_is_the_max_absolute_value() -> None:
    rnd = random.Random(1)
    values = [rnd.randrange(-(1 << 90), 1 << 90) for _ in range(257)]
    assert norms.linf(values) == max(abs(v) for v in values)


def test_l2_squared_is_the_sum_of_squares() -> None:
    rnd = random.Random(2)
    values = [rnd.randrange(-(1 << 90), 1 << 90) for _ in range(257)]
    assert norms.l2_squared(values) == sum(v * v for v in values)


@pytest.mark.parametrize("func", [norms.linf, norms.l2_squared])
def test_empty_input_is_zero(func) -> None:
    assert func([]) == 0


def test_exactness_survives_numpy_int64_input() -> None:
    """`to_balanced_limb0` hands back `int64`; the square must not stay there.

    `(2**40)**2` overflows `int64` — a lazy `v * v` on the numpy scalar wraps
    negative where the exact path returns `2**80`. The coercion to Python int
    inside the norm is load-bearing, and this is the test that keeps it.
    """
    values = np.array([1 << 40, -(1 << 40)], dtype=np.int64)
    assert norms.l2_squared(values) == 2 * (1 << 80)
    assert norms.linf(values) == 1 << 40


def test_the_two_reconstructions_norm_the_tie_one_apart() -> None:
    """The one-off the two readings produce at exactly `Q/2` — the reason
    the norms take a lift, not residues (rationale in `norms.py`)."""
    q_moduli = (5, 7)
    Q = 35
    coeffs = rns.set_big_coeffs([Q >> 1], q_moduli)

    centered = rns.reconstruct_centered(coeffs, q_moduli)
    mixed = rns.reconstruct_signed_mixed_radix(coeffs, q_moduli)

    assert norms.linf(centered) == (Q + 1) // 2  # 18: centered to -18
    assert norms.linf(mixed) == (Q - 1) // 2  # 17: left positive
