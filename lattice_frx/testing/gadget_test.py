"""The digit view's contract: exact round-trips, pinned ranges, honest edges.

The decomposition interval is not the symmetric `[-B^t/2, B^t/2)` intuition
suggests — balanced digits in `[-B/2, B/2)` reach exactly
`[-(B/2)·S, (B/2-1)·S]` with `S = (B^t-1)/(B-1)`, asymmetric because `+B/2`
is the excluded endpoint. These tests hold the algorithm to that exact
interval, endpoints included and one-past rejected, so a consumer sizing
`num_digits` for a bound gets a documented answer instead of a lucky one.
"""

import random

import pytest

from lattice_frx import gadget

_LOG_BASE = 4
_BASE = 1 << _LOG_BASE
_DIGITS = 5


def _span(log_base: int, num_digits: int) -> int:
    """`S = (B^t - 1)/(B - 1)`, the geometric factor both endpoints scale."""
    base = 1 << log_base
    return (base**num_digits - 1) // (base - 1)


def _lo(log_base: int, num_digits: int) -> int:
    return -(1 << (log_base - 1)) * _span(log_base, num_digits)


def _hi(log_base: int, num_digits: int) -> int:
    return ((1 << (log_base - 1)) - 1) * _span(log_base, num_digits)


def test_balanced_round_trips_and_bounds_its_digits() -> None:
    rnd = random.Random(1)
    half = _BASE >> 1
    for _ in range(500):
        value = rnd.randint(_lo(_LOG_BASE, _DIGITS), _hi(_LOG_BASE, _DIGITS))
        digits = gadget.decompose(value, _LOG_BASE, _DIGITS)
        assert len(digits) == _DIGITS
        assert all(-half <= d < half for d in digits)
        assert gadget.recompose(digits, _LOG_BASE) == value


def test_balanced_covers_its_exact_interval_and_nothing_more() -> None:
    lo, hi = _lo(_LOG_BASE, _DIGITS), _hi(_LOG_BASE, _DIGITS)
    assert gadget.recompose(gadget.decompose(lo, _LOG_BASE, _DIGITS), _LOG_BASE) == lo
    assert gadget.recompose(gadget.decompose(hi, _LOG_BASE, _DIGITS), _LOG_BASE) == hi
    with pytest.raises(ValueError):
        gadget.decompose(lo - 1, _LOG_BASE, _DIGITS)
    with pytest.raises(ValueError):
        gadget.decompose(hi + 1, _LOG_BASE, _DIGITS)


def test_balanced_exhaustively_at_a_small_base() -> None:
    """Every value of the tiny case, so the interval claim is checked, not
    trusted: base 4, two digits, `[-10, 5]` and exactly that."""
    for value in range(-10, 6):
        digits = gadget.decompose(value, 2, 2)
        assert all(-2 <= d < 2 for d in digits)
        assert gadget.recompose(digits, 2) == value
    for value in (-11, 6, 7):
        with pytest.raises(ValueError):
            gadget.decompose(value, 2, 2)


def test_unsigned_round_trips_over_its_interval() -> None:
    rnd = random.Random(2)
    top = (1 << (_LOG_BASE * _DIGITS)) - 1
    for value in [0, top, *(rnd.randint(0, top) for _ in range(500))]:
        digits = gadget.decompose_unsigned(value, _LOG_BASE, _DIGITS)
        assert len(digits) == _DIGITS
        assert all(0 <= d < _BASE for d in digits)
        assert gadget.recompose(digits, _LOG_BASE) == value
    with pytest.raises(ValueError):
        gadget.decompose_unsigned(-1, _LOG_BASE, _DIGITS)
    with pytest.raises(ValueError):
        gadget.decompose_unsigned(top + 1, _LOG_BASE, _DIGITS)


def test_zero_is_all_zero_digits_in_both_conventions() -> None:
    assert gadget.decompose(0, _LOG_BASE, _DIGITS) == [0] * _DIGITS
    assert gadget.decompose_unsigned(0, _LOG_BASE, _DIGITS) == [0] * _DIGITS


def test_the_vector_form_is_digit_major() -> None:
    """Row `i` is every value's `i`-th digit — one future ring element per
    row, which is the orientation a gadget consumer feeds key switching."""
    rnd = random.Random(3)
    values = [
        rnd.randint(_lo(_LOG_BASE, _DIGITS), _hi(_LOG_BASE, _DIGITS))
        for _ in range(17)
    ]
    rows = gadget.decompose_vector(values, _LOG_BASE, _DIGITS)
    assert len(rows) == _DIGITS
    assert all(len(row) == len(values) for row in rows)
    for j, value in enumerate(values):
        assert [row[j] for row in rows] == gadget.decompose(
            value, _LOG_BASE, _DIGITS
        )
    assert gadget.recompose_vector(rows, _LOG_BASE) == values


def test_the_vector_form_survives_empty_input() -> None:
    rows = gadget.decompose_vector([], _LOG_BASE, _DIGITS)
    assert rows == [[] for _ in range(_DIGITS)]
    assert gadget.recompose_vector(rows, _LOG_BASE) == []


@pytest.mark.parametrize("func", [gadget.decompose, gadget.decompose_unsigned])
def test_degenerate_parameters_are_rejected(func) -> None:
    with pytest.raises(ValueError):
        func(1, 0, 4)  # log_base < 1
    with pytest.raises(ValueError):
        func(1, 4, -1)  # negative digit count
