"""The digit view held to its exact interval — endpoints included, one-past
rejected — so a consumer sizing `num_digits` for a bound gets the documented
answer (the asymmetric-interval derivation lives in `gadget.py`)."""

import random

from absl.testing import absltest
from absl.testing import parameterized

from lattice_frx import gadget

_LOG_BASE = 4
_BASE = 1 << _LOG_BASE
_DIGITS = 5
# `S = (B^t - 1)/(B - 1)`, the geometric factor both endpoints scale.
_SPAN = (_BASE**_DIGITS - 1) // (_BASE - 1)
_LO = -(_BASE >> 1) * _SPAN
_HI = ((_BASE >> 1) - 1) * _SPAN


class GadgetTest(parameterized.TestCase):

    def test_balanced_round_trips_and_bounds_its_digits(self) -> None:
        rnd = random.Random(1)
        half = _BASE >> 1
        for _ in range(500):
            value = rnd.randint(_LO, _HI)
            digits = gadget.decompose(value, _LOG_BASE, _DIGITS)
            self.assertLen(digits, _DIGITS)
            self.assertTrue(all(-half <= d < half for d in digits))
            self.assertEqual(gadget.recompose(digits, _LOG_BASE), value)

    def test_balanced_covers_its_exact_interval_and_nothing_more(self) -> None:
        for end in (_LO, _HI):
            digits = gadget.decompose(end, _LOG_BASE, _DIGITS)
            self.assertEqual(gadget.recompose(digits, _LOG_BASE), end)
        for past in (_LO - 1, _HI + 1):
            with self.assertRaises(ValueError):
                gadget.decompose(past, _LOG_BASE, _DIGITS)

    def test_balanced_exhaustively_at_a_small_base(self) -> None:
        """Every value of the tiny case, so the interval claim is checked, not
        trusted: base 4, two digits, `[-10, 5]` and exactly that."""
        for value in range(-10, 6):
            digits = gadget.decompose(value, 2, 2)
            self.assertTrue(all(-2 <= d < 2 for d in digits))
            self.assertEqual(gadget.recompose(digits, 2), value)
        for value in (-11, 6, 7):
            with self.assertRaises(ValueError):
                gadget.decompose(value, 2, 2)

    def test_unsigned_round_trips_over_its_interval(self) -> None:
        rnd = random.Random(2)
        top = (1 << (_LOG_BASE * _DIGITS)) - 1
        for value in [0, top, *(rnd.randint(0, top) for _ in range(500))]:
            digits = gadget.decompose_unsigned(value, _LOG_BASE, _DIGITS)
            self.assertLen(digits, _DIGITS)
            self.assertTrue(all(0 <= d < _BASE for d in digits))
            self.assertEqual(gadget.recompose(digits, _LOG_BASE), value)
        with self.assertRaises(ValueError):
            gadget.decompose_unsigned(-1, _LOG_BASE, _DIGITS)
        with self.assertRaises(ValueError):
            gadget.decompose_unsigned(top + 1, _LOG_BASE, _DIGITS)

    def test_zero_is_all_zero_digits_in_both_conventions(self) -> None:
        self.assertEqual(gadget.decompose(0, _LOG_BASE, _DIGITS), [0] * _DIGITS)
        self.assertEqual(
            gadget.decompose_unsigned(0, _LOG_BASE, _DIGITS), [0] * _DIGITS
        )

    def test_the_vector_form_is_digit_major(self) -> None:
        """Row `i` is every value's `i`-th digit — one future ring element per
        row, which is the orientation a gadget consumer feeds key switching."""
        rnd = random.Random(3)
        values = [rnd.randint(_LO, _HI) for _ in range(17)]
        rows = gadget.decompose_vector(values, _LOG_BASE, _DIGITS)
        self.assertLen(rows, _DIGITS)
        self.assertTrue(all(len(row) == len(values) for row in rows))
        for j, value in enumerate(values):
            self.assertEqual(
                [row[j] for row in rows],
                gadget.decompose(value, _LOG_BASE, _DIGITS),
            )
        self.assertEqual(gadget.recompose_vector(rows, _LOG_BASE), values)

    def test_the_vector_form_survives_empty_input(self) -> None:
        rows = gadget.decompose_vector([], _LOG_BASE, _DIGITS)
        self.assertEqual(rows, [[] for _ in range(_DIGITS)])
        self.assertEqual(gadget.recompose_vector(rows, _LOG_BASE), [])

    @parameterized.parameters(gadget.decompose, gadget.decompose_unsigned)
    def test_degenerate_parameters_are_rejected(self, func) -> None:
        with self.assertRaises(ValueError):
            func(1, 0, 4)  # log_base < 1
        with self.assertRaises(ValueError):
            func(1, 4, -1)  # negative digit count


if __name__ == "__main__":
    absltest.main()
