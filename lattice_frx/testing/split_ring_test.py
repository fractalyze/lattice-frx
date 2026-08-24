"""Tests for `lattice_frx.split_ring`'s partial-split host ring.

The multiplication cross-check is deliberately two-sided: the ring's own
`mul` is a coefficient-domain negacyclic schoolbook convolution, and this
file re-derives the product independently through the two-factor CRT
(`to_split` → per-half twisted convolution → `from_split`) — two different
algorithms agreeing on random inputs is the correctness gate, mirroring how
`ring_test.py` checks the traced ring against `HostRnsRing`.

The invertibility tests are the property the LNP consumer buys (eprint
2022/284, Lemma 2.6): nonzero σ₋₁-invariant small elements are invertible at
a partial-split modulus. Invertibility is cross-checked against a test-side
extended-Euclid over each half field.
"""
import math

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from lattice_frx import canonical
from lattice_frx import primes
from lattice_frx import roots
from lattice_frx.split_ring import HostSplitRing

_D = 16
_Q_MODULI = tuple(primes.find_nearest_split_primes(30.0, 2))


def _ring() -> HostSplitRing:
    return HostSplitRing(_Q_MODULI, _D)


def _random_canonical(rng, ring) -> np.ndarray:
    rows = [rng.integers(0, q, ring.d, dtype=np.uint64) for q in ring.q_moduli]
    return np.stack(rows)


def _twisted_mul(u: list[int], v: list[int], s: int, q: int) -> list[int]:
    """Independent product in `F_q[X]/(X^{d/2} - s)`: full convolution, then
    fold the top half back with the twist `X^{d/2} ≡ s`."""
    n = len(u)
    conv = [0] * (2 * n)
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            conv[i + j] += ui * vj
    return [(conv[k] + s * conv[k + n]) % q for k in range(n)]


def _half_inverse(u: list[int], s: int, q: int):
    """Test-side extended Euclid over `F_q[X]` modulo `X^{d/2} - s`; returns
    an inverse of `u` or None when `gcd` is not a unit."""
    n = len(u)
    modulus = [(-s) % q] + [0] * (n - 1) + [1]  # X^n - s, ascending coeffs

    def deg(p):
        for i in range(len(p) - 1, -1, -1):
            if p[i] % q:
                return i
        return -1

    # Euclid with Bezout tracking on the first argument.
    r0, r1 = modulus, [x % q for x in u]
    t0, t1 = [0], [1]
    while deg(r1) > 0:
        d0, d1 = deg(r0), deg(r1)
        if d0 < d1:
            r0, r1, t0, t1 = r1, r0, t1, t0
            continue
        f = r0[d0] * pow(r1[d1], -1, q) % q
        shift = d0 - d1
        r0 = [(c - f * (r1[i - shift] if 0 <= i - shift <= d1 else 0)) % q for i, c in enumerate(r0)]
        t0 = [((t0[i] if i < len(t0) else 0) - f * (t1[i - shift] if 0 <= i - shift < len(t1) else 0)) % q
              for i in range(max(len(t0), len(t1) + shift))]
    if deg(r1) < 0:
        return None
    c_inv = pow(r1[0], -1, q)
    out = [(x * c_inv) % q for x in t1]
    return (out + [0] * n)[:n]


class SplitRootTest(absltest.TestCase):

    def test_split_root_squares_to_minus_one(self):
        for q in _Q_MODULI:
            r = roots.split_root(q)
            self.assertEqual(r * r % q, q - 1)
            # Deterministic canonical pick: the smaller of the two roots.
            self.assertEqual(r, min(r, q - r))

    def test_split_root_rejects_an_ntt_friendly_modulus(self):
        # An NTT-friendly limb is ≡ 1 (mod 2d) hence ≡ 1 (mod 8) — the other
        # ring mode. The error must name the mode confusion, since silently
        # mixing the two is a soundness bug in the consumer, not a crash.
        [q] = primes.find_nearest_ntt_primes(_D, 30.0, 1)
        with self.assertRaisesRegex(ValueError, "NTT"):
            roots.split_root(q)

    def test_split_root_rejects_a_non_prime(self):
        for bad in (0, 1, 21, 4):  # 21 ≡ 5 (mod 8), still rejected
            with self.assertRaisesRegex(ValueError, "prime"):
                roots.split_root(bad)


class SplitRingConstructionTest(parameterized.TestCase):

    def test_rejects_an_ntt_friendly_modulus_with_the_mode_message(self):
        [q] = primes.find_nearest_ntt_primes(_D, 30.0, 1)
        with self.assertRaisesRegex(ValueError, "NTT"):
            HostSplitRing((q,), _D)

    def test_rejects_a_non_prime_modulus(self):
        with self.assertRaisesRegex(ValueError, "prime"):
            HostSplitRing((21,), _D)  # 21 ≡ 5 (mod 8), not prime

    @parameterized.parameters(2, 3, 12)
    def test_rejects_a_bad_degree(self, d):
        with self.assertRaisesRegex(ValueError, "degree"):
            HostSplitRing(_Q_MODULI, d)


class SplitRingOpsTest(parameterized.TestCase):

    def test_contract_failure_modes_carry_the_split_prefix(self):
        ring = _ring()
        good = _random_canonical(np.random.default_rng(0), ring)
        with self.assertRaisesRegex(TypeError, r"^SplitRing\.mul"):
            ring.mul(good.astype(np.int64), good)
        bad = good.copy()
        bad[0, 0] = np.uint64(ring.q_moduli[0])
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.mul"):
            ring.mul(bad, good)

    def test_add_neg_roundtrip_and_no_mutation(self):
        ring = _ring()
        rng = np.random.default_rng(1)
        a = _random_canonical(rng, ring)
        b = _random_canonical(rng, ring)
        a0, b0 = a.copy(), b.copy()
        np.testing.assert_array_equal(ring.sub(ring.add(a, b), b), a)
        np.testing.assert_array_equal(ring.add(a, ring.neg(a)), np.zeros_like(a))
        np.testing.assert_array_equal(a, a0)
        np.testing.assert_array_equal(b, b0)

    def test_mul_matches_the_independent_twisted_crt_path(self):
        ring = _ring()
        rng = np.random.default_rng(2)
        for _ in range(8):
            a = _random_canonical(rng, ring)
            b = _random_canonical(rng, ring)
            got = ring.mul(a, b)
            sa, sb = ring.to_split(a), ring.to_split(b)
            prod = np.zeros_like(sa)
            for l, q in enumerate(ring.q_moduli):
                for h, sgn in enumerate((1, -1)):
                    s = sgn * ring.split_roots[l] % q
                    prod[l, h] = _twisted_mul(
                        [int(x) for x in sa[l, h]], [int(x) for x in sb[l, h]], s, q)
            np.testing.assert_array_equal(ring.from_split(prod), got)

    def test_mul_ring_axioms(self):
        ring = _ring()
        rng = np.random.default_rng(3)
        a = _random_canonical(rng, ring)
        b = _random_canonical(rng, ring)
        c = _random_canonical(rng, ring)
        np.testing.assert_array_equal(ring.mul(a, b), ring.mul(b, a))
        np.testing.assert_array_equal(
            ring.mul(a, ring.add(b, c)), ring.add(ring.mul(a, b), ring.mul(a, c)))
        one = ring.from_signed([1] + [0] * (ring.d - 1))
        np.testing.assert_array_equal(ring.mul(a, one), a)

    def test_mul_wraps_negacyclically(self):
        # X^{d-1} * X = X^d = -1.
        ring = _ring()
        x_last = ring.from_signed([0] * (ring.d - 1) + [1])
        x_one = ring.from_signed([0, 1] + [0] * (ring.d - 2))
        minus_one = ring.from_signed([-1] + [0] * (ring.d - 1))
        np.testing.assert_array_equal(ring.mul(x_last, x_one), minus_one)

    def test_split_roundtrip(self):
        ring = _ring()
        rng = np.random.default_rng(4)
        a = _random_canonical(rng, ring)
        np.testing.assert_array_equal(ring.from_split(ring.to_split(a)), a)

    def test_galois_sigma_minus_one_is_an_involution(self):
        ring = _ring()
        rng = np.random.default_rng(5)
        a = _random_canonical(rng, ring)
        k = 2 * ring.d - 1  # σ₋₁ : X ↦ X^{-1}
        np.testing.assert_array_equal(ring.galois(ring.galois(a, k), k), a)


class SplitRingInvertibilityTest(absltest.TestCase):

    def test_nonzero_sigma_invariant_small_elements_are_invertible(self):
        # Lemma 2.6 (eprint 2022/284): at q ≡ 5 (mod 8), σ₋₁-invariant c ≠ 0
        # with small coefficients is invertible — the LNP challenge-space
        # property this ring exists to give its consumer.
        ring = _ring()
        rng = np.random.default_rng(6)
        k = 2 * ring.d - 1
        for _ in range(50):
            free = rng.integers(-2, 3, ring.d // 2)
            if not free.any():
                continue
            coeffs = [0] * ring.d
            coeffs[0] = int(free[0])
            for i in range(1, ring.d // 2):
                coeffs[i] = int(free[i])
                coeffs[ring.d - i] = -int(free[i])  # σ₋₁-invariant embedding
            c = ring.from_signed(coeffs)
            np.testing.assert_array_equal(ring.galois(c, k), c)  # sanity: fixed by σ₋₁
            self.assertTrue(ring.is_invertible(c))

    def test_zero_and_half_zero_elements_are_not_invertible(self):
        ring = _ring()
        zero = ring.from_signed([0] * ring.d)
        self.assertFalse(ring.is_invertible(zero))
        # An element that is a multiple of one CRT half: zero in half 0.
        rng = np.random.default_rng(7)
        sp = ring.to_split(_random_canonical(rng, ring))
        sp[:, 0, :] = 0
        self.assertFalse(ring.is_invertible(ring.from_split(sp)))

    def test_is_invertible_agrees_with_extended_euclid(self):
        ring = _ring()
        rng = np.random.default_rng(8)
        for _ in range(10):
            a = _random_canonical(rng, ring)
            sp = ring.to_split(a)
            expected = True
            for l, q in enumerate(ring.q_moduli):
                for h, sgn in enumerate((1, -1)):
                    s = sgn * ring.split_roots[l] % q
                    inv = _half_inverse([int(x) for x in sp[l, h]], s, q)
                    if inv is None:
                        expected = False
                    else:
                        one = _twisted_mul([int(x) for x in sp[l, h]], inv, s, q)
                        self.assertEqual(one, [1] + [0] * (ring.d // 2 - 1))
            self.assertEqual(ring.is_invertible(a), expected)


def _random_stack(rng, ring, *lead: int) -> np.ndarray:
    """A `lead + (limbs, d)` stack of independent canonical elements."""
    flat = [_random_canonical(rng, ring) for _ in range(math.prod(lead))]
    return np.stack(flat).reshape(*lead, len(ring.q_moduli), ring.d)


class SplitRingModuleTest(absltest.TestCase):
    """The module layer as a shape convention: `matvec` over stacks, and
    the leading-batch-axis contract of the elementwise ops."""

    def test_matvec_matches_the_per_entry_composition(self):
        ring = _ring()
        rng = np.random.default_rng(10)
        matrix = _random_stack(rng, ring, 2, 3)
        vector = _random_stack(rng, ring, 3)
        got = ring.matvec(matrix, vector)
        for r in range(2):
            want = ring.mul(matrix[r, 0], vector[0])
            for c in range(1, 3):
                want = ring.add(want, ring.mul(matrix[r, c], vector[c]))
            np.testing.assert_array_equal(got[r], want)

    def test_matvec_is_linear(self):
        ring = _ring()
        rng = np.random.default_rng(11)
        matrix = _random_stack(rng, ring, 2, 3)
        x = _random_stack(rng, ring, 3)
        y = _random_stack(rng, ring, 3)
        # The right-hand sides also exercise the batched elementwise add.
        np.testing.assert_array_equal(
            ring.matvec(matrix, ring.add(x, y)),
            ring.add(ring.matvec(matrix, x), ring.matvec(matrix, y)),
        )

    def test_batched_elementwise_ops_apply_per_element(self):
        ring = _ring()
        rng = np.random.default_rng(12)
        a = _random_stack(rng, ring, 4)
        b = _random_stack(rng, ring, 4)
        k = 2 * ring.d - 1  # σ₋₁ : X ↦ X^{-1}
        for got, want_fn in [
            (ring.add(a, b), lambda i: ring.add(a[i], b[i])),
            (ring.sub(a, b), lambda i: ring.sub(a[i], b[i])),
            (ring.neg(a), lambda i: ring.neg(a[i])),
            (ring.mul_scalar(a, 7), lambda i: ring.mul_scalar(a[i], 7)),
            (ring.galois(a, k), lambda i: ring.galois(a[i], k)),
        ]:
            for i in range(4):
                np.testing.assert_array_equal(got[i], want_fn(i))

    def test_galois_batches_over_nested_axes(self):
        """`σ_k` over a matrix-shaped stack: the batch axes are carried,
        not read as limbs. LNP's Fig. 6 lifts whole module vectors through
        `σ`, so the op has to survive more than one leading axis."""
        ring = _ring()
        rng = np.random.default_rng(17)
        stack = _random_stack(rng, ring, 2, 3)
        k = 2 * ring.d - 1
        got = ring.galois(stack, k)
        self.assertEqual(got.shape, stack.shape)
        for i in range(2):
            for j in range(3):
                np.testing.assert_array_equal(got[i, j], ring.galois(stack[i, j], k))

    def test_matvec_shape_mismatches_raise(self):
        ring = _ring()
        rng = np.random.default_rng(13)
        matrix = _random_stack(rng, ring, 2, 3)
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.matvec"):
            ring.matvec(matrix, _random_stack(rng, ring, 2))  # cols mismatch
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.matvec"):
            ring.matvec(matrix[0], _random_stack(rng, ring, 3))  # 3-D matrix
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.matvec"):
            ring.matvec(matrix[:, :0], np.empty((0,) + matrix.shape[2:], np.uint64))

    def test_a_per_element_op_rejects_a_stack(self):
        """`mul`/`to_split` read axis 0 as limbs — a stack there must raise,
        not silently misread the batch axis. `galois` is deliberately absent:
        it is elementwise, so it batches (see the tests above)."""
        ring = _ring()
        rng = np.random.default_rng(14)
        stack = _random_stack(rng, ring, 2)
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.mul"):
            ring.mul(stack, stack)
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.to_split"):
            ring.to_split(stack)

    def test_scale_matches_per_row_mul(self):
        ring = _ring()
        rng = np.random.default_rng(16)
        element = _random_canonical(rng, ring)
        stack = _random_stack(rng, ring, 3)
        got = ring.scale(element, stack)
        for i in range(3):
            np.testing.assert_array_equal(got[i], ring.mul(element, stack[i]))
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.scale"):
            ring.scale(element, stack[0])  # a bare element is not a stack

    def test_from_signed_stack_matches_per_row_from_signed(self):
        ring = _ring()
        rng = np.random.default_rng(17)
        rows = rng.integers(-3, 4, size=(3, ring.d))
        got = ring.from_signed_stack(rows)
        for i in range(3):
            np.testing.assert_array_equal(got[i], ring.from_signed(rows[i]))

    def test_a_batched_op_rejects_wrong_trailing_axes(self):
        ring = _ring()
        rng = np.random.default_rng(15)
        stack = _random_stack(rng, ring, 2)
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.add"):
            ring.add(np.swapaxes(stack, -1, -2), np.swapaxes(stack, -1, -2))


class SplitRingModuleConstantsTest(absltest.TestCase):
    """The rest of the module surface: the ring's own constants, its
    `Z_q`-weighted sum, the constant coefficient, and a uniform stack —
    the ops a proof layer would otherwise spell against the array layout.
    """

    def test_combine_matches_the_per_term_fold(self):
        """Whatever the stack carries past its leading axis rides along, so
        one call aggregates elements and whole matrix rows alike."""
        ring = _ring()
        rng = np.random.default_rng(20)
        weights = rng.integers(0, ring.q_moduli[0], size=4)
        for lead in ((), (3,), (2, 3)):
            values = _random_stack(rng, ring, 4, *lead)
            want = ring.mul_scalar(values[0], int(weights[0]))
            for value, weight in zip(values[1:], weights[1:]):
                want = ring.add(want, ring.mul_scalar(value, int(weight)))
            np.testing.assert_array_equal(ring.combine(weights, values), want)

    def test_combine_over_no_terms_is_the_zero_it_sums_to(self):
        ring = _ring()
        got = ring.combine([], ring.zeros(0, 3))
        np.testing.assert_array_equal(got, ring.zeros(3))

    def test_combine_rejects_a_weight_count_mismatch_and_a_bare_element(self):
        ring = _ring()
        rng = np.random.default_rng(21)
        values = _random_stack(rng, ring, 4)
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.combine"):
            ring.combine([1, 2, 3], values)
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.combine"):
            ring.combine([1] * len(ring.q_moduli), values[0])

    def test_combine_batches_its_weights_like_the_rows_it_would_be_called_with(
        self,
    ):
        """A rank-2 `weights` is a batch of the rank-1 sums, in order.

        Pinned against `combine` itself rather than against a re-derived
        fold: the arithmetic is already gated by the per-term-fold test
        above, and what a batch axis can get wrong is the *layout* — a
        transposed or misordered batch still sums correctly, so only the
        per-row call it claims to batch can catch it."""
        ring = _ring()
        rng = np.random.default_rng(23)
        gamma = rng.integers(0, ring.q_moduli[0], size=(3, 4))
        for lead in ((), (3,), (2, 3)):
            values = _random_stack(rng, ring, 4, *lead)
            want = np.stack([ring.combine(row, values) for row in gamma])
            np.testing.assert_array_equal(ring.combine(gamma, values), want)

    def test_combine_over_no_terms_batches_the_zero_it_sums_to(self):
        ring = _ring()
        got = ring.combine(np.zeros((2, 0), dtype=np.uint64), ring.zeros(0, 3))
        np.testing.assert_array_equal(got, ring.zeros(2, 3))

    def test_combine_rejects_weights_that_are_neither_a_sum_nor_a_batch(self):
        ring = _ring()
        rng = np.random.default_rng(24)
        values = _random_stack(rng, ring, 4)
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.combine"):
            ring.combine(np.zeros((2, 2, 4), dtype=np.uint64), values)

    def test_combine_leaves_positions_with_no_support_zero(self):
        """Positions the stack never occupies contract to zero, and the
        occupied ones are unchanged.

        `combine` skips the unoccupied positions instead of carrying them
        through the exact-integer fold. The saving is only sound if the
        skip is exactly the all-zero set."""
        ring = _ring()
        rng = np.random.default_rng(25)
        values = ring.zeros(4, 5)
        occupied = (1, 3)
        for pos in occupied:
            values[:, pos] = _random_stack(rng, ring, 4)
        weights = rng.integers(0, ring.q_moduli[0], size=4)

        got = ring.combine(weights, values)

        want = ring.mul_scalar(values[0], int(weights[0]))
        for value, weight in zip(values[1:], weights[1:]):
            want = ring.add(want, ring.mul_scalar(value, int(weight)))
        np.testing.assert_array_equal(got, want)
        for pos in range(values.shape[1]):
            if pos not in occupied:
                np.testing.assert_array_equal(got[pos], ring.zeros())

    def test_combine_finds_a_position_only_a_later_term_occupies(self):
        """Occupancy is a property of the whole contracted axis.

        A scan that settled for the first term would report this position
        empty and drop a real contribution — silently, since the result
        stays canonical and every round-trip built on it still agrees. It
        is the one way the saving above can turn from exact into wrong, so
        it gets its own case: term 0 is zero exactly where terms 1-3 are
        not."""
        ring = _ring()
        rng = np.random.default_rng(27)
        values = ring.zeros(4, 5)
        values[1:, 2] = _random_stack(rng, ring, 3)
        weights = rng.integers(1, ring.q_moduli[0], size=4)

        got = ring.combine(weights, values)

        want = ring.mul_scalar(values[0], int(weights[0]))
        for value, weight in zip(values[1:], weights[1:]):
            want = ring.add(want, ring.mul_scalar(value, int(weight)))
        np.testing.assert_array_equal(got, want)
        self.assertTrue(got[2].any(), "the later terms' contribution was dropped")

    def test_combine_over_a_wholly_unoccupied_stack_is_zero(self):
        """The empty-support edge of the skip above: nothing to contract."""
        ring = _ring()
        values = ring.zeros(4, 5)
        got = ring.combine([1, 2, 3, 4], values)
        np.testing.assert_array_equal(got, ring.zeros(5))

    def test_combine_checks_the_array_contract_over_the_whole_stack(self):
        """A stack that breaks the array contract is rejected, narrowing
        or no narrowing.

        The skip cannot itself hide a bad residue — one is `>= q >= 2`,
        so it is nonzero, so it is always occupied. What this pins is that
        `combine` still routes the stack through the contract gate at all:
        deleting that call is otherwise invisible here, since the fold
        neither reads nor produces a residue out of range."""
        ring = _ring()
        rng = np.random.default_rng(26)
        values = _random_stack(rng, ring, 4, 5)
        values[2, 4, 0] = ring.q_moduli[0]
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.combine"):
            ring.combine([1, 2, 3, 4], values)

    def test_zeros_is_the_additive_and_one_the_multiplicative_identity(self):
        ring = _ring()
        rng = np.random.default_rng(22)
        a = _random_canonical(rng, ring)
        np.testing.assert_array_equal(ring.add(a, ring.zeros()), a)
        np.testing.assert_array_equal(ring.mul(a, ring.one()), a)
        self.assertEqual(ring.zeros(2, 3).shape, (2, 3, len(ring.q_moduli), ring.d))

    def test_constant_coeff_is_the_coefficient_a_signed_embedding_put_there(self):
        ring = _ring()
        rows = [[7] + [0] * (ring.d - 1), [-3] + list(range(1, ring.d))]
        got = ring.constant_coeff(ring.from_signed_stack(rows))
        np.testing.assert_array_equal(
            got,
            np.array(
                [[7 % q for q in ring.q_moduli], [-3 % q for q in ring.q_moduli]],
                dtype=np.uint64,
            ),
        )

    def test_constant_coeff_does_not_alias_its_input(self):
        ring = _ring()
        stack = ring.from_signed_stack([[5] + [0] * (ring.d - 1)])
        got = ring.constant_coeff(stack)
        got[...] = 0
        self.assertEqual(int(stack[0, 0, 0]), 5)

    def test_uniform_stack_is_canonical_reproducible_and_varied(self):
        ring = _ring()
        limbs = len(ring.q_moduli)
        stack = ring.uniform_stack(np.random.default_rng(24), 3, 2)
        self.assertEqual(stack.shape, (3, 2, limbs, ring.d))
        self.assertTrue(canonical.is_canonical(stack, ring.q_moduli))
        np.testing.assert_array_equal(
            stack, ring.uniform_stack(np.random.default_rng(24), 3, 2)
        )
        self.assertFalse(
            np.array_equal(stack, ring.uniform_stack(np.random.default_rng(25), 3, 2))
        )
        bare = ring.uniform_stack(np.random.default_rng(26))
        self.assertEqual(bare.shape, (limbs, ring.d))

    def test_matvec_and_scale_return_the_empty_stack_rather_than_raising(self):
        """No rows is a real statement for a proof layer — an opening with
        no linear relations attached — and the empty answer is the shape
        convention's own, not something each consumer should special-case.
        """
        ring = _ring()
        rng = np.random.default_rng(27)
        limbs = len(ring.q_moduli)
        got = ring.matvec(ring.zeros(0, 3), _random_stack(rng, ring, 3))
        self.assertEqual((got.shape, got.dtype), ((0, limbs, ring.d), np.uint64))
        got = ring.scale(_random_canonical(rng, ring), ring.zeros(0))
        self.assertEqual((got.shape, got.dtype), ((0, limbs, ring.d), np.uint64))

    def test_the_empty_case_still_enforces_the_operand_contract(self):
        """How many rows the answer has must not decide whether the
        operands are checked, or the no-rows path is a hole in the
        boundary — with nothing to multiply, `mul` never runs, and `mul` is
        where these would otherwise meet the contract."""
        ring = _ring()
        rng = np.random.default_rng(28)
        signed = _random_stack(rng, ring, 3).astype(np.int64)
        with self.assertRaisesRegex(TypeError, r"^SplitRing\.matvec"):
            ring.matvec(ring.zeros(0, 3), signed)
        with self.assertRaisesRegex(TypeError, r"^SplitRing\.scale"):
            ring.scale(signed[0], ring.zeros(0))


if __name__ == "__main__":
    absltest.main()
