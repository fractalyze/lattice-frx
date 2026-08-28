"""Tests for `lattice_frx.split_ring`'s traced partial-split ring.

The correctness gate is `SplitRing.mul` against `HostSplitRing.mul`, and the
two reach the product by genuinely different routes: the host multiplies by
negacyclic **schoolbook convolution over exact Python ints**, while the ring
under test moves to the two-factor CRT domain and contracts a **gathered
twisted-circulant** per half, in field arithmetic. Neither is a spelling of
the other, which is what the repo's oracle rule requires — an oracle written
like the code's own other branch would agree with a wrong implementation.

`to_split` is the one place the two *do* share a formula (`low ± r·high`),
so it is checked for agreement and then, more usefully, for being inverted
by `from_split` and for carrying `mul` to the schoolbook answer.

`jit`/`vmap` are asserted here rather than left to the consumer: the whole
reason this ring exists beside the host one is that it composes into a
traced zone, and a ring that silently falls out of one is the failure this
file is meant to catch.
"""

import frx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from lattice_frx import primes
from lattice_frx.domains import Coeff, Eval, Split
from lattice_frx.split_ring import HostSplitRing, SplitRing

_D = 16
_Q_MODULI = tuple(primes.find_nearest_split_primes(30.0, 2))


def _rings() -> tuple[SplitRing, HostSplitRing]:
    return SplitRing(_Q_MODULI, _D), HostSplitRing(_Q_MODULI, _D)


def _random_canonical(rng, *lead: int) -> np.ndarray:
    """A canonical host element, or a `lead`-shaped stack of them.

    Drawn through the oracle's own `uniform_stack` rather than a local
    reimplementation: the `(*lead, limbs, d)` axis order is the module
    convention's, and `_HostRingBase` is what owns it.
    """
    return HostSplitRing(_Q_MODULI, _D).uniform_stack(rng, *lead)



def _host_split_stack(host, arrays: np.ndarray) -> np.ndarray:
    """`HostSplitRing.to_split` applied over a stack.

    The host view is rank-one — its `_coerce` refuses a batched array, where
    the traced `to_split` carries every leading axis — so a batched
    expectation has to be walked. Worth doing rather than weakening the
    assertion: it keeps the oracle the host's own `low ± r·high`, instead of
    a second spelling of the thing under test.
    """
    flat = arrays.reshape(-1, *arrays.shape[-2:])
    out = np.stack([host.to_split(element) for element in flat])
    return out.reshape(*arrays.shape[:-2], *out.shape[1:])


def _host_scalars(ring, per_limb) -> np.ndarray:
    """A per-limb tuple of `Z_q` arrays in the host's `lead + (limbs,)` layout.

    `constant_coeff` returns one array per limb because a constant
    coefficient is a `Z_q` value rather than a ring element — so the
    comparison against `_HostRingBase.constant_coeff` has to name the axis
    order the host puts the limbs in, which is last.
    """
    del ring
    rows = [np.asarray(limb).astype(object) for limb in per_limb]
    return np.stack(
        [np.array([int(v) for v in row.reshape(-1)], dtype=np.uint64).reshape(row.shape)
         for row in rows],
        axis=-1,
    )


def _traced_product(ring, a: np.ndarray, b: np.ndarray, mul=None) -> np.ndarray:
    """`a·b` the way a consumer would spell it: host arrays in, the `Split`
    domain across the product, host bytes out.

    `mul` overrides the multiplication so the tracing tests push a `jit` or
    `vmap` wrapper through this exact path rather than a second spelling of it.
    """
    multiply = ring.mul if mul is None else mul
    x, y = ring.coeff_from_host(a), ring.coeff_from_host(b)
    return ring.to_host(ring.from_split(multiply(ring.to_split(x), ring.to_split(y))))


class TracedSplitRingConstructionTest(parameterized.TestCase):

    def test_rejects_an_ntt_friendly_modulus_with_the_mode_message(self):
        [q] = primes.find_nearest_ntt_primes(_D, 30.0, 1)
        with self.assertRaisesRegex(ValueError, "NTT"):
            SplitRing((q,), _D)

    def test_rejects_a_non_prime_modulus(self):
        with self.assertRaisesRegex(ValueError, "prime"):
            SplitRing((21,), _D)  # 21 ≡ 5 (mod 8), not prime

    @parameterized.parameters(2, 3, 12)
    def test_rejects_a_bad_degree(self, d):
        with self.assertRaisesRegex(ValueError, "degree"):
            SplitRing(_Q_MODULI, d)

    def test_the_two_rings_agree_on_the_split_roots(self):
        ring, host = _rings()
        self.assertEqual(ring.split_roots, host.split_roots)


class TracedSplitRingDomainTest(parameterized.TestCase):
    """The domain is a type: what each op refuses is the contract."""

    def setUp(self):
        super().setUp()
        self.ring, _ = _rings()
        self.coeff = self.ring.coeff_from_host(_random_canonical(np.random.default_rng(0)))
        self.split = self.ring.to_split(self.coeff)

    def test_mul_refuses_the_coefficient_domain(self):
        with self.assertRaisesRegex(TypeError, r"^mul: expected Split, got Coeff"):
            self.ring.mul(self.coeff, self.coeff)

    def test_to_split_refuses_an_already_split_element(self):
        with self.assertRaisesRegex(TypeError, r"^to_split: expected Coeff, got Split"):
            self.ring.to_split(self.split)

    def test_from_split_refuses_the_coefficient_domain(self):
        with self.assertRaisesRegex(TypeError, r"^from_split: expected Split, got Coeff"):
            self.ring.from_split(self.coeff)

    def test_a_generic_op_refuses_mixed_domains(self):
        with self.assertRaisesRegex(TypeError, "must share a domain"):
            self.ring.add(self.coeff, self.split)

    def test_a_generic_op_refuses_the_other_rings_domain(self):
        """`Eval` belongs to the NTT ring, whose moduli this one rejects. It
        is structurally identical to `Coeff`, so only the allowed-domain set
        keeps it from sailing through and producing well-shaped nonsense."""
        stray = Eval(self.coeff.limbs)
        with self.assertRaisesRegex(TypeError, "must share a domain"):
            self.ring.add(stray, stray)

    def test_the_domains_are_distinct_types(self):
        self.assertIsInstance(self.coeff, Coeff)
        self.assertIsInstance(self.split, Split)
        self.assertNotIsInstance(self.split, Coeff)


class TracedSplitRingHostBoundaryTest(parameterized.TestCase):

    @parameterized.named_parameters(("single", ()), ("vector", (3,)), ("matrix", (2, 3)))
    def test_the_coefficient_host_round_trip_preserves_shape_and_bytes(self, lead):
        ring, _ = _rings()
        arr = _random_canonical(np.random.default_rng(1), *lead)
        got = ring.to_host(ring.coeff_from_host(arr))
        self.assertEqual(got.shape, lead + (len(_Q_MODULI), _D))
        np.testing.assert_array_equal(got, arr)

    @parameterized.named_parameters(("single", ()), ("vector", (3,)))
    def test_the_split_host_round_trip_preserves_shape_and_bytes(self, lead):
        ring, host = _rings()
        arr = _random_canonical(np.random.default_rng(2), *lead)
        wire = (
            host.to_split(arr)
            if not lead
            else np.stack([host.to_split(arr[i]) for i in range(lead[0])])
        )
        got = ring.to_host(ring.split_from_host(wire))
        self.assertEqual(got.shape, lead + (len(_Q_MODULI), 2, _D // 2))
        np.testing.assert_array_equal(got, wire)

    def test_a_host_array_of_the_wrong_tail_is_rejected(self):
        ring, _ = _rings()
        arr = _random_canonical(np.random.default_rng(3))
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.coeff_from_host"):
            ring.coeff_from_host(arr[..., :-1])
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.split_from_host"):
            ring.split_from_host(arr)

    def test_a_host_array_with_the_wrong_limb_count_is_rejected(self):
        ring, _ = _rings()
        arr = _random_canonical(np.random.default_rng(4))
        with self.assertRaisesRegex(ValueError, "limbs"):
            ring.coeff_from_host(arr[:1])

    def test_from_signed_matches_the_host_embedding(self):
        ring, host = _rings()
        rng = np.random.default_rng(5)
        vals = [int(v) for v in rng.integers(-9, 10, _D)]
        np.testing.assert_array_equal(ring.to_host(ring.from_signed(vals)), host.from_signed(vals))

    def test_from_signed_rejects_a_wrong_length(self):
        ring, _ = _rings()
        with self.assertRaisesRegex(ValueError, r"^SplitRing\.from_signed"):
            ring.from_signed([0] * (_D - 1))

    def test_the_host_array_is_not_mutated_by_a_round_trip(self):
        ring, _ = _rings()
        arr = _random_canonical(np.random.default_rng(6))
        before = arr.copy()
        ring.to_host(ring.to_split(ring.coeff_from_host(arr)))
        np.testing.assert_array_equal(arr, before)


class TracedSplitRingOpsTest(parameterized.TestCase):

    @parameterized.parameters(0, 1, 2)
    def test_to_split_agrees_with_the_host_view(self, seed):
        ring, host = _rings()
        arr = _random_canonical(np.random.default_rng(seed))
        np.testing.assert_array_equal(
            ring.to_host(ring.to_split(ring.coeff_from_host(arr))), host.to_split(arr)
        )

    @parameterized.named_parameters(("single", ()), ("vector", (4,)))
    def test_from_split_inverts_to_split(self, lead):
        ring, _ = _rings()
        arr = _random_canonical(np.random.default_rng(7), *lead)
        element = ring.coeff_from_host(arr)
        np.testing.assert_array_equal(
            ring.to_host(ring.from_split(ring.to_split(element))), arr
        )

    @parameterized.parameters(0, 1, 2, 3)
    def test_mul_matches_the_schoolbook_oracle(self, seed):
        """The gate: a CRT-domain gather-and-contract against a
        coefficient-domain convolution over exact Python ints."""
        ring, host = _rings()
        rng = np.random.default_rng(100 + seed)
        a, b = _random_canonical(rng), _random_canonical(rng)
        np.testing.assert_array_equal(_traced_product(ring, a, b), host.mul(a, b))

    def test_mul_matches_the_oracle_over_a_batch(self):
        ring, host = _rings()
        rng = np.random.default_rng(11)
        k = 3
        a, b = _random_canonical(rng, k), _random_canonical(rng, k)
        want = np.stack([host.mul(a[i], b[i]) for i in range(k)])
        np.testing.assert_array_equal(_traced_product(ring, a, b), want)

    def test_mul_wraps_negacyclically(self):
        """`X^{d-1} · X = X^d = -1` — the quotient itself, through the CRT
        halves rather than through the fold `mul` spells on coefficients."""
        ring, _ = _rings()
        x_last = ring.to_split(ring.from_signed([0] * (_D - 1) + [1]))
        x_one = ring.to_split(ring.from_signed([0, 1] + [0] * (_D - 2)))
        minus_one = ring.from_signed([-1] + [0] * (_D - 1))
        np.testing.assert_array_equal(
            ring.to_host(ring.from_split(ring.mul(x_last, x_one))), ring.to_host(minus_one)
        )

    def test_mul_obeys_the_ring_axioms(self):
        ring, _ = _rings()
        rng = np.random.default_rng(12)
        a, b, c = (ring.to_split(ring.coeff_from_host(_random_canonical(rng))) for _ in range(3))
        one = ring.to_split(ring.from_signed([1] + [0] * (_D - 1)))
        np.testing.assert_array_equal(
            ring.to_host(ring.mul(a, b)), ring.to_host(ring.mul(b, a))
        )
        np.testing.assert_array_equal(
            ring.to_host(ring.mul(a, ring.add(b, c))),
            ring.to_host(ring.add(ring.mul(a, b), ring.mul(a, c))),
        )
        np.testing.assert_array_equal(ring.to_host(ring.mul(a, one)), ring.to_host(a))

    @parameterized.named_parameters(("add", "add"), ("sub", "sub"))
    def test_the_binary_elementwise_ops_match_the_host(self, op):
        ring, host = _rings()
        rng = np.random.default_rng(13)
        a, b = _random_canonical(rng), _random_canonical(rng)
        got = getattr(ring, op)(ring.coeff_from_host(a), ring.coeff_from_host(b))
        np.testing.assert_array_equal(ring.to_host(got), getattr(host, op)(a, b))

    def test_neg_matches_the_host(self):
        ring, host = _rings()
        a = _random_canonical(np.random.default_rng(13))
        np.testing.assert_array_equal(
            ring.to_host(ring.neg(ring.coeff_from_host(a))), host.neg(a)
        )

    def test_mul_scalar_matches_the_host_and_commutes_with_to_split(self):
        ring, host = _rings()
        rng = np.random.default_rng(14)
        a = _random_canonical(rng)
        s = int(rng.integers(1, _Q_MODULI[0]))
        element = ring.coeff_from_host(a)
        np.testing.assert_array_equal(
            ring.to_host(ring.mul_scalar(element, s)), host.mul_scalar(a, s)
        )
        # Scaling is linear, so it does not matter which side of `to_split` it
        # happens on — the property that lets `mul_scalar` stay domain-generic.
        np.testing.assert_array_equal(
            ring.to_host(ring.to_split(ring.mul_scalar(element, s))),
            ring.to_host(ring.mul_scalar(ring.to_split(element), s)),
        )

    def test_stack_assembles_the_module_convention(self):
        ring, _ = _rings()
        rng = np.random.default_rng(15)
        arrays = [_random_canonical(rng) for _ in range(3)]
        stacked = ring.stack([ring.coeff_from_host(a) for a in arrays])
        np.testing.assert_array_equal(ring.to_host(stacked), np.stack(arrays))



class TracedSplitRingConstructorTest(parameterized.TestCase):
    """The constructors a consumer needs to build operands without dropping
    to the host ring.

    Every one is pinned byte-for-byte against `HostSplitRing`'s counterpart
    through `to_host`. That is a real oracle rather than a restatement: the
    host builds `(limbs, d)` `uint64` arrays out of exact Python ints, while
    these build one `prime_field` array per limb and never share a line of
    it — the repo's rule that an oracle spelled like the code's own other
    branch proves nothing.
    """

    @parameterized.named_parameters(("element", ()), ("vector", (3,)), ("matrix", (2, 3)))
    def test_coefficient_zeros_match_the_host(self, lead):
        ring, host = _rings()
        np.testing.assert_array_equal(
            ring.to_host(ring.zeros(Coeff, *lead)), host.zeros(*lead)
        )

    @parameterized.named_parameters(("element", ()), ("vector", (3,)))
    def test_split_zeros_are_the_host_zero_in_the_crt_view(self, lead):
        ring, host = _rings()
        np.testing.assert_array_equal(
            ring.to_host(ring.zeros(Split, *lead)),
            _host_split_stack(host, host.zeros(*lead)),
        )

    def test_the_empty_stack_survives_the_host_boundary(self):
        """`zeros(domain, 0)` is `matvec`'s answer when it contracts nothing —
        a real statement, so the shape has to round-trip rather than raise."""
        ring, host = _rings()
        np.testing.assert_array_equal(ring.to_host(ring.zeros(Coeff, 0)), host.zeros(0))
        self.assertEqual(
            ring.to_host(ring.zeros(Split, 0)).shape, (0, len(_Q_MODULI), 2, _D // 2)
        )

    def test_zeros_refuses_the_other_rings_domain(self):
        ring, _ = _rings()
        with self.assertRaisesRegex(TypeError, r"SplitRing\.zeros"):
            ring.zeros(Eval, 2)

    def test_zeros_is_the_additive_identity_in_both_domains(self):
        ring, _ = _rings()
        element = ring.coeff_from_host(_random_canonical(np.random.default_rng(30)))
        np.testing.assert_array_equal(
            ring.to_host(ring.add(element, ring.zeros(Coeff))), ring.to_host(element)
        )
        split = ring.to_split(element)
        np.testing.assert_array_equal(
            ring.to_host(ring.add(split, ring.zeros(Split))), ring.to_host(split)
        )

    def test_multiplying_by_the_split_zero_annihilates(self):
        ring, _ = _rings()
        arr = _random_canonical(np.random.default_rng(31))
        split = ring.to_split(ring.coeff_from_host(arr))
        np.testing.assert_array_equal(
            ring.to_host(ring.mul(split, ring.zeros(Split))), ring.to_host(ring.zeros(Split))
        )

    def test_one_matches_the_host_and_is_the_coefficient_domain(self):
        ring, host = _rings()
        np.testing.assert_array_equal(ring.to_host(ring.one()), host.one())
        self.assertIsInstance(ring.one(), Coeff)

    def test_one_is_the_multiplicative_identity_through_the_split_domain(self):
        """`one` is named for the ring element, so the identity it claims has
        to hold where this ring's product actually lives."""
        ring, _ = _rings()
        arr = _random_canonical(np.random.default_rng(32))
        split = ring.to_split(ring.coeff_from_host(arr))
        np.testing.assert_array_equal(
            ring.to_host(ring.mul(split, ring.to_split(ring.one()))), ring.to_host(split)
        )

    @parameterized.named_parameters(("element", ()), ("vector", (4,)), ("matrix", (2, 3)))
    def test_constant_coeff_matches_the_host(self, lead):
        ring, host = _rings()
        arr = _random_canonical(np.random.default_rng(33), *lead)
        got = ring.constant_coeff(ring.coeff_from_host(arr))
        np.testing.assert_array_equal(_host_scalars(ring, got), host.constant_coeff(arr))

    def test_constant_coeff_returns_one_array_per_limb(self):
        """Not a domain type: a constant coefficient is a `Z_q` value, and
        dressing it as a ring element would be a lie in a plausible shape."""
        ring, _ = _rings()
        arr = _random_canonical(np.random.default_rng(34), 5)
        got = ring.constant_coeff(ring.coeff_from_host(arr))
        self.assertIsInstance(got, tuple)
        self.assertNotIsInstance(got, (Coeff, Split))
        self.assertLen(got, len(_Q_MODULI))
        self.assertEqual(tuple(np.asarray(got[0]).shape), (5,))

    def test_constant_coeff_refuses_the_split_domain(self):
        """A CRT half holds no coefficient, so asking one for its constant
        term is the bug this domain typing exists to catch."""
        ring, _ = _rings()
        arr = _random_canonical(np.random.default_rng(35))
        split = ring.to_split(ring.coeff_from_host(arr))
        with self.assertRaisesRegex(TypeError, r"constant_coeff.*Coeff"):
            ring.constant_coeff(split)

    @parameterized.named_parameters(("one_row", 1), ("many_rows", 5))
    def test_from_signed_stack_matches_the_host(self, k):
        ring, host = _rings()
        rows = np.random.default_rng(36).integers(-40, 41, size=(k, _D))
        np.testing.assert_array_equal(
            ring.to_host(ring.from_signed_stack(rows)), host.from_signed_stack(rows)
        )

    def test_from_signed_stack_takes_a_sequence_of_rows(self):
        """The host's own signature: a `(k, d)` array *or* a sequence of
        length-`d` rows, since a consumer assembling one row at a time is the
        caller it exists for."""
        ring, host = _rings()
        rows = [[i - 3] * _D for i in range(4)]
        np.testing.assert_array_equal(
            ring.to_host(ring.from_signed_stack(rows)), host.from_signed_stack(rows)
        )

    def test_from_signed_stack_rejects_an_empty_stack(self):
        """No rows is not the empty stack `zeros(domain, 0)` is. Without the
        guard the limbs come out shaped `(0,)` rather than `(0, d)` — a
        well-shaped nonsense that every later op would carry silently, which
        is why the empty case is refused here and spelled by `zeros`."""
        ring, _ = _rings()
        with self.assertRaisesRegex(ValueError, r"SplitRing\.from_signed_stack"):
            ring.from_signed_stack([])

    def test_from_signed_stack_rejects_a_rank_one_input(self):
        """One element's coefficients where a stack was wanted. It reads as a
        shape mistake and has to be reported as one — the comprehension would
        otherwise surface it as `not iterable`, which names neither of the two
        failure modes this package keeps apart."""
        ring, _ = _rings()
        with self.assertRaisesRegex(ValueError, r"SplitRing\.from_signed_stack"):
            ring.from_signed_stack(np.zeros(_D, dtype=np.int64))

    def test_from_signed_stack_rejects_a_wrong_row_length(self):
        ring, _ = _rings()
        with self.assertRaisesRegex(ValueError, r"SplitRing\.from_signed_stack"):
            ring.from_signed_stack(np.zeros((2, _D - 1), dtype=np.int64))

    @parameterized.named_parameters(("element", ()), ("vector", (3,)), ("matrix", (2, 3)))
    def test_uniform_stack_is_byte_identical_to_the_host(self, lead):
        """Same seed, same bytes — which pins the order the limbs are drawn
        in. A distribution test would pass with the limbs swapped, and a
        consumer whose transcript replays these draws would not."""
        ring, host = _rings()
        got = ring.uniform_stack(np.random.default_rng(37), *lead)
        want = host.uniform_stack(np.random.default_rng(37), *lead)
        np.testing.assert_array_equal(ring.to_host(got), want)

    def test_uniform_stack_is_the_coefficient_domain(self):
        ring, _ = _rings()
        self.assertIsInstance(ring.uniform_stack(np.random.default_rng(38), 2), Coeff)


class TracedSplitRingTracingTest(parameterized.TestCase):
    """The reason this ring exists beside the host one."""

    def test_mul_holds_under_jit(self):
        ring, host = _rings()
        rng = np.random.default_rng(21)
        a, b = _random_canonical(rng), _random_canonical(rng)
        got = _traced_product(ring, a, b, mul=frx.jit(ring.mul))
        np.testing.assert_array_equal(got, host.mul(a, b))

    def test_mul_holds_under_vmap(self):
        ring, host = _rings()
        rng = np.random.default_rng(22)
        k = 4
        a, b = _random_canonical(rng, k), _random_canonical(rng, k)
        got = _traced_product(ring, a, b, mul=frx.vmap(ring.mul))
        want = np.stack([host.mul(a[i], b[i]) for i in range(k)])
        np.testing.assert_array_equal(got, want)

    def test_the_interior_constructors_hold_under_jit(self):
        """`zeros`/`one`/`constant_coeff` are the three that have to survive
        tracing — the other two take host data (a generator, raw signed ints)
        and are boundary constructors by nature."""
        ring, host = _rings()
        arr = _random_canonical(np.random.default_rng(39), 3)

        def leading_coeffs(x):
            shifted = ring.add(x, ring.zeros(Coeff, 3))
            return ring.constant_coeff(shifted)

        got = frx.jit(leading_coeffs)(ring.coeff_from_host(arr))
        np.testing.assert_array_equal(_host_scalars(ring, got), host.constant_coeff(arr))
        np.testing.assert_array_equal(
            ring.to_host(frx.jit(ring.one)()), host.one()
        )

    def test_the_interior_constructors_hold_under_vmap(self):
        ring, host = _rings()
        k = 4
        arr = _random_canonical(np.random.default_rng(40), k)

        def annihilate(x):
            return ring.mul(ring.to_split(x), ring.zeros(Split))

        got = ring.to_host(frx.vmap(annihilate)(ring.coeff_from_host(arr)))
        np.testing.assert_array_equal(got, _host_split_stack(host, host.zeros(k)))

    def test_a_ring_survives_being_traced_first(self):
        """The ring caches trace-time constants, and the call that fills the
        cache decides what lands in it. Caching a *converted* constant leaks
        the first trace's tracer into every later call, so this drives a ring
        whose very first `mul` is inside a `jit` and then keeps using it —
        eagerly too. A single jitted call cannot catch that; the second one
        is where the leak surfaces.
        """
        ring, host = _rings()
        rng = np.random.default_rng(24)
        a, b = _random_canonical(rng), _random_canonical(rng)
        want = host.mul(a, b)
        fused = frx.jit(ring.mul)
        np.testing.assert_array_equal(_traced_product(ring, a, b, mul=fused), want)
        np.testing.assert_array_equal(_traced_product(ring, a, b, mul=fused), want)
        np.testing.assert_array_equal(_traced_product(ring, a, b), want)

    def test_a_ring_used_eagerly_first_still_traces(self):
        """The other order, for the same reason."""
        ring, host = _rings()
        rng = np.random.default_rng(25)
        a, b = _random_canonical(rng), _random_canonical(rng)
        want = host.mul(a, b)
        np.testing.assert_array_equal(_traced_product(ring, a, b), want)
        np.testing.assert_array_equal(
            _traced_product(ring, a, b, mul=frx.jit(ring.mul)), want
        )

    def test_a_whole_domain_round_trip_composes_into_one_jit_zone(self):
        """`to_split` → `mul` → `from_split` with no host boundary inside —
        which is the shape a consumer's proof layer wants."""
        ring, host = _rings()
        rng = np.random.default_rng(23)
        a, b = _random_canonical(rng), _random_canonical(rng)

        def product(x, y):
            return ring.from_split(ring.mul(ring.to_split(x), ring.to_split(y)))

        got = ring.to_host(frx.jit(product)(ring.coeff_from_host(a), ring.coeff_from_host(b)))
        np.testing.assert_array_equal(got, host.mul(a, b))


if __name__ == "__main__":
    absltest.main()
