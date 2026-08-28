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
from lattice_frx.split_ring import HostSplitRing, SplitRing, limbs_to_host

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


def host_split_stack(host, arrays: np.ndarray) -> np.ndarray:
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


def _host_scalars(per_limb) -> np.ndarray:
    """A per-limb tuple of `Z_q` arrays in the host's `lead + (limbs,)` layout.

    `constant_coeff` returns one array per limb because a constant coefficient
    is a `Z_q` value rather than a ring element, so the comparison against
    `_HostRingBase.constant_coeff` has to put the limbs where the host does,
    which is last. The conversion itself is the ring's own — restating it here
    would leave the oracle on the old spelling the day it changes.
    """
    return np.stack(limbs_to_host(per_limb), axis=-1)


def _split(ring, arr: np.ndarray):
    """A host array as a `Split` element — the domain the module layer takes."""
    return ring.to_split(ring.coeff_from_host(arr))


def _from_split(ring, element) -> np.ndarray:
    """A `Split` result back as the host contract, for comparison."""
    return ring.to_host(ring.from_split(element))


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
        wire = host_split_stack(host, arr)
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

    @parameterized.named_parameters(
        ("single", ()), ("vector", (3,)), ("matrix", (2, 3)), ("empty", (0,))
    )
    def test_coefficient_zeros_match_the_host(self, lead):
        ring, host = _rings()
        np.testing.assert_array_equal(
            ring.to_host(ring.zeros(Coeff, *lead)), host.zeros(*lead)
        )

    @parameterized.named_parameters(("single", ()), ("vector", (3,)))
    def test_split_zeros_are_the_host_zero_in_the_crt_view(self, lead):
        ring, host = _rings()
        np.testing.assert_array_equal(
            ring.to_host(ring.zeros(Split, *lead)),
            host_split_stack(host, host.zeros(*lead)),
        )

    def test_the_empty_stack_survives_the_host_boundary(self):
        """The `Split` half of `zeros(domain, 0)`, whose shape no host oracle
        can state — `np.stack([])` raises, so it is asserted directly. The
        `Coeff` half rides the parameterized test above."""
        ring, _ = _rings()
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
        np.testing.assert_array_equal(_host_scalars(got), host.constant_coeff(arr))

    def test_constant_coeff_returns_one_array_per_limb(self):
        """One array per limb, and not a domain type — see the method."""
        ring, _ = _rings()
        arr = _random_canonical(np.random.default_rng(34), 5)
        got = ring.constant_coeff(ring.coeff_from_host(arr))
        self.assertIsInstance(got, tuple)
        self.assertNotIsInstance(got, (Coeff, Split))
        self.assertLen(got, len(_Q_MODULI))
        self.assertEqual(tuple(np.asarray(got[0]).shape), (5,))

    def test_constant_coeff_refuses_the_split_domain(self):
        """A CRT half holds no coefficient to read."""
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

    @parameterized.named_parameters(
        # No rows is not the empty stack `zeros(domain, 0)` is: without the
        # guard the limbs come out shaped `(0,)` rather than `(0, d)`, a
        # well-shaped nonsense every later op would carry silently.
        ("empty", []),
        # One element's coefficients where a stack was wanted — a shape
        # mistake, so it has to arrive as a shape error. See the method for
        # why the two failure modes are kept apart.
        ("rank_one", np.zeros(_D, dtype=np.int64)),
        ("wrong_row_length", np.zeros((2, _D - 1), dtype=np.int64)),
    )
    def test_from_signed_stack_rejects(self, rows):
        ring, _ = _rings()
        with self.assertRaisesRegex(ValueError, r"SplitRing\.from_signed_stack"):
            ring.from_signed_stack(rows)

    @parameterized.named_parameters(("single", ()), ("vector", (3,)), ("matrix", (2, 3)))
    def test_uniform_stack_is_byte_identical_to_the_host(self, lead):
        """Same seed, same bytes. This pins the *order* the limbs are drawn
        in, which a distribution test would not — see the method for who
        depends on it."""
        ring, host = _rings()
        got = ring.uniform_stack(np.random.default_rng(37), *lead)
        want = host.uniform_stack(np.random.default_rng(37), *lead)
        np.testing.assert_array_equal(ring.to_host(got), want)
        self.assertIsInstance(got, Coeff)



class TracedSplitRingModuleTest(parameterized.TestCase):
    """The module layer: `matvec`/`matmul`/`scale` over `Split`, `combine`
    over either domain, `galois` over `Coeff`.

    `Split`-only for the three that carry the product, because that is where
    this ring's multiplication is defined — the same rule `RnsRing`'s
    `Eval`-only `matvec` states for the sibling ring. Each is pinned against
    `HostSplitRing`, which reaches the same answer by schoolbook convolution
    over exact Python ints and shares no step with the CRT contraction.
    """

    @parameterized.named_parameters(("single", ()), ("batched", (2,)))
    def test_matvec_matches_the_host(self, lead):
        ring, host = _rings()
        rng = np.random.default_rng(50)
        m, k = 3, 2
        mat = _random_canonical(rng, *lead, m, k)
        vec = _random_canonical(rng, *lead, k)
        got = _from_split(ring, ring.matvec(_split(ring, mat), _split(ring, vec)))
        if lead:
            want = np.stack([host.matvec(mat[i], vec[i]) for i in range(lead[0])])
        else:
            want = host.matvec(mat, vec)
        np.testing.assert_array_equal(got, want)

    def test_matvec_contracts_nothing_into_the_empty_stack(self):
        """No rows is a real statement — an opening with no linear relations
        attached — so it is the empty stack, not an error."""
        ring, _ = _rings()
        rng = np.random.default_rng(51)
        mat = _random_canonical(rng, 0, 2)
        vec = _random_canonical(rng, 2)
        got = ring.matvec(_split(ring, mat), _split(ring, vec))
        self.assertEqual(ring.to_host(got).shape, (0, len(_Q_MODULI), 2, _D // 2))

    def test_matvec_refuses_a_k_extent_that_does_not_contract(self):
        ring, _ = _rings()
        rng = np.random.default_rng(52)
        mat = _split(ring, _random_canonical(rng, 3, 2))
        vec = _split(ring, _random_canonical(rng, 3))
        with self.assertRaisesRegex(ValueError, r"matvec"):
            ring.matvec(mat, vec)

    def test_matvec_refuses_a_vector_of_the_matrix_rank(self):
        """A rank check alone would pass a `(k,)` vector given as `(1, k)`,
        so the guard states the contracted extent, not just the ranks."""
        ring, _ = _rings()
        rng = np.random.default_rng(53)
        mat = _split(ring, _random_canonical(rng, 3, 2))
        with self.assertRaisesRegex(ValueError, r"matvec"):
            ring.matvec(mat, mat)

    @parameterized.named_parameters(("coefficients", "coeff"), ("mixed", "mixed"))
    def test_matvec_refuses_the_wrong_domain(self, kind):
        ring, _ = _rings()
        rng = np.random.default_rng(54)
        mat_c = ring.coeff_from_host(_random_canonical(rng, 3, 2))
        vec_c = ring.coeff_from_host(_random_canonical(rng, 2))
        args = (mat_c, vec_c) if kind == "coeff" else (ring.to_split(mat_c), vec_c)
        with self.assertRaisesRegex(TypeError, r"matvec"):
            ring.matvec(*args)

    @parameterized.named_parameters(("one_column", 1), ("several", 3))
    def test_matmul_matches_the_host(self, w):
        ring, host = _rings()
        rng = np.random.default_rng(55)
        m, k = 3, 2
        mat = _random_canonical(rng, m, k)
        other = _random_canonical(rng, k, w)
        got = _from_split(ring, ring.matmul(_split(ring, mat), _split(ring, other)))
        np.testing.assert_array_equal(got, host.matmul(mat, other))

    def test_matmul_at_one_column_is_matvec(self):
        """The relationship `HostSplitRing.matmul`'s docstring states —
        `matmul(A, v[:, None])[:, 0]` — holds here too, because both go
        through one contraction rather than two spellings of it."""
        ring, _ = _rings()
        rng = np.random.default_rng(56)
        mat = _split(ring, _random_canonical(rng, 4, 2))
        vec_host = _random_canonical(rng, 2)
        vec = _split(ring, vec_host)
        column = _split(ring, vec_host[:, None])
        np.testing.assert_array_equal(
            _from_split(ring, ring.matvec(mat, vec)),
            _from_split(ring, ring.matmul(mat, column))[:, 0],
        )

    def test_matmul_refuses_an_inner_extent_that_does_not_contract(self):
        ring, _ = _rings()
        rng = np.random.default_rng(57)
        mat = _split(ring, _random_canonical(rng, 3, 2))
        other = _split(ring, _random_canonical(rng, 3, 2))
        with self.assertRaisesRegex(ValueError, r"matmul"):
            ring.matmul(mat, other)

    def test_scale_matches_the_host(self):
        ring, host = _rings()
        rng = np.random.default_rng(58)
        element = _random_canonical(rng)
        stack = _random_canonical(rng, 4)
        got = _from_split(ring, ring.scale(_split(ring, element), _split(ring, stack)))
        np.testing.assert_array_equal(got, host.scale(element, stack))

    def test_scale_refuses_a_stack_that_is_not_one(self):
        """`mul` would broadcast these into a well-shaped answer to a
        question nobody asked, which is the whole reason `scale` is named."""
        ring, _ = _rings()
        rng = np.random.default_rng(59)
        element = _split(ring, _random_canonical(rng))
        with self.assertRaisesRegex(ValueError, r"scale"):
            ring.scale(element, element)

    @parameterized.named_parameters(("one_sum", 1), ("a_batch", 3))
    def test_combine_matches_the_host(self, rows):
        ring, host = _rings()
        rng = np.random.default_rng(60)
        terms = 4
        stack = _random_canonical(rng, terms, 2)
        weights = rng.integers(0, _Q_MODULI[0], size=(rows, terms))
        flat = weights[0] if rows == 1 else weights
        got = ring.combine(flat, _split(ring, stack))
        np.testing.assert_array_equal(_from_split(ring, got), host.combine(flat, stack))

    def test_combine_is_domain_generic(self):
        """A `Z_q` scalar acts entrywise in either domain, so the answer does
        not depend on which side of `to_split` the aggregation happens."""
        ring, _ = _rings()
        rng = np.random.default_rng(61)
        stack = ring.coeff_from_host(_random_canonical(rng, 3))
        weights = rng.integers(0, _Q_MODULI[0], size=3)
        np.testing.assert_array_equal(
            ring.to_host(ring.to_split(ring.combine(weights, stack))),
            ring.to_host(ring.combine(weights, ring.to_split(stack))),
        )

    def test_combine_refuses_a_weight_count_the_stack_does_not_match(self):
        ring, _ = _rings()
        rng = np.random.default_rng(62)
        stack = _split(ring, _random_canonical(rng, 3))
        with self.assertRaisesRegex(ValueError, r"combine"):
            ring.combine(np.arange(4), stack)

    def test_combine_refuses_weights_above_rank_two(self):
        ring, _ = _rings()
        rng = np.random.default_rng(63)
        stack = _split(ring, _random_canonical(rng, 3))
        with self.assertRaisesRegex(ValueError, r"combine"):
            ring.combine(np.zeros((2, 2, 3), dtype=np.int64), stack)

    @parameterized.parameters(1, 3, 5, -1)
    def test_galois_matches_the_host(self, k):
        ring, host = _rings()
        arr = _random_canonical(np.random.default_rng(64))
        got = ring.galois(ring.coeff_from_host(arr), k)
        np.testing.assert_array_equal(ring.to_host(got), host.galois(arr, k))

    def test_galois_carries_batch_axes(self):
        """σ_k of a module vector is σ_k of each element — a consumer lifting
        a whole vector through an automorphism should not spell that loop."""
        ring, host = _rings()
        arr = _random_canonical(np.random.default_rng(65), 3)
        got = ring.galois(ring.coeff_from_host(arr), 3)
        np.testing.assert_array_equal(ring.to_host(got), host.galois(arr, 3))

    @parameterized.parameters(2, 0, 2 * _D)
    def test_galois_refuses_an_even_exponent(self, k):
        """`gcd(k, 2d) = 1` is what makes σ_k an automorphism, and `2d` is a
        power of two — so an even `k` is a projection, not a ring map."""
        ring, _ = _rings()
        element = ring.coeff_from_host(_random_canonical(np.random.default_rng(67)))
        with self.assertRaisesRegex(ValueError, r"k must be odd"):
            ring.galois(element, k)

    @parameterized.parameters((3, 3 + 2 * _D), (-1, 2 * _D - 1))
    def test_galois_reads_the_exponent_modulo_two_d(self, k, equivalent):
        """σ_k depends on `k mod 2d` only, so the two spellings are one map."""
        ring, _ = _rings()
        element = ring.coeff_from_host(_random_canonical(np.random.default_rng(68)))
        np.testing.assert_array_equal(
            ring.to_host(ring.galois(element, k)),
            ring.to_host(ring.galois(element, equivalent)),
        )

    def test_galois_refuses_the_split_domain(self):
        """σ_k permutes coefficients; a CRT half has none to permute, and the
        automorphism does not act half-wise anyway."""
        ring, _ = _rings()
        split = _split(ring, _random_canonical(np.random.default_rng(66)))
        with self.assertRaisesRegex(TypeError, r"galois"):
            ring.galois(split, 3)


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
        k = 3
        arr = _random_canonical(np.random.default_rng(39), k)

        def leading_coeffs(x):
            unchanged = ring.add(x, ring.zeros(Coeff, k))
            return ring.constant_coeff(unchanged)

        got = frx.jit(leading_coeffs)(ring.coeff_from_host(arr))
        np.testing.assert_array_equal(_host_scalars(got), host.constant_coeff(arr))
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
        np.testing.assert_array_equal(got, host_split_stack(host, host.zeros(k)))

    def test_the_module_layer_holds_under_jit_and_vmap(self):
        ring, host = _rings()
        rng = np.random.default_rng(70)
        b, m, k = 2, 3, 2
        mat, vec = _random_canonical(rng, b, m, k), _random_canonical(rng, b, k)
        want = np.stack([host.matvec(mat[i], vec[i]) for i in range(b)])
        jitted = frx.jit(ring.matvec)
        np.testing.assert_array_equal(
            _from_split(ring, jitted(_split(ring, mat), _split(ring, vec))), want
        )
        mapped = frx.vmap(ring.matvec)
        np.testing.assert_array_equal(
            _from_split(ring, mapped(_split(ring, mat), _split(ring, vec))), want
        )

    def test_a_whole_module_contraction_composes_into_one_jit_zone(self):
        """The shape a consumer's proof layer wants: host arrays at the edges
        and no boundary crossing anywhere inside."""
        ring, host = _rings()
        rng = np.random.default_rng(71)
        mat, vec = _random_canonical(rng, 3, 2), _random_canonical(rng, 2)

        def contract(a, v):
            return ring.from_split(ring.matvec(ring.to_split(a), ring.to_split(v)))

        got = ring.to_host(
            frx.jit(contract)(ring.coeff_from_host(mat), ring.coeff_from_host(vec))
        )
        np.testing.assert_array_equal(got, host.matvec(mat, vec))

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
