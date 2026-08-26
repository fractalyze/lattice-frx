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

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from lattice_frx import primes
from lattice_frx.domains import Coeff, Eval, Split
from lattice_frx.split_ring import HostSplitRing, SplitRing

try:  # `frx` is the ring under test's array layer; `jit`/`vmap` come from it.
    import frx
except ImportError:  # pragma: no cover - the bazel/pip lanes both provide it
    frx = None

_D = 16
_Q_MODULI = tuple(primes.find_nearest_split_primes(30.0, 2))


def _rings() -> tuple[SplitRing, HostSplitRing]:
    return SplitRing(_Q_MODULI, _D), HostSplitRing(_Q_MODULI, _D)


def _random_canonical(rng, *lead: int) -> np.ndarray:
    """A canonical host element, or a `lead`-shaped stack of them."""
    rows = [rng.integers(0, q, lead + (_D,), dtype=np.uint64) for q in _Q_MODULI]
    return np.moveaxis(np.stack(rows), 0, len(lead))


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

    @parameterized.named_parameters(
        ("add", "add"), ("sub", "sub"), ("neg", "neg"),
    )
    def test_the_elementwise_ops_match_the_host(self, op):
        ring, host = _rings()
        rng = np.random.default_rng(13)
        a, b = _random_canonical(rng), _random_canonical(rng)
        ca, cb = ring.coeff_from_host(a), ring.coeff_from_host(b)
        args = (ca,) if op == "neg" else (ca, cb)
        host_args = (a,) if op == "neg" else (a, b)
        np.testing.assert_array_equal(
            ring.to_host(getattr(ring, op)(*args)), getattr(host, op)(*host_args)
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


class TracedSplitRingTracingTest(parameterized.TestCase):
    """The reason this ring exists beside the host one."""

    def setUp(self):
        super().setUp()
        if frx is None:
            self.skipTest("frx is not importable")

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
