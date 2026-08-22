"""The ring against the host implementation, which is the only oracle it needs.

`host_ring.HostRnsRing` computes over exact Python integers, one coefficient at
a time, reproducing lattigo's tables and loop structure — slow and obviously
right. `ring.RnsRing` computes the same ring through `frx.lax.ntt` and field
dtypes so a tracer can enter it. Every case here is the same question: do the
two agree.

**That question also settles the output order**, which is the property this
package says no property test can catch. It cannot be caught *within* one
implementation — a consistently permuted transform round-trips and convolves
like the correct one — but it is caught *between* two, because the host side's
order is lattigo's natively and a permutation would have to be present in both
to hide. The chain is: host == lattigo (the consumer's golden, at its own moduli
chain), and ring == host (here). What that does not cover is the two drifting
together, which is what the consumer's cross-verification on a pin bump is for.

The moduli are 36-bit on purpose. Past a 32-bit lane, so a ring that narrows a residue fails here rather than at a consumer's wire boundary — and at
that width the host side cannot use `uint64` arithmetic either, which is why
the reference carries exact Python integers and the comparison below goes
through `to_host`.
"""

import functools
import random

import frx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from lattice_frx import host_ring as host_mod
from lattice_frx import roots as roots_mod
from lattice_frx import ring as ring_mod

# NTT-friendly, 36-bit, and `1 mod 2d` at d=256.
_Q_MODULI = (34359753217, 34359754753)
_D = 256


@functools.cache
def rings():
    return host_mod.HostRnsRing(_Q_MODULI, _D), ring_mod.RnsRing(_Q_MODULI, _D)


def _random_host(seed: int) -> np.ndarray:
    rnd = random.Random(seed)
    return np.array(
        [[rnd.randrange(0, q) for _ in range(_D)] for q in _Q_MODULI], dtype=np.uint64
    )


def _row(batched, index: int):
    """One element back out of a batched one, in the same domain."""
    return type(batched)(tuple(limb[index] for limb in batched.limbs))


class RingTest(parameterized.TestCase):

    def test_the_field_carries_the_full_limb_width(self) -> None:
        """The reason this backend exists: 36-bit residues survive the round trip.

        An integer lane would not carry them — frx runs without x64, so a `uint64`
        array narrows to `uint32` and truncates. A field's storage follows its
        modulus instead.
        """
        _, dev = rings()
        host = _random_host(1)
        self.assertTrue(np.array_equal(dev.to_host(dev.coeff_from_host(host)), host))
        self.assertGreater(max(int(v) for v in host[0]), 1 << 32)

    def test_ntt_agrees_with_the_reference(self) -> None:
        """The order gate, transitively: the reference's order is lattigo's."""
        ref, dev = rings()
        host = _random_host(2)
        got = dev.to_host(dev.ntt(dev.coeff_from_host(host)))
        self.assertTrue(np.array_equal(got, ref.ntt(host)))

    def test_intt_agrees_with_the_reference(self) -> None:
        ref, dev = rings()
        host = _random_host(3)
        got = dev.to_host(dev.intt(dev.eval_from_host(host)))
        self.assertTrue(np.array_equal(got, ref.intt(host)))

    def test_intt_inverts_ntt(self) -> None:
        _, dev = rings()
        host = _random_host(4)
        element = dev.coeff_from_host(host)
        self.assertTrue(np.array_equal(dev.to_host(dev.intt(dev.ntt(element))), host))

    @parameterized.parameters(
        "add", "sub", "mul", "mul_add", "mul_scalar", "mul_scalar_then_sub", "neg"
    )
    def test_arithmetic_agrees_with_the_reference(self, op: str) -> None:
        ref, dev = rings()
        a_host, b_host, c_host = _random_host(5), _random_host(6), _random_host(7)
        # `mul`/`mul_add` are pointwise products, defined only in the NTT domain;
        # everything else is domain-generic and exercised on the coefficient side.
        embed = dev.eval_from_host if op in ("mul", "mul_add") else dev.coeff_from_host
        a, b, c = embed(a_host), embed(b_host), embed(c_host)
        scalar = 123456789

        if op == "neg":
            got, want = dev.neg(a), ref.neg(a_host)
        elif op in ("add", "sub", "mul"):
            got = getattr(dev, op)(a, b)
            want = getattr(ref, op)(a_host, b_host)
        elif op == "mul_add":
            got, want = dev.mul_add(a, b, c), ref.mul_add(a_host, b_host, c_host)
        elif op == "mul_scalar":
            got, want = dev.mul_scalar(a, scalar), ref.mul_scalar(a_host, scalar)
        else:
            got = dev.mul_scalar_then_sub(a, scalar, c)
            want = ref.mul_scalar_then_sub(a_host, scalar, c_host)

        self.assertTrue(np.array_equal(dev.to_host(got), want))

    def test_from_signed_agrees_with_the_reference(self) -> None:
        ref, dev = rings()
        rnd = random.Random(8)
        values = [rnd.randrange(-(1 << 20), 1 << 20) for _ in range(_D)]
        self.assertTrue(
            np.array_equal(dev.to_host(dev.from_signed(values)), ref.from_signed(values))
        )

    def test_to_balanced_limb0_agrees_with_the_reference(self) -> None:
        ref, dev = rings()
        host = _random_host(9)
        got = dev.to_balanced_limb0(dev.coeff_from_host(host))
        want = ref.to_balanced_limb0(host)
        self.assertEqual(got.dtype, np.int64)
        self.assertTrue(np.array_equal(got, want))

    def test_the_ring_is_negacyclic(self) -> None:
        """`X * X^(d-1) = -1`, through the ring's own transform."""
        _, dev = rings()
        x = dev.from_signed([0, 1] + [0] * (_D - 2))
        x_pow = dev.from_signed([0] * (_D - 1) + [1])
        product = dev.intt(dev.mul(dev.ntt(x), dev.ntt(x_pow)))
        want = dev.from_signed([-1] + [0] * (_D - 1))
        self.assertTrue(np.array_equal(dev.to_host(product), dev.to_host(want)))

    def test_the_transform_traces_as_one_computation(self) -> None:
        """The point of the backend: `ntt` composes into a `jit` zone.

        `Coeff` and `Eval` are `NamedTuple`s over per-limb field arrays — still
        pytrees, so they cross the boundary without a registered type of their own,
        and the domain guard runs at trace time.
        """
        ref, dev = rings()
        host = _random_host(10)
        compiled = frx.jit(dev.ntt)
        got = dev.to_host(compiled(dev.coeff_from_host(host)))
        self.assertTrue(np.array_equal(got, ref.ntt(host)))

    def test_an_element_survives_a_vmapped_batch(self) -> None:
        """A batch axis over elements, which is what a consumer's hot path adds."""
        ref, dev = rings()
        hosts = [_random_host(11), _random_host(12), _random_host(13)]
        batched = frx.jit(frx.vmap(dev.ntt))(
            dev.stack([dev.coeff_from_host(h) for h in hosts])
        )
        for index, host in enumerate(hosts):
            self.assertTrue(
                np.array_equal(dev.to_host(_row(batched, index)), ref.ntt(host))
            )

    def test_a_leading_batch_axis_needs_no_vmap(self) -> None:
        """The batch convention directly: ops read `[..., d]` limbs as they are."""
        ref, dev = rings()
        hosts = [_random_host(14), _random_host(15)]
        batched = dev.ntt(dev.stack([dev.coeff_from_host(h) for h in hosts]))
        for index, host in enumerate(hosts):
            self.assertTrue(
                np.array_equal(dev.to_host(_row(batched, index)), ref.ntt(host))
            )

    def test_domain_confusion_is_a_typeerror(self) -> None:
        """The reason the element is two types instead of one tuple."""
        _, dev = rings()
        coeff = dev.coeff_from_host(_random_host(16))
        evaled = dev.ntt(coeff)

        with self.assertRaises(TypeError):
            dev.mul(coeff, coeff)  # pointwise in the coefficient domain: not mul
        with self.assertRaises(TypeError):
            dev.add(coeff, evaled)  # mixed domains never add
        with self.assertRaises(TypeError):
            dev.ntt(evaled)  # already transformed
        with self.assertRaises(TypeError):
            dev.intt(coeff)  # not transformed yet
        with self.assertRaises(TypeError):
            dev.to_balanced_limb0(evaled)  # a balanced lift of NTT values
        with self.assertRaises(TypeError):
            dev.mul(coeff.limbs, coeff.limbs)  # a bare tuple asserts no domain
        with self.assertRaises(TypeError):
            dev.stack([coeff, evaled])  # a batch never mixes domains

    def test_matvec_agrees_with_elementwise_mul_add(self) -> None:
        """`A·s` per the module convention equals the same sum taken one ring
        element at a time through the reference."""
        ref, dev = rings()
        m, k = 3, 2
        a_hosts = [[_random_host(20 + i * k + j) for j in range(k)] for i in range(m)]
        s_hosts = [_random_host(30 + j) for j in range(k)]

        mat = dev.stack(
            [dev.stack([dev.eval_from_host(h) for h in row]) for row in a_hosts]
        )
        vec = dev.stack([dev.eval_from_host(h) for h in s_hosts])
        got = dev.matvec(mat, vec)

        for i in range(m):
            want = ref.mul(a_hosts[i][0], s_hosts[0])
            for j in range(1, k):
                want = ref.mul_add(a_hosts[i][j], s_hosts[j], want)
            self.assertTrue(np.array_equal(dev.to_host(_row(got, i)), want))

    def test_host_matvec_agrees_with_the_traced_one(self) -> None:
        """The host module matvec (composed `mul`/`add`, NTT-domain for this
        ring) against `RnsRing.matvec` on the same values — the same
        cross-implementation gate the elementwise ops get above."""
        ref, dev = rings()
        m, k = 2, 3
        a_hosts = [[_random_host(70 + i * k + j) for j in range(k)] for i in range(m)]
        s_hosts = [_random_host(80 + j) for j in range(k)]
        got = ref.matvec(
            np.stack([np.stack(row) for row in a_hosts]), np.stack(s_hosts)
        )
        mat = dev.stack(
            [dev.stack([dev.eval_from_host(h) for h in row]) for row in a_hosts]
        )
        vec = dev.stack([dev.eval_from_host(h) for h in s_hosts])
        want = dev.matvec(mat, vec)
        for i in range(m):
            self.assertTrue(np.array_equal(got[i], dev.to_host(_row(want, i))))

    def test_matvec_rejects_a_shape_mismatch(self) -> None:
        _, dev = rings()
        a, b, c = (dev.eval_from_host(_random_host(s)) for s in (40, 41, 42))
        with self.assertRaises(ValueError):
            dev.matvec(a, a)  # both rank-1: no `m` axis to contract
        with self.assertRaises(ValueError):
            # k = 1 against k' = 2: a size-1 axis would broadcast straight past a
            # rank check into well-shaped wrong values, so the guard checks the
            # contracted extent itself.
            dev.matvec(dev.stack([dev.stack([a])]), dev.stack([b, c]))

    @parameterized.parameters(1, 3, 5, 2 * _D - 1, 2 * _D + 3)
    def test_galois_agrees_with_the_reference(self, k: int) -> None:
        """The automorphism against the host oracle, `k` normalized mod `2d`."""
        ref, dev = rings()
        host = _random_host(50)
        got = dev.to_host(dev.galois(dev.coeff_from_host(host), k))
        self.assertTrue(np.array_equal(got, ref.galois(host, k)))

    def test_galois_is_a_ring_automorphism(self) -> None:
        """`σ_k(a·b) = σ_k(a)·σ_k(b)`, through the ring's own transform."""
        _, dev = rings()
        k = 5
        a = dev.coeff_from_host(_random_host(51))
        b = dev.coeff_from_host(_random_host(52))
        prod = dev.intt(dev.mul(dev.ntt(a), dev.ntt(b)))
        lhs = dev.galois(prod, k)
        rhs = dev.intt(dev.mul(dev.ntt(dev.galois(a, k)), dev.ntt(dev.galois(b, k))))
        self.assertTrue(np.array_equal(dev.to_host(lhs), dev.to_host(rhs)))

    def test_galois_composes_and_inverts(self) -> None:
        _, dev = rings()
        a = dev.coeff_from_host(_random_host(53))
        k, j = 3, 7
        lhs = dev.galois(dev.galois(a, k), j)
        rhs = dev.galois(a, (k * j) % (2 * _D))
        self.assertTrue(np.array_equal(dev.to_host(lhs), dev.to_host(rhs)))
        k_inv = pow(k, -1, 2 * _D)
        round_trip = dev.galois(dev.galois(a, k), k_inv)
        self.assertTrue(np.array_equal(dev.to_host(round_trip), dev.to_host(a)))

    def test_galois_eval_never_leaves_the_ntt_domain(self) -> None:
        """The contract-order permutation: `galois_eval ∘ ntt == ntt ∘ galois`,
        with no output permutation beyond the pinned adapter — the property that
        fails if the slot-exponent table and the order contract ever disagree,
        on either limb."""
        _, dev = rings()
        a = dev.coeff_from_host(_random_host(54))
        for k in (3, 5, 2 * _D - 1):
            lhs = dev.galois_eval(dev.ntt(a), k)
            rhs = dev.ntt(dev.galois(a, k))
            self.assertTrue(np.array_equal(dev.to_host(lhs), dev.to_host(rhs)))

    def test_galois_rejects_an_even_k_and_the_wrong_domain(self) -> None:
        _, dev = rings()
        a = dev.coeff_from_host(_random_host(55))
        for k in (0, 2, 2 * _D):
            with self.assertRaises(ValueError):
                dev.galois(a, k)
        with self.assertRaises(TypeError):
            dev.galois(dev.ntt(a), 3)  # an NTT-domain element in the coeff op
        with self.assertRaises(TypeError):
            dev.galois_eval(a, 3)  # a coefficient element in the eval op

    def test_galois_by_hand_at_a_tiny_ring(self) -> None:
        """`d = 4`, `q = 17`: `σ_3(X) = X³` and `σ_3(X²) = X⁶ = -X²`."""
        ref = host_mod.HostRnsRing((17,), 4)
        x = ref.from_signed([0, 1, 0, 0])
        self.assertTrue(np.array_equal(ref.galois(x, 3), ref.from_signed([0, 0, 0, 1])))
        x_sq = ref.from_signed([0, 0, 1, 0])
        self.assertTrue(
            np.array_equal(ref.galois(x_sq, 3), ref.from_signed([0, 0, -1, 0]))
        )

    def test_slot_exponents_match_the_oracle(self) -> None:
        """The closed form `e(j) = 2·brv(j) + 1` re-derived from the host ring,
        per limb: the slot values of `ntt(X)` are the roots themselves, so a
        discrete log base `ψ` reads the exponent table off the pinned order.
        This is the insurance that `roots.slot_exponents` and the order contract
        never drift apart — and that one modulus-free table serves every limb."""
        ref, _ = rings()
        want = roots_mod.slot_exponents(_D)
        mono = np.zeros((len(_Q_MODULI), _D), dtype=np.uint64)
        mono[:, 1] = 1
        slots = ref.ntt(mono)
        for limb, q in enumerate(_Q_MODULI):
            generator = roots_mod.primitive_root(q, roots_mod.prime_factors(q - 1))
            psi = pow(generator, (q - 1) // (2 * _D), q)
            dlog = {pow(psi, t, q): t for t in range(2 * _D)}
            self.assertEqual([dlog[int(v)] for v in slots[limb]], want)


if __name__ == "__main__":
    absltest.main()
