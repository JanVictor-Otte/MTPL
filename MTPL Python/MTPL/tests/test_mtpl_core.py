"""
Python port of test_mtpl_core.cpp — verifies the Python MTPL core.

Run with:  python -m pytest python/tests/test_mtpl_core.py -v
"""
from __future__ import annotations

import math
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from mtpl.core.event import Event
from mtpl.core.signal import (
    Signal,
    ConstantSignal,
    ConstantFrozenSignal,
    FrozenSignal,
    SignalTransform,
    Pushforward,
    CastTransform,
    is_signal,
    is_constant_signal,
    is_varying_signal,
)
from mtpl.core.morphism import (
    Morphism,
    EventLeaf,
    MultiLaneLeaf,
    Compose,
    Tensor,
    Project,
    Identity,
    Variadic,
)
from mtpl.core.source import PrimitiveSource, Source, merge, apply as source_apply
from mtpl.timed.timed_event import TimedEvent, join, evaluate


# ============================================================================
#  Helpers
# ============================================================================

class TestEvent(Event):
    """Simple test event wrapping an int payload."""
    pass


def te(val: int) -> TestEvent:
    return TestEvent(payload=val)


# ============================================================================
#  Signal Tests
# ============================================================================

class TestConstantSignal:
    def test_returns_correct_value(self):
        sig = SignalTransform.Constant(42)
        frozen = sig()
        assert frozen() == 42

    def test_ignores_event(self):
        sig = SignalTransform.Constant(42)
        frozen = sig()
        assert frozen(te(10)) == 42


class TestSignalArithmetic:
    def test_addition(self):
        a = SignalTransform.Constant(10)
        b = SignalTransform.Constant(5)
        s = a + b
        assert s()(te(0)) == 15

    def test_subtraction(self):
        a = SignalTransform.Constant(10)
        b = SignalTransform.Constant(5)
        s = a - b
        assert s()(te(0)) == 5

    def test_multiplication(self):
        a = SignalTransform.Constant(10)
        b = SignalTransform.Constant(5)
        s = a * b
        assert s()(te(0)) == 50

    def test_division(self):
        a = SignalTransform.Constant(10)
        b = SignalTransform.Constant(5)
        s = a / b
        assert s()(te(0)) == 2.0

    def test_floor_division(self):
        a = SignalTransform.Constant(10)
        b = SignalTransform.Constant(3)
        s = a // b
        assert s()(te(0)) == 3


class TestSignalWithScalar:
    def test_signal_plus_scalar(self):
        sig = SignalTransform.Constant(10)
        s = sig + 5
        assert s()(te(0)) == 15

    def test_signal_times_scalar(self):
        sig = SignalTransform.Constant(10)
        s = sig * 3
        assert s()(te(0)) == 30

    def test_scalar_minus_signal(self):
        sig = SignalTransform.Constant(10)
        s = 20 - sig
        assert s()(te(0)) == 10


class TestPushforward:
    def test_basic(self):
        a = SignalTransform.Constant(3)
        b = SignalTransform.Constant(4)
        hyp = Pushforward(lambda x, y: int(math.sqrt(x*x + y*y)), arity=2)(a, b)
        assert hyp()(te(0)) == 5

    def test_mixed_types(self):
        i = SignalTransform.Constant(10)
        d = SignalTransform.Constant(3.5)
        r = Pushforward(lambda x, y: x * y, arity=2)(i, d)
        assert r()(te(0)) == 35.0

    def test_constant_propagation(self):
        c1 = SignalTransform.Constant(6)
        c2 = SignalTransform.Constant(4)
        result = Pushforward(lambda x, y: x + y, arity=2)(c1, c2)
        assert is_constant_signal(result)
        assert result()() == 10

    def test_three_args(self):
        a = SignalTransform.Constant(5)
        b = SignalTransform.Constant(3)
        c = SignalTransform.Constant(2)
        r = Pushforward(lambda x, y, z: x * y + z, arity=3)(a, b, c)
        assert r()(te(0)) == 17

    def test_composition(self):
        x = SignalTransform.Constant(3)
        y = SignalTransform.Constant(4)
        sos = Pushforward(lambda a, b: a*a + b*b, arity=2)(x, y)
        mag = Pushforward(lambda s: int(math.sqrt(s)), arity=1)(sos)
        assert mag()(te(0)) == 5

    def test_with_operators(self):
        a = SignalTransform.Constant(10)
        b = SignalTransform.Constant(3)
        s = a + b
        d = a - b
        p = a * b
        assert s()(te(0)) == 13
        assert d()(te(0)) == 7
        assert p()(te(0)) == 30

    def test_unary_neg(self):
        a = SignalTransform.Constant(5)
        r = -a
        assert r()(te(0)) == -5


class TestVaryingSignal:
    def test_varying_signal(self):
        sig = Signal(lambda: FrozenSignal(lambda e: e.payload * 2))
        frozen = sig()
        assert frozen(te(5)) == 10

    def test_varying_is_not_constant(self):
        sig = Signal(lambda: FrozenSignal(lambda e: e.payload))
        assert is_signal(sig)
        assert not is_constant_signal(sig)
        assert is_varying_signal(sig)

    def test_constant_is_signal(self):
        sig = SignalTransform.Constant(42)
        assert is_signal(sig)
        assert is_constant_signal(sig)
        assert not is_varying_signal(sig)

    def test_to_constant(self):
        # Use a factory that doesn't depend on the event payload
        # (the C++ ConstantifiedElement evaluates at E{} by design)
        sig = Signal(lambda: FrozenSignal(lambda e: 42))
        csig = sig.to_constant()
        assert is_constant_signal(csig)
        assert csig()() == 42

    def test_to_constant_event_independent(self):
        # Key invariant: once frozen, a ConstantSignal is event-independent
        csig = SignalTransform.Constant(99)
        frozen = csig()
        assert frozen(te(1)) == 99
        assert frozen(te(2)) == 99


class TestCastTransform:
    def test_int_to_float(self):
        sig = SignalTransform.Constant(42)
        cast = CastTransform(float)
        result = cast(sig)
        assert result()(te(0)) == 42.0
        assert isinstance(result()(te(0)), float)


# ============================================================================
#  Morphism Tests
# ============================================================================

class TestLeafMorphism:
    def test_identity(self):
        identity = MultiLaneLeaf(fn=lambda lanes: lanes)
        inp = [[te(1), te(2)]]
        out = identity(inp)
        assert len(out) == 1
        assert len(out[0]) == 2
        assert out[0][0].payload == 1

    def test_fixed_arity(self):
        combiner = MultiLaneLeaf(
            fn=lambda lanes: [[e for lane in lanes for e in lane]],
            in_arity=2,
            out_arity_fn=1,
        )
        assert combiner.in_arity() == 2
        assert combiner.out_arity(2) == 1
        inp = [[te(1), te(2)], [te(3), te(4)]]
        out = combiner(inp)
        assert len(out) == 1
        assert len(out[0]) == 4


class TestCompose:
    def test_basic(self):
        dup = MultiLaneLeaf(
            fn=lambda lanes: [lanes[0], lanes[0]],
            in_arity=1, out_arity_fn=2,
        )
        sel = MultiLaneLeaf(
            fn=lambda lanes: [lanes[0]],
            in_arity=2, out_arity_fn=1,
        )
        composed = Compose(dup, sel)
        assert composed.in_arity() == 1
        assert composed.out_arity(1) == 1
        out = composed([[te(42)]])
        assert len(out) == 1
        assert out[0][0].payload == 42

    def test_flattening(self):
        m1 = MultiLaneLeaf(fn=lambda l: l)
        m2 = MultiLaneLeaf(fn=lambda l: l)
        m3 = MultiLaneLeaf(fn=lambda l: l)
        c1 = Compose(m1, m2)
        c2 = Compose(c1, m3)
        out = c2([[te(1)]])
        assert len(out) == 1

    def test_arity_validation(self):
        m1 = MultiLaneLeaf(fn=lambda l: [[e for lane in l for e in lane]], in_arity=2, out_arity_fn=1)
        m2 = MultiLaneLeaf(fn=lambda l: l, in_arity=1, out_arity_fn=1)
        # m1 outputs 1 lane, m2 expects 1 → OK
        Compose(m1, m2)

        m3 = MultiLaneLeaf(fn=lambda l: l, in_arity=1, out_arity_fn=1)
        m4 = MultiLaneLeaf(fn=lambda l: l, in_arity=2, out_arity_fn=1)
        # m3 outputs 1 lane, m4 expects 2 → fail
        with pytest.raises(RuntimeError, match="incompatible arities"):
            Compose(m3, m4)


class TestTensor:
    def test_basic(self):
        m1 = MultiLaneLeaf(fn=lambda l: l)
        m2 = MultiLaneLeaf(fn=lambda l: l)
        tensored = Tensor(m1, m2)
        inp = [[te(1), te(2)]]
        out = tensored(inp)
        assert len(out) == 2
        assert tensored.out_arity(1) == 2


class TestProject:
    def test_single_index(self):
        proj = Project(1)
        inp = [[te(10)], [te(20)], [te(30)]]
        out = proj(inp)
        assert len(out) == 1
        assert out[0][0].payload == 20

    def test_multiple_indices(self):
        proj = Project(0, 2)
        inp = [[te(10)], [te(20)], [te(30)]]
        out = proj(inp)
        assert len(out) == 2
        assert out[0][0].payload == 10
        assert out[1][0].payload == 30

    def test_out_of_range(self):
        proj = Project(5)
        with pytest.raises(RuntimeError, match="out of range"):
            proj([[te(1)]])


class TestIdentity:
    def test_pass_through(self):
        ident = Identity()
        inp = [[te(1)], [te(2)]]
        out = ident(inp)
        assert out == inp


# ============================================================================
#  Source Tests
# ============================================================================

class TestPrimitiveSource:
    def test_basic(self):
        prim = PrimitiveSource(2, lambda: [[te(1), te(2)], [te(3), te(4)]])
        assert prim.out_arity() == 2
        out = prim()
        assert len(out) == 2
        assert len(out[0]) == 2


class TestSource:
    def test_from_primitive(self):
        prim = PrimitiveSource(1, lambda: [[te(42)]])
        src = Source(prim)
        out = src()
        assert len(out) == 1
        assert out[0][0].payload == 42

    def test_with_morphism(self):
        prim = PrimitiveSource(1, lambda: [[te(10)]])
        dup = MultiLaneLeaf(
            fn=lambda l: [l[0], l[0]],
            in_arity=1, out_arity_fn=2,
        )
        src = Source(prim, morphism=dup)
        assert src.out_arity() == 2
        out = src()
        assert len(out) == 2

    def test_morphism_application_operator(self):
        prim = PrimitiveSource(1, lambda: [[te(5)]])
        tripler = MultiLaneLeaf(
            fn=lambda l: [l[0], l[0], l[0]],
            in_arity=1, out_arity_fn=3,
        )
        src = tripler(prim)
        assert src.out_arity() == 3
        out = src()
        assert len(out) == 3

    def test_chained_application(self):
        prim = PrimitiveSource(1, lambda: [[te(1)]])
        m1 = MultiLaneLeaf(
            fn=lambda l: [l[0], l[0]],
            in_arity=1, out_arity_fn=2,
        )
        m2 = MultiLaneLeaf(
            fn=lambda l: [l[0] + l[1]],
            in_arity=2, out_arity_fn=1,
        )
        src = m2(m1(prim))
        assert src.out_arity() == 1
        out = src()
        assert len(out[0]) == 2  # duplicated then concatenated


class TestMerge:
    def test_merge_two_sources(self):
        p1 = PrimitiveSource(1, lambda: [[te(1)]])
        p2 = PrimitiveSource(1, lambda: [[te(2)]])
        s1 = Source(p1)
        s2 = Source(p2)
        merged = merge(s1, s2)
        out = merged()
        assert len(out) == 2
        assert out[0][0].payload == 1
        assert out[1][0].payload == 2

    def test_merge_single(self):
        p1 = PrimitiveSource(1, lambda: [[te(42)]])
        s1 = Source(p1)
        merged = merge(s1)
        out = merged()
        assert len(out) == 1
        assert out[0][0].payload == 42


# ============================================================================
#  EventLeaf Tests
# ============================================================================

class TestEventLeaf:
    def test_double_payload(self):
        def double(event, factor):
            return TestEvent(payload=event.payload * factor)

        doubler = EventLeaf(
            fn=double,
            signals=[SignalTransform.Constant(2)],
            in_arity=1,
        )
        prim = PrimitiveSource(1, lambda: [[te(5), te(10)]])
        src = Source(prim, morphism=doubler)
        out = src()
        assert out[0][0].payload == 10
        assert out[0][1].payload == 20

    def test_multi_signal(self):
        def adder(event, a, b):
            return TestEvent(payload=event.payload + a + b)

        leaf = EventLeaf(
            fn=adder,
            signals=[SignalTransform.Constant(3), SignalTransform.Constant(7)],
            in_arity=1,
        )
        prim = PrimitiveSource(1, lambda: [[te(5)]])
        src = Source(prim, morphism=leaf)
        out = src()
        assert out[0][0].payload == 15


# ============================================================================
#  MultiLaneLeaf with signals Tests
# ============================================================================

class TestMultiLaneLeafSignals:
    def test_parameterised(self):
        def scale_fn(lanes, factor):
            return [[TestEvent(payload=e.payload * factor) for e in lane] for lane in lanes]

        scale = MultiLaneLeaf(
            fn=scale_fn,
            signals=[SignalTransform.Constant(3)],
            in_arity=1,
        )
        prim = PrimitiveSource(1, lambda: [[te(2), te(4)]])
        src = Source(prim, morphism=scale)
        out = src()
        assert out[0][0].payload == 6
        assert out[0][1].payload == 12

    def test_multi_signal(self):
        def offset_scale_fn(lanes, offset, scale):
            return [[TestEvent(payload=e.payload * scale + offset) for e in lane] for lane in lanes]

        m = MultiLaneLeaf(
            fn=offset_scale_fn,
            signals=[SignalTransform.Constant(10), SignalTransform.Constant(2)],
            in_arity=1,
        )
        prim = PrimitiveSource(1, lambda: [[te(5)]])
        src = Source(prim, morphism=m)
        out = src()
        assert out[0][0].payload == 20  # 5 * 2 + 10


# ============================================================================
#  Integration Tests
# ============================================================================

class TestFullPipeline:
    def test_pipeline(self):
        gen = PrimitiveSource(1, lambda: [[te(1), te(2), te(3)]])
        dup = MultiLaneLeaf(
            fn=lambda l: [l[0], l[0]],
            in_arity=1, out_arity_fn=2,
        )
        sel = MultiLaneLeaf(
            fn=lambda l: [l[0]],
            in_arity=2, out_arity_fn=1,
        )
        pipeline = Compose(dup, sel)(gen)
        out = pipeline()
        assert len(out) == 1
        assert len(out[0]) == 3
        assert out[0][0].payload == 1


class TestSignalIntrospection:
    def test_inputs_tracked(self):
        a = SignalTransform.Constant(5)
        b = SignalTransform.Constant(3)
        s = Pushforward(lambda x, y: x + y, arity=2)(a, b)
        assert len(s.inputs) == 2

    def test_metadata(self):
        a = SignalTransform.Constant(5)
        b = SignalTransform.Constant(3)
        s = Pushforward(lambda x, y: x + y, arity=2)(a, b)
        s.metadata = "Addition operation"
        assert s.metadata == "Addition operation"

    def test_nested_computation(self):
        a = SignalTransform.Constant(5)
        b = SignalTransform.Constant(3)
        c = SignalTransform.Constant(2)
        s = Pushforward(lambda x, y: x + y, arity=2)(a, b)
        p = Pushforward(lambda x, y: x * y, arity=2)(s, c)
        assert len(p.inputs) == 2
        result = p()(te(0))
        assert result == 16  # (5+3)*2


# ============================================================================
#  TimedEvent Tests
# ============================================================================

class TestTimedEvent:
    def test_join(self):
        j = join()
        lanes = [
            [TimedEvent(time=0.5, payload="b"), TimedEvent(time=0.1, payload="a")],
            [TimedEvent(time=0.3, payload="c")],
        ]
        out = j(lanes)
        assert len(out) == 1
        assert [e.payload for e in out[0]] == ["a", "c", "b"]

    def test_evaluate(self):
        prim = PrimitiveSource(1, lambda: [[TimedEvent(time=0.0, payload="x")]])
        src = Source(prim)
        result = evaluate(src, period=1.0, loops=3)
        assert len(result) == 3
        assert [e.time for e in result] == [0.0, 1.0, 2.0]


# ============================================================================
#  Arity enforcement edge cases
# ============================================================================

class TestArityEnforcement:
    def test_morphism_wrong_lane_count(self):
        m = MultiLaneLeaf(fn=lambda l: l, in_arity=2)
        with pytest.raises(RuntimeError, match="expected 2 input lanes"):
            m([[te(1)]])

    def test_signal_transform_wrong_arity(self):
        st = SignalTransform(lambda a, b: a, arity=2)
        with pytest.raises(RuntimeError):
            st(SignalTransform.Constant(1))  # only 1 arg, expects 2

    def test_compose_wrong_arity_at_runtime(self):
        m = MultiLaneLeaf(fn=lambda l: l, in_arity=3)
        with pytest.raises(RuntimeError, match="expected 3 input lanes"):
            m([[te(1)], [te(2)]])
