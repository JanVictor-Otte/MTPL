"""
Signal system — the core of MTPL.

Mirrors the C++ EventSignal<T,E> / LaneSignal<T,E> / MultiLaneSignal<T,E>
hierarchy but drops all template parameters (Python is dynamically typed).
Arity checks and the level-propagation rule are preserved as runtime
assertions.

Three-tier signal hierarchy
---------------------------
  EventSignal      — varies per event
  LaneSignal       — varies per lane, constant within lane
  MultiLaneSignal  — constant across the entire multilane

Three-tier frozen hierarchy
---------------------------
  FrozenSignal          — evaluator: event → value
  LaneFrozenSignal      — constant within lane, callable without event
  MultiLaneFrozenSignal — constant across multilane

Key classes
-----------
SignalTransformBase — abstract base for transforms (erased value types)
SignalTransform     — concrete family of morphisms on signals
Element             — leaf transform holding a factory () → FrozenSignal
MultiLaneElement    — leaf transform holding a baked value
LaneifiedElement    — re-samples each freeze, collapses to event-independent
Pushforward         — lifts a plain function into a SignalTransform
CastTransform       — type-casting transform (static_cast equivalent)
"""
from __future__ import annotations

import copy
import operator as _op
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

from mtpl.core.event import Event


# ============================================================================
#  FrozenSignal / LaneFrozenSignal / MultiLaneFrozenSignal
# ============================================================================

class FrozenSignal:
    """Evaluator: event → value."""

    __slots__ = ("_fn",)

    def __init__(self, fn: Callable[[Any], Any]):
        self._fn = fn

    def __call__(self, event: Any = None) -> Any:     # noqa: D401
        return self._fn(event)


class LaneFrozenSignal(FrozenSignal):
    """Lane-constant value.  Callable with or without an event."""

    __slots__ = ("_val",)

    def __init__(self, value: Any):
        super().__init__(lambda _e, v=value: v)
        self._val = value

    def __call__(self, event: Any = None) -> Any:      # noqa: D401
        return self._val


class MultiLaneFrozenSignal(LaneFrozenSignal):
    """Constant across the entire multilane.  Callable with or without an event."""
    pass


# ============================================================================
#  SignalTransformBase (abstract)
# ============================================================================

class SignalTransformBase(ABC):
    """Abstract base that erases value types.

    A Signal holds a reference to a ``SignalTransformBase`` instance — this is
    what makes computation and introspection one and the same mechanism.
    """

    @abstractmethod
    def apply(self, children: list[Any]) -> FrozenSignal:
        """Given children (type-erased Signals), produce a FrozenSignal."""

    @abstractmethod
    def to_lane_parts(self, children: list[Any], event: Any) -> tuple[SignalTransformBase, list[Any]]:
        """Produce (transform, children) for building a LaneSignal or MultiLaneSignal."""

    @abstractmethod
    def clone(self) -> SignalTransformBase:
        """Deep copy preserving derived type."""


# ============================================================================
#  EventSignal / LaneSignal / MultiLaneSignal
# ============================================================================

class EventSignal:
    """Factory: ``() → FrozenSignal``, backed by a computation tree.

    Construction
    ------------
    * ``EventSignal(factory)``  — wraps a zero-arg callable in an *Element* leaf.
    * Internal: ``EventSignal(_transform=…, _children=…)`` — build from parts.
    """

    def __init__(self, factory: Callable[[], FrozenSignal] | None = None, *,
                 _transform: SignalTransformBase | None = None,
                 _children: list[Any] | None = None):
        if _transform is not None:
            # Internal tree-based construction
            self._transform: SignalTransformBase = _transform
            self._children: list[Any] = _children if _children is not None else []
        elif factory is not None:
            # Leaf construction: wrap factory in an Element
            self._transform = Element(factory)
            self._children = [_SIGNAL_ATOM]
        else:
            # Bare construction (used by subclasses)
            self._transform = None  # type: ignore[assignment]
            self._children = []
        self._metadata: Any = None

    # -- Freeze: computation IS tree traversal --
    def __call__(self) -> FrozenSignal:
        return self._transform.apply(self._children)

    def to_lane_signal(self, event: Any = None) -> 'LaneSignal':
        """Freeze event variation, keep lane variation."""
        if event is None:
            event = Event(payload=None)
        t, c = self._transform.to_lane_parts(self._children, event)
        return LaneSignal(_transform=t, _children=c)

    def to_multilane_signal(self, event: Any = None) -> 'MultiLaneSignal':
        """Freeze all variation (event + lane)."""
        if event is None:
            event = Event(payload=None)
        t, c = self._transform.to_lane_parts(self._children, event)
        return MultiLaneSignal(_transform=t, _children=c)

    # Backward-compat alias
    def to_constant(self, event: Any = None) -> 'MultiLaneSignal':
        """Backward-compat alias for ``to_multilane_signal``."""
        return self.to_multilane_signal(event)

    # -- Introspection (same data used for computation) --
    @property
    def children(self) -> list[Any]:
        return self._children

    @property
    def inputs(self) -> list[Any]:
        return self._children

    @property
    def transform_ptr(self) -> SignalTransformBase:
        return self._transform

    @property
    def metadata(self) -> Any:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Any) -> None:
        self._metadata = value

    # -- Arithmetic helpers (defined after MultiLaneSignal) --


class LaneSignal(EventSignal):
    """Lane-constant signal: different per lane, constant within lane.

    Construction
    ------------
    * ``LaneSignal(factory)``  — *factory* returns a ``LaneFrozenSignal``.
    * Internal: ``LaneSignal(_transform=…, _children=…)``
    """

    def __init__(self,
                 factory: Callable[[], LaneFrozenSignal] | None = None, *,
                 _transform: SignalTransformBase | None = None,
                 _children: list[Any] | None = None):
        super().__init__()  # bare init
        if _transform is not None:
            self._transform = _transform
            self._children = _children if _children is not None else []
        elif factory is not None:
            def _fac(f=factory):
                val = f()()
                return FrozenSignal(lambda _e, v=val: v)
            self._transform = LaneifiedElement(_fac)
            self._children = [_LANE_SIGNAL_ATOM]
        # else: bare for subclass use

    def __call__(self) -> LaneFrozenSignal:
        frozen = self._transform.apply(self._children)
        val = frozen(Event(payload=None))
        return LaneFrozenSignal(val)

    # Already lane-constant — no work needed.
    def to_lane_signal(self, event: Any = None) -> 'LaneSignal':
        return self

    def to_multilane_signal(self, event: Any = None) -> 'MultiLaneSignal':
        if event is None:
            event = Event(payload=None)
        t, c = self._transform.to_lane_parts(self._children, event)
        return MultiLaneSignal(_transform=t, _children=c)


class MultiLaneSignal(LaneSignal):
    """Constant across the entire multilane.

    Construction
    ------------
    * ``MultiLaneSignal(factory)``  — *factory* returns a ``MultiLaneFrozenSignal``.
    * Internal: ``MultiLaneSignal(_transform=…, _children=…)``
    """

    def __init__(self,
                 factory: Callable[[], MultiLaneFrozenSignal] | None = None, *,
                 _transform: SignalTransformBase | None = None,
                 _children: list[Any] | None = None):
        EventSignal.__init__(self)  # skip LaneSignal, bare
        if _transform is not None:
            self._transform = _transform
            self._children = _children if _children is not None else []
        elif factory is not None:
            def _fac(f=factory):
                val = f()()
                return FrozenSignal(lambda _e, v=val: v)
            self._transform = LaneifiedElement(_fac)
            self._children = [_MULTILANE_SIGNAL_ATOM]
        # else: bare

    def __call__(self) -> MultiLaneFrozenSignal:
        frozen = self._transform.apply(self._children)
        val = frozen(Event(payload=None))
        return MultiLaneFrozenSignal(val)

    # Already multilane-constant — no work needed.
    def to_lane_signal(self, event: Any = None) -> 'LaneSignal':
        return self

    def to_multilane_signal(self, event: Any = None) -> 'MultiLaneSignal':
        return self


# Sentinel atoms for leaf children (mirrors C++ SignalAtom hierarchy)
_SIGNAL_ATOM = object()
_LANE_SIGNAL_ATOM = object()
_MULTILANE_SIGNAL_ATOM = object()


# ============================================================================
#  Type checks
# ============================================================================

def is_signal(x: Any) -> bool:
    """``AnySignalArg`` — is *x* any kind of signal?"""
    return isinstance(x, EventSignal)


def is_event_signal(x: Any) -> bool:
    """Alias for ``is_signal``."""
    return isinstance(x, EventSignal)


def is_lane_signal(x: Any) -> bool:
    """``LaneSignalArg`` — is *x* a LaneSignal or MultiLaneSignal?"""
    return isinstance(x, LaneSignal)


def is_multilane_signal(x: Any) -> bool:
    """``MultiLaneSignalArg`` — is *x* a MultiLaneSignal?"""
    return isinstance(x, MultiLaneSignal)


def is_event_only_signal(x: Any) -> bool:
    """``EventOnlySignalArg`` — EventSignal but not LaneSignal."""
    return isinstance(x, EventSignal) and not isinstance(x, LaneSignal)


def is_lane_only_signal(x: Any) -> bool:
    """``LaneOnlySignalArg`` — LaneSignal but not MultiLaneSignal."""
    return isinstance(x, LaneSignal) and not isinstance(x, MultiLaneSignal)


def is_varying_signal(x: Any) -> bool:
    """``VaryingSignalArg`` — any signal but not MultiLaneSignal."""
    return isinstance(x, EventSignal) and not isinstance(x, MultiLaneSignal)


# Backward-compat alias
is_constant_signal = is_multilane_signal


# ============================================================================
#  SignalTransform
# ============================================================================

class SignalTransform(SignalTransformBase):
    """Family of morphisms on signals, parameterised by input arity.

    The ``__call__`` method dispatches to the appropriate signal level:

    * All MultiLaneSignal → MultiLaneSignal
    * All LaneSignal+, at least one LaneOnly → LaneSignal
    * At least one EventOnly → EventSignal

    Parameters
    ----------
    fn : callable
        ``(Signal, Signal, …) → FrozenSignal``  — one arg per input signal.
    arity : int
        Number of expected input signals (checked at apply time).
    """

    def __init__(self, fn: Callable[..., FrozenSignal], arity: int):
        self._fn = fn
        self._arity = arity
        self._metadata: Any = None

    # -- apply --
    def apply(self, children: list[Any]) -> FrozenSignal:
        if len(children) != self._arity:
            raise RuntimeError(
                f"SignalTransform: expected {self._arity} children, got {len(children)}")
        return self._fn(*children)

    # -- to_lane_parts --
    def to_lane_parts(self, children: list[Any], event: Any) -> tuple[SignalTransformBase, list[Any]]:
        new_children: list[Any] = []
        for child in children:
            if isinstance(child, EventSignal):
                new_children.append(child.to_multilane_signal(event))
            else:
                new_children.append(child)
        return (self.clone(), new_children)

    # -- clone --
    def clone(self) -> SignalTransform:
        return SignalTransform(self._fn, self._arity)

    # -- operator() : three-way dispatch --
    def __call__(self, *args: Any) -> EventSignal | LaneSignal | MultiLaneSignal:
        """Apply the transform to signal (or raw-value) arguments.

        Returns MultiLaneSignal when all inputs are multilane or non-signal,
        LaneSignal when all inputs are at least lane-level (with at least one
        lane-only), otherwise EventSignal.
        """
        normalized = [self._normalize(a) for a in args]
        if len(normalized) != self._arity:
            raise RuntimeError(
                f"SignalTransform: expected {self._arity} args, got {len(normalized)}")

        all_multilane = all(is_multilane_signal(s) for s in normalized)
        if all_multilane:
            return MultiLaneSignal(_transform=self.clone(), _children=list(normalized))

        all_lane = all(is_lane_signal(s) for s in normalized)
        if all_lane:
            return LaneSignal(_transform=self.clone(), _children=list(normalized))

        return EventSignal(_transform=self.clone(), _children=list(normalized))

    # -- Constant factories --
    @staticmethod
    def Constant(value_or_signal: Any) -> MultiLaneSignal:
        """Wrap a plain value (or Signal) in a MultiLaneSignal."""
        if isinstance(value_or_signal, MultiLaneSignal):
            return value_or_signal
        if isinstance(value_or_signal, EventSignal):
            return value_or_signal.to_multilane_signal()
        # Plain value → MultiLaneSignal via MultiLaneElement
        val = value_or_signal
        return MultiLaneSignal(
            _transform=MultiLaneElement(val),
            _children=[_MULTILANE_SIGNAL_ATOM],
        )

    # -- internal --
    @staticmethod
    def _normalize(arg: Any) -> EventSignal:
        """Ensure *arg* is a signal, wrapping plain values in Constant."""
        if isinstance(arg, EventSignal):
            return arg
        return SignalTransform.Constant(arg)

    @property
    def metadata(self) -> Any:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Any) -> None:
        self._metadata = value


# ============================================================================
#  Leaf transforms
# ============================================================================

class MultiLaneElement(SignalTransformBase):
    """Leaf transform for multilane-constant signals — holds a baked value."""

    __slots__ = ("_value",)

    def __init__(self, value: Any):
        self._value = value

    def apply(self, children: list[Any]) -> FrozenSignal:
        v = self._value
        return FrozenSignal(lambda _e, v=v: v)

    def to_lane_parts(self, children: list[Any], event: Any) -> tuple[SignalTransformBase, list[Any]]:
        return (self.clone(), children)

    def clone(self) -> MultiLaneElement:
        return MultiLaneElement(self._value)


class LaneifiedElement(SignalTransformBase):
    """Produced by ``Element.to_lane_parts``.

    Holds the original varying factory.  Each ``apply()`` re-calls the factory
    and collapses to an event-independent value.  Different freezes can produce
    different values (e.g. random signals), but within one freeze the value is
    event-independent — exactly the lane-constant contract.
    """

    __slots__ = ("_factory",)

    def __init__(self, factory: Callable[[], FrozenSignal]):
        self._factory = factory

    def apply(self, children: list[Any]) -> FrozenSignal:
        val = self._factory()(Event(payload=None))
        return FrozenSignal(lambda _e, v=val: v)

    def to_lane_parts(self, children: list[Any], event: Any) -> tuple[SignalTransformBase, list[Any]]:
        return (self.clone(), children)

    def clone(self) -> LaneifiedElement:
        return LaneifiedElement(self._factory)


class Element(SignalTransformBase):
    """Leaf transform for varying signals — holds a factory ``() → FrozenSignal``."""

    __slots__ = ("_factory",)

    def __init__(self, factory: Callable[[], FrozenSignal]):
        self._factory = factory

    def apply(self, children: list[Any]) -> FrozenSignal:
        return self._factory()

    def to_lane_parts(self, children: list[Any], event: Any) -> tuple[SignalTransformBase, list[Any]]:
        return (
            LaneifiedElement(self._factory),
            [_MULTILANE_SIGNAL_ATOM],
        )

    def clone(self) -> Element:
        return Element(self._factory)


# ============================================================================
#  Pushforward
# ============================================================================

class Pushforward(SignalTransform):
    """Lifts a plain value-level function into a SignalTransform.

    ``Pushforward(fn, arity)`` where *fn* is ``(v1, v2, …) → result``.
    """

    def __init__(self, fn: Callable[..., Any], arity: int):
        def signal_fn(*signals: EventSignal) -> FrozenSignal:
            frozen = [s() for s in signals]
            def evaluate(event: Any) -> Any:
                return fn(*(f(event) for f in frozen))
            return FrozenSignal(evaluate)
        super().__init__(signal_fn, arity)
        self._value_fn = fn

    def clone(self) -> Pushforward:
        return Pushforward(self._value_fn, self._arity)


# ============================================================================
#  CastTransform
# ============================================================================

class CastTransform(SignalTransform):
    """Signal transform that casts values via a target type constructor."""

    def __init__(self, target_type: type):
        self._target_type = target_type
        def signal_fn(sig: EventSignal) -> FrozenSignal:
            frozen = sig()
            def evaluate(event: Any) -> Any:
                return target_type(frozen(event))
            return FrozenSignal(evaluate)
        super().__init__(signal_fn, arity=1)

    def clone(self) -> CastTransform:
        return CastTransform(self._target_type)


# ============================================================================
#  Arithmetic operators on EventSignal (inherited by Lane/MultiLane)
#
#  In C++ these are template overloads.  In Python we use __add__ etc.
#  The level-propagation rule is enforced: if BOTH operands are
#  MultiLaneSignal, the result is MultiLaneSignal; if both are at least
#  LaneSignal, the result is LaneSignal; otherwise EventSignal.
# ============================================================================

def _make_binop(op: Callable[[Any, Any], Any]) -> Callable:
    """Build a binary arithmetic helper for Signals."""
    def _binop(a: EventSignal, b: Any) -> EventSignal:
        pf = Pushforward(op, arity=2)
        if not isinstance(b, EventSignal):
            b = SignalTransform.Constant(b)
        return pf(a, b)
    return _binop


def _make_rbinop(op: Callable[[Any, Any], Any]) -> Callable:
    """Build a reflected binary op (e.g. ``5 + signal``)."""
    def _rbinop(self: EventSignal, other: Any) -> EventSignal:
        if isinstance(other, EventSignal):
            return NotImplemented
        pf = Pushforward(op, arity=2)
        return pf(SignalTransform.Constant(other), self)
    return _rbinop


def _safe_div(x: Any, y: Any) -> Any:
    return x / y if y else type(x)()

def _safe_floordiv(x: Any, y: Any) -> Any:
    return x // y if y else type(x)()

def _safe_mod(x: Any, y: Any) -> Any:
    return x % y if y else type(x)()


# Attach operators to EventSignal (inherited by LaneSignal, MultiLaneSignal)
EventSignal.__add__  = _make_binop(_op.add)
EventSignal.__radd__ = _make_rbinop(_op.add)
EventSignal.__sub__  = _make_binop(_op.sub)
EventSignal.__rsub__ = _make_rbinop(_op.sub)
EventSignal.__mul__  = _make_binop(_op.mul)
EventSignal.__rmul__ = _make_rbinop(_op.mul)
EventSignal.__truediv__  = _make_binop(_safe_div)
EventSignal.__rtruediv__ = _make_rbinop(_safe_div)
EventSignal.__floordiv__  = _make_binop(_safe_floordiv)
EventSignal.__rfloordiv__ = _make_rbinop(_safe_floordiv)
EventSignal.__mod__  = _make_binop(_safe_mod)
EventSignal.__rmod__ = _make_rbinop(_safe_mod)

def _signal_neg(self: EventSignal) -> EventSignal:
    return Pushforward(_op.neg, arity=1)(self)

EventSignal.__neg__ = _signal_neg


# ============================================================================
#  Backward-compatibility aliases
# ============================================================================

Signal = EventSignal
ConstantSignal = MultiLaneSignal
ConstantFrozenSignal = MultiLaneFrozenSignal
ConstantElement = MultiLaneElement
ConstantifiedElement = LaneifiedElement
_CONSTANT_SIGNAL_ATOM = _MULTILANE_SIGNAL_ATOM
