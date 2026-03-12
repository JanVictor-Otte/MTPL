"""
Morphism system — MultiLane → MultiLane transformations with arity tracking.

Mirrors the C++ Morphism<E> hierarchy.  Templates are replaced by plain
Python classes; arity checks and composability validation are preserved.

Key classes
-----------
Morphism        — abstract base (clone, in_arity, out_arity, apply)
EventLeaf       — per-event transformation with frozen signals
LaneLeaf        — per-lane transformation with lane-frozen signals
MultiLaneLeaf   — whole-multilane transformation with frozen constant signals
Compose         — sequential chaining (flattens nested Composes)
Tensor          — parallel execution (flattens nested Tensors)
Project         — signal-parameterised lane selection
Identity        — pass-through
Sink            — dead-end morphism (out_arity 0)
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from mtpl.core.event import Event
from mtpl.core.signal import (
    EventSignal,
    LaneSignal,
    MultiLaneSignal,
    SignalTransform,
    is_signal,
    is_lane_signal,
    is_multilane_signal,
    # backward compat
    Signal,
    ConstantSignal,
    is_constant_signal,
)

# ============================================================================
#  Arity descriptors
# ============================================================================

Variadic: Optional[int] = None
"""Morphism accepts any number of input lanes."""

Preserving: Callable[[int], int] = lambda n: n
"""Output arity equals input arity."""

_UNRESOLVED = -1


# ============================================================================
#  Morphism (abstract base)
# ============================================================================

class Morphism(ABC):
    """Abstract base for ``MultiLane → MultiLane`` transformations.

    Subclasses must override ``clone``, ``in_arity``, ``out_arity``, and
    the protected ``_apply`` method.
    """

    def __init__(self) -> None:
        self._signals: list[Any] = []
        self._in_arity: Optional[int] = None
        self._out_arity_fn: Callable[[int], int] | None = None
        self._fn: Any = None
        self._children: list[Morphism] = []
        self._metadata: Any = None

    # -- abstract --
    @abstractmethod
    def clone(self) -> Morphism:
        """Deep copy preserving derived type."""

    @abstractmethod
    def in_arity(self) -> Optional[int]:
        """Expected input arity, or ``None`` for variadic."""

    @abstractmethod
    def out_arity(self, in_: int) -> int:
        """Output arity given *in_* input lanes."""

    @abstractmethod
    def _apply(self, lanes: list[list]) -> list[list]:
        """Core transformation — override in subclass."""

    # -- public API --
    def __call__(self, in_: Any) -> Any:
        """Apply to a MultiLane *or* a Source / PrimitiveSource.

        Arity is checked for MultiLane input.  Source / PrimitiveSource
        handling is in ``source.py`` but we support it here via duck-typing.
        """
        # Import here to avoid circular dependency
        from mtpl.core.source import Source, PrimitiveSource
        if isinstance(in_, PrimitiveSource):
            return Source(in_, morphism=self)
        if isinstance(in_, Source):
            return Source(in_, morphism=self)
        # Must be a MultiLane (list of lists)
        expected = self.in_arity()
        if expected is not None and len(in_) != expected:
            raise RuntimeError(
                f"Morphism: expected {expected} input lanes but got {len(in_)}")
        return self._apply(in_)

    def on_resolve(self, concrete_in: int) -> None:
        """Called after arity propagation — override for e.g. Project."""

    def resolve_arity(self, concrete_in: int) -> None:
        if concrete_in == _UNRESOLVED:
            raise RuntimeError("resolve_arity: cannot resolve")
        expected = self.in_arity()
        if expected is not None and expected != concrete_in:
            raise RuntimeError("Morphism: arity mismatch")
        self.on_resolve(concrete_in)


# ============================================================================
#  EventLeaf
# ============================================================================

class EventLeaf(Morphism):
    """Per-event transformation with frozen (possibly varying) signals.

    Parameters
    ----------
    fn : callable
        ``(event, *signal_values) → event``
    signals : list[Signal]
        Signals to freeze per evaluation; values are passed to *fn*.
    in_arity : int | None
        Expected input lane count (``None`` for variadic).
    """

    def __init__(self, fn: Callable, signals: list[Signal] | None = None, *,
                 in_arity: int | None = Variadic):
        super().__init__()
        self._fn = fn
        self._signals = list(signals) if signals else []
        self._in_arity = in_arity
        self._out_arity_fn = Preserving

    def clone(self) -> EventLeaf:
        c = EventLeaf.__new__(EventLeaf)
        Morphism.__init__(c)
        c._fn = self._fn
        c._signals = list(self._signals)
        c._in_arity = self._in_arity
        c._out_arity_fn = self._out_arity_fn
        return c

    def in_arity(self) -> Optional[int]:
        return self._in_arity

    def out_arity(self, in_: int) -> int:
        return self._out_arity_fn(in_)

    def _apply(self, lanes: list[list]) -> list[list]:
        # Freeze all signals
        frozen = [s() for s in self._signals]
        out = []
        for lane in lanes:
            new_lane = []
            for event in lane:
                vals = [f(event) for f in frozen]
                new_lane.append(self._fn(event, *vals))
            out.append(new_lane)
        return out


# ============================================================================
#  MultiLaneLeaf
# ============================================================================

class MultiLaneLeaf(Morphism):
    """Whole-multilane transformation with frozen constant signals.

    Several construction modes (mirroring C++):

    * **MultiLane → MultiLane**: ``fn(multilane, *params) → multilane``
    * **Lane → Lane** (auto-wrapped): ``fn(lane, *params) → lane``
    * **Lane → MultiLane**: ``fn(lane, *params) → multilane``

    Parameters
    ----------
    fn : callable
        The transformation function.
    signals : list[ConstantSignal]
        Constant signals whose frozen values are passed as extra args.
    in_arity : int | None
        Expected input lane count (``None`` for variadic).
    out_arity_fn : callable | int | None
        Function ``int → int`` or fixed int for output arity.
    mode : str
        ``"multi"`` (default), ``"lane"``, or ``"lane_multi"``.
    """

    def __init__(self, fn: Callable, signals: list | None = None, *,
                 in_arity: int | None = Variadic,
                 out_arity_fn: Callable[[int], int] | int | None = None,
                 mode: str = "multi"):
        super().__init__()
        self._signals = list(signals) if signals else []
        self._in_arity = in_arity
        self._mode = mode

        if mode == "lane":
            # Wrap lane-fn into multi-fn
            lane_fn = fn
            def multi_fn(lanes, *params):
                return [lane_fn(lane, *params) for lane in lanes]
            self._fn = multi_fn
            if out_arity_fn is None:
                self._out_arity_fn = Preserving
            elif isinstance(out_arity_fn, int):
                self._out_arity_fn = lambda _n, o=out_arity_fn: o
            else:
                self._out_arity_fn = out_arity_fn
        elif mode == "lane_multi":
            lane_multi_fn = fn
            def multi_fn(lanes, *params):
                out = []
                for lane in lanes:
                    result = lane_multi_fn(lane, *params)
                    out.extend(result)
                return out
            self._fn = multi_fn
            if out_arity_fn is None:
                self._out_arity_fn = Preserving
            elif isinstance(out_arity_fn, int):
                self._out_arity_fn = lambda _n, o=out_arity_fn: o
            else:
                self._out_arity_fn = out_arity_fn
        else:
            # mode == "multi"
            self._fn = fn
            if out_arity_fn is None:
                self._out_arity_fn = Preserving
            elif isinstance(out_arity_fn, int):
                self._out_arity_fn = lambda _n, o=out_arity_fn: o
            else:
                self._out_arity_fn = out_arity_fn

    def clone(self) -> MultiLaneLeaf:
        c = MultiLaneLeaf.__new__(MultiLaneLeaf)
        Morphism.__init__(c)
        c._fn = self._fn
        c._signals = list(self._signals)
        c._in_arity = self._in_arity
        c._out_arity_fn = self._out_arity_fn
        c._mode = self._mode
        return c

    def in_arity(self) -> Optional[int]:
        return self._in_arity

    def out_arity(self, in_: int) -> int:
        return self._out_arity_fn(in_)

    def _apply(self, lanes: list[list]) -> list[list]:
        # Freeze constant signals
        frozen_vals = [s()() for s in self._signals]
        return self._fn(lanes, *frozen_vals)


# ============================================================================
#  Compose
# ============================================================================

class Compose(Morphism):
    """Sequential chaining of morphisms, flattens nested Composes.

    ``Compose(m1, m2, …)`` means ``m2(m1(input))``.
    """

    def __init__(self, *morphisms: Morphism):
        super().__init__()
        if len(morphisms) < 2:
            raise RuntimeError("Compose requires at least 2 morphisms")

        # Flatten nested Composes and collect for validation
        flat: list[Morphism] = []
        for m in morphisms:
            if isinstance(m, Compose):
                flat.extend(child.clone() for child in m._children)
            else:
                flat.append(m)

        self._validate_composability(flat)
        self._children = flat

    @staticmethod
    def _validate_composability(morphisms: list[Morphism]) -> None:
        for i in range(len(morphisms) - 1):
            f_in = morphisms[i].in_arity()
            g_in = morphisms[i + 1].in_arity()
            if f_in is not None and g_in is not None:
                f_out = morphisms[i].out_arity(f_in)
                if f_out != g_in:
                    raise RuntimeError(
                        f"Compose: incompatible arities — morphism {i} outputs "
                        f"{f_out} lanes but morphism {i+1} expects {g_in}")

    def clone(self) -> Compose:
        c = Compose.__new__(Compose)
        Morphism.__init__(c)
        c._children = [child.clone() for child in self._children]
        return c

    def in_arity(self) -> Optional[int]:
        return self._children[0].in_arity()

    def out_arity(self, in_: int) -> int:
        current = in_
        for child in self._children:
            current = child.out_arity(current)
        return current

    def on_resolve(self, concrete_in: int) -> None:
        current = concrete_in
        for i, child in enumerate(self._children):
            child.resolve_arity(current)
            out = child.out_arity(current)
            if i < len(self._children) - 1:
                next_in = self._children[i + 1].in_arity()
                if next_in is not None and out != next_in:
                    raise RuntimeError(
                        f"Compose: morphism {i} outputs {out} lanes but "
                        f"morphism {i+1} expects {next_in}")
            current = out

    def _apply(self, lanes: list[list]) -> list[list]:
        current = lanes
        for child in self._children:
            current = child(current)
        return current


# ============================================================================
#  Tensor
# ============================================================================

class Tensor(Morphism):
    """Parallel execution of morphisms — each child sees the full input.

    ``Tensor(m1, m2)`` concatenates the outputs of m1 and m2.
    Flattens nested Tensors.
    """

    def __init__(self, *morphisms: Morphism):
        super().__init__()
        if len(morphisms) < 2:
            raise RuntimeError("Tensor requires at least 2 morphisms")

        flat: list[Morphism] = []
        for m in morphisms:
            if isinstance(m, Tensor):
                flat.extend(child.clone() for child in m._children)
            else:
                flat.append(m)

        self._validate_arity_compat(flat)
        self._children = flat

    @staticmethod
    def _validate_arity_compat(morphisms: list[Morphism]) -> None:
        fixed: int | None = None
        for i, m in enumerate(morphisms):
            a = m.in_arity()
            if a is not None:
                if fixed is not None and fixed != a:
                    raise RuntimeError(
                        f"Tensor: incompatible input arities — child {i} "
                        f"expects {a} but another expects {fixed}")
                fixed = a

    def clone(self) -> Tensor:
        c = Tensor.__new__(Tensor)
        Morphism.__init__(c)
        c._children = [child.clone() for child in self._children]
        return c

    def in_arity(self) -> Optional[int]:
        for child in self._children:
            a = child.in_arity()
            if a is not None:
                return a
        return Variadic

    def out_arity(self, in_: int) -> int:
        return sum(child.out_arity(in_) for child in self._children)

    def on_resolve(self, concrete_in: int) -> None:
        for child in self._children:
            child.resolve_arity(concrete_in)

    def _apply(self, lanes: list[list]) -> list[list]:
        out: list[list] = []
        for child in self._children:
            out.extend(child(lanes))
        return out


# ============================================================================
#  Project
# ============================================================================

class Project(Morphism):
    """Signal-parameterised lane selection.

    ``Project(0)`` selects lane 0.  ``Project(0, 2)`` selects lanes 0 and 2.
    Accepts ints or ConstantSignal[int].
    """

    def __init__(self, *indices: int | ConstantSignal):
        super().__init__()
        self._in_arity = Variadic
        self._index_signals: list[ConstantSignal] = []
        for idx in indices:
            if isinstance(idx, list):
                # Accept a single list of ints
                for i in idx:
                    self._add_index(i)
            else:
                self._add_index(idx)
        n = len(self._index_signals)
        self._out_arity_fn = lambda _in, n=n: n

    def _add_index(self, idx: int | ConstantSignal) -> None:
        if isinstance(idx, ConstantSignal):
            self._index_signals.append(idx)
        elif isinstance(idx, int):
            self._index_signals.append(SignalTransform.Constant(idx))
        else:
            raise TypeError(f"Project index must be int or ConstantSignal, got {type(idx)}")

    def clone(self) -> Project:
        c = Project.__new__(Project)
        Morphism.__init__(c)
        c._in_arity = self._in_arity
        c._index_signals = list(self._index_signals)
        c._out_arity_fn = self._out_arity_fn
        return c

    def in_arity(self) -> Optional[int]:
        return self._in_arity

    def out_arity(self, in_: int) -> int:
        return self._out_arity_fn(in_)

    def _apply(self, lanes: list[list]) -> list[list]:
        result: list[list] = []
        for sig in self._index_signals:
            i = sig()()
            if i < 0 or i >= len(lanes):
                raise RuntimeError(
                    f"Project: index {i} out of range [0, {len(lanes)})")
            result.append(lanes[i])
        return result


# ============================================================================
#  Identity
# ============================================================================

class Identity(MultiLaneLeaf):
    """Pass-through morphism."""

    def __init__(self, arity: int | None = Variadic):
        super().__init__(
            fn=lambda lanes: lanes,
            in_arity=arity,
            out_arity_fn=Preserving,
        )

    def clone(self) -> Identity:
        return Identity(self._in_arity)


# ============================================================================
#  LaneLeaf
# ============================================================================

class LaneLeaf(Morphism):
    """Per-lane transformation with lane-frozen signals.

    Like EventLeaf, but signals are frozen **per lane** instead of once for
    the entire multilane.  Accepts ``LaneSignal`` (or ``MultiLaneSignal``).

    Parameters
    ----------
    fn : callable
        ``(event, *signal_values) → event``
    signals : list[LaneSignal]
        Signals to freeze per lane; values are passed to *fn*.
    in_arity : int | None
        Expected input lane count (``None`` for variadic).
    """

    def __init__(self, fn: Callable, signals: list | None = None, *,
                 in_arity: int | None = Variadic):
        super().__init__()
        self._fn = fn
        self._signals = list(signals) if signals else []
        self._in_arity = in_arity
        self._out_arity_fn = Preserving

    def clone(self) -> LaneLeaf:
        c = LaneLeaf.__new__(LaneLeaf)
        Morphism.__init__(c)
        c._fn = self._fn
        c._signals = list(self._signals)
        c._in_arity = self._in_arity
        c._out_arity_fn = self._out_arity_fn
        return c

    def in_arity(self) -> Optional[int]:
        return self._in_arity

    def out_arity(self, in_: int) -> int:
        return self._out_arity_fn(in_)

    def _apply(self, lanes: list[list]) -> list[list]:
        out = []
        for lane in lanes:
            # Freeze signals per-lane (each lane gets a fresh freeze)
            frozen = [s() for s in self._signals]
            new_lane = []
            for event in lane:
                vals = [f(event) for f in frozen]
                new_lane.append(self._fn(event, *vals))
            out.append(new_lane)
        return out


# ============================================================================
#  Sink
# ============================================================================

class Sink(Morphism):
    """Dead-end morphism with out_arity 0.

    Applies a side-effect function and returns an empty multilane.
    Useful for saving data, logging, etc.

    Parameters
    ----------
    fn : callable or None
        ``(multilane) → None``  — side-effect function.
    in_arity : int | None
        Expected input lane count (``None`` for variadic).
    """

    def __init__(self, fn: Callable | None = None, *,
                 in_arity: int | None = Variadic):
        super().__init__()
        self._fn = fn
        self._in_arity = in_arity

    def clone(self) -> Sink:
        return Sink(self._fn, in_arity=self._in_arity)

    def in_arity(self) -> Optional[int]:
        return self._in_arity

    def out_arity(self, in_: int) -> int:
        return 0

    def _apply(self, lanes: list[list]) -> list[list]:
        if self._fn is not None:
            self._fn(lanes)
        return []


# ============================================================================
#  move_morphism helper  (less relevant in Python — no unique_ptr)
# ============================================================================

def move_morphism(m: Morphism) -> Morphism:
    """Return a (shallow) reference — Python GC handles the rest."""
    return m
