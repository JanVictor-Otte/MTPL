"""
TimedEvent — Event with a float timestamp.

Mirrors the C++ TimedEvent<P> + free functions ``join()`` and ``evaluate()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mtpl.core.event import Event
from mtpl.core.morphism import MultiLaneLeaf, Variadic
from mtpl.core.source import Source


# ============================================================================
#  TimedEvent
# ============================================================================

@dataclass
class TimedEvent(Event):
    """Event with a ``time`` field (float seconds)."""
    time: float = 0.0

    def __init__(self, time: float = 0.0, payload: Any = None):
        super().__init__(payload=payload)
        self.time = time


# ============================================================================
#  join — collapse N lanes into 1, sorted by time
# ============================================================================

def join() -> MultiLaneLeaf:
    """MultiLaneLeaf that collapses all lanes into one, sorted by time."""
    def _join_fn(lanes: list[list]) -> list[list]:
        joined: list = []
        for lane in lanes:
            joined.extend(lane)
        joined.sort(key=lambda e: e.time)
        return [joined]

    return MultiLaneLeaf(
        fn=_join_fn,
        in_arity=Variadic,
        out_arity_fn=lambda _n: 1,
    )


# ============================================================================
#  evaluate — run N loops, return flat time-sorted lane
# ============================================================================

def evaluate(src: Source, period: float, loops: int = 1) -> list:
    """Run *src* for *loops* iterations, offset by *period*, return sorted lane."""
    joined: list = []
    for i in range(loops):
        result = src()
        offset = i * period
        for lane in result:
            for e in lane:
                joined.append(TimedEvent(time=e.time + offset, payload=e.payload))
    joined.sort(key=lambda e: e.time)
    return joined
