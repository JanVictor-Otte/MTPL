"""
Event, Lane, MultiLane — the currency of morphisms.

Event<P> in C++ becomes a simple dataclass wrapper around a payload.
Lane<E> = list[E], MultiLane<E> = list[Lane[E]].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar, Generic, List

# ---------------------------------------------------------------------------
#  Event
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """Pure payload wrapper — no time, no metadata."""
    payload: Any

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return self.payload == other.payload

# ---------------------------------------------------------------------------
#  Type aliases (for documentation; Python lists are untyped at runtime)
# ---------------------------------------------------------------------------

# Lane  = list[Event]       — a sequence of events
# MultiLane = list[Lane]    — multiple parallel lanes

Lane = list          # Lane[E]
MultiLane = list     # list[Lane[E]]
