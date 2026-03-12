"""mtpl.core — Event, Signal, Morphism, Source."""
from mtpl.core.event import Event, Lane, MultiLane
from mtpl.core.signal import (
    # frozen signals
    FrozenSignal,
    LaneFrozenSignal,
    MultiLaneFrozenSignal,
    # signal factories
    EventSignal,
    LaneSignal,
    MultiLaneSignal,
    # transforms
    SignalTransformBase,
    SignalTransform,
    Element,
    MultiLaneElement,
    LaneifiedElement,
    Pushforward,
    CastTransform,
    # predicates
    is_signal,
    is_event_signal,
    is_lane_signal,
    is_multilane_signal,
    is_event_only_signal,
    is_lane_only_signal,
    is_varying_signal,
    # backward-compat aliases
    Signal,
    ConstantSignal,
    ConstantFrozenSignal,
    ConstantElement,
    ConstantifiedElement,
    is_constant_signal,
)
from mtpl.core.morphism import (
    Variadic,
    Preserving,
    Morphism,
    EventLeaf,
    LaneLeaf,
    MultiLaneLeaf,
    Compose,
    Tensor,
    Project,
    Identity,
    Sink,
    move_morphism,
)
from mtpl.core.source import (
    PrimitiveSource,
    Source,
    apply,
    merge,
)

__all__ = [
    # event
    "Event", "Lane", "MultiLane",
    # signal — frozen
    "FrozenSignal", "LaneFrozenSignal", "MultiLaneFrozenSignal",
    "ConstantFrozenSignal",
    # signal — factories
    "EventSignal", "LaneSignal", "MultiLaneSignal",
    "Signal", "ConstantSignal",
    # signal — transforms
    "SignalTransformBase", "SignalTransform",
    "Element", "MultiLaneElement", "LaneifiedElement",
    "ConstantElement", "ConstantifiedElement",
    "Pushforward", "CastTransform",
    # signal — predicates
    "is_signal", "is_event_signal", "is_lane_signal", "is_multilane_signal",
    "is_event_only_signal", "is_lane_only_signal", "is_varying_signal",
    "is_constant_signal",
    # morphism
    "Variadic", "Preserving",
    "Morphism", "EventLeaf", "LaneLeaf", "MultiLaneLeaf",
    "Compose", "Tensor", "Project", "Identity", "Sink",
    "move_morphism",
    # source
    "PrimitiveSource", "Source", "apply", "merge",
]
