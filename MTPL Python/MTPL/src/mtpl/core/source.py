"""
Source system — composite of primitive generators + accumulated morphism.

Mirrors the C++ PrimitiveSource<E> / Source<E> hierarchy.

Key classes
-----------
PrimitiveSource — leaf generator: ``() → MultiLane``
Source          — vector of primitives + accumulated morphism chain

Free functions
--------------
merge   — concatenate primitives from multiple sources, build Tensor
apply   — compose multiple morphisms onto a Source
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from mtpl.core.morphism import (
    Morphism, Compose, Tensor, Project, Identity
)


# ============================================================================
#  PrimitiveSource
# ============================================================================

class PrimitiveSource:
    """Leaf source with a generator function and fixed output arity.

    Parameters
    ----------
    arity : int
        Number of lanes this primitive generates.
    fn : callable
        ``() → MultiLane``
    """

    def __init__(self, arity: int, fn: Callable[[], list[list]]):
        self._out_arity = arity
        self._fn = fn
        self._metadata: Any = None

    def __call__(self) -> list[list]:
        return self._fn()

    def out_arity(self) -> int:
        return self._out_arity


# ============================================================================
#  Source
# ============================================================================

class Source:
    """Composite of primitives + accumulated morphism chain.

    Construction modes (mirrors C++ overloads)
    ------------------------------------------
    * ``Source(primitive)``
    * ``Source(primitive, morphism=m)``
    * ``Source(source, morphism=m)``
    * ``Source(primitives=prims, morphism=m)``
    """

    def __init__(self, first: PrimitiveSource | 'Source' | None = None, *,
                 morphism: Morphism | None = None,
                 primitives: list[PrimitiveSource] | None = None,
                 _raw_morphism: Morphism | None = None):
        self._primitives: list[PrimitiveSource] = []
        self._morphism: Morphism = Identity()
        self._metadata: Any = None

        if primitives is not None:
            # From explicit list of primitives + optional morphism
            self._primitives = list(primitives)
            if _raw_morphism is not None:
                self._morphism = _raw_morphism
            elif morphism is not None:
                morphism.resolve_arity(self.primitives_out_arity())
                self._morphism = morphism.clone()
            return

        if first is None:
            return

        if isinstance(first, PrimitiveSource):
            self._primitives = [first]
            if morphism is not None:
                morphism.resolve_arity(first.out_arity())
                self._morphism = morphism.clone()
            # else keep Identity
        elif isinstance(first, Source):
            src = first
            self._primitives = list(src._primitives)
            if morphism is not None:
                morphism.resolve_arity(src.out_arity())
                self._morphism = Compose(src._morphism.clone(), morphism.clone())
            else:
                self._morphism = src._morphism.clone()
        else:
            raise TypeError(f"Source: unsupported first argument type {type(first)}")

    # -- accessors --
    def primitives_out_arity(self) -> int:
        return sum(p.out_arity() for p in self._primitives)

    def sample_primitives(self) -> list[list]:
        combined: list[list] = []
        for p in self._primitives:
            combined.extend(p())
        return combined

    def __call__(self) -> list[list]:
        return self._morphism(self.sample_primitives())

    @property
    def primitives(self) -> list[PrimitiveSource]:
        return self._primitives

    def morphism(self) -> Morphism:
        return self._morphism

    def out_arity(self) -> int:
        return self._morphism.out_arity(self.primitives_out_arity())


# ============================================================================
#  apply — compose multiple morphisms onto a Source
# ============================================================================

def apply(src: Source, *morphisms: Morphism) -> Source:
    """Compose multiple morphisms onto *src*."""
    if len(morphisms) < 2:
        raise RuntimeError("apply requires at least 2 morphisms")
    composed = Compose(*morphisms)
    return Source(src, morphism=composed)


# ============================================================================
#  merge — concatenate primitives from multiple sources
# ============================================================================

def merge(*sources: Source) -> Source:
    """Concatenate primitives from multiple sources, build Tensor.

    All sources must be of the same type.  Returns a Source whose primitives
    are the concatenation and whose morphism is a Tensor of projected
    per-source morphisms.
    """
    if len(sources) == 0:
        raise RuntimeError("merge requires at least 1 source")

    # Same-type enforcement
    first_type = type(sources[0])
    for i, s in enumerate(sources):
        if type(s) is not first_type:
            raise TypeError(
                f"merge: all sources must have the same type, "
                f"got {first_type.__name__} and {type(s).__name__}")

    if len(sources) == 1:
        return first_type(sources[0])

    # Compute offsets
    offsets = [0]
    for s in sources:
        offsets.append(offsets[-1] + s.primitives_out_arity())

    # Concatenate all primitives
    all_primitives: list[PrimitiveSource] = []
    for s in sources:
        all_primitives.extend(s.primitives)

    # Build projected morphisms
    projected: list[Morphism] = []
    for i, s in enumerate(sources):
        start = offsets[i]
        end = offsets[i + 1]
        indices = list(range(start, end))
        proj = Project(*indices) if len(indices) > 1 else Project(indices[0])
        projected.append(Compose(proj, s.morphism().clone()))

    tensor = Tensor(*projected)

    return first_type(
        Source(primitives=all_primitives, _raw_morphism=tensor)
    )
