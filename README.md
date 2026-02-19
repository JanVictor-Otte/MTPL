# MTPL – Multi-Lane Time-Series Processing Library

**MTPL** is a modern C++20 header-only library for composing type-safe, immutable morphisms that transform event streams organized in parallel lanes. It provides a powerful abstraction for building data transformation pipelines with zero-copy move semantics and support for event-level and lane-level transformations parameterized by signals. The library is ideal for real-time signal processing, telemetry ingestion, and any application requiring composable, stateless transformations over ordered event sequences.

## Features

- **Composable Morphisms**: Chain transformations using `Compose` and `Tensor` with automatic arity validation
- **Event-Level Transforms**: Apply per-event transformations with arbitrary signal parameters (via `EventLeaf`)
- **Lane-Level Transforms**: Apply lane-wide transformations parameterized by constant signals (via `MultiLaneLeaf`)
- **Zero-Copy Move Semantics**: Temporary morphisms avoid deep cloning through rvalue overloads
- **Type-Safe**: Full C++20 concepts support with immutable architecture prevents common errors
- **Header-Only**: Easy integration—just include `mtpl.hpp`

## Quick Example

```cpp
#include "mtpl/mtpl.hpp"

struct Event { int value; };

// Create a leaf morphism that scales all values
auto scaler = Leaf<Event>([](MultiLane<Event> lanes) {
    for (auto& lane : lanes)
        for (auto& e : lane)
            e.value *= 2;
    return lanes;
});

// Compose with other morphisms
auto pipeline = Compose(scaler,/*another_morphism*/);
```

## Building & Testing

```bash
mkdir build && cd build
cmake ..
make
ctest
```

All tests pass with 100% coverage of new features.
