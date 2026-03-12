# MTPL – Monoidal Typed Pipeline Library

**MTPL** is a library for composing type-safe, immutable morphisms that transform event streams organized in parallel lanes. It provides an abstraction for building data transformation pipelines with support for event-level and lane-level transformations parameterized by signals. Ideal for real-time signal processing, telemetry ingestion, and any application requiring composable, stateless transformations over ordered event sequences.

## Implementations

| Language | Location | Install |
|----------|----------|---------|
| **C++20** | [`MTPL C++/MTPL/`](MTPL%20C%2B%2B/MTPL/) | Header-only — include `mtpl.hpp` |
| **Python** | [`MTPL Python/MTPL/`](MTPL%20Python/MTPL/) | `pip install mtpl` |

Each implementation has its own README with build instructions and examples.

## Features

- **Composable Morphisms**: Chain transformations using `Compose` and `Tensor` with automatic arity validation
- **Event-Level & Lane-Level Transforms**: Apply per-event or lane-wide transformations parameterized by signals
- **Type-Safe**: Immutable architecture with runtime (Python) or compile-time (C++) safeguards
- **Zero-Copy** (C++): Move semantics avoid deep cloning of temporary morphisms

## License

MIT — see [LICENSE](LICENSE).
