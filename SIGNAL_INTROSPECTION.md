# Signal Introspection

## Overview

Signals now support **introspection** - the ability to inspect the computation graph that produced them. This is analogous to how `Morphism` tracks its constituent parts and `Source` remembers its primitives and morphisms.

## Features

### 1. Input Tracking

Every signal constructed through `pushforward`, `map`, `apply`, or `constant(signal)` automatically tracks its input signals:

```cpp
auto a = constant<int, Event>(5);
auto b = constant<int, Event>(3);
auto sum = pushforward<int, Event>(
    std::function<int(int, int)>([](int x, int y) { return x + y; }),
    a, b
);

// Access tracked inputs
const std::vector<std::any>& inputs = sum.inputs();
// inputs.size() == 2
```

### 2. Metadata Attachment

Arbitrary metadata can be attached to any signal for use by visualization or debugging tools:

```cpp
sum.setMetadata(std::string("Addition operation"));
const std::any& meta = sum.metadata();
auto description = std::any_cast<std::string>(meta);
// description == "Addition operation"
```

### 3. Type-Erased Storage

Inputs are stored as `std::vector<std::any>`, allowing signals with different value types to be tracked together:

```cpp
auto int_signal = constant<int, Event>(42);
auto double_signal = constant<double, Event>(3.14);
auto mixed = pushforward<double, Event>(
    std::function<double(int, double)>([](int i, double d) { return i + d; }),
    int_signal, double_signal
);
// mixed.inputs() contains both signals in type-erased form
```

## API

### Signal<T,E>

```cpp
class Signal<T,E> {
public:
    // Introspection API
    const std::any& metadata() const;
    void setMetadata(std::any meta);
    
    const std::vector<std::any>& inputs() const;
    void setInputs(std::vector<std::any> ins);
    
protected:
    std::any metadata_;
    std::vector<std::any> inputs_;
};
```

### ConstantSignal<T,E>

Inherits all introspection capabilities from `Signal<T,E>`:

```cpp
class ConstantSignal<T,E> : public Signal<T,E> {
public:
    // Convenient alias for inputs (all inputs must be ConstantSignals)
    const std::vector<std::any>& constantInputs() const { return this->inputs(); }
};
```

## Automatic Tracking

The following functions automatically populate the `inputs_` vector:

- **`pushforward(f, args...)`** - tracks all signal arguments
- **`map(f, signal)`** - tracks the input signal
- **`apply(f, signal)`** - tracks the tuple signal
- **`constant(signal, dummy)`** - tracks the source signal

Primitive signals created with `constant(value)` have empty input vectors.

## Use Cases

### 1. Computation Graph Visualization

A future `mtpl-viz` library could traverse the signal graph:

```cpp
void visualize(const Signal<T,E>& sig) {
    std::cout << "Signal with " << sig.inputs().size() << " inputs\n";
    if (sig.metadata().has_value()) {
        // Render metadata
    }
    for (const auto& input : sig.inputs()) {
        // Recursively visualize inputs
    }
}
```

### 2. Nested Signal Construction

Example: `randomBernoulli(randomUniform(0.0f, 1.0f))`

```cpp
auto uniform = randomUniform<float, Event>(0.0f, 1.0f);
uniform.setMetadata(std::string("Uniform RNG"));

auto bernoulli = randomBernoulli(uniform);
bernoulli.setMetadata(std::string("Bernoulli with random p"));

// bernoulli.inputs() contains uniform
// Can trace back through the entire random generation pipeline
```

### 3. Pipeline Documentation

```cpp
auto pipeline = buildComplexSignal();
pipeline.setMetadata(json{
    {"description", "Multi-stage signal processing"},
    {"author", "Jan"},
    {"version", "2.0"}
});
```

## Design Notes

- **Type Erasure**: Using `std::any` allows heterogeneous signal storage without template complexity
- **Immutability**: Matches MTPL's philosophy - signals are immutable, metadata/inputs set at construction
- **Zero-Cost When Unused**: The `inputs_` vector is only populated by factory functions; user code can ignore it
- **Consistency**: Mirrors the design of `Morphism` and `Source` which also track their constituents

## Future Work

- **Graph serialization**: Export signal computation graphs to JSON/GraphML
- **Interactive visualization**: Web-based tool to explore signal pipelines
- **Optimization**: Graph analysis to detect redundant computations
- **Debugging**: Stack traces through signal evaluation chains
