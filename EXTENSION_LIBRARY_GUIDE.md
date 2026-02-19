# MTPL Extension Library Guide

## Overview

MTPL provides **zero-intrusion extensibility** for metadata tracking through a global metadata provider hook. Extension libraries (like `mtpl-viz`) can inject their own metadata into signals without modifying MTPL core.

## The Problem

You want signals to track metadata (like operation types, descriptions, visualization hints) but:
- ❌ Don't want to hardcode metadata logic into MTPL core
- ❌ Don't want MTPL to depend on visualization libraries
- ❌ Want extension libraries to remain optional
- ✅ Need clean separation of concerns

## The Solution: Metadata Provider Hook

MTPL calls an optional global **MetadataProvider** function whenever it constructs a signal through `pushforward`, `map`, `apply`, or `constant(signal)`.

### API

```cpp
namespace mtpl {
    // Provider signature: (operation_name, input_signals) -> metadata
    using MetadataProvider = std::function<std::any(const char* operation, 
                                                     const std::vector<std::any>& inputs)>;
    
    // Install your metadata provider (or nullptr to disable)
    void setMetadataProvider(MetadataProvider provider);
    
    // Get current provider (for advanced use)
    MetadataProvider& getMetadataProvider();
}
```

## Example: Building mtpl-viz

Here's how an extension library would use the hook:

### Step 1: Define Your Metadata Structure

```cpp
// mtpl_viz.hpp
#pragma once
#include "mtpl/mtpl.hpp"
#include <string>
#include <vector>

namespace mtpl_viz {
    struct OperationMetadata {
        std::string operation;      // "pushforward", "map", "apply", "constant"
        int input_count;            // Number of input signals
        std::string description;    // Human-readable description
        
        // Add whatever visualization data you need:
        std::string color_hint;
        int depth_level;
        // ... custom fields
    };
}
```

### Step 2: Install Your Provider

```cpp
namespace mtpl_viz {
    void enableTracking() {
        mtpl::setMetadataProvider([](const char* operation, 
                                     const std::vector<std::any>& inputs) {
            OperationMetadata meta;
            meta.operation = operation;
            meta.input_count = static_cast<int>(inputs.size());
            meta.description = std::string(operation) + " with " + 
                             std::to_string(inputs.size()) + " inputs";
            
            // Add visualization hints
            if (std::string(operation) == "pushforward") {
                meta.color_hint = "blue";
            } else if (std::string(operation) == "map") {
                meta.color_hint = "green";
            }
            
            return std::any(meta);
        });
    }
    
    void disableTracking() {
        mtpl::setMetadataProvider(nullptr);
    }
}
```

### Step 3: Use in Your Application

```cpp
#include "mtpl/mtpl.hpp"
#include "mtpl_viz.hpp"

int main() {
    using namespace mtpl;
    
    // Enable visualization metadata tracking
    mtpl_viz::enableTracking();
    
    // Now MTPL automatically populates metadata!
    auto a = constant<int, Event>(5);
    auto b = constant<int, Event>(3);
    
    auto sum = pushforward<int, Event>(
        std::function<int(int,int)>([](int x, int y) { return x + y; }),
        a, b
    );
    
    // Access the metadata populated by mtpl-viz
    auto& meta = sum.metadata();
    if (meta.has_value()) {
        auto op_meta = std::any_cast<mtpl_viz::OperationMetadata>(meta);
        std::cout << "Operation: " << op_meta.description << "\n";
        std::cout << "Color: " << op_meta.color_hint << "\n";
    }
    
    // ... build your visualization from the metadata
    
    return 0;
}
```

## Advanced: Recursive Graph Traversal

Since `inputs_` is also tracked, you can traverse the entire computation graph:

```cpp
namespace mtpl_viz {
    template<typename T, typename E>
    void visualizeGraph(const Signal<T,E>& signal, int depth = 0) {
        std::string indent(depth * 2, ' ');
        
        // Extract metadata if present
        if (signal.metadata().has_value()) {
            auto meta = std::any_cast<OperationMetadata>(signal.metadata());
            std::cout << indent << meta.description << " [" << meta.color_hint << "]\n";
        } else {
            std::cout << indent << "Primitive signal\n";
        }
        
        // Recursively visualize inputs
        for (const auto& input_any : signal.inputs()) {
            // Try to extract as Signal<int,E>, Signal<double,E>, etc.
            // (You'd need type-specific handling or RTTI here)
            if (auto* sig = std::any_cast<Signal<int,E>>(&input_any)) {
                visualizeGraph(*sig, depth + 1);
            }
            // ... handle other types
        }
    }
}
```

## What Operations Are Tracked?

MTPL automatically calls your provider for these operations:

| Function | Operation Name | Inputs Tracked |
|----------|---------------|----------------|
| `pushforward(f, args...)` | `"pushforward"` | All signal arguments |
| `map(f, signal)` | `"map"` | The input signal |
| `apply(f, signal)` | `"apply"` | The tuple signal |
| `constant(signal, dummy)` | `"constant"` | The source signal |

**Not tracked:** `constant(value)` - it's a primitive with no inputs.

## Best Practices

### 1. Thread Safety
The global provider is shared. If you need thread-local tracking:

```cpp
thread_local MetadataProvider local_provider = nullptr;
```

### 2. Performance
The provider is called for **every** signal construction. Keep it fast:

```cpp
// ✅ Good - lightweight metadata
meta.operation = operation;
meta.input_count = inputs.size();

// ❌ Bad - expensive computation in provider
meta.graph_hash = computeExpensiveHash(inputs);  // Do this lazily instead
```

### 3. Graceful Fallback
Always check if metadata exists:

```cpp
if (signal.metadata().has_value()) {
    // Use it
} else {
    // Fallback to default behavior
}
```

### 4. Multiple Extensions
If multiple libraries want to track metadata, they need to coordinate:

```cpp
// Option 1: Chain providers
auto old_provider = getMetadataProvider();
setMetadataProvider([old_provider](const char* op, const auto& inputs) {
    auto my_meta = /* ... */;
    if (old_provider) {
        // Combine or wrap old metadata
    }
    return my_meta;
});

// Option 2: Use a metadata registry pattern (advanced)
```

## Example: Complete mtpl-viz Library

```cpp
// mtpl_viz.hpp
#pragma once
#include "mtpl/mtpl.hpp"
#include <unordered_map>
#include <memory>

namespace mtpl_viz {
    class GraphVisualizer {
    public:
        static void enable() {
            mtpl::setMetadataProvider(&GraphVisualizer::metadataFactory);
        }
        
        static void disable() {
            mtpl::setMetadataProvider(nullptr);
        }
        
    private:
        static std::any metadataFactory(const char* operation, 
                                       const std::vector<std::any>& inputs) {
            // Your visualization logic here
            return OperationMetadata{operation, inputs.size(), /* ... */};
        }
    };
}

// Usage
int main() {
    mtpl_viz::GraphVisualizer::enable();
    
    // Build signal pipeline...
    
    mtpl_viz::GraphVisualizer::renderToGraphviz(my_signal, "pipeline.dot");
    mtpl_viz::GraphVisualizer::renderToHTML(my_signal, "pipeline.html");
    
    return 0;
}
```

## Summary

✅ **MTPL Core**: Provides infrastructure (`metadata_`, `inputs_`, provider hook)  
✅ **MTPL Core**: Remains agnostic of metadata content  
✅ **Extension Libraries**: Inject custom metadata via provider  
✅ **Zero Cost**: If no provider installed, just a fast nullptr check  
✅ **Clean Separation**: No MTPL modifications needed for new extensions

This pattern allows unlimited extensibility while keeping MTPL's core focused and dependency-free!
