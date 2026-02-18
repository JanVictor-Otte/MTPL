#pragma once
#include "Event.hpp"
#include "Signal.hpp"
#include <functional>
#include <vector>
#include <optional>
#include <stdexcept>
#include <string>

// ============================================================================
//  Arity descriptors
//
//  Variadic   — morphism accepts any number of input lanes
//  Preserving — morphism produces exactly as many lanes as it receives
//
//  resolvedIn / resolvedOut:
//    -1 = unresolved (variadic, not yet wired to a source)
//    >= 0 = concrete
// ============================================================================
namespace mtpl {
inline const std::optional<int>      Variadic   = std::nullopt;
inline const std::function<int(int)> Preserving = [](int n){ return n; };

static constexpr int kUnresolved = -1;

template<typename E>
struct Morphism {
    std::optional<int>                        inArity;
    std::function<int(int)>                   outArityFn;
    int                                       resolvedIn  = kUnresolved;
    int                                       resolvedOut = kUnresolved;
    std::vector<Morphism<E>>                  children;
    std::function<MultiLane<E>(MultiLane<E>)> fn;

    // Called after arity propagation — no-op by default, set by e.g. project()
    std::function<void(int)> onResolve = [](int){};

    bool isLeaf()     const { return children.empty(); }
    bool isResolved() const { return resolvedIn != kUnresolved; }

    // Make morphism callable like a function
    MultiLane<E> operator()(MultiLane<E> in) const {
        return fn(in);
    }

    // Non-virtual: checks inArity, propagates concrete arity, calls onResolve
    void resolveArity(int concreteIn) {
        if (concreteIn == kUnresolved)
            throw std::runtime_error("resolveArity: cannot resolve with unresolved input");
        if (inArity && *inArity != concreteIn)
            throw std::runtime_error(
                "resolveArity: morphism expects " + std::to_string(*inArity) +
                " lanes but predecessor produces " + std::to_string(concreteIn));
        resolvedIn  = concreteIn;
        resolvedOut = outArityFn(concreteIn);
        onResolve(concreteIn);
    }
};

// ============================================================================
//  makeLeaf overloads
// ============================================================================

template<typename E>
Morphism<E> makeLeaf(std::optional<int> inArity,
                     std::function<int(int)> outArityFn,
                     std::function<MultiLane<E>(MultiLane<E>)> fn)
{
    // Fixed arity: resolve immediately. Variadic: leave unresolved.
    int resolvedIn  = inArity.has_value() ? *inArity  : kUnresolved;
    int resolvedOut = inArity.has_value() ? outArityFn(*inArity) : kUnresolved;
    return Morphism<E>{ inArity, outArityFn, resolvedIn, resolvedOut, {}, std::move(fn) };
}

template<typename E>
Morphism<E> makeLeaf(int in, int out,
                     std::function<MultiLane<E>(MultiLane<E>)> fn)
{
    std::function<int(int)> outFn = [out](int){ return out; };
    return Morphism<E>{ in, outFn, in, out, {}, std::move(fn) };
}

template<typename E>
Morphism<E> makeLeaf(int in,
                     std::function<MultiLane<E>(MultiLane<E>)> fn)
{
    return Morphism<E>{ in, Preserving, in, in, {}, std::move(fn) };
}

template<typename E>
Morphism<E> makeLeaf(std::function<MultiLane<E>(MultiLane<E>)> fn)
{
    // Variadic preserving — unresolved until wired to a source
    return Morphism<E>{ Variadic, Preserving, kUnresolved, kUnresolved, {}, std::move(fn) };
}

// ============================================================================
//  identity — variadic preserving
// ============================================================================

template<typename E>
Morphism<E> identity() {
    return makeLeaf<E>([](MultiLane<E> in) { return in; });
}

// ============================================================================
//  compose
//
//  If f.resolvedOut is unresolved (f is variadic and not yet wired),
//  defer arity checking — the check will happen at apply() time when
//  the source provides a concrete arity.
// ============================================================================

template<typename E>
Morphism<E> compose(Morphism<E> f, Morphism<E> g) {
    if (f.resolvedOut != kUnresolved)
        g.resolveArity(f.resolvedOut);  // can check now
    // else: defer — will be resolved when apply() calls resolveArity on the chain

    // The composed morphism's outArityFn chains f then g
    auto fOutFn = f.outArityFn;
    auto gOutFn = g.outArityFn;
    std::function<int(int)> composedOutFn = [fOutFn, gOutFn](int n){
        return gOutFn(fOutFn(n));
    };

    int resolvedIn  = f.resolvedIn;
    int resolvedOut = (f.resolvedOut != kUnresolved) ? g.resolvedOut : kUnresolved;

    Morphism<E> result{
        f.inArity, composedOutFn,
        resolvedIn, resolvedOut,
        { f, g },
        [f, g](MultiLane<E> in) { return g(f(in)); }
    };

    // onResolve for the composed morphism: re-resolve both children
    result.onResolve = [f, g](int concreteIn) mutable {
        f.resolveArity(concreteIn);
        g.resolveArity(f.resolvedOut);
    };

    return result;
}

template<typename E>
Morphism<E> compose(Morphism<E> m) { return m; }

template<typename E, typename... Morphisms>
Morphism<E> compose(Morphism<E> first, Morphism<E> second, Morphisms... rest) {
    return compose(compose(first, second), rest...);
}

// ============================================================================
//  tensor
// ============================================================================

template<typename E>
Morphism<E> tensor(std::vector<Morphism<E>> morphisms) {
    if (morphisms.empty()) throw std::runtime_error("tensor: empty");
    if (morphisms.size() == 1) return morphisms[0];

    int sharedIn = morphisms[0].resolvedIn;
    if (sharedIn != kUnresolved)
        for (auto& m : morphisms)
            m.resolveArity(sharedIn);

    int totalOut = kUnresolved;
    if (sharedIn != kUnresolved) {
        totalOut = 0;
        for (const auto& m : morphisms) totalOut += m.resolvedOut;
    }

    auto capturedMorphisms = morphisms;
    std::function<int(int)> outFn = [capturedMorphisms](int n) {
        int total = 0;
        for (const auto& m : capturedMorphisms) total += m.outArityFn(n);
        return total;
    };

    Morphism<E> result{
        morphisms[0].inArity, outFn,
        sharedIn, totalOut,
        morphisms,
        [morphisms](MultiLane<E> in) {
            MultiLane<E> out;
            for (const auto& m : morphisms) {
                auto r = m(in);
                out.insert(out.end(), r.begin(), r.end());
            }
            return out;
        }
    };

    result.onResolve = [morphisms](int concreteIn) mutable {
        for (auto& m : morphisms) m.resolveArity(concreteIn);
    };

    return result;
}

template<typename E, typename... Morphisms>
Morphism<E> tensor(Morphism<E> first, Morphisms... rest) {
    return tensor<E>(std::vector<Morphism<E>>{ first, rest... });
}

// ============================================================================
//  project
// ============================================================================

template<typename E>
Morphism<E> project(std::vector<int> indices) {
    int out = (int)indices.size();
    auto m  = makeLeaf<E>(Variadic,
                          std::function<int(int)>([out](int){ return out; }),
                          [indices](MultiLane<E> in) {
                              MultiLane<E> result;
                              for (int i : indices) result.push_back(in[i]);
                              return result;
                          });
    m.onResolve = [indices](int concreteIn) {
        for (int i : indices)
            if (i < 0 || i >= concreteIn)
                throw std::runtime_error(
                    "project: index " + std::to_string(i) +
                    " out of range [0, " + std::to_string(concreteIn) + ")");
    };
    return m;
}

template<typename E>
Morphism<E> project(int i) {
    return project<E>(std::vector<int>{i});
}

// Dynamic index — ConstantSignal<int,E>, outArity=1
template<typename E>
Morphism<E> project(ConstantSignal<int,E> index) {
    return makeLeaf<E>(
        Variadic,
        std::function<int(int)>([](int){ return 1; }),
        [index](MultiLane<E> in) {
            int i = index()();  // freeze → ConstantFrozenSignal → value, no E needed
            if (i < 0 || i >= (int)in.size())
                throw std::runtime_error(
                    "project: index " + std::to_string(i) +
                    " out of range [0, " + std::to_string(in.size()) + ")");
            return MultiLane<E>{ in[i] };
        });
}

// Dynamic multi-index — vector<ConstantSignal<int,E>>
// outArity = indices.size(), known at construction time
// each output lane gets its source lane chosen fresh each loop
template<typename E>
Morphism<E> project(std::vector<ConstantSignal<int,E>> indices) {
    int out = (int)indices.size();
    return makeLeaf<E>(
        Variadic,
        std::function<int(int)>([out](int){ return out; }),
        [indices](MultiLane<E> in) {
            MultiLane<E> result;
            for (const auto& idx : indices) {
                int i = idx()();  // freeze → ConstantFrozenSignal → value
                if (i < 0 || i >= (int)in.size())
                    throw std::runtime_error(
                        "project: index " + std::to_string(i) +
                        " out of range [0, " + std::to_string(in.size()) + ")");
                result.push_back(in[i]);
            }
            return result;
        });
}

}