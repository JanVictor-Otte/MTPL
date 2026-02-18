#pragma once
#include "mtpl/core/Morphism.hpp"
#include <stdexcept>
#include <string>

// ============================================================================
//  Source<E> — outArity is always a concrete int, never Variadic
// ============================================================================
namespace mtpl {
template<typename E>
struct Source {
    int                           outArity;
    std::function<MultiLane<E>()> fn;
};

template<typename E>
Source<E> apply(Source<E> src, Morphism<E> m) {
    m.resolveArity(src.outArity);
    return Source<E>{ m.resolvedOut, [src, m]() { return m.fn(src.fn()); } };
}

template<typename E, typename... Morphisms>
Source<E> apply(Source<E> src, Morphism<E> first, Morphisms... rest) {
    return apply(apply(src, first), rest...);
}

template<typename E>
Source<E> merge(std::vector<Source<E>> sources) {
    if (sources.empty()) throw std::runtime_error("merge: empty");
    int totalOut = 0;
    for (const auto& s : sources) totalOut += s.outArity;
    return Source<E>{
        totalOut,
        [sources]() {
            MultiLane<E> out;
            for (const auto& s : sources) {
                auto r = s.fn();
                out.insert(out.end(), r.begin(), r.end());
            }
            return out;
        }
    };
}
}