#pragma once
#include "mtpl/core/Morphism.hpp"
#include <stdexcept>
#include <functional>
#include <vector>
#include <memory>
#include <type_traits>


namespace mtpl {
// ============================================================================
//  PrimitiveSource — leaf source with generator function, no morphism
// ============================================================================

template<typename E>
class PrimitiveSource {
public:
    PrimitiveSource(int arity, std::function<MultiLane<E>()> function)
        : outArity_(arity), fn_(function) {}


    MultiLane<E> operator()() {
        return fn_();
    }
    int outArity() const { return outArity_; }

protected:
    std::function<MultiLane<E>()> fn_;
    int outArity_;  // Constant - number of lanes this primitive generates
    std::any metadata_;  // Optional metadata (e.g., for mtpl-viz)
};

// ============================================================================
//  Source — composite of primitives + accumulated morphism
// ============================================================================

template<typename E>
class Source {
public:
    // Default constructor
    Source() = default;
    
    // Copy constructor
    Source(const Source& other)
        : primitives_(other.primitives_),
          morphism_(other.morphism_ ? other.morphism_->clone() : nullptr) {}
    
    // Move constructor
    Source(Source&&) = default;
    
    // Copy assignment
    Source& operator=(const Source& other) {
        if (this != &other) {
            primitives_ = other.primitives_;
            morphism_ = other.morphism_ ? other.morphism_->clone() : nullptr;
        }
        return *this;
    }
    
    // Move assignment
    Source& operator=(Source&&) = default;
    
    // Construct from single PrimitiveSource (upcast)
    explicit Source(PrimitiveSource<E> prim)
        : morphism_(std::make_unique<Identity<E>>()) {
        primitives_.push_back(prim);
    }
    
    // Construct from vector of PrimitiveSource (merge primitives)
    explicit Source(std::vector<PrimitiveSource<E>> prims)
        : morphism_(std::make_unique<Identity<E>>()), primitives_(prims) { }

    // Construct from PrimitiveSource + Morphism (lvalue - clone)
    explicit Source(PrimitiveSource<E> prim, const Morphism<E>& m) {
        m.resolveArity(prim.outArity());
        primitives_.push_back(prim);
        morphism_ = m.clone();
    }
    
    // Construct from PrimitiveSource + unique_ptr (move - no clone)
    explicit Source(PrimitiveSource<E> prim, std::unique_ptr<Morphism<E>> m) {
        m->resolveArity(prim.outArity());
        primitives_.push_back(prim);
        morphism_ = std::move(m);
    }
    
    // Construct from Source + Morphism (lvalue - clone)
    explicit Source(const Source<E>& src, const Morphism<E>& m) {
        m.resolveArity(src.outArity());
        primitives_ = src.primitives();
        morphism_ = std::make_unique<Compose<E>>(src.morphism_->clone(), m);
    }
    
    // Construct from Source + unique_ptr (move - no clone for second morphism)
    explicit Source(const Source<E>& src, std::unique_ptr<Morphism<E>> m) {
        m->resolveArity(src.outArity());
        primitives_ = src.primitives();
        morphism_ = std::make_unique<Compose<E>>(src.morphism_->clone(), std::move(m));
    }
    
    // Construct from Source&& + Morphism (move source morphism - no clone for first)
    explicit Source(Source<E>&& src, const Morphism<E>& m) {
        m.resolveArity(src.outArity());
        primitives_ = std::move(src.primitives_);
        morphism_ = std::make_unique<Compose<E>>(std::move(src.morphism_), m);
    }
    
    // Construct from Source&& + unique_ptr (move both - no cloning)
    explicit Source(Source<E>&& src, std::unique_ptr<Morphism<E>> m) {
        m->resolveArity(src.outArity());
        primitives_ = std::move(src.primitives_);
        morphism_ = std::make_unique<Compose<E>>(std::move(src.morphism_), std::move(m));
    }

    // Helper: sum of primitive output arities
    int primitivesOutArity() const {
        int total = 0;
        for (const auto& prim : primitives_)
            total += prim.outArity();
        return total;
    }

    // Helper: call all primitives and concatenate
    MultiLane<E> samplePrimitives() {
        MultiLane<E> combined;
        for (auto& prim : primitives_) {
            auto lanes = prim();
            combined.insert(combined.end(), lanes.begin(), lanes.end());
        }
        return combined;
    }

    MultiLane<E> operator()() {
        return (*morphism_)(samplePrimitives());
    }
    
    const std::vector<PrimitiveSource<E>>& primitives() const { return primitives_; }
    Morphism<E>* morphism() const { return morphism_.get(); }
    int outArity() const { return morphism_->outArity(primitivesOutArity()); }
protected:
    std::vector<PrimitiveSource<E>> primitives_;
    std::unique_ptr<Morphism<E>> morphism_;
    std::any metadata_;  // Optional metadata (e.g., for mtpl-viz)
};

// ============================================================================
//  Morphism::operator()(Source)
// ============================================================================


template<typename E>
Source<E> Morphism<E>::operator()(const Source<E>& src) const {
    return Source<E>(src, *this);
}
template<typename E>
Source<E> Morphism<E>::operator()(Source<E>&& src) const {
    return Source<E>(std::move(src), *this);
}
template<typename E>
Source<E> Morphism<E>::operator()(PrimitiveSource<E> prim) const {
    return Source<E>(prim, *this);
}


// ============================================================================
//  apply
// ============================================================================

template<typename E, typename... Ms>
requires (sizeof...(Ms) >= 2) && (MorphismLike<Ms, E> && ...)
Source<E> apply(Source<E> src, Ms... morphisms) {
    return Compose<E>(morphisms...)(src);
}

// ============================================================================
//  merge — concatenate primitives, build tensor from offsets
// ============================================================================


template<typename E, typename... Sources>
requires (std::is_same_v<std::decay_t<Sources>, Source<E>> && ...)
Source<E> merge(Sources... sources) {
    if constexpr (sizeof...(Sources) == 0) {
        throw std::runtime_error("merge: empty");
    }
    
    if constexpr (sizeof...(Sources) == 1) {
        return Source<E>(sources...);
    }
    
    // Compute offsets based on primitivesOutArity of each source
    auto computeOffsets = [](const auto&... srcs) {
        std::array<int, sizeof...(srcs) + 1> offsets = {0};
        int idx = 0;
        ((offsets[++idx] = offsets[idx - 1] + srcs.primitivesOutArity()), ...);
        return offsets;
    };
    
    auto offsets = computeOffsets(sources...);
    
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        // Concatenate all primitives
        std::vector<PrimitiveSource<E>> allPrimitives;
        ((allPrimitives.insert(allPrimitives.end(), 
                               sources.primitives().begin(), 
                               sources.primitives().end())), ...);
        
        Source<E> result(std::move(allPrimitives));
        
        // Build projected morphisms using primitive offsets
        auto buildProjectedAt = [&]<std::size_t I>(const auto& src) {
            int startIdx = offsets[I];
            int endIdx = offsets[I + 1];
            std::vector<int> indices;
            for (int i = startIdx; i < endIdx; ++i)
                indices.push_back(i);
            return Compose<E>(Project<E>(indices), *src.morphism());
        };
        
        // Tensor all projected morphisms
        result.morphism_ = std::make_unique<Tensor<E>>(buildProjectedAt.template operator()<Is>(sources)...);
        
        return result;
    }(std::index_sequence_for<Sources...>{});
}
}