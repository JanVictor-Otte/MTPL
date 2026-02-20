#pragma once
#include "mtpl/core/Event.hpp"
#include "mtpl/core/Signal.hpp"
#include <functional>
#include <vector>
#include <optional>
#include <stdexcept>
#include <string>
#include <memory>
#include <any>
#include <type_traits>
#include <concepts>
#include <tuple>

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

// Forward declaration for Morphism::operator()(Source)
template<typename E> class Source;
// Forward declaration for Morphism::operator()(PrimitiveSource)
template<typename E> class PrimitiveSource;

inline const std::optional<int>      Variadic   = std::nullopt;
inline const std::function<int(int)> Preserving = [](int n){ return n; };
static constexpr int kUnresolved = -1;


// Base class for Morphism implementation. Must override these three methods:
// virtual std::optional<int> inArity()
// virtual int outArity(int in)
// protected:
// virtual MultiLane<E> apply(MultiLane<E> in)
template<typename E>
class Morphism {
public:
    Morphism() = default;
    
    // Copy constructor: clone all children
    Morphism(const Morphism& other) 
        : signals_(other.signals_),
          inArity_(other.inArity_),
          outArityFn_(other.outArityFn_),
          fn_(other.fn_),
          metadata_(other.metadata_) {
        for (const auto& child : other.children_)
            children_.push_back(child->clone());
    }
    
    // Move constructor
    Morphism(Morphism&&) = default;
    
    // Copy assignment
    Morphism& operator=(const Morphism& other) {
        if (this != &other) {
            signals_ = other.signals_;
            inArity_ = other.inArity_;
            outArityFn_ = other.outArityFn_;
            fn_ = other.fn_;
            metadata_ = other.metadata_;
            children_.clear();
            for (const auto& child : other.children_)
                children_.push_back(child->clone());
        }
        return *this;
    }
    
    // Move assignment
    Morphism& operator=(Morphism&&) = default;

    virtual ~Morphism() = default;
    
    // Virtual clone for polymorphic copying
    virtual std::unique_ptr<Morphism<E>> clone() const = 0;
    
    // Pure virtual: must override
    virtual std::optional<int> inArity() const = 0;
    virtual int outArity(int in) const = 0;

    // Application operator for MultiLanes
    MultiLane<E> operator()(MultiLane<E> in) const {
        auto expectedIn = inArity();
        if (expectedIn && in.size() != static_cast<size_t>(*expectedIn)) {
            throw std::runtime_error(
                "Morphism: expected " + std::to_string(*expectedIn) +
                " input lanes but got " + std::to_string(in.size()));
        }
        return apply(in);
    }

    // Called after arity propagation — no-op by default, set by e.g. project()
    virtual void onResolve(int conreteIn) const {}

    // Application operator for Source
    Source<E> operator()(const Source<E>& src) const;
    Source<E> operator()(Source<E>&& src) const;
    // Application operator for PrimitiveSource
    Source<E> operator()(PrimitiveSource<E> prim) const;

    void resolveArity(int concreteIn) const {
        if (concreteIn == kUnresolved)
            throw std::runtime_error("resolveArity: cannot resolve");
        
        auto in = inArity();
        if (in && *in != concreteIn)
            throw std::runtime_error("Morphism: arity mismatch");
        
        onResolve(concreteIn);
    }

protected:
    // Optional primitive (used by Signal parameterised Morphisms)
    std::vector<std::any> signals_;  // Type-erased signals
    // Optional primitives (used by leaves, ignored by composites)
    std::optional<int> inArity_; 
    std::function<int(int)> outArityFn_;
    // Function signature varies based on number of signals:
    // 0 signals: MultiLane<E> fn(MultiLane<E>)
    // 1 signal:  MultiLane<E> fn(MultiLane<E>, Signal)
    // N signals: MultiLane<E> fn(MultiLane<E>, Signals...)
    std::any fn_; // Type erased function
    // Optional primitive (used by composites, ignored by leaves)
    std::vector<std::unique_ptr<Morphism<E>>> children_;
    
    // Optional metadata to be used for instance in MTPL-Viz
    std::any metadata_;
    
    // Pure virtual: must override
    virtual MultiLane<E> apply(MultiLane<E> in) const = 0;
};

// ============================================================================
//  MorphismLike concept
// ============================================================================

template<typename T, typename E>
struct is_unique_ptr_to_morphism : std::false_type {};

template<typename U, typename E>
struct is_unique_ptr_to_morphism<std::unique_ptr<U>, E>
    : std::bool_constant<std::is_base_of_v<Morphism<E>, U>> {};

template<typename T, typename E>
concept MorphismLike = std::is_base_of_v<Morphism<E>, std::decay_t<T>>;

template<typename T, typename E>
concept MorphismArg = MorphismLike<T, E> || is_unique_ptr_to_morphism<std::decay_t<T>, E>::value;

// ============================================================================
//  Note: Leaf has been removed - it was just MultiLaneLeaf with 0 parameters
//  Use MultiLaneLeaf<E> instead
// ============================================================================

// ============================================================================
//  EventLeaf — per-event transformation with frozen signals
// ============================================================================

template<typename E, typename... Ss>
class EventLeaf : public Morphism<E> {
public:
    using Fn = std::function<E(E, Ss...)>;

    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (AnySignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    EventLeaf(std::optional<int> in, Fn f, SigArgs&&... sigs) {
        this->fn_ = std::move(f);
        addSignals<Ss...>(std::forward<SigArgs>(sigs)...);
        this->inArity_ = in;
        this->outArityFn_ = Preserving;
    }

    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (AnySignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    EventLeaf(int in, Fn f, SigArgs&&... sigs)
        : EventLeaf(std::optional<int>(in), std::move(f), std::forward<SigArgs>(sigs)...) {}

    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (AnySignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    explicit EventLeaf(Fn f, SigArgs&&... sigs)
        : EventLeaf(Variadic, std::move(f), std::forward<SigArgs>(sigs)...) {}

    std::unique_ptr<Morphism<E>> clone() const override {
        return std::make_unique<EventLeaf>(*this);
    }

    std::optional<int> inArity() const override {
        return this->inArity_;
    }

    int outArity(int in) const override {
        return this->outArityFn_(in);
    }

protected:
    MultiLane<E> apply(MultiLane<E> in) const override {
        const auto* fn_ptr = std::any_cast<Fn>(&this->fn_);
        if (!fn_ptr) throw std::bad_any_cast();
        if (this->signals_.size() != sizeof...(Ss))
            throw std::runtime_error("EventLeaf: signal arity mismatch");

        auto frozen = makeFrozenTuple(std::index_sequence_for<Ss...>{});
        MultiLane<E> out = in;
        for (auto& lane : out) {
            for (auto& e : lane) {
                e = std::apply([&](const auto&... fs) { return (*fn_ptr)(e, fs(e)...); }, frozen);
            }
        }
        return out;
    }

private:
    template<typename S, typename SigArg>
    void addSignal(SigArg&& sig) {
        // Always store as Signal<S, E> (base type)
        if constexpr (std::is_pointer_v<std::decay_t<SigArg>>) {
            this->signals_.push_back(Signal<S, E>(*sig));
        } else {
            this->signals_.push_back(Signal<S, E>(std::forward<SigArg>(sig)));
        }
    }

    template<typename... SignalTypes, typename... SigArgs>
    void addSignals(SigArgs&&... sigs) {
        (addSignal<SignalTypes>(std::forward<SigArgs>(sigs)), ...);
    }

    template<std::size_t... Is>
    std::tuple<FrozenSignal<Ss, E>...> makeFrozenTuple(std::index_sequence<Is...>) const {
        return std::make_tuple(std::any_cast<Signal<Ss, E>>(this->signals_[Is])()...);
    }
};

// ============================================================================
//  MultiLaneLeaf — MultiLane transformation with frozen constant signals
// ============================================================================

template<typename E, typename... Ss>
class MultiLaneLeaf : public Morphism<E> {
public:
    using Fn = std::function<MultiLane<E>(MultiLane<E>, Ss...)>;
    using LaneFn = std::function<Lane<E>(Lane<E>, Ss...)>;
    using LaneMultiFn = std::function<MultiLane<E>(Lane<E>, Ss...)>;

    // MultiLane → MultiLane constructors
    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (ConstantSignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    MultiLaneLeaf(std::optional<int> in, Fn f, SigArgs&&... sigs) {
        this->fn_ = std::move(f);
        addSignals<Ss...>(std::forward<SigArgs>(sigs)...);
        this->inArity_ = in;
        this->outArityFn_ = Preserving;
    }

    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (ConstantSignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    MultiLaneLeaf(int in, Fn f, SigArgs&&... sigs)
        : MultiLaneLeaf(std::optional<int>(in), std::move(f), std::forward<SigArgs>(sigs)...) {}

    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (ConstantSignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    explicit MultiLaneLeaf(Fn f, SigArgs&&... sigs)
        : MultiLaneLeaf(Variadic, std::move(f), std::forward<SigArgs>(sigs)...) {}

    // Lane → Lane constructors (auto-wrap to map over MultiLane)
    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (ConstantSignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    MultiLaneLeaf(std::optional<int> in, LaneFn laneF, SigArgs&&... sigs) {
        // Auto-wrap Lane → Lane to MultiLane → MultiLane
        auto multiF = [laneF](MultiLane<E> in, Ss... params) -> MultiLane<E> {
            MultiLane<E> out;
            for (auto& lane : in) {
                out.push_back(laneF(lane, params...));
            }
            return out;
        };
        this->fn_ = std::move(multiF);
        addSignals<Ss...>(std::forward<SigArgs>(sigs)...);
        this->inArity_ = in;
        this->outArityFn_ = Preserving;  // Lane → Lane preserves arity
    }

    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (ConstantSignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    MultiLaneLeaf(int in, LaneFn laneF, SigArgs&&... sigs)
        : MultiLaneLeaf(std::optional<int>(in), laneF, std::forward<SigArgs>(sigs)...) {}

    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (ConstantSignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    explicit MultiLaneLeaf(LaneFn laneF, SigArgs&&... sigs)
        : MultiLaneLeaf(Variadic, laneF, std::forward<SigArgs>(sigs)...) {}

    // Lane → MultiLane constructors (for lane-splitting operations)
    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (ConstantSignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    MultiLaneLeaf(std::optional<int> in, std::function<int(int)> outFn, LaneMultiFn laneMultiF, SigArgs&&... sigs) {
        // Wrap Lane → MultiLane to MultiLane → MultiLane
        auto multiF = [laneMultiF](MultiLane<E> in, Ss... params) -> MultiLane<E> {
            MultiLane<E> out;
            for (auto& lane : in) {
                auto result = laneMultiF(lane, params...);
                out.insert(out.end(), result.begin(), result.end());
            }
            return out;
        };
        this->fn_ = std::move(multiF);
        addSignals<Ss...>(std::forward<SigArgs>(sigs)...);
        this->inArity_ = in;
        this->outArityFn_ = outFn;  // User-specified output arity function
    }

    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (ConstantSignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    MultiLaneLeaf(int in, int outPerLane, LaneMultiFn laneMultiF, SigArgs&&... sigs)
        : MultiLaneLeaf(std::optional<int>(in), [outPerLane, in](int n){ return n * outPerLane; }, laneMultiF, std::forward<SigArgs>(sigs)...) {}

    template<typename... SigArgs>
    requires (sizeof...(SigArgs) == sizeof...(Ss)) && 
             (ConstantSignalArg<SigArgs, E> && ...) &&
             (std::same_as<ExtractValueType_t<SigArgs, E>, Ss> && ...)
    explicit MultiLaneLeaf(LaneMultiFn laneMultiF, SigArgs&&... sigs) 
        : MultiLaneLeaf(Variadic, Preserving, laneMultiF, std::forward<SigArgs>(sigs)...) {}

    // Additional constructors for 0 signal parameters (formerly Leaf functionality)
    // These allow specifying custom outArityFn when no signals are present
    
    MultiLaneLeaf(std::optional<int> in, std::function<int(int)> outFn, Fn f)
    requires (sizeof...(Ss) == 0) {
        this->fn_ = std::move(f);
        this->inArity_ = in;
        this->outArityFn_ = outFn;
    }

    MultiLaneLeaf(std::optional<int> in, int out, Fn f)
    requires (sizeof...(Ss) == 0) {
        this->fn_ = std::move(f);
        this->inArity_ = in;
        this->outArityFn_ = [out](int){ return out; };
    }

    MultiLaneLeaf(int in, int out, Fn f)
    requires (sizeof...(Ss) == 0) {
        this->fn_ = std::move(f);
        this->inArity_ = in;
        this->outArityFn_ = [out](int){ return out; };
    }

    std::unique_ptr<Morphism<E>> clone() const override {
        return std::make_unique<MultiLaneLeaf>(*this);
    }

    std::optional<int> inArity() const override {
        return this->inArity_;
    }

    int outArity(int in) const override {
        return this->outArityFn_(in);
    }

protected:
    MultiLane<E> apply(MultiLane<E> in) const override {
        const auto* fn_ptr = std::any_cast<Fn>(&this->fn_);
        if (!fn_ptr) throw std::bad_any_cast();
        if (this->signals_.size() != sizeof...(Ss))
            throw std::runtime_error("MultiLaneLeaf: signal arity mismatch");

        auto frozen = makeFrozenTuple(std::index_sequence_for<Ss...>{});
        return std::apply([&](const auto&... fs) { return (*fn_ptr)(in, fs()...); }, frozen);
    }

private:
    template<typename S, typename SigArg>
    void addSignal(SigArg&& sig) {
        // Always store as ConstantSignal<S, E>
        if constexpr (std::is_pointer_v<std::decay_t<SigArg>>) {
            this->signals_.push_back(ConstantSignal<S, E>(*sig));
        } else {
            this->signals_.push_back(ConstantSignal<S, E>(std::forward<SigArg>(sig)));
        }
    }

    template<typename... SignalTypes, typename... SigArgs>
    void addSignals(SigArgs&&... sigs) {
        (addSignal<SignalTypes>(std::forward<SigArgs>(sigs)), ...);
    }

    template<std::size_t... Is>
    std::tuple<ConstantFrozenSignal<Ss, E>...> makeFrozenTuple(std::index_sequence<Is...>) const {
        return std::make_tuple(std::any_cast<ConstantSignal<Ss, E>>(this->signals_[Is])()...);
    }
};

// ============================================================================
//  Compose — chains morphisms sequentially, flattens nested Compose children
// ============================================================================

template<typename E>
class Compose : public Morphism<E> {
public:
    // Default constructor for cloning
    Compose() = default;
    
    std::unique_ptr<Morphism<E>> clone() const override {
        Compose<E> copy;
        for (const auto& child : this->children_)
            copy.children_.push_back(child->clone());
        return std::make_unique<Compose>(std::move(copy));
    }
    
    // Variadic constructor - just takes values, copies them
    template<typename... Ms>
    requires (sizeof...(Ms) >= 2) && (MorphismArg<Ms, E> && ...)
    Compose(Ms&&... morphisms) {
        std::vector<const Morphism<E>*> temp;
        (collectForValidation(temp, morphisms), ...);
        
        validateComposability(temp);
        
        (processAndAdd(std::forward<Ms>(morphisms)), ...);
    }

    std::optional<int> inArity() const override {
        return this->children_[0]->inArity();
    }

    int outArity(int in) const override {
        int current = in;
        for (auto& child : this->children_)
            current = child->outArity(current);
        return current;
    }

    void onResolve(int concreteIn) const override {
        int current = concreteIn;
        for (size_t i = 0; i < this->children_.size(); ++i) {
            auto& child = this->children_[i];
            child->resolveArity(current);
            auto outArity_ = child->outArity(current);
            
            if (i < this->children_.size() - 1) {
                auto& next = this->children_[i + 1];
                auto nextIn = next->inArity();
                if (nextIn && outArity_ != *nextIn) {
                    throw std::runtime_error(
                        "Compose: morphism " + std::to_string(i) + " outputs " +
                        std::to_string(outArity_) + " lanes but morphism " +
                        std::to_string(i + 1) + " expects " + std::to_string(*nextIn));
                }
            }
            
            current = outArity_;
        }
    }

protected:
    MultiLane<E> apply(MultiLane<E> in) const override {
        MultiLane<E> current = in;
        for (auto& child : this->children_)
            current = (*child)(current);
        return current;
    }

private:
    template<typename M>
    requires MorphismLike<M, E> && (!std::is_same_v<std::decay_t<M>, Compose<E>>)
    void collectForValidation(std::vector<const Morphism<E>*>& vec, M& m) const {
        vec.push_back(&m);
    }
    
    void collectForValidation(std::vector<const Morphism<E>*>& vec, const Morphism<E>& m) const {
        vec.push_back(&m);
    }

    template<typename M>
    requires std::is_base_of_v<Morphism<E>, M>
    void collectForValidation(std::vector<const Morphism<E>*>& vec, const std::unique_ptr<M>& m) const {
        if (!m)
            throw std::runtime_error("Compose: null morphism pointer");
        vec.push_back(m.get());
    }
    
    void collectForValidation(std::vector<const Morphism<E>*>& vec, const Compose<E>& c) const {
        for (auto& child : c.children_)
            vec.push_back(child.get());
    }
    
    void validateComposability(std::vector<const Morphism<E>*>& morphisms) const {
        for (size_t i = 0; i < morphisms.size() - 1; ++i) {
            auto fIn = morphisms[i]->inArity();
            auto gIn = morphisms[i + 1]->inArity();
            
            if (fIn && gIn) {
                int fOutput = morphisms[i]->outArity(*fIn);
                if (fOutput != *gIn) {
                    throw std::runtime_error(
                        "Compose: incompatible arities - morphism " + std::to_string(i) +
                        " outputs " + std::to_string(fOutput) + " lanes but morphism " +
                        std::to_string(i + 1) + " expects " + std::to_string(*gIn));
                }
            }
        }
    }
    
    template<typename M>
    requires MorphismLike<M, E> && (!std::is_same_v<std::decay_t<M>, Compose<E>>)
    void processAndAdd(M&& m) {
        this->children_.push_back(std::unique_ptr<Morphism<E>>(new std::decay_t<M>(std::forward<M>(m))));
    }
    
    void processAndAdd(const Morphism<E>& m) {
        this->children_.push_back(m.clone());
    }
    
    void processAndAdd(std::unique_ptr<Morphism<E>> m) {
        this->children_.push_back(std::move(m));
    }
    
    void processAndAdd(Compose<E>&& c) {
        // Flatten: move children from nested Compose
        for (auto& child : c.children_)
            this->children_.push_back(std::move(child));
    }
    
    void processAndAdd(Compose<E>& c) {
        // Flatten: clone children from nested Compose
        for (auto& child : c.children_)
            this->children_.push_back(child->clone());
    }
};



// ============================================================================
//  Tensor - parallel execution of morphisms, flattens nested Tensor children
// ============================================================================

template<typename E>
class Tensor : public Morphism<E> {
public:
    // Default constructor for cloning
    Tensor() = default;
    
    std::unique_ptr<Morphism<E>> clone() const override {
        Tensor<E> copy;
        for (const auto& child : this->children_)
            copy.children_.push_back(child->clone());
        return std::make_unique<Tensor>(std::move(copy));
    }
    
    // Variadic constructor with flattening
    template<typename... Ms>
    requires (sizeof...(Ms) >= 2) && (MorphismArg<Ms, E> && ...)
    Tensor(Ms&&... morphisms) {
        std::vector<const Morphism<E>*> temp;
        (collectForValidation(temp, morphisms), ...);
        
        validateArityCompatibility(temp);
        
        (processAndAdd(std::forward<Ms>(morphisms)), ...);
    }

    std::optional<int> inArity() const override {
        for (auto& child : this->children_) {
            auto childIn = child->inArity();
            if (childIn) return childIn;
        }
        return Variadic;
    }

    int outArity(int in) const override {
        int total = 0;
        for (auto& child : this->children_)
            total += child->outArity(in);
        return total;
    }

    void onResolve(int concreteIn) const override {
        for (auto& child : this->children_)
            child->resolveArity(concreteIn);
    }

protected:
    MultiLane<E> apply(MultiLane<E> in) const override {
        MultiLane<E> out;
        for (auto& child : this->children_) {
            auto result = (*child)(in);
            out.insert(out.end(), result.begin(), result.end());
        }
        return out;
    }

private:
    template<typename M>
    requires MorphismLike<M, E> && (!std::is_same_v<std::decay_t<M>, Tensor<E>>)
    void collectForValidation(std::vector<const Morphism<E>*>& vec, M& m) const {
        vec.push_back(&m);
    }
    
    void collectForValidation(std::vector<const Morphism<E>*>& vec, const Morphism<E>& m) const {
        vec.push_back(&m);
    }

    template<typename M>
    requires std::is_base_of_v<Morphism<E>, M>
    void collectForValidation(std::vector<const Morphism<E>*>& vec, const std::unique_ptr<M>& m) const {
        if (!m)
            throw std::runtime_error("Tensor: null morphism pointer");
        vec.push_back(m.get());
    }
    
    void collectForValidation(std::vector<const Morphism<E>*>& vec, const Tensor<E>& t) const {
        for (auto& child : t.children_)
            vec.push_back(child.get());
    }
    
    void validateArityCompatibility(std::vector<const Morphism<E>*>& morphisms) const {
        std::optional<int> fixedArity;
        
        for (size_t i = 0; i < morphisms.size(); ++i) {
            auto childIn = morphisms[i]->inArity();
            
            if (childIn) {
                if (fixedArity && *fixedArity != *childIn) {
                    throw std::runtime_error(
                        "Tensor: incompatible input arities - child " +
                        std::to_string(i) + " expects " + std::to_string(*childIn) +
                        " but another child expects " + std::to_string(*fixedArity));
                }
                fixedArity = childIn;
            }
        }
    }
    
    template<typename M>
    requires MorphismLike<M, E> && (!std::is_same_v<std::decay_t<M>, Tensor<E>>)
    void processAndAdd(M&& m) {
        this->children_.push_back(std::unique_ptr<Morphism<E>>(new std::decay_t<M>(std::forward<M>(m))));
    }
    
    void processAndAdd(const Morphism<E>& m) {
        this->children_.push_back(m.clone());
    }
    
    void processAndAdd(std::unique_ptr<Morphism<E>> m) {
        this->children_.push_back(std::move(m));
    }
    
    void processAndAdd(Tensor<E>&& t) {
        // Flatten: move children from nested Tensor
        for (auto& child : t.children_)
            this->children_.push_back(std::move(child));
    }
    
    void processAndAdd(Tensor<E>& t) {
        // Flatten: clone children from nested Tensor
        for (auto& child : t.children_)
            this->children_.push_back(child->clone());
    }
};




// ============================================================================
//  Project — signal-parameterized lane selection
// ============================================================================

// Type trait for Project indices
template<typename T, typename E>
concept ProjectIndex = std::is_same_v<std::decay_t<T>, int> || 
                       ConstantSignalArg<T, E>;


template<typename E>
class Project : public Morphism<E> {
public:
    std::unique_ptr<Morphism<E>> clone() const override {
        return std::make_unique<Project>(*this);
    }
    
    // Single int
    explicit Project(int index) {
        this->inArity_ = Variadic;
        this->outArityFn_ = [](int){ return 1; };
        this->signals_.push_back(SignalTransform<int, E>::Constant(index));
    }
    
    // Vector of ints - convert all to ConstantSignal
    explicit Project(std::vector<int> indices) {
        this->inArity_ = Variadic;
        this->outArityFn_ = [n=indices.size()](int){ return (int)n; };
        for (int idx : indices)
            this->signals_.push_back(SignalTransform<int, E>::Constant(idx));
    }
    
    // Variadic: any mix of int and ConstantSignal<int, E> (by value, reference, or pointer)
    template<typename... Indices>
    requires (sizeof...(Indices) >= 1) && (ProjectIndex<Indices, E> && ...)
    explicit Project(Indices&&... indices) {
        this->inArity_ = Variadic;
        this->outArityFn_ = [n=sizeof...(Indices)](int){ return (int)n; };
        (addIndex(std::forward<Indices>(indices)), ...);
    }

    std::optional<int> inArity() const override {
        return this->inArity_;
    }

    int outArity(int in) const override {
        return this->outArityFn_(in);
    }

protected:
    MultiLane<E> apply(MultiLane<E> in) const override {
        MultiLane<E> result;
        
        for (auto& sig : this->signals_) {
            auto& signal = std::any_cast<ConstantSignal<int, E>&>(sig);
            int i = signal()();
            
            if (i < 0 || i >= (int)in.size()) {
                throw std::runtime_error(
                    "project: index " + std::to_string(i) +
                    " out of range [0, " + std::to_string(in.size()) + ")");
            }
            
            result.push_back(in[i]);
        }
        
        return result;
    }

private:
    void addIndex(int idx) {
        this->signals_.push_back(SignalTransform<int, E>::Constant(idx));
    }
    
    template<typename SigArg>
    requires ConstantSignalArg<SigArg, E>
    void addIndex(SigArg&& sig) {
        if constexpr (std::is_pointer_v<std::decay_t<SigArg>>) {
            this->signals_.push_back(*sig);
        } else {
            this->signals_.push_back(std::forward<SigArg>(sig));
        }
    }
};


// ============================================================================
//  Identity
// ============================================================================

template<typename E>
class Identity : public MultiLaneLeaf<E> {
public:
    Identity() : MultiLaneLeaf<E>([](MultiLane<E> in) { return in; }) {}
    Identity(int arity) : MultiLaneLeaf<E>(arity, [](MultiLane<E> in) { return in; }) {}
    Identity(std::optional<int> arity) : MultiLaneLeaf<E>(arity, [](MultiLane<E> in) { return in; }) {}
    
    std::unique_ptr<Morphism<E>> clone() const override {
        return std::make_unique<Identity>(*this);
    }
};

// ============================================================================
//  Helper function to move morphisms into unique_ptr for zero-copy semantics
// ============================================================================

template<typename M, typename E>
requires MorphismLike<M, E>
std::unique_ptr<Morphism<E>> move_morphism(M&& m) {
    return std::make_unique<std::decay_t<M>>(std::move(m));
}

}