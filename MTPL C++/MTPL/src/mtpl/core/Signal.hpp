#pragma once
#include <functional>
#include <type_traits>
#include <tuple>
#include <memory>
#include <vector>
#include <any>

// ============================================================================
//  FrozenSignal<T,E>              — evaluator: E → T  (event-level)
//  LaneFrozenSignal<T,E>          — constant within lane, callable without E
//  MultiLaneFrozenSignal<T,E>     — constant across multilane
//  EventSignal<T,E>               — factory: () → FrozenSignal<T,E>
//  LaneSignal<T,E>                — factory: () → LaneFrozenSignal<T,E>
//  MultiLaneSignal<T,E>           — factory: () → MultiLaneFrozenSignal<T,E>
// ============================================================================
namespace mtpl {


// ============================================================================
//  Category-theoretic notes (long form, intentionally verbose)
//
//  1) Objects and morphisms in code
//     - A single EventSignal<T,E> value corresponds to one object in the category.
//     - A SignalTransform<T,E,ValueTypes...> represents a family of morphisms
//       that share the same shape: given inputs of types (ValueTypes...), it
//       yields an output of type T. In code, this is a callable that can be
//       applied to many concrete EventSignal objects of those types.
//     - This is why a SignalTransform is not a single morphism but a structured
//       family of morphisms parameterized by the particular input Signals.
//
//  2) Three categories: ES, LS, MLS
//     - ES: objects are all EventSignals; morphisms are SignalTransforms
//       that produce EventSignals.
//     - LS: objects are LaneSignals; morphisms are SignalTransforms that
//       preserve lane-constancy. LS is a subcategory of ES.
//     - MLS: objects are MultiLaneSignals; morphisms are SignalTransforms that
//       preserve multilane-constancy. MLS is a subcategory of LS.
//
//  3) Laneification and MultiLaneification as functors
//     - toLaneSignal() is the object part of a family of functors
//       F_e: ES → LS, indexed by a dummy event e.
//     - toMultiLaneSignal() is the object part of a functor LS → MLS.
//     - Composing both gives ES → MLS (full freezing).
//
//  4) The restricted subcategory we actually implement
//     - We restrict to a subcategory of ES that has the same objects
//       as ES, but only those morphism-families that preserve the hierarchy:
//       if all inputs are MultiLaneSignals, the output must be a MultiLaneSignal;
//       if all inputs are LaneSignals (or higher), the output is a LaneSignal.
//     - Intuition: we disallow morphisms that break level guarantees.
//       If inputs live in MLS, outputs must land in MLS.
//       If inputs live in LS, outputs must land in LS.
//
//  5) Why this matches the implementation
//     - SignalTransform exposes three overloads: if all inputs are MultiLane,
//       it returns MultiLaneSignal; if at least one LaneOnly, it returns
//       LaneSignal; if at least one EventOnly, it returns EventSignal.
//       This encodes the restriction above and keeps the hierarchy coherent.
//     - The action of F_e on morphisms is not represented directly because a
//       SignalTransform groups many morphisms together. Still, for any concrete
//       morphism picked out by a particular call, the level is preserved.
//
//  6) Actual object model in code
//     - The practical category has objects that are finite collections of
//       EventSignals, and morphisms whose target is always a singleton EventSignal.
//     - This is the "multi-input, single-output" perspective used by the API.
//       The above discussion generalizes immediately to that setting.
// ============================================================================

    
// --- FrozenSignal ---

template<typename T, typename E>
class LaneFrozenSignal;  // forward

template<typename T, typename E>
class MultiLaneFrozenSignal;  // forward

template<typename T, typename E>
class FrozenSignal {
public:
    using Fn = std::function<T(const E&)>;

    FrozenSignal(Fn fn) : fn_(std::move(fn)) {}

    template<typename F,
             typename = std::enable_if_t<
                 !std::is_same_v<std::decay_t<F>, FrozenSignal> &&
                 !std::is_same_v<std::decay_t<F>, LaneFrozenSignal<T,E>> &&
                 !std::is_same_v<std::decay_t<F>, MultiLaneFrozenSignal<T,E>>>>
    FrozenSignal(F&& fn) : fn_(std::forward<F>(fn)) {}

    T operator()(const E& e) const { return fn_(e); }

protected:
    Fn fn_;
};

// Lane-level frozen signal: constant within a lane, callable without event.
// Each freeze (one per lane) may produce a different value.
template<typename T, typename E>
class LaneFrozenSignal : public FrozenSignal<T,E> {
public:
    LaneFrozenSignal(T val)
        : FrozenSignal<T,E>([val](const E&) -> T { return val; })
        , val_(val) {}

    T operator()()         const { return val_; }
    T operator()(const E&) const { return val_; }

protected:
    T val_;
};

// MultiLane-level frozen signal: same value across the entire multilane.
template<typename T, typename E>
class MultiLaneFrozenSignal : public LaneFrozenSignal<T,E> {
public:
    MultiLaneFrozenSignal(T val) : LaneFrozenSignal<T,E>(val) {}

    using LaneFrozenSignal<T,E>::operator();
};

// ============================================================================
//  EventSignal type hierarchy base classes (tag types for concept dispatch)
// ============================================================================

template<typename E>
struct SignalBase {};

template<typename E>
struct LaneSignalBase {};

template<typename E>
struct MultiLaneSignalBase {};


// Singleton atoms stored in inputs_ for leaf signals
template<typename E>
struct SignalAtom : SignalBase<E> {};

template<typename E>
struct LaneSignalAtom : SignalAtom<E>, LaneSignalBase<E> {};

template<typename E>
struct MultiLaneSignalAtom : LaneSignalAtom<E>, MultiLaneSignalBase<E> {};

// --- SignalTransformBase ---
//
// Abstract base that erases ValueTypes. EventSignal<T,E> holds a
// shared_ptr<const SignalTransformBase<T,E>> — this is what makes
// computation and introspection one and the same mechanism.

template<typename T, typename E>
class SignalTransformBase {
public:
    struct LaneParts {
        std::shared_ptr<const SignalTransformBase> transform;
        std::vector<std::any> children;
    };

    virtual ~SignalTransformBase() = default;

    // Given children (type-erased Signals), produce a FrozenSignal.
    virtual FrozenSignal<T,E> apply(const std::vector<std::any>& children) const = 0;

    // Produce the pieces needed to build a MultiLaneSignal from this
    // transform. For tree transforms: same transform + constantified children.
    // For leaf Elements: a new MultiLaneElement + MultiLaneSignalAtom.
    virtual LaneParts toLaneParts(const std::vector<std::any>& children, const E& e) const = 0;

    // Deep copy (preserves derived type).
    virtual std::shared_ptr<const SignalTransformBase> clone() const = 0;
};

// --- EventSignal ---

template<typename T, typename E>
class LaneSignal;  // forward

template<typename T, typename E>
class MultiLaneSignal;  // forward

template<typename T, typename E, typename... ValueTypes>
class SignalTransform;  // forward

template<typename T, typename E>
class Element;  // forward

template<typename T, typename E>
class MultiLaneElement;  // forward

template<typename T, typename E>
class EventSignal : public SignalBase<E> {
public:
    // Leaf constructor: wraps a factory function in an Element transform.
    // Body is deferred to after Element is defined.
    template<typename F,
             typename = std::enable_if_t<
                 !std::is_same_v<std::decay_t<F>, EventSignal> &&
                 !std::is_same_v<std::decay_t<F>, LaneSignal<T,E>> &&
                 !std::is_same_v<std::decay_t<F>, MultiLaneSignal<T,E>>>>
    EventSignal(F&& factory);

    // Computation IS tree traversal — one mechanism.
    FrozenSignal<T,E> operator()() const {
        return transform_->apply(children_);
    }

    // Freeze to lane-constant (deferred — needs LaneSignal complete)
    LaneSignal<T,E> toLaneSignal(E e = {}) const;

    // Freeze to multilane-constant (deferred — needs MultiLaneSignal complete)
    MultiLaneSignal<T,E> toMultiLaneSignal(E e = {}) const;

    // Backward-compat alias
    MultiLaneSignal<T,E> toConstant(E e = {}) const { return toMultiLaneSignal(e); }

    // Introspection — the SAME data used for computation
    const std::vector<std::any>& children() const { return children_; }
    const std::vector<std::any>& inputs() const { return children_; }  // alias
    const std::shared_ptr<const SignalTransformBase<T,E>>& transformPtr() const { return transform_; }
    const std::any& metadata() const { return metadata_; }
    void setMetadata(std::any m) { metadata_ = std::move(m); }

    // Backward-compat: return transform wrapped in any for old test code
    std::any transform() const { return transform_ ? std::any(transform_) : std::any(); }

protected:
    struct internal_tag {};
    EventSignal() = default;
    EventSignal(internal_tag) {}

    std::shared_ptr<const SignalTransformBase<T,E>> transform_;
    std::vector<std::any> children_;
    std::any metadata_;

    template<typename, typename, typename...>
    friend class SignalTransform;

    template<typename, typename>
    friend class Element;

    template<typename, typename>
    friend class MultiLaneElement;

    template<typename, typename>
    friend class LaneSignal;
};

// ============================================================================
//  LaneSignal — lane-constant signal: different per lane, constant within lane
// ============================================================================

template<typename T, typename E>
class LaneSignal : public EventSignal<T,E>, public LaneSignalBase<E> {
public:
    using LaneFn = std::function<LaneFrozenSignal<T,E>()>;

    // Leaf lane-signal from factory — deferred to after element defs.
    LaneSignal(LaneFn fn);

    // Tree-based constructor (produced by SignalTransform).
    LaneSignal(std::shared_ptr<const SignalTransformBase<T,E>> t,
               std::vector<std::any> c)
        : EventSignal<T,E>(typename EventSignal<T,E>::internal_tag{})
    {
        this->transform_ = std::move(t);
        this->children_ = std::move(c);
    }

    // Lane-level freeze: one value per lane.
    LaneFrozenSignal<T,E> operator()() const {
        FrozenSignal<T,E> frozen = this->transform_->apply(this->children_);
        T val = frozen(E{});
        return LaneFrozenSignal<T,E>(val);
    }

    // Already lane-constant — no work needed.
    LaneSignal<T,E> toLaneSignal(E = {}) const { return *this; }

    // Promote to MultiLaneSignal — deferred.
    MultiLaneSignal<T,E> toMultiLaneSignal(E e = {}) const;

protected:
    LaneSignal() = default;
    LaneSignal(typename EventSignal<T,E>::internal_tag tag) : EventSignal<T,E>(tag) {}
};

// ============================================================================
//  MultiLaneSignal — constant across the entire multilane
// ============================================================================

template<typename T, typename E>
class MultiLaneSignal : public LaneSignal<T,E>, public MultiLaneSignalBase<E> {
public:
    // Leaf constant from factory — wraps in MultiLaneElement.
    // Body deferred to after MultiLaneElement is defined.
    using MultiLaneFn = std::function<MultiLaneFrozenSignal<T,E>()>;
    MultiLaneSignal(MultiLaneFn fn);

    // Tree-based constructor: same transform, constantified children.
    MultiLaneSignal(std::shared_ptr<const SignalTransformBase<T,E>> t,
                   std::vector<std::any> c)
        : LaneSignal<T,E>(typename EventSignal<T,E>::internal_tag{})
    {
        this->transform_ = std::move(t);
        this->children_ = std::move(c);
    }

    // Already multilane-constant — no work needed.
    LaneSignal<T,E> toLaneSignal(E = {}) const { return *this; }
    MultiLaneSignal<T,E> toMultiLaneSignal(E = {}) const { return *this; }

    // ONE mechanism: tree-walk, evaluate at E{}.
    MultiLaneFrozenSignal<T,E> operator()() const {
        FrozenSignal<T,E> frozen = this->transform_->apply(this->children_);
        T val = frozen(E{});
        return MultiLaneFrozenSignal<T,E>(val);
    }
};

// ============================================================================
//  EventSignal concept hierarchy
//
//  bare_t<T> strips references, cv-qualifiers, and one pointer level,
//  yielding the underlying value type for concept checking.
//
//  Three-level hierarchy:
//    EventSignal                        — varies per event
//    LaneSignal                          — varies per lane, constant within lane
//    MultiLaneSignal                     — constant across the entire multilane
//
//  Type suffix — matches the type directly (after decay)
//  Arg  suffix — matches any argument form: value, reference, pointer, etc.
// ============================================================================

// Recursively strip references, cv-qualifiers, and all pointer indirections
template<typename T>
struct bare_impl {
    using type = std::decay_t<T>;
};
template<typename T>
struct bare_impl<T*> {
    using type = typename bare_impl<std::decay_t<T>>::type;
};
template<typename T>
using bare_t = typename bare_impl<std::decay_t<T>>::type;

// --- Type concepts (direct type, no pointer unwrapping) ---

template<typename T, typename E>
concept AnySignalType = std::is_base_of_v<SignalBase<E>, std::decay_t<T>>;

template<typename T, typename E>
concept LaneSignalType = std::is_base_of_v<LaneSignalBase<E>, std::decay_t<T>>;

template<typename T, typename E>
concept MultiLaneSignalType = std::is_base_of_v<MultiLaneSignalBase<E>, std::decay_t<T>>;

template<typename T, typename E>
concept EventOnlySignalType = AnySignalType<T, E> && !LaneSignalType<T, E>;

template<typename T, typename E>
concept LaneOnlySignalType = LaneSignalType<T, E> && !MultiLaneSignalType<T, E>;

template<typename T, typename E>
concept VaryingSignalType = AnySignalType<T, E> && !MultiLaneSignalType<T, E>;

// --- Arg concepts (any argument form: value, pointer, reference) ---

template<typename T, typename E>
concept AnySignalArg = std::is_base_of_v<SignalBase<E>, bare_t<T>>;

template<typename T, typename E>
concept LaneSignalArg = std::is_base_of_v<LaneSignalBase<E>, bare_t<T>>;

template<typename T, typename E>
concept MultiLaneSignalArg = std::is_base_of_v<MultiLaneSignalBase<E>, bare_t<T>>;

template<typename T, typename E>
concept EventOnlySignalArg = AnySignalArg<T, E> && !LaneSignalArg<T, E>;

template<typename T, typename E>
concept LaneOnlySignalArg = LaneSignalArg<T, E> && !MultiLaneSignalArg<T, E>;

template<typename T, typename E>
concept VaryingSignalArg = AnySignalArg<T, E> && !MultiLaneSignalArg<T, E>;

// --- Non-signal concept ---

template<typename T, typename E>
concept NonSignal = !AnySignalArg<T, E>;

// --- Aliases for semantic naming (Constant* are backward-compat for MultiLane*) ---

// Type trait to extract the value type T from EventSignal<T,E>, LaneSignal<T,E>,
// or MultiLaneSignal<T,E>
template<typename S, typename E>
struct ExtractValueType { using type = void; };

template<typename T, typename E>
struct ExtractValueType<EventSignal<T,E>, E> { using type = T; };

template<typename T, typename E>
struct ExtractValueType<LaneSignal<T,E>, E> { using type = T; };

template<typename T, typename E>
struct ExtractValueType<MultiLaneSignal<T,E>, E> { using type = T; };

template<typename S, typename E>
using ExtractValueType_t = typename ExtractValueType<bare_t<S>, E>::type;

// --- SignalTransform and subclasses ---

template<typename T, typename E, typename... ValueTypes>
class SignalTransform : public SignalTransformBase<T, E> {
public:
    using SignalFn = std::function<FrozenSignal<T,E>(EventSignal<ValueTypes,E>...)>;

    virtual ~SignalTransform() = default;

    SignalTransform() = default;
    explicit SignalTransform(SignalFn fn) : fn_(std::move(fn)) {}

    // --- apply: extract typed children from vector<any>, call fn_ ---
    FrozenSignal<T,E> apply(const std::vector<std::any>& children) const override {
        return applyImpl(children, std::index_sequence_for<ValueTypes...>{});
    }

    // --- toLaneParts: recursively constantify each child ---
    typename SignalTransformBase<T,E>::LaneParts
    toLaneParts(const std::vector<std::any>& children, const E& e) const override {
        return toLanePartsImpl(children, e, std::index_sequence_for<ValueTypes...>{});
    }

    // --- clone: deep copy preserving derived type ---
    std::shared_ptr<const SignalTransformBase<T,E>> clone() const override {
        return std::make_shared<const SignalTransform>(*this);
    }

    // --- Constant factories ---

    // Constant from non-signal value
    static MultiLaneSignal<T, E> Constant(T value) {
        return MultiLaneSignal<T, E>([value]() {
            return MultiLaneFrozenSignal<T, E>(value);
        });
    }

    // Recursive Constant functor F_e : S → CS
    static MultiLaneSignal<T, E> Constant(EventSignal<T, E> s, E e = {}) {
        return s.toMultiLaneSignal(e);
    }

    // --- operator() ---

    // All args constant-or-nonsignal, none varying → MultiLaneSignal
    template<typename... Args>
    requires ((MultiLaneSignalArg<Args, E> || NonSignal<Args, E>) && ...)
          && (!(VaryingSignalArg<Args, E> || ...))
    MultiLaneSignal<T, E>
    operator()(const Args&... args) const {
        auto normalized = std::make_tuple(normalize(args)...);
        std::vector<std::any> children;
        std::apply([&](const auto&... sigs) {
            (children.push_back(std::any(sigs)), ...);
        }, normalized);
        return MultiLaneSignal<T, E>(clone(), std::move(children));
    }

    // All lane-or-higher or nonsignal, at least one lane-only → LaneSignal
    template<typename... Args>
    requires ((LaneSignalArg<Args, E> || NonSignal<Args, E>) && ...)
          && (LaneOnlySignalArg<Args, E> || ...)
    LaneSignal<T, E>
    operator()(const Args&... args) const {
        auto normalized = std::make_tuple(normalize(args)...);
        std::vector<std::any> children;
        std::apply([&](const auto&... sigs) {
            (children.push_back(std::any(sigs)), ...);
        }, normalized);
        return LaneSignal<T, E>(clone(), std::move(children));
    }

    // At least one event-only signal → EventSignal
    template<typename... Args>
    requires (EventOnlySignalArg<Args, E> || ...)
    EventSignal<T, E>
    operator()(const Args&... args) const {
        auto normalized = std::make_tuple(normalize(args)...);
        std::vector<std::any> children;
        std::apply([&](const auto&... sigs) {
            (children.push_back(std::any(sigs)), ...);
        }, normalized);
        EventSignal<T, E> result(typename EventSignal<T,E>::internal_tag{});
        result.transform_ = clone();
        result.children_ = std::move(children);
        return result;
    }

    // Metadata
    void setMetadata(std::any m) { metadata_ = m; }
    std::any metadata() const { return metadata_; }

protected:
    SignalFn fn_;
    std::any metadata_;

private:
    template<std::size_t... Is>
    FrozenSignal<T,E> applyImpl(const std::vector<std::any>& children,
                                 std::index_sequence<Is...>) const {
        return fn_(std::any_cast<EventSignal<ValueTypes, E>>(children[Is])...);
    }

    template<std::size_t... Is>
    typename SignalTransformBase<T,E>::LaneParts
    toLanePartsImpl(const std::vector<std::any>& children, const E& e,
                        std::index_sequence<Is...>) const {
        std::vector<std::any> const_children;
        // Recursively constantify each child and store as EventSignal<V,E>
        (const_children.push_back(std::any(
            EventSignal<ValueTypes, E>(
                std::any_cast<EventSignal<ValueTypes, E>>(children[Is]).toMultiLaneSignal(e)
            )
        )), ...);
        return { clone(), std::move(const_children) };
    }

    // Normalize an argument: unwrap pointers, wrap NonSignals in Constant
    template<typename Arg>
    static auto normalize(const Arg& arg) {
        if constexpr (std::is_pointer_v<std::decay_t<Arg>>) {
            return normalize(*arg);
        } else if constexpr (AnySignalType<Arg, E>) {
            using V = ExtractValueType_t<Arg, E>;
            return EventSignal<V, E>(static_cast<const EventSignal<V, E>&>(arg));
        } else {
            // NonSignal — wrap in Constant
            using V = std::decay_t<Arg>;
            return EventSignal<V, E>(SignalTransform<V, E>::Constant(arg));
        }
    }
};

// --- MultiLaneElement ---
//
// Leaf transform for constant signals. Holds a baked value T.
// apply() returns MultiLaneFrozenSignal(value), ignoring children.

template<typename T, typename E>
class MultiLaneElement : public SignalTransformBase<T, E> {
public:
    explicit MultiLaneElement(T val) : value_(std::move(val)) {}

    FrozenSignal<T,E> apply(const std::vector<std::any>&) const override {
        T v = value_;
        return FrozenSignal<T,E>([v](const E&) -> T { return v; });
    }

    typename SignalTransformBase<T,E>::LaneParts
    toLaneParts(const std::vector<std::any>& children, const E&) const override {
        // Already constant — return self
        return { clone(), children };
    }

    std::shared_ptr<const SignalTransformBase<T,E>> clone() const override {
        return std::make_shared<const MultiLaneElement>(*this);
    }

private:
    T value_;
};

// --- LaneifiedElement ---
//
// Produced by Element::toLaneParts.  Holds the original varying factory.
// Each apply() re-calls the factory → FrozenSignal → evaluate at E{} → constant.
// This means each freeze draws a fresh sample, but within one freeze the value
// does not depend on the event — exactly the MultiLaneSignal contract.

template<typename T, typename E>
class LaneifiedElement : public SignalTransformBase<T, E> {
public:
    using Factory = std::function<FrozenSignal<T,E>()>;

    explicit LaneifiedElement(Factory f) : factory_(std::move(f)) {}

    FrozenSignal<T,E> apply(const std::vector<std::any>&) const override {
        T val = factory_()(E{});
        return FrozenSignal<T,E>([val](const E&) -> T { return val; });
    }

    typename SignalTransformBase<T,E>::LaneParts
    toLaneParts(const std::vector<std::any>& children, const E&) const override {
        // Already constantified — return self
        return { clone(), children };
    }

    std::shared_ptr<const SignalTransformBase<T,E>> clone() const override {
        return std::make_shared<const LaneifiedElement>(*this);
    }

private:
    Factory factory_;
};

// --- Element Transform ---
//
// Leaf transform for varying signals. Holds a factory () → FrozenSignal<T,E>.
// apply() calls the factory, ignoring children.
// toLaneParts() evaluates the factory at e, produces a MultiLaneElement.

template<typename T, typename E>
class Element : public SignalTransformBase<T, E> {
public:
    using Factory = std::function<FrozenSignal<T,E>()>;

    explicit Element(Factory f) : factory_(std::move(f)) {}

    FrozenSignal<T,E> apply(const std::vector<std::any>&) const override {
        return factory_();
    }

    typename SignalTransformBase<T,E>::LaneParts
    toLaneParts(const std::vector<std::any>&, const E& e) const override {
        // Wrap the factory in a LaneifiedElement: each freeze re-calls
        // the factory and collapses to a constant. Different freezes can
        // produce different values (e.g. random signals), but within one
        // freeze the value is event-independent.
        return {
            std::make_shared<const LaneifiedElement<T,E>>(factory_),
            { std::any(MultiLaneSignalAtom<E>{}) }
        };
    }

    std::shared_ptr<const SignalTransformBase<T,E>> clone() const override {
        return std::make_shared<const Element>(*this);
    }

private:
    Factory factory_;
};

// --- Deferred definitions ---
//
// These need Element, MultiLaneElement, LaneSignal, and MultiLaneSignal to be complete.

// EventSignal leaf constructor: wraps factory in an Element
template<typename T, typename E>
template<typename F, typename>
EventSignal<T,E>::EventSignal(F&& factory) {
    auto f = std::function<FrozenSignal<T,E>()>(std::forward<F>(factory));
    transform_ = std::make_shared<const Element<T, E>>(f);
    children_ = { std::any(SignalAtom<E>{}) };
}

// EventSignal::toLaneSignal — freeze event variation, keep lane variation
template<typename T, typename E>
LaneSignal<T,E> EventSignal<T,E>::toLaneSignal(E e) const {
    auto [t, c] = transform_->toLaneParts(children_, e);
    return LaneSignal<T,E>(std::move(t), std::move(c));
}

// EventSignal::toMultiLaneSignal — freeze all variation
template<typename T, typename E>
MultiLaneSignal<T,E> EventSignal<T,E>::toMultiLaneSignal(E e) const {
    auto [t, c] = transform_->toLaneParts(children_, e);
    return MultiLaneSignal<T,E>(std::move(t), std::move(c));
}



// LaneSignal::toMultiLaneSignal — freeze lane variation
template<typename T, typename E>
MultiLaneSignal<T,E> LaneSignal<T,E>::toMultiLaneSignal(E e) const {
    auto [t, c] = this->transform_->toLaneParts(this->children_, e);
    return MultiLaneSignal<T,E>(std::move(t), std::move(c));
}

// LaneSignal leaf constructor: wraps factory in a LaneifiedElement.
// Each freeze (one per lane) re-calls the factory and collapses to a
// lane-constant value. Different lanes may get different values.
template<typename T, typename E>
LaneSignal<T,E>::LaneSignal(LaneFn fn)
    : EventSignal<T,E>(typename EventSignal<T,E>::internal_tag{})
{
    auto factory = [fn]() -> FrozenSignal<T,E> {
        T val = fn()();
        return FrozenSignal<T,E>([val](const E&) -> T { return val; });
    };
    this->transform_ = std::make_shared<const LaneifiedElement<T,E>>(factory);
    this->children_ = { std::any(LaneSignalAtom<E>{}) };
}

// MultiLaneSignal leaf constructor: wraps factory in a LaneifiedElement.
// Each freeze re-calls the factory and collapses to a constant value,
// so different freezes can produce different values (e.g. random draws)
// while each individual frozen signal is event-independent.
template<typename T, typename E>
MultiLaneSignal<T,E>::MultiLaneSignal(MultiLaneFn fn)
    : LaneSignal<T,E>(typename EventSignal<T,E>::internal_tag{})
{
    // Wrap MultiLaneFn → Element-compatible factory (FrozenSignal)
    auto factory = [fn]() -> FrozenSignal<T,E> {
        T val = fn()();
        return FrozenSignal<T,E>([val](const E&) -> T { return val; });
    };
    this->transform_ = std::make_shared<const LaneifiedElement<T,E>>(factory);
    this->children_ = { std::any(MultiLaneSignalAtom<E>{}) };
}

// --- Pushforward Transform ---
//
// Wraps a value-level function T(ValueTypes...) into a SignalFn that
// freezes each input signal and applies the function to the frozen values.

template<typename T, typename E, typename... ValueTypes>
class Pushforward : public SignalTransform<T, E, ValueTypes...> {
public:
    using ValueFn = std::function<T(ValueTypes...)>;

    Pushforward(ValueFn f)
        : SignalTransform<T, E, ValueTypes...>(
            [f](EventSignal<ValueTypes, E>... signals) -> FrozenSignal<T, E> {
                auto frozen = std::make_tuple(signals()...);
                return [f, frozen](const E& e) -> T {
                    return std::apply([&](const auto&... fs) {
                        return f(fs(e)...);
                    }, frozen);
                };
            }
        ) {}

    std::shared_ptr<const SignalTransformBase<T,E>> clone() const override {
        return std::make_shared<const Pushforward>(*this);
    }
};

// --- CastTransform ---
//
// A SignalTransform that casts Source → Target via static_cast.

template<typename Target, typename E, typename Source>
class CastTransform : public SignalTransform<Target, E, Source> {
public:
    CastTransform()
        : SignalTransform<Target, E, Source>(
            [](EventSignal<Source, E> arg) -> FrozenSignal<Target, E> {
                FrozenSignal<Source, E> frozen = arg();
                return [frozen](const E& e) -> Target {
                    return static_cast<Target>(frozen(e));
                };
            }
        ) {}

    std::shared_ptr<const SignalTransformBase<Target,E>> clone() const override {
        return std::make_shared<const CastTransform>(*this);
    }
};

// --- ApplyTransform (stub for now) ---

template<typename T, typename E, typename... Args>
class ApplyTransform : public SignalTransform<T, E, Args...> {
    // Stub - not fully implemented yet
};




// ============================================================================
//  Arithmetic operators
// ============================================================================

// (EventSignal, EventSignal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
EventSignal<R,E> operator+(EventSignal<S,E> a, EventSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x+y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
EventSignal<R,E> operator-(EventSignal<S,E> a, EventSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x-y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
EventSignal<R,E> operator*(EventSignal<S,E> a, EventSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x*y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
EventSignal<R,E> operator/(EventSignal<S,E> a, EventSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return y?x/y:S{}; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
EventSignal<R,E> operator%(EventSignal<S,E> a, EventSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return y?x%y:S{}; }))(a, b);
}

// (EventSignal, non-EventSignal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
requires NonSignal<T, E>
EventSignal<R,E> operator+(EventSignal<S,E> a, T b) { return a + SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
requires NonSignal<T, E>
EventSignal<R,E> operator-(EventSignal<S,E> a, T b) { return a - SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
requires NonSignal<T, E>
EventSignal<R,E> operator*(EventSignal<S,E> a, T b) { return a * SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
requires NonSignal<T, E>
EventSignal<R,E> operator/(EventSignal<S,E> a, T b) { return a / SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
requires NonSignal<T, E>
EventSignal<R,E> operator%(EventSignal<S,E> a, T b) { return a % SignalTransform<T,E>::Constant(b); }

// (non-EventSignal, EventSignal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
requires NonSignal<S, E>
EventSignal<R,E> operator+(S a, EventSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) + b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
requires NonSignal<S, E>
EventSignal<R,E> operator-(S a, EventSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) - b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
requires NonSignal<S, E>
EventSignal<R,E> operator*(S a, EventSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) * b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
requires NonSignal<S, E>
EventSignal<R,E> operator/(S a, EventSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) / b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
requires NonSignal<S, E>
EventSignal<R,E> operator%(S a, EventSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) % b; }

// Unary minus operator

template<typename T, typename E, typename R = decltype(-std::declval<T>())>
EventSignal<R,E> operator-(EventSignal<T,E> a) {
    return Pushforward<R,E,T>(std::function<R(T)>([](T x){ return -x; }))(a);
}

// MultiLaneSignal arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
MultiLaneSignal<R,E> operator+(MultiLaneSignal<S,E> a, MultiLaneSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x+y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
MultiLaneSignal<R,E> operator-(MultiLaneSignal<S,E> a, MultiLaneSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x-y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
MultiLaneSignal<R,E> operator*(MultiLaneSignal<S,E> a, MultiLaneSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x*y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
MultiLaneSignal<R,E> operator/(MultiLaneSignal<S,E> a, MultiLaneSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return y?x/y:S{}; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
MultiLaneSignal<R,E> operator%(MultiLaneSignal<S,E> a, MultiLaneSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return y?x%y:S{}; }))(a, b);
}

// (MultiLaneSignal, non-EventSignal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
requires NonSignal<T, E>
MultiLaneSignal<R,E> operator+(MultiLaneSignal<S,E> a, T b) { return a + SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
requires NonSignal<T, E>
MultiLaneSignal<R,E> operator-(MultiLaneSignal<S,E> a, T b) { return a - SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
requires NonSignal<T, E>
MultiLaneSignal<R,E> operator*(MultiLaneSignal<S,E> a, T b) { return a * SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
requires NonSignal<T, E>
MultiLaneSignal<R,E> operator/(MultiLaneSignal<S,E> a, T b) { return a / SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
requires NonSignal<T, E>
MultiLaneSignal<R,E> operator%(MultiLaneSignal<S,E> a, T b) { return a % SignalTransform<T,E>::Constant(b); }

// (non-EventSignal, MultiLaneSignal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
requires NonSignal<S, E>
MultiLaneSignal<R,E> operator+(S a, MultiLaneSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) + b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
requires NonSignal<S, E>
MultiLaneSignal<R,E> operator-(S a, MultiLaneSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) - b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
requires NonSignal<S, E>
MultiLaneSignal<R,E> operator*(S a, MultiLaneSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) * b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
requires NonSignal<S, E>
MultiLaneSignal<R,E> operator/(S a, MultiLaneSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) / b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
requires NonSignal<S, E>
MultiLaneSignal<R,E> operator%(S a, MultiLaneSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) % b; }

// Unary minus operator for MultiLaneSignal

template<typename T, typename E, typename R = decltype(-std::declval<T>())>
MultiLaneSignal<R,E> operator-(MultiLaneSignal<T,E> a) {
    return Pushforward<R,E,T>(std::function<R(T)>([](T x){ return -x; }))(a);
}

// ============================================================================
//  LaneSignal arithmetic operators
// ============================================================================

// (LaneSignal, LaneSignal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
LaneSignal<R,E> operator+(LaneSignal<S,E> a, LaneSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x+y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
LaneSignal<R,E> operator-(LaneSignal<S,E> a, LaneSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x-y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
LaneSignal<R,E> operator*(LaneSignal<S,E> a, LaneSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x*y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
LaneSignal<R,E> operator/(LaneSignal<S,E> a, LaneSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return y?x/y:S{}; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
LaneSignal<R,E> operator%(LaneSignal<S,E> a, LaneSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return y?x%y:S{}; }))(a, b);
}

// (LaneSignal, non-EventSignal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
requires NonSignal<T, E>
LaneSignal<R,E> operator+(LaneSignal<S,E> a, T b) { return a + LaneSignal<T,E>(SignalTransform<T,E>::Constant(b)); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
requires NonSignal<T, E>
LaneSignal<R,E> operator-(LaneSignal<S,E> a, T b) { return a - LaneSignal<T,E>(SignalTransform<T,E>::Constant(b)); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
requires NonSignal<T, E>
LaneSignal<R,E> operator*(LaneSignal<S,E> a, T b) { return a * LaneSignal<T,E>(SignalTransform<T,E>::Constant(b)); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
requires NonSignal<T, E>
LaneSignal<R,E> operator/(LaneSignal<S,E> a, T b) { return a / LaneSignal<T,E>(SignalTransform<T,E>::Constant(b)); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
requires NonSignal<T, E>
LaneSignal<R,E> operator%(LaneSignal<S,E> a, T b) { return a % LaneSignal<T,E>(SignalTransform<T,E>::Constant(b)); }

// (non-EventSignal, LaneSignal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
requires NonSignal<S, E>
LaneSignal<R,E> operator+(S a, LaneSignal<T,E> b) { return LaneSignal<S,E>(SignalTransform<S,E>::Constant(a)) + b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
requires NonSignal<S, E>
LaneSignal<R,E> operator-(S a, LaneSignal<T,E> b) { return LaneSignal<S,E>(SignalTransform<S,E>::Constant(a)) - b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
requires NonSignal<S, E>
LaneSignal<R,E> operator*(S a, LaneSignal<T,E> b) { return LaneSignal<S,E>(SignalTransform<S,E>::Constant(a)) * b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
requires NonSignal<S, E>
LaneSignal<R,E> operator/(S a, LaneSignal<T,E> b) { return LaneSignal<S,E>(SignalTransform<S,E>::Constant(a)) / b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
requires NonSignal<S, E>
LaneSignal<R,E> operator%(S a, LaneSignal<T,E> b) { return LaneSignal<S,E>(SignalTransform<S,E>::Constant(a)) % b; }

// Unary minus operator for LaneSignal

template<typename T, typename E, typename R = decltype(-std::declval<T>())>
LaneSignal<R,E> operator-(LaneSignal<T,E> a) {
    return Pushforward<R,E,T>(std::function<R(T)>([](T x){ return -x; }))(a);
}

// (backward-compat aliases added after bulk rename below)

// ============================================================================
//  Backward-compatibility aliases
//  Old names → new primary names (keep existing code compiling)
// ============================================================================
template<typename T, typename E> using Signal              = EventSignal<T, E>;
template<typename T, typename E> using ConstantSignal      = MultiLaneSignal<T, E>;
template<typename T, typename E> using ConstantFrozenSignal = MultiLaneFrozenSignal<T, E>;

template<typename T, typename E>
concept ConstantSignalArg  = MultiLaneSignalArg<T, E>;

template<typename T, typename E>
concept ConstantSignalType = MultiLaneSignalType<T, E>;


}