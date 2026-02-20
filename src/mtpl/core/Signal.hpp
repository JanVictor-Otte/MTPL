#pragma once
#include <functional>
#include <type_traits>
#include <tuple>
#include <memory>
#include <vector>
#include <any>

// ============================================================================
//  FrozenSignal<T,E>        — evaluator: E → T
//  ConstantFrozenSignal<T,E>— evaluator constant in E, callable without E
//  Signal<T,E>              — factory: () → FrozenSignal<T,E>, one per loop
//  ConstantSignal<T,E>      — factory: () → ConstantFrozenSignal<T,E>
// ============================================================================
namespace mtpl {


// ============================================================================
//  Category-theoretic notes (long form, intentionally verbose)
//
//  1) Objects and morphisms in code
//     - A single Signal<T,E> value corresponds to one object in the category.
//     - A SignalTransform<T,E,ValueTypes...> represents a family of morphisms
//       that share the same shape: given inputs of types (ValueTypes...), it
//       yields an output of type T. In code, this is a callable that can be
//       applied to many concrete Signal objects of those types.
//     - This is why a SignalTransform is not a single morphism but a structured
//       family of morphisms parameterized by the particular input Signals.
//
//  2) Two categories: S and CS
//     - S: objects are all Signals; morphisms are SignalTransforms (families of
//       morphisms as described above) that produce Signals.
//     - CS: objects are ConstantSignals; morphisms are ConstantSignalTransforms
//       (not yet a separate type in the codebase, but conceptually present).
//       CS is a subcategory of S because every ConstantSignal is also a Signal
//       and every ConstantSignalTransform is a SignalTransform with additional
//       structure.
//
//  3) Constant as a family of functors
//     - Constant() is not itself a morphism in S. It is the object part of a
//       family of functors F_e: S -> CS, indexed by a dummy event e.
//     - Concretely, F_e on objects is just Constant(s, e): evaluate Signal s
//       at the dummy event e to produce a ConstantSignal.
//     - The family aspect matters: different e can give different ConstantSignals.
//       This is why we avoid pretending Constant() is a morphism in S.
//
//  4) The restricted subcategory we actually implement
//     - We explicitly restrict to a subcategory of S that has the same objects
//       as S, but only those morphism-families that preserve constancy:
//       if all inputs are ConstantSignals, the output must be a ConstantSignal.
//     - Intuition: we disallow morphisms f : i(x) -> y where x is constant but
//       y is not constant. That is, if inputs live in the embedded subcategory
//       CS (via the inclusion i), then outputs must also land in CS.
//     - This makes the "all-constant inputs -> ConstantSignal output" rule well
//       defined and consistent with the functors F_e.
//
//  5) Why this matches the implementation
//     - SignalTransform exposes a convenience: if all inputs are constant, the
//       call returns a ConstantSignal. This encodes the restriction above and
//       keeps the inspection tree coherent.
//     - The action of F_e on morphisms is not represented directly because a
//       SignalTransform groups many morphisms together. Still, for any concrete
//       morphism picked out by a particular call, F_e(f) does not depend on e.
//       We enforce the restriction at the API boundary: constant inputs produce
//       constant outputs.
//
//  6) Actual object model in code
//     - The practical category has objects that are finite collections of
//       Signals, and morphisms whose target is always a singleton Signal.
//     - This is the "multi-input, single-output" perspective used by the API.
//       The above discussion generalizes immediately to that setting.
// ============================================================================

    
// --- FrozenSignal ---

template<typename T, typename E>
class ConstantFrozenSignal;  // forward

template<typename T, typename E>
class FrozenSignal {
public:
    using Fn = std::function<T(const E&)>;

    FrozenSignal(Fn fn) : fn_(std::move(fn)) {}

    template<typename F,
             typename = std::enable_if_t<
                 !std::is_same_v<std::decay_t<F>, FrozenSignal> &&
                 !std::is_same_v<std::decay_t<F>, ConstantFrozenSignal<T,E>>>>
    FrozenSignal(F&& fn) : fn_(std::forward<F>(fn)) {}

    T operator()(const E& e) const { return fn_(e); }

protected:
    Fn fn_;
};

template<typename T, typename E>
class ConstantFrozenSignal : public FrozenSignal<T,E> {
public:
    ConstantFrozenSignal(T val)
        : FrozenSignal<T,E>([val](const E&) -> T { return val; })
        , val_(val) {}

    T operator()()         const { return val_; }
    T operator()(const E&) const { return val_; }

private:
    T val_;
};

// ============================================================================
//  Signal type hierarchy base classes (tag types for concept dispatch)
// ============================================================================

template<typename E>
struct SignalBase {};

template<typename E>
struct ConstantSignalBase {};


// Singleton atoms stored in inputs_ for leaf signals
template<typename E>
struct SignalAtom : SignalBase<E> {};

template<typename E>
struct ConstantSignalAtom : SignalAtom<E>, ConstantSignalBase<E> {};

// --- SignalTransformBase ---
//
// Abstract base that erases ValueTypes. Signal<T,E> holds a
// shared_ptr<const SignalTransformBase<T,E>> — this is what makes
// computation and introspection one and the same mechanism.

template<typename T, typename E>
class SignalTransformBase {
public:
    struct ConstantParts {
        std::shared_ptr<const SignalTransformBase> transform;
        std::vector<std::any> children;
    };

    virtual ~SignalTransformBase() = default;

    // Given children (type-erased Signals), produce a FrozenSignal.
    virtual FrozenSignal<T,E> apply(const std::vector<std::any>& children) const = 0;

    // Produce the pieces needed to build a ConstantSignal from this
    // transform. For tree transforms: same transform + constantified children.
    // For leaf Elements: a new ConstantElement + ConstantSignalAtom.
    virtual ConstantParts toConstantParts(const std::vector<std::any>& children, const E& e) const = 0;

    // Deep copy (preserves derived type).
    virtual std::shared_ptr<const SignalTransformBase> clone() const = 0;
};

// --- Signal ---

template<typename T, typename E>
class ConstantSignal;  // forward

template<typename T, typename E, typename... ValueTypes>
class SignalTransform;  // forward

template<typename T, typename E>
class Element;  // forward

template<typename T, typename E>
class ConstantElement;  // forward

template<typename T, typename E>
class Signal : public SignalBase<E> {
public:
    // Leaf constructor: wraps a factory function in an Element transform.
    // Body is deferred to after Element is defined.
    template<typename F,
             typename = std::enable_if_t<
                 !std::is_same_v<std::decay_t<F>, Signal> &&
                 !std::is_same_v<std::decay_t<F>, ConstantSignal<T,E>>>>
    Signal(F&& factory);

    // Computation IS tree traversal — one mechanism.
    FrozenSignal<T,E> operator()() const {
        return transform_->apply(children_);
    }

    // Recursive Constant functor F_e — deferred (needs ConstantSignal complete)
    ConstantSignal<T,E> toConstant(E e = {}) const;

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
    Signal() = default;
    Signal(internal_tag) {}

    std::shared_ptr<const SignalTransformBase<T,E>> transform_;
    std::vector<std::any> children_;
    std::any metadata_;

    template<typename, typename, typename...>
    friend class SignalTransform;

    template<typename, typename>
    friend class Element;

    template<typename, typename>
    friend class ConstantElement;
};

template<typename T, typename E>
class ConstantSignal : public Signal<T,E>, public ConstantSignalBase<E> {
public:
    // Leaf constant from factory — wraps in ConstantElement.
    // Body deferred to after ConstantElement is defined.
    using ConstFn = std::function<ConstantFrozenSignal<T,E>()>;
    ConstantSignal(ConstFn fn);

    // Tree-based constructor: same transform, constantified children.
    ConstantSignal(std::shared_ptr<const SignalTransformBase<T,E>> t,
                   std::vector<std::any> c)
        : Signal<T,E>(typename Signal<T,E>::internal_tag{})
    {
        this->transform_ = std::move(t);
        this->children_ = std::move(c);
    }

    // ONE mechanism: tree-walk, evaluate at E{}.
    ConstantFrozenSignal<T,E> operator()() const {
        FrozenSignal<T,E> frozen = this->transform_->apply(this->children_);
        T val = frozen(E{});
        return ConstantFrozenSignal<T,E>(val);
    }
};

// ============================================================================
//  Signal concept hierarchy
//
//  bare_t<T> strips references, cv-qualifiers, and one pointer level,
//  yielding the underlying value type for concept checking.
//
//  Type suffix — matches the type directly (after decay)
//  Arg  suffix — matches any argument form: value, reference, pointer, etc.
//
//  AnySignalType / AnySignalArg       — any Signal or ConstantSignal
//  ConstantSignalType / ConstantSignalArg — ConstantSignal only
//  VaryingSignalType / VaryingSignalArg   — Signal but not ConstantSignal
//  NonSignal                              — anything that is not AnySignalArg
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
concept ConstantSignalType = std::is_base_of_v<ConstantSignalBase<E>, std::decay_t<T>>;

template<typename T, typename E>
concept VaryingSignalType = AnySignalType<T, E> && !ConstantSignalType<T, E>;

// --- Arg concepts (any argument form: value, pointer, reference) ---

template<typename T, typename E>
concept AnySignalArg = std::is_base_of_v<SignalBase<E>, bare_t<T>>;

template<typename T, typename E>
concept ConstantSignalArg = std::is_base_of_v<ConstantSignalBase<E>, bare_t<T>>;

template<typename T, typename E>
concept VaryingSignalArg = AnySignalArg<T, E> && !ConstantSignalArg<T, E>;

// --- Non-signal concept ---

template<typename T, typename E>
concept NonSignal = !AnySignalArg<T, E>;

// Type trait to extract the value type T from Signal<T,E> or ConstantSignal<T,E>
template<typename S, typename E>
struct ExtractValueType { using type = void; };

template<typename T, typename E>
struct ExtractValueType<Signal<T,E>, E> { using type = T; };

template<typename T, typename E>
struct ExtractValueType<ConstantSignal<T,E>, E> { using type = T; };

template<typename S, typename E>
using ExtractValueType_t = typename ExtractValueType<bare_t<S>, E>::type;

// --- SignalTransform and subclasses ---

template<typename T, typename E, typename... ValueTypes>
class SignalTransform : public SignalTransformBase<T, E> {
public:
    using SignalFn = std::function<FrozenSignal<T,E>(Signal<ValueTypes,E>...)>;

    virtual ~SignalTransform() = default;

    SignalTransform() = default;
    explicit SignalTransform(SignalFn fn) : fn_(std::move(fn)) {}

    // --- apply: extract typed children from vector<any>, call fn_ ---
    FrozenSignal<T,E> apply(const std::vector<std::any>& children) const override {
        return applyImpl(children, std::index_sequence_for<ValueTypes...>{});
    }

    // --- toConstantParts: recursively constantify each child ---
    typename SignalTransformBase<T,E>::ConstantParts
    toConstantParts(const std::vector<std::any>& children, const E& e) const override {
        return toConstantPartsImpl(children, e, std::index_sequence_for<ValueTypes...>{});
    }

    // --- clone: deep copy preserving derived type ---
    std::shared_ptr<const SignalTransformBase<T,E>> clone() const override {
        return std::make_shared<const SignalTransform>(*this);
    }

    // --- Constant factories ---

    // Constant from non-signal value
    static ConstantSignal<T, E> Constant(T value) {
        return ConstantSignal<T, E>([value]() {
            return ConstantFrozenSignal<T, E>(value);
        });
    }

    // Recursive Constant functor F_e : S → CS
    static ConstantSignal<T, E> Constant(Signal<T, E> s, E e = {}) {
        return s.toConstant(e);
    }

    // --- operator() ---

    // All args constant-or-nonsignal, none varying → ConstantSignal
    template<typename... Args>
    requires ((ConstantSignalArg<Args, E> || NonSignal<Args, E>) && ...)
          && (!(VaryingSignalArg<Args, E> || ...))
    ConstantSignal<T, E>
    operator()(const Args&... args) const {
        auto normalized = std::make_tuple(normalize(args)...);
        std::vector<std::any> children;
        std::apply([&](const auto&... sigs) {
            (children.push_back(std::any(sigs)), ...);
        }, normalized);
        return ConstantSignal<T, E>(clone(), std::move(children));
    }

    // At least one varying signal → Signal
    template<typename... Args>
    requires (VaryingSignalArg<Args, E> || ...)
    Signal<T, E>
    operator()(const Args&... args) const {
        auto normalized = std::make_tuple(normalize(args)...);
        std::vector<std::any> children;
        std::apply([&](const auto&... sigs) {
            (children.push_back(std::any(sigs)), ...);
        }, normalized);
        Signal<T, E> result(typename Signal<T,E>::internal_tag{});
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
        return fn_(std::any_cast<Signal<ValueTypes, E>>(children[Is])...);
    }

    template<std::size_t... Is>
    typename SignalTransformBase<T,E>::ConstantParts
    toConstantPartsImpl(const std::vector<std::any>& children, const E& e,
                        std::index_sequence<Is...>) const {
        std::vector<std::any> const_children;
        // Recursively constantify each child and store as Signal<V,E>
        (const_children.push_back(std::any(
            Signal<ValueTypes, E>(
                std::any_cast<Signal<ValueTypes, E>>(children[Is]).toConstant(e)
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
            return Signal<V, E>(static_cast<const Signal<V, E>&>(arg));
        } else {
            // NonSignal — wrap in Constant
            using V = std::decay_t<Arg>;
            return Signal<V, E>(SignalTransform<V, E>::Constant(arg));
        }
    }
};

// --- ConstantElement ---
//
// Leaf transform for constant signals. Holds a baked value T.
// apply() returns ConstantFrozenSignal(value), ignoring children.

template<typename T, typename E>
class ConstantElement : public SignalTransformBase<T, E> {
public:
    explicit ConstantElement(T val) : value_(std::move(val)) {}

    FrozenSignal<T,E> apply(const std::vector<std::any>&) const override {
        T v = value_;
        return FrozenSignal<T,E>([v](const E&) -> T { return v; });
    }

    typename SignalTransformBase<T,E>::ConstantParts
    toConstantParts(const std::vector<std::any>& children, const E&) const override {
        // Already constant — return self
        return { clone(), children };
    }

    std::shared_ptr<const SignalTransformBase<T,E>> clone() const override {
        return std::make_shared<const ConstantElement>(*this);
    }

private:
    T value_;
};

// --- Element Transform ---
//
// Leaf transform for varying signals. Holds a factory () → FrozenSignal<T,E>.
// apply() calls the factory, ignoring children.
// toConstantParts() evaluates the factory at e, produces a ConstantElement.

template<typename T, typename E>
class Element : public SignalTransformBase<T, E> {
public:
    using Factory = std::function<FrozenSignal<T,E>()>;

    explicit Element(Factory f) : factory_(std::move(f)) {}

    FrozenSignal<T,E> apply(const std::vector<std::any>&) const override {
        return factory_();
    }

    typename SignalTransformBase<T,E>::ConstantParts
    toConstantParts(const std::vector<std::any>&, const E& e) const override {
        // Evaluate the factory at e, produce a ConstantElement
        T val = factory_()(e);
        return {
            std::make_shared<const ConstantElement<T,E>>(val),
            { std::any(ConstantSignalAtom<E>{}) }
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
// These need Element, ConstantElement, and ConstantSignal to be complete.

// Signal leaf constructor: wraps factory in an Element
template<typename T, typename E>
template<typename F, typename>
Signal<T,E>::Signal(F&& factory) {
    auto f = std::function<FrozenSignal<T,E>()>(std::forward<F>(factory));
    transform_ = std::make_shared<const Element<T, E>>(f);
    children_ = { std::any(SignalAtom<E>{}) };
}

// Signal::toConstant — delegates to the transform
template<typename T, typename E>
ConstantSignal<T,E> Signal<T,E>::toConstant(E e) const {
    auto [t, c] = transform_->toConstantParts(children_, e);
    return ConstantSignal<T,E>(std::move(t), std::move(c));
}

// ConstantSignal leaf constructor: wraps factory value in a ConstantElement
template<typename T, typename E>
ConstantSignal<T,E>::ConstantSignal(ConstFn fn)
    : Signal<T,E>(typename Signal<T,E>::internal_tag{})
{
    T val = fn()();  // evaluate once to get the baked value
    this->transform_ = std::make_shared<const ConstantElement<T,E>>(val);
    this->children_ = { std::any(ConstantSignalAtom<E>{}) };
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
            [f](Signal<ValueTypes, E>... signals) -> FrozenSignal<T, E> {
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
            [](Signal<Source, E> arg) -> FrozenSignal<Target, E> {
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

// (Signal, Signal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
Signal<R,E> operator+(Signal<S,E> a, Signal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x+y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
Signal<R,E> operator-(Signal<S,E> a, Signal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x-y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
Signal<R,E> operator*(Signal<S,E> a, Signal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x*y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
Signal<R,E> operator/(Signal<S,E> a, Signal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return y?x/y:S{}; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
Signal<R,E> operator%(Signal<S,E> a, Signal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return y?x%y:S{}; }))(a, b);
}

// (Signal, non-Signal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
requires NonSignal<T, E>
Signal<R,E> operator+(Signal<S,E> a, T b) { return a + SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
requires NonSignal<T, E>
Signal<R,E> operator-(Signal<S,E> a, T b) { return a - SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
requires NonSignal<T, E>
Signal<R,E> operator*(Signal<S,E> a, T b) { return a * SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
requires NonSignal<T, E>
Signal<R,E> operator/(Signal<S,E> a, T b) { return a / SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
requires NonSignal<T, E>
Signal<R,E> operator%(Signal<S,E> a, T b) { return a % SignalTransform<T,E>::Constant(b); }

// (non-Signal, Signal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
requires NonSignal<S, E>
Signal<R,E> operator+(S a, Signal<T,E> b) { return SignalTransform<S,E>::Constant(a) + b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
requires NonSignal<S, E>
Signal<R,E> operator-(S a, Signal<T,E> b) { return SignalTransform<S,E>::Constant(a) - b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
requires NonSignal<S, E>
Signal<R,E> operator*(S a, Signal<T,E> b) { return SignalTransform<S,E>::Constant(a) * b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
requires NonSignal<S, E>
Signal<R,E> operator/(S a, Signal<T,E> b) { return SignalTransform<S,E>::Constant(a) / b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
requires NonSignal<S, E>
Signal<R,E> operator%(S a, Signal<T,E> b) { return SignalTransform<S,E>::Constant(a) % b; }

// Unary minus operator

template<typename T, typename E, typename R = decltype(-std::declval<T>())>
Signal<R,E> operator-(Signal<T,E> a) {
    return Pushforward<R,E,T>(std::function<R(T)>([](T x){ return -x; }))(a);
}

// ConstantSignal arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
ConstantSignal<R,E> operator+(ConstantSignal<S,E> a, ConstantSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x+y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
ConstantSignal<R,E> operator-(ConstantSignal<S,E> a, ConstantSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x-y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
ConstantSignal<R,E> operator*(ConstantSignal<S,E> a, ConstantSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return x*y; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
ConstantSignal<R,E> operator/(ConstantSignal<S,E> a, ConstantSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return y?x/y:S{}; }))(a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
ConstantSignal<R,E> operator%(ConstantSignal<S,E> a, ConstantSignal<T,E> b) {
    return Pushforward<R,E,S,T>(std::function<R(S,T)>([](S x, T y){ return y?x%y:S{}; }))(a, b);
}

// (ConstantSignal, non-Signal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
requires NonSignal<T, E>
ConstantSignal<R,E> operator+(ConstantSignal<S,E> a, T b) { return a + SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
requires NonSignal<T, E>
ConstantSignal<R,E> operator-(ConstantSignal<S,E> a, T b) { return a - SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
requires NonSignal<T, E>
ConstantSignal<R,E> operator*(ConstantSignal<S,E> a, T b) { return a * SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
requires NonSignal<T, E>
ConstantSignal<R,E> operator/(ConstantSignal<S,E> a, T b) { return a / SignalTransform<T,E>::Constant(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
requires NonSignal<T, E>
ConstantSignal<R,E> operator%(ConstantSignal<S,E> a, T b) { return a % SignalTransform<T,E>::Constant(b); }

// (non-Signal, ConstantSignal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
requires NonSignal<S, E>
ConstantSignal<R,E> operator+(S a, ConstantSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) + b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
requires NonSignal<S, E>
ConstantSignal<R,E> operator-(S a, ConstantSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) - b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
requires NonSignal<S, E>
ConstantSignal<R,E> operator*(S a, ConstantSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) * b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
requires NonSignal<S, E>
ConstantSignal<R,E> operator/(S a, ConstantSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) / b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
requires NonSignal<S, E>
ConstantSignal<R,E> operator%(S a, ConstantSignal<T,E> b) { return SignalTransform<S,E>::Constant(a) % b; }

// Unary minus operator for ConstantSignal

template<typename T, typename E, typename R = decltype(-std::declval<T>())>
ConstantSignal<R,E> operator-(ConstantSignal<T,E> a) {
    return Pushforward<R,E,T>(std::function<R(T)>([](T x){ return -x; }))(a);
}




}