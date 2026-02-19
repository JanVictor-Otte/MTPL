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

// --- Signal ---

template<typename T, typename E>
class ConstantSignal;  // forward

template<typename T, typename E, typename... ValueTypes>
class SignalTransform;  // forward

template<typename T, typename E>
class Signal {
public:
    using Fn = std::function<FrozenSignal<T,E>()>;

    Signal(Fn fn) : fn_(std::move(fn)) {}

    template<typename F,
             typename = std::enable_if_t<
                 !std::is_same_v<std::decay_t<F>, Signal> &&
                 !std::is_same_v<std::decay_t<F>, ConstantSignal<T,E>>>>
    Signal(F&& fn) : fn_(std::forward<F>(fn)) {}

    FrozenSignal<T,E> operator()() const { return fn_(); }

    // Introspection API
    const std::vector<std::any>& inputs() const { return inputs_; }
    const std::any& metadata() const { return metadata_; }
    const std::any& transform() const { return transform_; }

    // Helper to populate introspection from transform
    template<typename Transform>
    void setFromTransform(const Transform& t, const std::vector<std::any>& inp) {
        transform_ = std::any(t);
        inputs_ = inp;
    }

protected:
    std::function<FrozenSignal<T,E>()> fn_;
    std::vector<std::any> inputs_;
    std::any metadata_;
    std::any transform_;

    friend class SignalTransform<T, E>;
};

template<typename T, typename E>
class ConstantSignal : public Signal<T,E> {
public:
    using ConstFn = std::function<ConstantFrozenSignal<T,E>()>;

    // Construct from a factory that returns ConstantFrozenSignal
    ConstantSignal(ConstFn fn)
        : Signal<T,E>([fn]() -> FrozenSignal<T,E> { return fn(); })
        , constFn_(std::move(fn)) {}

    // operator()() returns ConstantFrozenSignal — callable without E
    ConstantFrozenSignal<T,E> operator()() const { return constFn_(); }

private:
    ConstFn constFn_;
};

// Type trait to detect Signal and ConstantSignal types
template<typename T>
struct is_signal : std::false_type {};

template<typename T, typename E>
struct is_signal<Signal<T,E>> : std::true_type {};

template<typename T, typename E>
struct is_signal<ConstantSignal<T,E>> : std::true_type {};

template<typename T>
inline constexpr bool is_signal_v = is_signal<T>::value;

template<typename T>
struct is_constant_signal : std::false_type {};

template<typename T, typename E>
struct is_constant_signal<ConstantSignal<T,E>> : std::true_type {};

template<typename T>
inline constexpr bool is_constant_signal_v = is_constant_signal<T>::value;

// Helper trait to check if all args are "constant" (either ConstantSignal or primitive value)
template<typename... Args>
struct all_constant_or_primitive {
    static constexpr bool value = true;
};

template<typename T, typename E, typename... Rest>
struct all_constant_or_primitive<ConstantSignal<T, E>, Rest...> {
    static constexpr bool value = all_constant_or_primitive<Rest...>::value;
};

template<typename T, typename E, typename... Rest>
struct all_constant_or_primitive<ConstantSignal<T, E>*, Rest...> {
    static constexpr bool value = all_constant_or_primitive<Rest...>::value;
};

template<typename T, typename... Rest>
struct all_constant_or_primitive<T, Rest...> {
    static constexpr bool value = std::is_arithmetic_v<T> && all_constant_or_primitive<Rest...>::value;
};

template<>
struct all_constant_or_primitive<> {
    static constexpr bool value = true;
};

// Concepts for Morphism requirements
template<typename T, typename E>
concept SignalArg = is_signal_v<T> || is_constant_signal_v<T>;

template<typename T, typename E>
concept ConstantSignalArg = is_constant_signal_v<T>;

template<typename T, typename E>
concept ProjectIndex = std::is_integral_v<T>;

// --- SignalTransform and subclasses ---

template<typename T, typename E, typename... ValueTypes>
class SignalTransform {
public:
    virtual ~SignalTransform() = default;

    // Static factory for constant values
    static ConstantSignal<T, E> Constant(T value) {
        return ConstantSignal<T, E>([value]() {
            return ConstantFrozenSignal<T, E>(value);
        });
    }

    // Metadata
    void setMetadata(std::any m) { metadata_ = m; }
    std::any metadata() const { return metadata_; }

protected:
    std::any metadata_;
};

// --- Pushforward Transform ---

template<typename T, typename E, typename... ValueTypes>
class Pushforward : public SignalTransform<T, E, ValueTypes...> {
public:
    using Fn = std::function<T(ValueTypes...)>;

    Pushforward(Fn fn) : fn_(std::move(fn)) {}

    // All constant arguments (either ConstantSignal or primitive)
    template<typename... Args>
    std::enable_if_t<all_constant_or_primitive<Args...>::value, ConstantSignal<T, E>>
    operator()(const Args&... args) {
        auto inputs = std::vector<std::any>{std::any(args)...};
        ConstantSignal<T, E> result(
            [this, args...]() {
                T val = std::apply(fn_, std::make_tuple(this->extractValue(args)()...));
                return ConstantFrozenSignal<T, E>(val);
            }
        );
        result.setFromTransform(*this, inputs);
        return result;
    }

    // Mixed or contains at least one non-constant Signal
    template<typename... Args>
    std::enable_if_t<!all_constant_or_primitive<Args...>::value, Signal<T, E>>
    operator()(const Args&... args) {
        auto inputs = std::vector<std::any>{std::any(args)...};
        Signal<T, E> result(
            [this, args...]() -> FrozenSignal<T, E> {
                auto frozen = std::make_tuple(this->normalizeToFrozen(args)...);
                return [this, frozen](const E& e) -> T {
                    return std::apply([this, &e](const auto&... fs) {
                        return this->fn_(deref(fs, e)...);
                    }, frozen);
                };
            }
        );
        result.setFromTransform(*this, inputs);
        return result;
    }

private:
    Fn fn_;

    // Extract callable value from args - handles ConstantSignal, pointers, and primitives
    template<typename Arg>
    auto extractValue(const Arg& arg) const {
        if constexpr (is_constant_signal_v<Arg>) {
            // ConstantSignal - directly callable to get value
            return [arg]() { return arg()(); };
        } else if constexpr (std::is_pointer_v<Arg> && is_constant_signal_v<std::remove_pointer_t<Arg>>) {
            // Pointer to ConstantSignal
            return [arg]() { return (*arg)()(); };
        } else {
            // Primitive value - wrap to make callable
            return [arg]() { return arg; };
        }
    }

    // Normalize argument to FrozenSignal for evaluation
    template<typename Arg>
    auto normalizeToFrozen(const Arg& arg) const {
        using ArgDecay = std::decay_t<Arg>;
        
        if constexpr (std::is_pointer_v<ArgDecay>) {
            // Pointer to something - dereference and normalize
            return normalizeToFrozen(*arg);
        } else if constexpr (is_signal_v<ArgDecay>) {
            // Plain Signal
            return arg();
        } else if constexpr (is_constant_signal_v<ArgDecay>) {
            // ConstantSignal - wrap to yield constant
            return FrozenSignal<T, E>([arg](const E&) { return arg()(); });
        } else {
            // Primitive value - wrap in FrozenSignal that ignores E
            auto val = arg;
            using ValType = decltype(val);
            return FrozenSignal<T, E>([val](const E&) -> ValType { return val; });
        }
    }

    template<typename FrozenSig>
    auto deref(const FrozenSig& fs, const E& e) const {
        return fs(e);
    }

    friend class Signal<T, E>;
};

// --- CastTransform ---

template<typename Target, typename E, typename Source>
class CastTransform : public SignalTransform<Target, E, Source> {
public:
    // All constant
    ConstantSignal<Target, E> operator()(ConstantSignal<Source, E> arg) {
        auto inputs = std::vector<std::any>{std::any(arg)};
        ConstantSignal<Target, E> result(
            [arg]() {
                Source val = arg()();
                return ConstantFrozenSignal<Target, E>(static_cast<Target>(val));
            }
        );
        result.setFromTransform(*this, inputs);
        return result;
    }

    // Mixed or non-constant
    Signal<Target, E> operator()(Signal<Source, E> arg) {
        auto inputs = std::vector<std::any>{std::any(arg)};
        Signal<Target, E> result(
            [arg]() -> FrozenSignal<Target, E> {
                FrozenSignal<Source, E> frozen = arg();
                return [frozen](const E& e) -> Target {
                    return static_cast<Target>(frozen(e));
                };
            }
        );
        result.setFromTransform(*this, inputs);
        return result;
    }
};

// --- ApplyTransform (stub for now) ---

template<typename T, typename E, typename... Args>
class ApplyTransform : public SignalTransform<T, E, Args...> {
    // Stub - not fully implemented yet
};


// ============================================================================
//  map, pushforward, castSignal, apply
// ============================================================================

template<typename S, typename T, typename E>
Signal<T,E> map(std::function<T(S)> f, Signal<S,E> s) {
    return [f, s]() -> FrozenSignal<T,E> {
        FrozenSignal<S,E> fs = s();
        return [f, fs](const E& e) -> T { return f(fs(e)); };
    };
}

template<typename T, typename E, typename... Ss>
Signal<T,E> pushforward(std::function<T(Ss...)> f, Signal<Ss,E>... signals) {
    return [f, signals...]() -> FrozenSignal<T,E> {
        auto frozen = std::make_tuple(signals()...);
        return [f, frozen](const E& e) -> T {
            return std::apply([&](const auto&... fs) { return f(fs(e)...); }, frozen);
        };
    };
}

template<typename T, typename E, typename... Ss>
ConstantSignal<T,E> pushforward(std::function<T(Ss...)> f, ConstantSignal<Ss,E>... signals) {
    return ConstantSignal<T,E>([f, signals...]() {
        T result = std::apply(f, std::make_tuple(signals()()...));
        return ConstantFrozenSignal<T,E>(result);
    });
}

/* abstract pushforward turned off, as its not overloadable to return ConstantSignal when given only ConstantSignal arugments, and the explicit pushforward with Signal arguments is overloadable.
template<typename T, typename E, typename... Ss>
std::function<Signal<T,E>(Signal<Ss,E>...)> pushforward(std::function<T(Ss...)> f) {
    return [f](Signal<Ss,E>... signals) {
        return pushforward<T,E,Ss...>(f, signals...);
    };
}*/

template<typename B, typename A, typename E,
         typename = std::enable_if_t<std::is_convertible_v<A,B>>>
Signal<B,E> castSignal(Signal<A,E> s) {
    return pushforward<B,E>(std::function<B(A)>([](A x){ return static_cast<B>(x); }), s);
}

template<typename T, typename E, typename... Ss>
Signal<T,E> apply(std::function<Signal<T,E>(Ss...)> f, Signal<std::tuple<Ss...>,E> signal) {
    return [f, signal]() -> FrozenSignal<T,E> {
        FrozenSignal<std::tuple<Ss...>,E> frozenSignal = signal();
        return [f, frozenSignal](const E& e) -> T {
            std::tuple<Ss...> values = frozenSignal(e);
            return std::apply(f, values)()(e);
        };
    };
}

// ============================================================================
//  constant
//  constant(T value, E dummy = {})     — ConstantSignal from value
//  constant(Signal<T,E>, E dummy = {}) — freeze once, wrap in ConstantSignal
// ============================================================================

template<typename T, typename E>
ConstantSignal<T,E> constant(T value, E dummy = {}) {
    return ConstantSignal<T,E>([value]() {
        return ConstantFrozenSignal<T,E>(value);
    });
}

template<typename T, typename E>
ConstantSignal<T,E> constant(Signal<T,E> s, E dummy = {}) {
    return ConstantSignal<T,E>([s, dummy]() {
        T val = s()(dummy);
        return ConstantFrozenSignal<T,E>(val);
    });
}

// ============================================================================
//  Arithmetic operators
// ============================================================================

// (Signal, Signal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
Signal<R,E> operator+(Signal<S,E> a, Signal<T,E> b) {
    return pushforward<R,E>(std::function<R(S,T)>([](S x, T y){ return x+y; }), a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
Signal<R,E> operator-(Signal<S,E> a, Signal<T,E> b) {
    return pushforward<R,E>(std::function<R(S,T)>([](S x, T y){ return x-y; }), a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
Signal<R,E> operator*(Signal<S,E> a, Signal<T,E> b) {
    return pushforward<R,E>(std::function<R(S,T)>([](S x, T y){ return x*y; }), a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
Signal<R,E> operator/(Signal<S,E> a, Signal<T,E> b) {
    return pushforward<R,E>(std::function<R(S,T)>([](S x, T y){ return y?x/y:S{}; }), a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
Signal<R,E> operator%(Signal<S,E> a, Signal<T,E> b) {
    return pushforward<R,E>(std::function<R(S,T)>([](S x, T y){ return y?x%y:S{}; }), a, b);
}

// (Signal, non-Signal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>()), typename = std::enable_if_t<!is_signal_v<T>>>
Signal<R,E> operator+(Signal<S,E> a, T b) { return a + constant<T,E>(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>()), typename = std::enable_if_t<!is_signal_v<T>>>
Signal<R,E> operator-(Signal<S,E> a, T b) { return a - constant<T,E>(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>()), typename = std::enable_if_t<!is_signal_v<T>>>
Signal<R,E> operator*(Signal<S,E> a, T b) { return a * constant<T,E>(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>()), typename = std::enable_if_t<!is_signal_v<T>>>
Signal<R,E> operator/(Signal<S,E> a, T b) { return a / constant<T,E>(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>()), typename = std::enable_if_t<!is_signal_v<T>>>
Signal<R,E> operator%(Signal<S,E> a, T b) { return a % constant<T,E>(b); }

// (non-Signal, Signal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>()), typename = std::enable_if_t<!is_signal_v<S>>>
Signal<R,E> operator+(S a, Signal<T,E> b) { return constant<S,E>(a) + b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>()), typename = std::enable_if_t<!is_signal_v<S>>>
Signal<R,E> operator-(S a, Signal<T,E> b) { return constant<S,E>(a) - b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>()), typename = std::enable_if_t<!is_signal_v<S>>>
Signal<R,E> operator*(S a, Signal<T,E> b) { return constant<S,E>(a) * b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>()), typename = std::enable_if_t<!is_signal_v<S>>>
Signal<R,E> operator/(S a, Signal<T,E> b) { return constant<S,E>(a) / b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>()), typename = std::enable_if_t<!is_signal_v<S>>>
Signal<R,E> operator%(S a, Signal<T,E> b) { return constant<S,E>(a) % b; }

// Unary minus operator

template<typename T, typename E, typename R = decltype(-std::declval<T>())>
Signal<R,E> operator-(Signal<T,E> a) {
    return pushforward<R,E>(std::function<R(T)>([](T x){ return -x; }), a);
}

// ConstantSignal arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>())>
ConstantSignal<R,E> operator+(ConstantSignal<S,E> a, ConstantSignal<T,E> b) {
    return pushforward<R,E>(std::function<R(S,T)>([](S x, T y){ return x+y; }), a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>())>
ConstantSignal<R,E> operator-(ConstantSignal<S,E> a, ConstantSignal<T,E> b) {
    return pushforward<R,E>(std::function<R(S,T)>([](S x, T y){ return x-y; }), a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>())>
ConstantSignal<R,E> operator*(ConstantSignal<S,E> a, ConstantSignal<T,E> b) {
    return pushforward<R,E>(std::function<R(S,T)>([](S x, T y){ return x*y; }), a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>())>
ConstantSignal<R,E> operator/(ConstantSignal<S,E> a, ConstantSignal<T,E> b) {
    return pushforward<R,E>(std::function<R(S,T)>([](S x, T y){ return y?x/y:S{}; }), a, b);
}
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>())>
ConstantSignal<R,E> operator%(ConstantSignal<S,E> a, ConstantSignal<T,E> b) {
    return pushforward<R,E>(std::function<R(S,T)>([](S x, T y){ return y?x%y:S{}; }), a, b);
}

// (ConstantSignal, non-Signal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>()), typename = std::enable_if_t<!is_signal_v<T>>>
ConstantSignal<R,E> operator+(ConstantSignal<S,E> a, T b) { return a + constant<T,E>(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>()), typename = std::enable_if_t<!is_signal_v<T>>>
ConstantSignal<R,E> operator-(ConstantSignal<S,E> a, T b) { return a - constant<T,E>(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>()), typename = std::enable_if_t<!is_signal_v<T>>>
ConstantSignal<R,E> operator*(ConstantSignal<S,E> a, T b) { return a * constant<T,E>(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>()), typename = std::enable_if_t<!is_signal_v<T>>>
ConstantSignal<R,E> operator/(ConstantSignal<S,E> a, T b) { return a / constant<T,E>(b); }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>()), typename = std::enable_if_t<!is_signal_v<T>>>
ConstantSignal<R,E> operator%(ConstantSignal<S,E> a, T b) { return a % constant<T,E>(b); }

// (non-Signal, ConstantSignal) arithmetic operators

template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() + std::declval<T>()), typename = std::enable_if_t<!is_signal_v<S>>>
ConstantSignal<R,E> operator+(S a, ConstantSignal<T,E> b) { return constant<S,E>(a) + b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() - std::declval<T>()), typename = std::enable_if_t<!is_signal_v<S>>>
ConstantSignal<R,E> operator-(S a, ConstantSignal<T,E> b) { return constant<S,E>(a) - b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() * std::declval<T>()), typename = std::enable_if_t<!is_signal_v<S>>>
ConstantSignal<R,E> operator*(S a, ConstantSignal<T,E> b) { return constant<S,E>(a) * b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() / std::declval<T>()), typename = std::enable_if_t<!is_signal_v<S>>>
ConstantSignal<R,E> operator/(S a, ConstantSignal<T,E> b) { return constant<S,E>(a) / b; }
template<typename S, typename T, typename E, typename R = decltype(std::declval<S>() % std::declval<T>()), typename = std::enable_if_t<!is_signal_v<S>>>
ConstantSignal<R,E> operator%(S a, ConstantSignal<T,E> b) { return constant<S,E>(a) % b; }

// Unary minus operator for ConstantSignal

template<typename T, typename E, typename R = decltype(-std::declval<T>())>
ConstantSignal<R,E> operator-(ConstantSignal<T,E> a) {
    return pushforward<R,E>(std::function<R(T)>([](T x){ return -x; }), a);
}

/*
template<typename T, typename E, std::size_t N>
auto diagonal(Signal<T,E> s) {
    return pushforward(([](T x){ 
        return []<std::size_t... Is>(T v, std::index_sequence<Is...>){ 
            return std::make_tuple((Is, v)...);
        }(x, std::make_index_sequence<N>{});
    }), s);
}*/


}