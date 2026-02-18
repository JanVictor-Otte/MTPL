#pragma once
#include <functional>
#include <type_traits>
#include <tuple>
#include <memory>
#include <vector>

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

protected:
    std::function<FrozenSignal<T,E>()> fn_;
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