#include "mtpl/core/Signal.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

using namespace mtpl;

struct TestEvent {
    std::vector<int> data;
};

// ============================================================================
//  Helper
// ============================================================================

void assert_eq(int a, int b, const std::string& msg) {
    if (a != b) {
        std::cerr << "FAIL: " << msg << " (expected " << b << ", got " << a << ")" << std::endl;
        throw std::runtime_error("Test failed: " + msg);
    }
    std::cout << "  ✓ " << msg << std::endl;
}

void assert_deq(double a, double b, const std::string& msg, double eps = 1e-9) {
    if (std::abs(a - b) > eps) {
        std::cerr << "FAIL: " << msg << " (expected " << b << ", got " << a << ")" << std::endl;
        throw std::runtime_error("Test failed: " + msg);
    }
    std::cout << "  ✓ " << msg << std::endl;
}

// ============================================================================
//  1. Temporary Pushforward: "auto sig = Pushforward(f)(a, b)"
//     The Pushforward is destroyed immediately. The Signal must still work.
// ============================================================================

void test_temporary_pushforward() {
    std::cout << "\n=== Temporary Pushforward ===" << std::endl;

    auto a = SignalTransform<int, TestEvent>::Constant(7);
    auto b = SignalTransform<int, TestEvent>::Constant(3);

    // Pushforward is a temporary — destroyed after this line
    auto result = Pushforward<int, TestEvent, int, int>(
        std::function<int(int,int)>([](int x, int y){ return x + y; })
    )(a, b);

    // Must still evaluate correctly
    assert_eq(result()(), 10, "Temporary Pushforward: 7+3=10");

    // Evaluate a second time (no stale state)
    assert_eq(result()(), 10, "Temporary Pushforward: second eval still 10");
}

// ============================================================================
//  2. Temporary Pushforward with varying Signal
// ============================================================================

void test_temporary_pushforward_varying() {
    std::cout << "\n=== Temporary Pushforward (varying) ===" << std::endl;

    int counter = 0;
    Signal<int, TestEvent> varying([&counter]() -> FrozenSignal<int, TestEvent> {
        int c = ++counter;
        return [c](const TestEvent&) -> int { return c * 10; };
    });

    // Temporary Pushforward
    auto result = Pushforward<int, TestEvent, int>(
        std::function<int(int)>([](int x){ return x + 1; })
    )(varying);

    TestEvent e{{0}};
    int v1 = result()(e);
    int v2 = result()(e);
    // Each call to result() calls the varying signal factory anew
    assert_eq(v1, 11, "Temporary varying: first eval");
    assert_eq(v2, 21, "Temporary varying: second eval (different factory call)");
}

// ============================================================================
//  3. Shared transform across different Signals — mutating one doesn't
//     affect the other (value semantics via clone())
// ============================================================================

void test_shared_transform_independence() {
    std::cout << "\n=== Shared transform across signals ===" << std::endl;

    Pushforward<int, TestEvent, int, int> add(
        std::function<int(int,int)>([](int x, int y){ return x + y; })
    );

    auto a = SignalTransform<int, TestEvent>::Constant(10);
    auto b = SignalTransform<int, TestEvent>::Constant(20);
    auto c = SignalTransform<int, TestEvent>::Constant(5);

    // Same transform, different inputs
    auto sig1 = add(a, b);  // 10 + 20 = 30
    auto sig2 = add(a, c);  // 10 + 5  = 15
    auto sig3 = add(b, c);  // 20 + 5  = 25

    assert_eq(sig1()(), 30, "sig1 = 10+20 = 30");
    assert_eq(sig2()(), 15, "sig2 = 10+5 = 15");
    assert_eq(sig3()(), 25, "sig3 = 20+5 = 25");

    // Re-evaluate — values must not leak between signals
    assert_eq(sig1()(), 30, "sig1 re-eval still 30");
    assert_eq(sig2()(), 15, "sig2 re-eval still 15");
    assert_eq(sig3()(), 25, "sig3 re-eval still 25");
}

// ============================================================================
//  4. Mutating metadata on transform does NOT affect already-produced Signals
// ============================================================================

void test_transform_metadata_isolation() {
    std::cout << "\n=== Transform metadata isolation ===" << std::endl;

    Pushforward<int, TestEvent, int> doubler(
        std::function<int(int)>([](int x){ return x * 2; })
    );
    doubler.setMetadata(std::string("original"));

    auto a = SignalTransform<int, TestEvent>::Constant(5);
    auto sig = doubler(a);  // 10

    // Mutate transform metadata AFTER producing the signal
    doubler.setMetadata(std::string("mutated"));

    // Signal's transform must still reflect "original" (clone was taken)
    auto* pf = dynamic_cast<const Pushforward<int, TestEvent, int>*>(
        sig.transformPtr().get());
    assert(pf != nullptr && "transform should be a Pushforward");

    // The clone should have the metadata from construction time
    // (clone() copies the object, including metadata)
    auto meta = std::any_cast<std::string>(pf->metadata());
    assert(meta == "original" && "Signal's transform must have 'original' metadata");
    std::cout << "  ✓ Signal transform metadata isolated from source mutation" << std::endl;

    // And computation still works
    assert_eq(sig()(), 10, "Computation unaffected: 5*2=10");
}

// ============================================================================
//  5. Copy a Signal, mutate metadata on copy — original unaffected
// ============================================================================

void test_signal_copy_independence() {
    std::cout << "\n=== Signal copy independence ===" << std::endl;

    auto a = SignalTransform<int, TestEvent>::Constant(42);
    auto sum = Pushforward<int, TestEvent, int, int>(
        std::function<int(int,int)>([](int x, int y){ return x + y; })
    )(a, a);

    // Copy
    auto copy = sum;

    // Mutate metadata on copy
    copy.setMetadata(std::string("copy-meta"));

    // Original must not have metadata set
    assert(!sum.metadata().has_value() && "Original signal metadata must be empty");
    std::cout << "  ✓ Original signal metadata unaffected by copy mutation" << std::endl;

    // Both compute the same
    assert_eq(sum()(), 84, "Original: 42+42=84");
    assert_eq(copy()(), 84, "Copy: 42+42=84");
}

// ============================================================================
//  6. Deep composition with temporaries — Pushforward(f)(Pushforward(g)(x))
// ============================================================================

void test_deep_temporary_composition() {
    std::cout << "\n=== Deep composition with temporaries ===" << std::endl;

    auto x = SignalTransform<int, TestEvent>::Constant(3);

    // All Pushforwards are temporaries
    auto result = Pushforward<int, TestEvent, int>(
        std::function<int(int)>([](int v){ return v * v; })                // square
    )(
        Pushforward<int, TestEvent, int>(
            std::function<int(int)>([](int v){ return v + 1; })            // +1
        )(
            Pushforward<int, TestEvent, int>(
                std::function<int(int)>([](int v){ return v * 2; })        // *2
            )(x)
        )
    );

    // x=3 → *2=6 → +1=7 → square=49
    assert_eq(result()(), 49, "Deep temporaries: (3*2+1)^2 = 49");

    // Evaluate again
    assert_eq(result()(), 49, "Deep temporaries: re-eval still 49");
}

// ============================================================================
//  7. Same ConstantSignal used as input to many transforms
// ============================================================================

void test_shared_input_signal() {
    std::cout << "\n=== Shared input signal ===" << std::endl;

    auto shared = SignalTransform<int, TestEvent>::Constant(10);

    auto s1 = Pushforward<int, TestEvent, int>(
        std::function<int(int)>([](int x){ return x + 1; })
    )(shared);

    auto s2 = Pushforward<int, TestEvent, int>(
        std::function<int(int)>([](int x){ return x * 2; })
    )(shared);

    auto s3 = Pushforward<int, TestEvent, int>(
        std::function<int(int)>([](int x){ return x - 5; })
    )(shared);

    assert_eq(s1()(), 11, "shared+1 = 11");
    assert_eq(s2()(), 20, "shared*2 = 20");
    assert_eq(s3()(), 5, "shared-5 = 5");

    // Re-eval interleaved
    assert_eq(s2()(), 20, "re-eval shared*2 = 20");
    assert_eq(s1()(), 11, "re-eval shared+1 = 11");
    assert_eq(s3()(), 5, "re-eval shared-5 = 5");
}

// ============================================================================
//  8. Diamond dependency: sig1 and sig2 both depend on root,
//     sig3 depends on both sig1 and sig2 — no double-counting
// ============================================================================

void test_diamond_dependency() {
    std::cout << "\n=== Diamond dependency ===" << std::endl;

    auto root = SignalTransform<int, TestEvent>::Constant(5);

    auto left = Pushforward<int, TestEvent, int>(
        std::function<int(int)>([](int x){ return x + 1; })
    )(root);  // 6

    auto right = Pushforward<int, TestEvent, int>(
        std::function<int(int)>([](int x){ return x * 2; })
    )(root);  // 10

    auto merged = Pushforward<int, TestEvent, int, int>(
        std::function<int(int,int)>([](int a, int b){ return a + b; })
    )(left, right);  // 6 + 10 = 16

    assert_eq(merged()(), 16, "Diamond: (5+1) + (5*2) = 16");
    assert_eq(merged()(), 16, "Diamond: re-eval still 16");

    // Verify tree structure: merged has 2 children
    assert(merged.children().size() == 2 && "Merged should have 2 children");
    std::cout << "  ✓ Diamond tree structure correct (2 children)" << std::endl;
}

// ============================================================================
//  9. Temporary Pushforward with primitives (NonSignal auto-wrapping)
// ============================================================================

void test_temporary_with_primitives() {
    std::cout << "\n=== Temporary Pushforward with primitives ===" << std::endl;

    // All inputs are raw primitives, Pushforward is a temporary
    auto result = Pushforward<int, TestEvent, int, int>(
        std::function<int(int,int)>([](int x, int y){ return x * y; })
    )(6, 7);

    assert_eq(result()(), 42, "Temporary with primitives: 6*7=42");
}

// ============================================================================
// 10. CastTransform as temporary
// ============================================================================

void test_temporary_cast() {
    std::cout << "\n=== Temporary CastTransform ===" << std::endl;

    auto intSig = SignalTransform<int, TestEvent>::Constant(99);

    auto doubleSig = CastTransform<double, TestEvent, int>()(intSig);

    assert_deq(doubleSig()(), 99.0, "Temporary cast: int 99 → double 99.0");
}

// ============================================================================
// 11. Arithmetic operators produce independent signals
// ============================================================================

void test_arithmetic_independence() {
    std::cout << "\n=== Arithmetic operator independence ===" << std::endl;

    auto a = SignalTransform<int, TestEvent>::Constant(10);
    auto b = SignalTransform<int, TestEvent>::Constant(3);

    auto sum = a + b;     // 13
    auto diff = a - b;    // 7
    auto prod = a * b;    // 30
    auto quot = a / b;    // 3

    // All must evaluate independently
    assert_eq(sum()(), 13, "a+b = 13");
    assert_eq(diff()(), 7, "a-b = 7");
    assert_eq(prod()(), 30, "a*b = 30");
    assert_eq(quot()(), 3, "a/b = 3");

    // Re-evaluate in different order
    assert_eq(quot()(), 3, "re-eval a/b = 3");
    assert_eq(sum()(), 13, "re-eval a+b = 13");
    assert_eq(prod()(), 30, "re-eval a*b = 30");
    assert_eq(diff()(), 7, "re-eval a-b = 7");
}

// ============================================================================
// 12. toConstant produces independent ConstantSignal
// ============================================================================

void test_to_constant_independence() {
    std::cout << "\n=== toConstant independence ===" << std::endl;

    int counter = 0;
    Signal<int, TestEvent> varying([&counter]() -> FrozenSignal<int, TestEvent> {
        int c = ++counter;
        return [c](const TestEvent&) -> int { return c * 100; };
    });

    auto a = SignalTransform<int, TestEvent>::Constant(5);

    auto sum = Pushforward<int, TestEvent, int, int>(
        std::function<int(int,int)>([](int x, int y){ return x + y; })
    )(varying, a);

    // Freeze at a particular moment
    TestEvent e{{0}};
    auto frozen1 = sum.toConstant(e);
    auto frozen2 = sum.toConstant(e);

    int v1 = frozen1()();
    int v2 = frozen2()();

    // frozen1 should capture counter=1 → 100+5=105
    assert_eq(v1, 105, "toConstant #1: varying(1)+5 = 105");
    // frozen2 should capture counter=2 → 200+5=205
    assert_eq(v2, 205, "toConstant #2: varying(2)+5 = 205");

    // Each is truly constant — re-eval gives same value
    assert_eq(frozen1()(), 105, "re-eval frozen1 still 105");
    assert_eq(frozen2()(), 205, "re-eval frozen2 still 205");

    // Original varying signal still works and advances
    int v3 = sum()(e);
    assert_eq(v3, 305, "Original still varying: varying(3)+5 = 305");
}

// ============================================================================
// 13. Stress: many signals from the same temporary transform
// ============================================================================

void test_many_signals_from_temporary() {
    std::cout << "\n=== Many signals from one temporary ===" << std::endl;

    constexpr int N = 100;
    std::vector<ConstantSignal<int, TestEvent>> signals;

    for (int i = 0; i < N; ++i) {
        auto ci = SignalTransform<int, TestEvent>::Constant(i);
        // Every iteration creates and destroys a temporary Pushforward
        signals.push_back(
            Pushforward<int, TestEvent, int>(
                std::function<int(int)>([](int x){ return x * x; })
            )(ci)
        );
    }

    // Verify all signals are independent and correct
    bool all_correct = true;
    for (int i = 0; i < N; ++i) {
        if (signals[i]()() != i * i) {
            std::cerr << "  ✗ Signal " << i << ": expected " << i*i
                      << ", got " << signals[i]()() << std::endl;
            all_correct = false;
        }
    }
    assert(all_correct && "All 100 signals must compute i^2 correctly");
    std::cout << "  ✓ 100 signals from temporary Pushforwards all correct" << std::endl;
}

// ============================================================================
// 14. Double pointer dereferencing
// ============================================================================

void test_double_pointer() {
    std::cout << "\n=== Double pointer ===" << std::endl;

    auto a = SignalTransform<int, TestEvent>::Constant(8);
    auto b = SignalTransform<int, TestEvent>::Constant(2);
    auto* ptr_a = &a;
    auto** dptr_a = &ptr_a;

    auto result = Pushforward<int, TestEvent, int, int>(
        std::function<int(int,int)>([](int x, int y){ return x / y; })
    )(*dptr_a, b);

    assert_eq(result()(), 4, "Double pointer: 8/2 = 4");
}

// ============================================================================
// 15. Transform destroyed, signal survives scope
// ============================================================================

void test_transform_outlived_by_signal() {
    std::cout << "\n=== Transform destroyed, signal survives ===" << std::endl;

    Signal<int, TestEvent>* heap_sig = nullptr;

    {
        // Transform lives only in this scope
        Pushforward<int, TestEvent, int, int> ephemeral(
            std::function<int(int,int)>([](int x, int y){ return x - y; })
        );
        auto a = SignalTransform<int, TestEvent>::Constant(100);
        auto b = SignalTransform<int, TestEvent>::Constant(37);
        heap_sig = new Signal<int, TestEvent>(ephemeral(a, b));
    }
    // ephemeral is destroyed

    TestEvent e{{0}};
    assert_eq((*heap_sig)()(e), 63, "Signal survives transform: 100-37=63");
    delete heap_sig;
    std::cout << "  ✓ Signal outlives its producing transform" << std::endl;
}

// ============================================================================
// 16. Explicit destructor on named transform (heap-allocated)
// ============================================================================

void test_explicit_destructor_on_named_transform() {
    std::cout << "\n=== Explicit destructor on heap-allocated transform ===" << std::endl;

    auto a = SignalTransform<int, TestEvent>::Constant(50);
    auto b = SignalTransform<int, TestEvent>::Constant(8);

    // Heap-allocate the transform, produce a signal, then delete
    auto* transform = new Pushforward<int, TestEvent, int, int>(
        std::function<int(int,int)>([](int x, int y){ return x * y; })
    );
    auto sig = (*transform)(a, b);
    delete transform;
    transform = nullptr;

    TestEvent e{{0}};
    assert_eq(sig()(e), 400, "Heap transform deleted: 50*8=400");
    std::cout << "  ✓ Signal works after explicit delete of heap transform" << std::endl;
}

// ============================================================================
// 17. shared_ptr transform reset
// ============================================================================

void test_shared_ptr_transform_reset() {
    std::cout << "\n=== shared_ptr transform reset ===" << std::endl;

    auto a = SignalTransform<int, TestEvent>::Constant(12);
    auto b = SignalTransform<int, TestEvent>::Constant(3);

    auto transform = std::make_shared<Pushforward<int, TestEvent, int, int>>(
        std::function<int(int,int)>([](int x, int y){ return x + y; })
    );
    auto sig = (*transform)(a, b);

    // Reset the shared_ptr — transform object is destroyed
    transform.reset();

    TestEvent e{{0}};
    assert_eq(sig()(e), 15, "shared_ptr reset: 12+3=15");
    std::cout << "  ✓ Signal works after shared_ptr::reset() destroys transform" << std::endl;
}

// ============================================================================
// 18. Mutate (reassign) transform after signal creation
// ============================================================================

void test_mutate_transform_after_signal_creation() {
    std::cout << "\n=== Reassign transform after signal creation ===" << std::endl;

    auto a = SignalTransform<int, TestEvent>::Constant(20);
    auto b = SignalTransform<int, TestEvent>::Constant(4);

    Pushforward<int, TestEvent, int, int> m(
        std::function<int(int,int)>([](int x, int y){ return x - y; })
    );
    auto sig = m(a, b);  // Should compute 20-4=16

    // Reassign m to a completely different function
    m = Pushforward<int, TestEvent, int, int>(
        std::function<int(int,int)>([](int x, int y){ return x * y; })
    );
    auto sig2 = m(a, b);  // Should compute 20*4=80

    TestEvent e{{0}};
    assert_eq(sig()(e), 16, "Original signal after reassign: 20-4=16");
    assert_eq(sig2()(e), 80, "New signal from reassigned transform: 20*4=80");
    std::cout << "  ✓ Original signal unaffected by transform reassignment" << std::endl;
}

// ============================================================================
// 19. Multiple signals from transform destroyed in scope
// ============================================================================

void test_multiple_signals_transform_destroyed() {
    std::cout << "\n=== Multiple signals from scoped transform ===" << std::endl;

    auto a = SignalTransform<int, TestEvent>::Constant(10);
    auto b = SignalTransform<int, TestEvent>::Constant(5);
    auto c = SignalTransform<int, TestEvent>::Constant(3);

    Signal<int, TestEvent> sig1 = SignalTransform<int, TestEvent>::Constant(0);
    Signal<int, TestEvent> sig2 = SignalTransform<int, TestEvent>::Constant(0);
    Signal<int, TestEvent> sig3 = SignalTransform<int, TestEvent>::Constant(0);

    {
        Pushforward<int, TestEvent, int, int> add(
            std::function<int(int,int)>([](int x, int y){ return x + y; })
        );
        sig1 = add(a, b);   // 10+5=15
        sig2 = add(a, c);   // 10+3=13
        sig3 = add(b, c);   // 5+3=8
    }
    // add is destroyed here

    TestEvent e{{0}};
    assert_eq(sig1()(e), 15, "sig1 after destroy: 10+5=15");
    assert_eq(sig2()(e), 13, "sig2 after destroy: 10+3=13");
    assert_eq(sig3()(e), 8,  "sig3 after destroy: 5+3=8");
    std::cout << "  ✓ All 3 signals work after producing transform destroyed" << std::endl;
}

// ============================================================================
//  Main
// ============================================================================

int main() {
    try {
        std::cout << "=================================" << std::endl;
        std::cout << "Immutability Stress Tests" << std::endl;
        std::cout << "=================================" << std::endl;

        test_temporary_pushforward();
        test_temporary_pushforward_varying();
        test_shared_transform_independence();
        test_transform_metadata_isolation();
        test_signal_copy_independence();
        test_deep_temporary_composition();
        test_shared_input_signal();
        test_diamond_dependency();
        test_temporary_with_primitives();
        test_temporary_cast();
        test_arithmetic_independence();
        test_to_constant_independence();
        test_many_signals_from_temporary();
        test_double_pointer();
        test_transform_outlived_by_signal();
        test_explicit_destructor_on_named_transform();
        test_shared_ptr_transform_reset();
        test_mutate_transform_after_signal_creation();
        test_multiple_signals_transform_destroyed();

        std::cout << "\n=================================" << std::endl;
        std::cout << "ALL IMMUTABILITY TESTS PASSED!" << std::endl;
        std::cout << "=================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
