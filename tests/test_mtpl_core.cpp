#include "mtpl/mtpl.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace mtpl;

// Simple test event type
struct TestPayload {
    int value;
    
    bool operator==(const TestPayload& other) const {
        return value == other.value;
    }
};

using TestEvent = Event<TestPayload>;

// Helper functions
void assert_true(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        throw std::runtime_error("Test failed: " + message);
    }
    std::cout << "PASS: " << message << std::endl;
}

void assert_equal(int a, int b, const std::string& message) {
    if (a != b) {
        std::cerr << "FAIL: " << message << " (expected " << b << ", got " << a << ")" << std::endl;
        throw std::runtime_error("Test failed");
    }
    std::cout << "PASS: " << message << std::endl;
}

// ============================================================================
//  Signal Tests
// ============================================================================

void test_constant_signal() {
    std::cout << "\n=== Testing ConstantSignal ===" << std::endl;
    
    auto sig = SignalTransform<int, TestEvent>::Constant(42);
    auto frozen = sig();
    
    assert_equal(frozen(), 42, "ConstantSignal returns correct value");
    assert_equal(frozen(TestEvent{{10}}), 42, "ConstantSignal ignores event");
}

void test_signal_arithmetic() {
    std::cout << "\n=== Testing Signal Arithmetic ===" << std::endl;
    
    auto a = SignalTransform<int, TestEvent>::Constant(10);
    auto b = SignalTransform<int, TestEvent>::Constant(5);
    
    auto sum = a + b;
    auto diff = a - b;
    auto prod = a * b;
    auto quot = a / b;
    
    assert_equal(sum()(), 15, "Signal addition works");
    assert_equal(diff()(), 5, "Signal subtraction works");
    assert_equal(prod()(), 50, "Signal multiplication works");
    assert_equal(quot()(), 2, "Signal division works");
}

void test_signal_with_scalar() {
    std::cout << "\n=== Testing Signal + Scalar Arithmetic ===" << std::endl;
    
    auto sig = SignalTransform<int, TestEvent>::Constant(10);
    
    auto sum = sig + 5;
    auto prod = sig * 3;
    auto rsub = 20 - sig;
    
    assert_equal(sum()(), 15, "Signal + scalar works");
    assert_equal(prod()(), 30, "Signal * scalar works");
    assert_equal(rsub()(), 10, "Scalar - signal works");
}

void test_pushforward() {
    std::cout << "\n=== Testing Pushforward ===" << std::endl;
    
    auto a = SignalTransform<int, TestEvent>::Constant(3);
    auto b = SignalTransform<int, TestEvent>::Constant(4);
    
    auto hypotenuse = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int x, int y) { 
            return static_cast<int>(std::sqrt(x*x + y*y)); 
        })
    )(a, b);
    
    assert_equal(hypotenuse()(), 5, "Pushforward computes correctly");
}

void test_pushforward_mixed_types() {
    std::cout << "\n=== Testing Pushforward with Mixed Types ===" << std::endl;
    
    // Test 1: Mix Signal and ConstantSignal by reference
    auto sig1 = SignalTransform<int, TestEvent>::Constant(10);
    auto sig2 = SignalTransform<double, TestEvent>::Constant(3.5);
    
    auto result1 = Pushforward<double, TestEvent, int, double>(
        std::function<double(int, double)>([](int x, double y) { return x * y; })
    )(sig1, sig2);
    assert_equal(static_cast<int>(result1()()), 35, "Mixed Signal/ConstantSignal by reference");
    
    // Test 2: Mix pointers and values
    auto sig3 = SignalTransform<int, TestEvent>::Constant(7);
    auto* ptr_sig3 = &sig3;
    auto sig4 = SignalTransform<int, TestEvent>::Constant(3);
    
    auto result2 = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int x, int y) { return x + y; })
    )(ptr_sig3, sig4);
    assert_equal(result2()(), 10, "Mixed pointer and value");
    
    // Test 3: All pointers
    auto sig5 = SignalTransform<int, TestEvent>::Constant(8);
    auto sig6 = SignalTransform<int, TestEvent>::Constant(2);
    auto* ptr_sig5 = &sig5;
    auto* ptr_sig6 = &sig6;
    
    auto result3 = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int x, int y) { return x / y; })
    )(ptr_sig5, ptr_sig6);
    assert_equal(result3()(), 4, "All pointers");
    
    // Test 4: Three arguments with mixed types
    auto sig7 = SignalTransform<int, TestEvent>::Constant(5);
    auto sig8 = SignalTransform<int, TestEvent>::Constant(3);
    auto sig9 = SignalTransform<int, TestEvent>::Constant(2);
    auto* ptr_sig8 = &sig8;
    
    auto result4 = Pushforward<int, TestEvent, int, int, int>(
        std::function<int(int, int, int)>([](int x, int y, int z) { return x * y + z; })
    )(sig7, ptr_sig8, sig9);
    assert_equal(result4()(), 17, "Three arguments with mixed types");
    
    // Test 5: Const references
    const auto& const_ref_sig = sig1;
    
    auto result5 = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int x, int y) { return x - y; })
    )(const_ref_sig, sig4);
    assert_equal(result5()(), 7, "Const reference and value");
}

void test_pushforward_constant_propagation() {
    std::cout << "\n=== Testing Pushforward ConstantSignal Propagation ===" << std::endl;
    
    // When all inputs are ConstantSignal, result should be ConstantSignal
    auto c1 = SignalTransform<int, TestEvent>::Constant(6);
    auto c2 = SignalTransform<int, TestEvent>::Constant(4);
    
    auto result_const = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int x, int y) { return x + y; })
    )(c1, c2);
    
    // This should compile as a ConstantSignal
    ConstantSignal<int, TestEvent> verified_const = result_const;
    assert_equal(verified_const()(), 10, "All-constant inputs produce ConstantSignal");
    
    // Test with pointers to constants
    auto* ptr_c1 = &c1;
    auto* ptr_c2 = &c2;
    
    auto result_const2 = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int x, int y) { return x * y; })
    )(ptr_c1, ptr_c2);
    
    ConstantSignal<int, TestEvent> verified_const2 = result_const2;
    assert_equal(verified_const2()(), 24, "All-constant pointers produce ConstantSignal");
}

void test_pushforward_with_different_value_types() {
    std::cout << "\n=== Testing Pushforward with Different Value Types ===" << std::endl;
    
    // Test mixing int and double
    auto int_sig = SignalTransform<int, TestEvent>::Constant(10);
    auto double_sig = SignalTransform<double, TestEvent>::Constant(2.5);
    
    auto result_mixed = Pushforward<double, TestEvent, int, double>(
        std::function<double(int, double)>([](int x, double y) { return x / y; })
    )(int_sig, double_sig);
    assert_equal(static_cast<int>(result_mixed()()), 4, "Mixed int/double types");
    
    // Test with pointers to different types
    auto* ptr_int_sig = &int_sig;
    
    auto result_mixed2 = Pushforward<double, TestEvent, int, double>(
        std::function<double(int, double)>([](int x, double y) { return x + y; })
    )(ptr_int_sig, double_sig);
    assert_equal(static_cast<int>(result_mixed2()()), 12, "Pointer to int + value double");
    
    // Test returning different type than inputs
    auto sig_a = SignalTransform<int, TestEvent>::Constant(5);
    auto sig_b = SignalTransform<int, TestEvent>::Constant(2);
    
    auto result_bool = Pushforward<bool, TestEvent, int, int>(
        std::function<bool(int, int)>([](int x, int y) { return x > y; })
    )(sig_a, sig_b);
    assert_true(result_bool()(), "Result type different from input types");
}

void test_pushforward_composition() {
    std::cout << "\n=== Testing Pushforward Composition with Mixed Types ===" << std::endl;
    
    // Create base signals
    auto x = SignalTransform<int, TestEvent>::Constant(3);
    auto y = SignalTransform<int, TestEvent>::Constant(4);
    
    // First Pushforward: compute x^2 + y^2
    auto sum_of_squares = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int a, int b) { return a*a + b*b; })
    )(x, y);
    
    // Second Pushforward: take square root (using pointer to previous result)
    auto* ptr_sum = &sum_of_squares;
    
    auto magnitude = Pushforward<int, TestEvent, int>(
        std::function<int(int)>([](int s) { return static_cast<int>(std::sqrt(s)); })
    )(ptr_sum);
    
    assert_equal(magnitude()(), 5, "Composition with pointer to intermediate result");
    
    // More complex: combine results of multiple pushforwards
    auto diff = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int a, int b) { return a - b; })
    )(x, y);
    
    auto sum = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int a, int b) { return a + b; })
    )(x, y);
    
    auto* ptr_diff = &diff;
    auto* ptr_sum2 = &sum;
    
    // Combine the difference and sum
    auto combined = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int d, int s) { return d * s; })
    )(ptr_diff, ptr_sum2);
    
    assert_equal(combined()(), -7, "Composition of multiple Pushforward results via pointers");
}

void test_pushforward_with_operators() {
    std::cout << "\n=== Testing Operators with Mixed Types ===" << std::endl;
    
    // Test that operators work with pointers
    auto a = SignalTransform<int, TestEvent>::Constant(10);
    auto b = SignalTransform<int, TestEvent>::Constant(3);
    auto* ptr_a = &a;
    auto* ptr_b = &b;
    
    auto sum = (*ptr_a) + (*ptr_b);
    auto diff = (*ptr_a) - (*ptr_b);
    auto prod = (*ptr_a) * (*ptr_b);
    auto quot = (*ptr_a) / (*ptr_b);
    
    assert_equal(sum()(), 13, "Addition with dereferenced pointers");
    assert_equal(diff()(), 7, "Subtraction with dereferenced pointers");
    assert_equal(prod()(), 30, "Multiplication with dereferenced pointers");
    assert_equal(quot()(), 3, "Division with dereferenced pointers");
    
    // Test mixing pointer-based results with regular signals
    auto c = SignalTransform<int, TestEvent>::Constant(2);
    auto result = sum + c;
    
    assert_equal(result()(), 15, "Operator result combined with regular signal");
}

// ============================================================================
//  Morphism Tests
// ============================================================================

void test_leaf_morphism() {
    std::cout << "\n=== Testing Leaf Morphism ===" << std::endl;
    
    // Identity morphism
    auto identity = MultiLaneLeaf<TestEvent>([](MultiLane<TestEvent> in) { return in; });
    
    MultiLane<TestEvent> input = {{TestEvent{{1}}, TestEvent{{2}}}};
    auto output = identity(input);
    
    assert_equal(output.size(), 1, "Identity preserves number of lanes");
    assert_equal(output[0].size(), 2, "Identity preserves lane contents");
    assert_equal(output[0][0].payload.value, 1, "Identity preserves values");
}

void test_leaf_with_fixed_arity() {
    std::cout << "\n=== Testing Leaf with Fixed Arity ===" << std::endl;
    
    // Morphism that takes 2 lanes, outputs 1
    auto combiner = MultiLaneLeaf<TestEvent>(2, 1, [](MultiLane<TestEvent> in) {
        MultiLane<TestEvent> out;
        Lane<TestEvent> combined;
        for (auto& lane : in) {
            combined.insert(combined.end(), lane.begin(), lane.end());
        }
        out.push_back(combined);
        return out;
    });
    
    assert_true(combiner.inArity() == 2, "Fixed inArity is correct");
    assert_equal(combiner.outArity(2), 1, "Fixed outArity is correct");
    
    MultiLane<TestEvent> input = {
        {TestEvent{{1}}, TestEvent{{2}}},
        {TestEvent{{3}}, TestEvent{{4}}}
    };
    
    auto output = combiner(input);
    assert_equal(output.size(), 1, "Combiner produces 1 lane");
    assert_equal(output[0].size(), 4, "Combiner combines all events");
}

void test_compose() {
    std::cout << "\n=== Testing Compose ===" << std::endl;
    
    // First morphism: duplicate each lane
    auto duplicator = MultiLaneLeaf<TestEvent>(1, 2, [](MultiLane<TestEvent> in) {
        MultiLane<TestEvent> out;
        out.push_back(in[0]);
        out.push_back(in[0]);
        return out;
    });
    
    // Second morphism: take first lane only
    auto selector = MultiLaneLeaf<TestEvent>(2, 1, [](MultiLane<TestEvent> in) {
        MultiLane<TestEvent> out;
        out.push_back(in[0]);
        return out;
    });
    
    auto composed = Compose<TestEvent>(duplicator, selector);
    
    assert_equal(composed.inArity().value(), 1, "Composed inArity is first child's");
    assert_equal(composed.outArity(1), 1, "Composed outArity chains correctly");
    
    MultiLane<TestEvent> input = {{TestEvent{{42}}}};
    auto output = composed(input);
    
    assert_equal(output.size(), 1, "Compose produces correct number of lanes");
    assert_equal(output[0][0].payload.value, 42, "Compose preserves values");
}

void test_compose_flattening() {
    std::cout << "\n=== Testing Compose Flattening ===" << std::endl;
    
    auto m1 = MultiLaneLeaf<TestEvent>([](MultiLane<TestEvent> in) { return in; });
    auto m2 = MultiLaneLeaf<TestEvent>([](MultiLane<TestEvent> in) { return in; });
    auto m3 = MultiLaneLeaf<TestEvent>([](MultiLane<TestEvent> in) { return in; });
    
    auto c1 = Compose<TestEvent>(m1, m2);
    auto c2 = Compose<TestEvent>(c1, m3);
    
    // c2 should flatten c1's children
    // This is validated internally, test that it works
    MultiLane<TestEvent> input = {{TestEvent{{1}}}};
    auto output = c2(input);
    
    assert_equal(output.size(), 1, "Flattened compose works");
}

void test_tensor() {
    std::cout << "\n=== Testing Tensor ===" << std::endl;
    
    // Two morphisms that preserve lanes
    auto m1 = MultiLaneLeaf<TestEvent>([](MultiLane<TestEvent> in) { return in; });
    auto m2 = MultiLaneLeaf<TestEvent>([](MultiLane<TestEvent> in) { return in; });
    
    auto tensored = Tensor<TestEvent>(m1, m2);
    
    MultiLane<TestEvent> input = {{TestEvent{{1}}, TestEvent{{2}}}};
    auto output = tensored(input);
    
    assert_equal(output.size(), 2, "Tensor doubles output lanes");
    assert_equal(tensored.outArity(1), 2, "Tensor outArity sums children");
}

void test_arity_validation() {
    std::cout << "\n=== Testing Arity Validation ===" << std::endl;
    
    auto m1 = MultiLaneLeaf<TestEvent>(2, 1, [](MultiLane<TestEvent> in) {
        return MultiLane<TestEvent>{{in[0][0]}};
    });
    
    auto m2 = MultiLaneLeaf<TestEvent>(1, 1, [](MultiLane<TestEvent> in) {
        return in;
    });
    
    // This should work - m1 outputs 1, m2 expects 1
    try {
        auto composed = Compose<TestEvent>(m1, m2);
        std::cout << "PASS: Valid arity composition succeeded" << std::endl;
    } catch (...) {
        std::cerr << "FAIL: Valid arity composition threw" << std::endl;
        throw;
    }
    
    // This should fail - m2 outputs 1, m1 expects 2
    try {
        auto invalid = Compose<TestEvent>(m2, m1);
        std::cerr << "FAIL: Invalid arity composition should have thrown" << std::endl;
        throw std::runtime_error("Should have validated arity mismatch");
    } catch (const std::runtime_error& e) {
        std::cout << "PASS: Invalid arity composition threw correctly" << std::endl;
    }
}

// ============================================================================
//  Source Tests
// ============================================================================

void test_primitive_source() {
    std::cout << "\n=== Testing PrimitiveSource ===" << std::endl;
    
    auto prim = PrimitiveSource<TestEvent>(2, []() {
        return MultiLane<TestEvent>{
            {TestEvent{{1}}, TestEvent{{2}}},
            {TestEvent{{3}}, TestEvent{{4}}}
        };
    });
    
    assert_equal(prim.outArity(), 2, "PrimitiveSource has correct arity");
    
    auto output = prim();
    assert_equal(output.size(), 2, "PrimitiveSource generates correct lanes");
    assert_equal(output[0].size(), 2, "PrimitiveSource lane has correct events");
}

void test_source_from_primitive() {
    std::cout << "\n=== Testing Source from PrimitiveSource ===" << std::endl;
    
    auto prim = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{42}}}};
    });
    
    auto src = Source<TestEvent>(prim);
    
    auto output = src();
    assert_equal(output.size(), 1, "Source generates lanes");
    assert_equal(output[0][0].payload.value, 42, "Source preserves values");
}

void test_source_with_morphism() {
    std::cout << "\n=== Testing Source with Morphism ===" << std::endl;
    
    auto prim = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{10}}}};
    });
    
    auto doubler = MultiLaneLeaf<TestEvent>(1, 2, [](MultiLane<TestEvent> in) {
        MultiLane<TestEvent> out;
        out.push_back(in[0]);
        out.push_back(in[0]);
        return out;
    });
    
    auto src = Source<TestEvent>(prim, doubler);
    
    assert_equal(src.outArity(), 2, "Source with morphism has correct arity");
    
    auto output = src();
    assert_equal(output.size(), 2, "Morphism applied to source");
}

void test_morphism_application_operator() {
    std::cout << "\n=== Testing Morphism Application Operator ===" << std::endl;
    
    auto prim = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{5}}}};
    });
    
    auto tripler = MultiLaneLeaf<TestEvent>(1, 3, [](MultiLane<TestEvent> in) {
        MultiLane<TestEvent> out;
        out.push_back(in[0]);
        out.push_back(in[0]);
        out.push_back(in[0]);
        return out;
    });
    
    // Use operator() to apply morphism
    auto src = tripler(prim);
    
    assert_equal(src.outArity(), 3, "Operator application works");
    auto output = src();
    assert_equal(output.size(), 3, "Applied morphism generates lanes");
}

void test_chained_application() {
    std::cout << "\n=== Testing Chained Application ===" << std::endl;
    
    auto prim = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{1}}}};
    });
    
    auto m1 = MultiLaneLeaf<TestEvent>(1, 2, [](MultiLane<TestEvent> in) {
        return MultiLane<TestEvent>{in[0], in[0]};
    });
    
    auto m2 = MultiLaneLeaf<TestEvent>(2, 1, [](MultiLane<TestEvent> in) {
        Lane<TestEvent> combined;
        combined.insert(combined.end(), in[0].begin(), in[0].end());
        combined.insert(combined.end(), in[1].begin(), in[1].end());
        return MultiLane<TestEvent>{combined};
    });
    
    // Chain: prim -> m1 -> m2
    auto src = m2(m1(prim));
    
    assert_equal(src.outArity(), 1, "Chained application has correct arity");
    auto output = src();
    assert_equal(output[0].size(), 2, "Chained morphisms compose correctly");
}

// ============================================================================
//  Integration Tests
// ============================================================================

void test_full_pipeline() {
    std::cout << "\n=== Testing Full Pipeline ===" << std::endl;
    
    // Create a generator
    auto gen = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{
            TestEvent{{1}},
            TestEvent{{2}},
            TestEvent{{3}}
        }};
    });
    
    // Morphism to duplicate lanes
    auto dup = MultiLaneLeaf<TestEvent>(1, 2, [](MultiLane<TestEvent> in) {
        return MultiLane<TestEvent>{in[0], in[0]};
    });
    
    // Morphism to select first lane
    auto sel = MultiLaneLeaf<TestEvent>(2, 1, [](MultiLane<TestEvent> in) {
        return MultiLane<TestEvent>{in[0]};
    });
    
    // Build pipeline: gen -> dup -> sel
    auto pipeline = Compose<TestEvent>(dup, sel)(gen);
    
    auto output = pipeline();
    assert_equal(output.size(), 1, "Pipeline produces correct lanes");
    assert_equal(output[0].size(), 3, "Pipeline preserves events");
    assert_equal(output[0][0].payload.value, 1, "Pipeline maintains order");
}

// ============================================================================
//  EventLeaf Tests
// ============================================================================

void test_event_leaf() {
    std::cout << "\n=== Testing EventLeaf ===" << std::endl;
    
    // Create an event-level morphism that doubles payload values
    auto doubler = EventLeaf<TestEvent, int>(
        1,
        std::function<TestEvent(TestEvent, int)>([](TestEvent e, int factor) {
            e.payload.value *= factor;
            return e;
        }),
        SignalTransform<int, TestEvent>::Constant(2)
    );
    
    auto prim = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{5}}, TestEvent{{10}}}};
    });
    
    auto src = Source<TestEvent>(prim, doubler);
    auto output = src();
    
    assert_equal(output[0][0].payload.value, 10, "EventLeaf doubles first event");
    assert_equal(output[0][1].payload.value, 20, "EventLeaf doubles second event");
}

void test_event_leaf_multi_signal() {
    std::cout << "\n=== Testing EventLeaf with Multiple Signals ===" << std::endl;
    
    // Create an event-level morphism that adds two signals to payload
    auto adder = EventLeaf<TestEvent, int, int>(
        1,
        std::function<TestEvent(TestEvent, int, int)>([](TestEvent e, int a, int b) {
            e.payload.value += a + b;
            return e;
        }),
        SignalTransform<int, TestEvent>::Constant(3),
        SignalTransform<int, TestEvent>::Constant(7)
    );
    
    auto prim = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{5}}}};
    });
    
    auto src = Source<TestEvent>(prim, adder);
    auto output = src();
    
    assert_equal(output[0][0].payload.value, 15, "EventLeaf adds multiple signals correctly");
}

// ============================================================================
//  MultiLaneLeaf Tests
// ============================================================================

void test_parameterised_leaf() {
    std::cout << "\n=== Testing MultiLaneLeaf ===" << std::endl;
    
    // Create a lane-level morphism parameterised by a constant signal
    auto scale = MultiLaneLeaf<TestEvent, int>(
        1,
        std::function<MultiLane<TestEvent>(MultiLane<TestEvent>, int)>([](MultiLane<TestEvent> lanes, int factor) {
            MultiLane<TestEvent> out;
            for (const auto& lane : lanes) {
                Lane<TestEvent> scaled;
                for (const auto& e : lane) {
                    TestEvent se = e;
                    se.payload.value *= factor;
                    scaled.push_back(se);
                }
                out.push_back(scaled);
            }
            return out;
        }),
        SignalTransform<int, TestEvent>::Constant(3)
    );
    
    auto prim = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{2}}, TestEvent{{4}}}};
    });
    
    auto src = Source<TestEvent>(prim, scale);
    auto output = src();
    
    assert_equal(output[0][0].payload.value, 6, "MultiLaneLeaf scales first event");
    assert_equal(output[0][1].payload.value, 12, "MultiLaneLeaf scales second event");
}

void test_parameterised_leaf_multi_signal() {
    std::cout << "\n=== Testing MultiLaneLeaf with Multiple Signals ===" << std::endl;
    
    // Create a lane-level morphism parameterised by two signals
    auto offset_scale = MultiLaneLeaf<TestEvent, int, int>(
        1,
        std::function<MultiLane<TestEvent>(MultiLane<TestEvent>, int, int)>([](MultiLane<TestEvent> lanes, int offset, int scale) {
            MultiLane<TestEvent> out;
            for (const auto& lane : lanes) {
                Lane<TestEvent> transformed;
                for (const auto& e : lane) {
                    TestEvent te = e;
                    te.payload.value = te.payload.value * scale + offset;
                    transformed.push_back(te);
                }
                out.push_back(transformed);
            }
            return out;
        }),
        SignalTransform<int, TestEvent>::Constant(10),
        SignalTransform<int, TestEvent>::Constant(2)
    );
    
    auto prim = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{5}}}};
    });
    
    auto src = Source<TestEvent>(prim, offset_scale);
    auto output = src();
    
    assert_equal(output[0][0].payload.value, 20, "MultiLaneLeaf: 5 * 2 + 10 = 20");
}

// ============================================================================//  Signal Introspection Tests
// ============================================================================

void test_signal_introspection() {
    std::cout << "\n=== Testing Signal Introspection ===" << std::endl;
    
    // Create primitive signals
    auto a = SignalTransform<int, TestEvent>::Constant(5);
    auto b = SignalTransform<int, TestEvent>::Constant(3);
    
    // Create composite signal - this should track its inputs
    auto sum = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int x, int y) { return x + y; })
    )(a, b);
    
    // Verify that inputs are tracked
    if (sum.inputs().size() != 2) {
        throw std::runtime_error("Expected 2 inputs, got " + std::to_string(sum.inputs().size()));
    }
    std::cout << "  ✓ Pushforward tracks " << sum.inputs().size() << " input signals" << std::endl;
    
    // Test metadata
    sum.setMetadata(std::string("Addition operation"));
    if (!sum.metadata().has_value()) {
        throw std::runtime_error("Metadata should have a value");
    }
    if (std::any_cast<std::string>(sum.metadata()) != "Addition operation") {
        throw std::runtime_error("Metadata value mismatch");
    }
    std::cout << "  ✓ Metadata: " << std::any_cast<std::string>(sum.metadata()) << std::endl;
    
    // Test nested signal construction
    auto c = SignalTransform<int, TestEvent>::Constant(2);
    auto product = Pushforward<int, TestEvent, int, int>(
        std::function<int(int, int)>([](int x, int y) { return x * y; })
    )(sum, c);
    
    if (product.inputs().size() != 2) {
        throw std::runtime_error("Nested signal should track 2 immediate inputs");
    }
    std::cout << "  ✓ Nested signal construction maintains input tracking" << std::endl;
    
    // Verify the computation still works correctly
    auto frozen_product = product();
    TestEvent e{{0}};
    int result = frozen_product(e);
    assert_equal(result, 16, "Introspection doesn't affect computation: (5+3)*2 = 16");
}

// ============================================================================
//  Main Test Runner
// ============================================================================

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "MTPL Core Unit Tests" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Signal tests
        test_constant_signal();
        test_signal_arithmetic();
        test_signal_with_scalar();
        test_pushforward();
        test_pushforward_mixed_types();
        test_pushforward_constant_propagation();
        test_pushforward_with_different_value_types();
        test_pushforward_composition();
        test_pushforward_with_operators();
        
        // Morphism tests
        test_leaf_morphism();
        test_leaf_with_fixed_arity();
        test_compose();
        test_compose_flattening();
        test_tensor();
        test_arity_validation();
        
        // Source tests
        test_primitive_source();
        test_source_from_primitive();
        test_source_with_morphism();
        test_morphism_application_operator();
        test_chained_application();
        
        // Integration tests
        test_full_pipeline();
        
        // EventLeaf tests
        test_event_leaf();
        test_event_leaf_multi_signal();
        
        // MultiLaneLeaf tests
        test_parameterised_leaf();
        test_parameterised_leaf_multi_signal();
        
        // Signal introspection tests
        test_signal_introspection();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n========================================" << std::endl;
        std::cerr << "TEST SUITE FAILED" << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "========================================" << std::endl;
        return 1;
    }
}