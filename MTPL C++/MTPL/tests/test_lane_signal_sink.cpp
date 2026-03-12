#include "mtpl/mtpl.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <random>
#include <set>
#include <string>

using namespace mtpl;

// Simple test event type
struct TestPayload {
    int value;
    bool operator==(const TestPayload& other) const { return value == other.value; }
};
using TestEvent = Event<TestPayload>;

void assert_eq(int a, int b, const std::string& msg) {
    if (a != b) {
        std::cerr << "FAIL: " << msg << " (expected " << b << ", got " << a << ")" << std::endl;
        throw std::runtime_error("Test failed: " + msg);
    }
    std::cout << "  ✓ " << msg << std::endl;
}

void assert_true(bool c, const std::string& msg) {
    if (!c) {
        std::cerr << "FAIL: " << msg << std::endl;
        throw std::runtime_error("Test failed: " + msg);
    }
    std::cout << "  ✓ " << msg << std::endl;
}

// ============================================================================
//  FrozenSignal hierarchy tests
// ============================================================================

void test_lane_frozen_signal() {
    std::cout << "\n=== LaneFrozenSignal ===" << std::endl;

    LaneFrozenSignal<int, TestEvent> lf(42);
    assert_eq(lf(), 42, "LaneFrozenSignal callable without event");
    assert_eq(lf(TestEvent{{99}}), 42, "LaneFrozenSignal ignores event");

    // ConstantFrozenSignal IS-A LaneFrozenSignal
    ConstantFrozenSignal<int, TestEvent> cf(7);
    const LaneFrozenSignal<int, TestEvent>& as_lane = cf;
    assert_eq(as_lane(), 7, "ConstantFrozenSignal upcast to LaneFrozenSignal");
}

// ============================================================================
//  LaneSignal tests
// ============================================================================

void test_lane_signal_basic() {
    std::cout << "\n=== LaneSignal basic ===" << std::endl;

    // Create a LaneSignal from a ConstantSignal (IS-A)
    auto constant = SignalTransform<int, TestEvent>::Constant(10);
    LaneSignal<int, TestEvent> ls(constant);  // copy-construct (base slice is OK)

    auto frozen = ls();
    assert_eq(frozen(), 10, "LaneSignal from ConstantSignal evaluates correctly");
}

void test_lane_signal_from_factory() {
    std::cout << "\n=== LaneSignal from factory ===" << std::endl;

    int counter = 0;
    LaneSignal<int, TestEvent> ls([&counter]() -> LaneFrozenSignal<int, TestEvent> {
        return LaneFrozenSignal<int, TestEvent>(++counter);
    });

    // Each call to operator()() should give a new value (different "lane")
    auto f1 = ls();
    auto f2 = ls();
    assert_eq(f1(), 1, "First lane freeze returns 1");
    assert_eq(f2(), 2, "Second lane freeze returns 2");
}

void test_event_to_lane_signal_conversion() {
    std::cout << "\n=== EventSignal → LaneSignal conversion ===" << std::endl;

    int counter = 0;
    Signal<int, TestEvent> eventSig([&counter]() -> FrozenSignal<int, TestEvent> {
        int c = ++counter;
        return [c](const TestEvent&) -> int { return c * 10; };
    });

    // Convert to LaneSignal
    auto laneSig = eventSig.toLaneSignal();

    // Each call to laneSig() collapses event variation to lane-constant
    auto f1 = laneSig();
    auto f2 = laneSig();
    assert_eq(f1(), 10, "First lane freeze of converted signal");
    assert_eq(f2(), 20, "Second lane freeze (different value per lane)");

    // But within a "lane", the frozen value is constant
    assert_eq(f1(), 10, "Lane-frozen is stable");
}

void test_lane_to_multilane_signal_conversion() {
    std::cout << "\n=== LaneSignal → MultiLaneSignal conversion ===" << std::endl;

    int counter = 0;
    LaneSignal<int, TestEvent> ls([&counter]() -> LaneFrozenSignal<int, TestEvent> {
        return LaneFrozenSignal<int, TestEvent>(++counter);
    });

    auto ms = ls.toMultiLaneSignal();
    // Each call to ms() is a new freeze — may produce a new value
    // (like ConstantSignal wrapping a random source).
    // The key property: within a single freeze, the value is consistent.
    auto f1 = ms();
    assert_eq(f1(), f1(), "MultiLaneSignal freeze is internally consistent");

    // Verify it can be used as a ConstantSignal
    ConstantSignal<int, TestEvent> verified = ms;
    auto f2 = verified();
    assert_eq(f2(), f2(), "Verified ConstantSignal freeze is consistent");
}

void test_event_to_multilane_signal_conversion() {
    std::cout << "\n=== EventSignal → MultiLaneSignal (via toConstant) ===" << std::endl;

    int counter = 0;
    Signal<int, TestEvent> eventSig([&counter]() -> FrozenSignal<int, TestEvent> {
        int c = ++counter;
        return [c](const TestEvent&) -> int { return c; };
    });

    auto ms = eventSig.toMultiLaneSignal();
    // Each freeze may produce a new value (fresh sample).
    // Within a freeze, the value is consistent across all events.
    auto f1 = ms();
    assert_eq(f1(), f1(), "toMultiLaneSignal freeze is internally consistent");

    // Verify it works as ConstantSignal
    ConstantSignal<int, TestEvent> verified = ms;
    assert_true(verified()() > 0, "toMultiLaneSignal converted signal evaluates");
}

// ============================================================================
//  SignalTransform three-overload dispatch tests
// ============================================================================

void test_signal_transform_dispatch() {
    std::cout << "\n=== SignalTransform three-overload dispatch ===" << std::endl;

    Pushforward<int, TestEvent, int, int> add(
        std::function<int(int,int)>([](int x, int y){ return x + y; })
    );

    // All ConstantSignal → ConstantSignal
    auto c1 = SignalTransform<int, TestEvent>::Constant(5);
    auto c2 = SignalTransform<int, TestEvent>::Constant(3);
    ConstantSignal<int, TestEvent> r1 = add(c1, c2);
    assert_eq(r1()(), 8, "All constant → ConstantSignal");

    // Mix LaneSignal and ConstantSignal → LaneSignal
    LaneSignal<int, TestEvent> l1(c1);  // wrap constant in lane
    LaneSignal<int, TestEvent> r2 = add(l1, c2);
    assert_eq(r2()(), 8, "Lane + Constant → LaneSignal");

    // At least one EventSignal → EventSignal
    Signal<int, TestEvent> e1([](){ return [](const TestEvent& e) -> int { return e.payload.value; }; });
    Signal<int, TestEvent> r3 = add(e1, c2);
    TestEvent te{{10}};
    assert_eq(r3()(te), 13, "Event + Constant → EventSignal");
}

// ============================================================================
//  LaneSignal arithmetic tests
// ============================================================================

void test_lane_signal_arithmetic() {
    std::cout << "\n=== LaneSignal arithmetic ===" << std::endl;

    auto c1 = SignalTransform<int, TestEvent>::Constant(10);
    auto c2 = SignalTransform<int, TestEvent>::Constant(3);
    LaneSignal<int, TestEvent> l1(c1);
    LaneSignal<int, TestEvent> l2(c2);

    auto sum  = l1 + l2;
    auto diff = l1 - l2;
    auto prod = l1 * l2;
    auto quot = l1 / l2;

    assert_eq(sum()(), 13, "LaneSignal addition");
    assert_eq(diff()(), 7, "LaneSignal subtraction");
    assert_eq(prod()(), 30, "LaneSignal multiplication");
    assert_eq(quot()(), 3, "LaneSignal division");

    // LaneSignal + scalar
    auto sum2 = l1 + 5;
    assert_eq(sum2()(), 15, "LaneSignal + scalar");

    // scalar + LaneSignal
    auto sum3 = 20 - l1;
    assert_eq(sum3()(), 10, "scalar - LaneSignal");

    // Unary minus
    auto neg = -l1;
    assert_eq(neg()(), -10, "Unary minus LaneSignal");
}

// ============================================================================
//  LaneLeaf tests
// ============================================================================

void test_lane_leaf_basic() {
    std::cout << "\n=== LaneLeaf basic ===" << std::endl;

    // Per-lane transform: add constant offset to each event
    auto offset = SignalTransform<int, TestEvent>::Constant(100);
    // ConstantSignal IS-A LaneSignal, so it should be accepted

    auto leaf = LaneLeaf<TestEvent, int>(
        1,
        std::function<Lane<TestEvent>(Lane<TestEvent>, int)>(
            [](Lane<TestEvent> lane, int off) -> Lane<TestEvent> {
                for (auto& e : lane) e.payload.value += off;
                return lane;
            }),
        offset
    );

    MultiLane<TestEvent> input = {{TestEvent{{1}}, TestEvent{{2}}, TestEvent{{3}}}};
    auto output = leaf(input);

    assert_eq(output.size(), 1, "LaneLeaf preserves lane count");
    assert_eq(output[0][0].payload.value, 101, "LaneLeaf applies transform");
    assert_eq(output[0][1].payload.value, 102, "LaneLeaf applies to second event");
    assert_eq(output[0][2].payload.value, 103, "LaneLeaf applies to third event");
}

void test_lane_leaf_per_lane_variation() {
    std::cout << "\n=== LaneLeaf per-lane signal variation ===" << std::endl;

    // Create a lane signal that gives different values per lane
    int counter = 0;
    LaneSignal<int, TestEvent> laneSig([&counter]() -> LaneFrozenSignal<int, TestEvent> {
        return LaneFrozenSignal<int, TestEvent>(++counter * 10);
    });

    auto leaf = LaneLeaf<TestEvent, int>(
        std::function<Lane<TestEvent>(Lane<TestEvent>, int)>(
            [](Lane<TestEvent> lane, int scale) -> Lane<TestEvent> {
                for (auto& e : lane) e.payload.value *= scale;
                return lane;
            }),
        laneSig
    );

    MultiLane<TestEvent> input = {
        {TestEvent{{1}}, TestEvent{{2}}},
        {TestEvent{{3}}, TestEvent{{4}}}
    };
    auto output = leaf(input);

    assert_eq(output.size(), 2, "LaneLeaf preserves lane count with varying signal");
    // Lane 0: counter=1, scale=10
    assert_eq(output[0][0].payload.value, 10, "Lane 0 event 0: 1*10=10");
    assert_eq(output[0][1].payload.value, 20, "Lane 0 event 1: 2*10=20");
    // Lane 1: counter=2, scale=20
    assert_eq(output[1][0].payload.value, 60, "Lane 1 event 0: 3*20=60");
    assert_eq(output[1][1].payload.value, 80, "Lane 1 event 1: 4*20=80");
}

void test_lane_leaf_in_pipeline() {
    std::cout << "\n=== LaneLeaf in pipeline ===" << std::endl;

    auto prim = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{5}}, TestEvent{{10}}}};
    });

    auto offset = SignalTransform<int, TestEvent>::Constant(100);
    auto laneLeaf = LaneLeaf<TestEvent, int>(
        1,
        std::function<Lane<TestEvent>(Lane<TestEvent>, int)>(
            [](Lane<TestEvent> lane, int off) -> Lane<TestEvent> {
                for (auto& e : lane) e.payload.value += off;
                return lane;
            }),
        offset
    );

    auto src = Source<TestEvent>(prim, laneLeaf);
    auto output = src();

    assert_eq(output[0][0].payload.value, 105, "LaneLeaf in pipeline: 5+100=105");
    assert_eq(output[0][1].payload.value, 110, "LaneLeaf in pipeline: 10+100=110");
}

// ============================================================================
//  Sink tests
// ============================================================================

void test_sink_basic() {
    std::cout << "\n=== Sink basic ===" << std::endl;

    std::vector<int> collected;
    auto sink = Sink<TestEvent>(
        [&collected](const MultiLane<TestEvent>& ml) {
            for (const auto& lane : ml)
                for (const auto& e : lane)
                    collected.push_back(e.payload.value);
        }
    );

    assert_true(!sink.inArity().has_value(), "Sink default is variadic");
    assert_eq(sink.outArity(5), 0, "Sink outArity is always 0");

    MultiLane<TestEvent> input = {{TestEvent{{1}}, TestEvent{{2}}}, {TestEvent{{3}}}};
    auto output = sink(input);

    assert_eq(output.size(), 0, "Sink returns empty multilane");
    assert_eq(collected.size(), 3, "Sink collected all events");
    assert_eq(collected[0], 1, "Sink collected correct values");
    assert_eq(collected[1], 2, "Sink collected correct values");
    assert_eq(collected[2], 3, "Sink collected correct values");
}

void test_sink_per_lane() {
    std::cout << "\n=== Sink per-lane ===" << std::endl;

    int lane_count = 0;
    auto sink = Sink<TestEvent>(
        Sink<TestEvent>::LaneSinkFn([&lane_count](const Lane<TestEvent>& lane) {
            lane_count++;
        })
    );

    MultiLane<TestEvent> input = {
        {TestEvent{{1}}}, {TestEvent{{2}}}, {TestEvent{{3}}}
    };
    sink(input);

    assert_eq(lane_count, 3, "Per-lane sink iterated over all lanes");
}

void test_sink_per_event() {
    std::cout << "\n=== Sink per-event ===" << std::endl;

    std::vector<int> values;
    auto sink = Sink<TestEvent>(
        Sink<TestEvent>::EventSinkFn([&values](const TestEvent& e) {
            values.push_back(e.payload.value);
        })
    );

    MultiLane<TestEvent> input = {
        {TestEvent{{10}}, TestEvent{{20}}},
        {TestEvent{{30}}}
    };
    sink(input);

    assert_eq(values.size(), 3, "Per-event sink iterated over all events");
    assert_eq(values[0], 10, "Per-event sink correct order");
    assert_eq(values[2], 30, "Per-event sink correct last value");
}

void test_sink_with_fixed_arity() {
    std::cout << "\n=== Sink with fixed arity ===" << std::endl;

    std::vector<int> collected;
    auto sink = Sink<TestEvent>(
        2,
        [&collected](const MultiLane<TestEvent>& ml) {
            for (const auto& lane : ml)
                for (const auto& e : lane)
                    collected.push_back(e.payload.value);
        }
    );

    assert_eq(sink.inArity().value(), 2, "Sink with fixed arity is 2");

    // Correct input
    MultiLane<TestEvent> input = {{TestEvent{{1}}}, {TestEvent{{2}}}};
    sink(input);
    assert_eq(collected.size(), 2, "Sink accepted correct arity");

    // Wrong input should throw
    bool threw = false;
    try {
        MultiLane<TestEvent> bad = {{TestEvent{{1}}}};
        sink(bad);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert_true(threw, "Sink rejects wrong arity");
}

void test_sink_in_compose() {
    std::cout << "\n=== Sink in Compose ===" << std::endl;

    std::vector<int> collected;
    auto doubler = MultiLaneLeaf<TestEvent>(1, 2, [](MultiLane<TestEvent> in) {
        return MultiLane<TestEvent>{in[0], in[0]};
    });
    auto sink = Sink<TestEvent>(
        2,
        [&collected](const MultiLane<TestEvent>& ml) {
            for (const auto& lane : ml)
                for (const auto& e : lane)
                    collected.push_back(e.payload.value);
        }
    );

    auto pipeline = Compose<TestEvent>(doubler, sink);
    assert_eq(pipeline.outArity(1), 0, "Compose ending in Sink has outArity 0");

    MultiLane<TestEvent> input = {{TestEvent{{42}}}};
    auto output = pipeline(input);

    assert_eq(output.size(), 0, "Compose with Sink returns empty");
    assert_eq(collected.size(), 2, "Sink in compose collected duplicated data");
    assert_eq(collected[0], 42, "Collected correct value");
}

void test_sink_tensor_with_identity() {
    std::cout << "\n=== Tensor(Sink, Identity) — sink and propagate ===" << std::endl;

    std::vector<int> sunk;
    auto sink = Sink<TestEvent>(
        [&sunk](const MultiLane<TestEvent>& ml) {
            for (const auto& lane : ml)
                for (const auto& e : lane)
                    sunk.push_back(e.payload.value);
        }
    );
    auto id = Identity<TestEvent>();

    auto both = Tensor<TestEvent>(sink, id);

    MultiLane<TestEvent> input = {{TestEvent{{7}}, TestEvent{{8}}}};
    auto output = both(input);

    // Sink contributes 0 lanes, Identity contributes 1 lane
    assert_eq(output.size(), 1, "Tensor(Sink, Id) passes through identity lanes");
    assert_eq(output[0][0].payload.value, 7, "Identity preserved values");
    assert_eq(output[0][1].payload.value, 8, "Identity preserved second value");
    assert_eq(sunk.size(), 2, "Sink side-effect collected data");
    assert_eq(sunk[0], 7, "Sink got correct data");
}

void test_sink_in_pipeline() {
    std::cout << "\n=== Sink in full pipeline ===" << std::endl;

    std::vector<int> saved;
    auto prim = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{100}}, TestEvent{{200}}}};
    });

    auto sink = Sink<TestEvent>([&saved](const MultiLane<TestEvent>& ml) {
        for (const auto& lane : ml)
            for (const auto& e : lane)
                saved.push_back(e.payload.value);
    });

    auto src = sink(prim);
    assert_eq(src.outArity(), 0, "Source ending in sink has outArity 0");

    auto output = src();
    assert_eq(output.size(), 0, "Pipeline ending in sink produces nothing");
    assert_eq(saved.size(), 2, "Sink saved all primitives");
    assert_eq(saved[0], 100, "Saved correct first value");
    assert_eq(saved[1], 200, "Saved correct second value");
}

// ============================================================================
//  Concept verification tests (compile-time checks)
// ============================================================================

void test_concept_hierarchy() {
    std::cout << "\n=== Concept hierarchy ===" << std::endl;

    // ConstantSignal satisfies all three levels
    static_assert(AnySignalType<ConstantSignal<int, TestEvent>, TestEvent>);
    static_assert(LaneSignalType<ConstantSignal<int, TestEvent>, TestEvent>);
    static_assert(ConstantSignalType<ConstantSignal<int, TestEvent>, TestEvent>);
    std::cout << "  ✓ ConstantSignal satisfies all signal concepts" << std::endl;

    // LaneSignal satisfies Any and Lane, but not Constant
    static_assert(AnySignalType<LaneSignal<int, TestEvent>, TestEvent>);
    static_assert(LaneSignalType<LaneSignal<int, TestEvent>, TestEvent>);
    static_assert(!ConstantSignalType<LaneSignal<int, TestEvent>, TestEvent>);
    std::cout << "  ✓ LaneSignal satisfies Any + Lane but not Constant" << std::endl;

    // Signal (EventSignal) satisfies only Any
    static_assert(AnySignalType<Signal<int, TestEvent>, TestEvent>);
    static_assert(!LaneSignalType<Signal<int, TestEvent>, TestEvent>);
    static_assert(!ConstantSignalType<Signal<int, TestEvent>, TestEvent>);
    std::cout << "  ✓ EventSignal satisfies only AnySignal" << std::endl;

    // EventOnly / LaneOnly concepts
    static_assert(EventOnlySignalType<Signal<int, TestEvent>, TestEvent>);
    static_assert(!EventOnlySignalType<LaneSignal<int, TestEvent>, TestEvent>);
    static_assert(LaneOnlySignalType<LaneSignal<int, TestEvent>, TestEvent>);
    static_assert(!LaneOnlySignalType<ConstantSignal<int, TestEvent>, TestEvent>);
    std::cout << "  ✓ EventOnly / LaneOnly concepts correct" << std::endl;

    // Aliases
    static_assert(MultiLaneSignalType<ConstantSignal<int, TestEvent>, TestEvent>);
    static_assert(std::is_same_v<EventSignal<int, TestEvent>, Signal<int, TestEvent>>);
    static_assert(std::is_same_v<MultiLaneSignal<int, TestEvent>, ConstantSignal<int, TestEvent>>);
    std::cout << "  ✓ Type aliases correct" << std::endl;
}

// ============================================================================
//  Integration: microscopy-style pipeline sketch
// ============================================================================

void test_microscopy_pipeline_sketch() {
    std::cout << "\n=== Microscopy pipeline sketch ===" << std::endl;

    // Simulate: filename → read data → fork (snapshot for saving, data for processing)
    // → process → sink(save snapshot) ⊗ identity → combine → done

    std::vector<int> saved_snapshots;

    auto gen = PrimitiveSource<TestEvent>(1, []() {
        return MultiLane<TestEvent>{{TestEvent{{42}}, TestEvent{{84}}}};
    });

    // Fork: duplicate lane
    auto fork = MultiLaneLeaf<TestEvent>(1, 2, [](MultiLane<TestEvent> in) {
        return MultiLane<TestEvent>{in[0], in[0]};  // lane 0 = snapshot, lane 1 = processing
    });

    // Process lane 1 (multiply by 2)
    auto process = MultiLaneLeaf<TestEvent>(2, 2, [](MultiLane<TestEvent> in) {
        auto processed = in[1];
        for (auto& e : processed) e.payload.value *= 2;
        return MultiLane<TestEvent>{in[0], processed};
    });

    // Sink lane 0 (save snapshot), propagate lane 1
    auto saveLane0 = Sink<TestEvent>(
        2,
        [&saved_snapshots](const MultiLane<TestEvent>& ml) {
            // Only save lane 0
            for (const auto& e : ml[0])
                saved_snapshots.push_back(e.payload.value);
        }
    );

    // Project to get lane 1 only after saving
    auto getLane1 = MultiLaneLeaf<TestEvent>(2, 1, [](MultiLane<TestEvent> in) {
        return MultiLane<TestEvent>{in[1]};
    });

    // Use Tensor(Sink, Identity) pattern on the full multilane, then project
    auto save_all = Sink<TestEvent>(
        [&saved_snapshots](const MultiLane<TestEvent>& ml) {
            for (const auto& e : ml[0])
                saved_snapshots.push_back(e.payload.value);
        }
    );

    auto sink_and_pass = Tensor<TestEvent>(save_all, Identity<TestEvent>());
    auto pipeline = Compose<TestEvent>(fork, process, sink_and_pass, getLane1);

    auto src = pipeline(gen);
    auto output = src();

    assert_eq(saved_snapshots.size(), 2, "Saved 2 snapshot events");
    assert_eq(saved_snapshots[0], 42, "Snapshot preserved original value");
    assert_eq(saved_snapshots[1], 84, "Snapshot preserved second original value");

    assert_eq(output.size(), 1, "Pipeline produces 1 output lane");
    assert_eq(output[0][0].payload.value, 84, "Processed: 42*2=84");
    assert_eq(output[0][1].payload.value, 168, "Processed: 84*2=168");
}

// ============================================================================
//  Main
// ============================================================================

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "LaneSignal + Sink Tests" << std::endl;
        std::cout << "========================================" << std::endl;

        // FrozenSignal hierarchy
        test_lane_frozen_signal();

        // LaneSignal
        test_lane_signal_basic();
        test_lane_signal_from_factory();
        test_event_to_lane_signal_conversion();
        test_lane_to_multilane_signal_conversion();
        test_event_to_multilane_signal_conversion();

        // SignalTransform dispatch
        test_signal_transform_dispatch();

        // LaneSignal arithmetic
        test_lane_signal_arithmetic();

        // LaneLeaf
        test_lane_leaf_basic();
        test_lane_leaf_per_lane_variation();
        test_lane_leaf_in_pipeline();

        // Sink
        test_sink_basic();
        test_sink_per_lane();
        test_sink_per_event();
        test_sink_with_fixed_arity();
        test_sink_in_compose();
        test_sink_tensor_with_identity();
        test_sink_in_pipeline();

        // Concepts
        test_concept_hierarchy();

        // Integration
        test_microscopy_pipeline_sketch();

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
