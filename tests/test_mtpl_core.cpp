#include "mtpl/mtpl.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace mtpl;

int passed = 0, failed = 0;
#define TEST(name) \
    std::cout << "Testing " << name << "... "; \
    try {
#define PASS() \
        std::cout << "PASS" << std::endl; \
        ++passed; \
    } catch (const std::exception& e) { \
        std::cout << "FAIL: " << e.what() << std::endl; \
        ++failed; \
    }
#define ASSERT(cond) if (!(cond)) throw std::runtime_error(#cond)

struct TestPayload {
    int value = 0;
};

using TestEvent     = Event<TestPayload>;
using TestLane      = Lane<TestEvent>;
using TestMultiLane = MultiLane<TestEvent>;
using TestMorphism  = Morphism<TestEvent>;
using TestSource    = Source<TestEvent>;

void testEventBasics() {
    TEST("Event construction")
        TestEvent e{TestPayload{42}};
        ASSERT(e.payload.value == 42);
    PASS()
}

void testMorphismIdentity() {
    TEST("identity morphism")
        auto id = identity<TestEvent>();
        ASSERT(!id.inArity.has_value());  // Variadic
        
        TestMultiLane in = {
            TestLane{{TestPayload{1}}},
            TestLane{{TestPayload{2}}}
        };
        auto out = id(in);
        ASSERT(out.size() == 2);
        ASSERT(out[0][0].payload.value == 1);
        ASSERT(out[1][0].payload.value == 2);
    PASS()
}

void testMorphismCompose() {
    TEST("compose two morphisms")
        auto f = makeLeaf<TestEvent>(2, 1, [](TestMultiLane in) {
            TestLane merged;
            for (auto& lane : in)
                for (auto& e : lane) merged.push_back(e);
            return TestMultiLane{merged};
        });
        auto g = makeLeaf<TestEvent>(1, 1, [](TestMultiLane in) {
            auto lane = in[0];
            for (auto& e : lane) e.payload.value *= 2;
            return TestMultiLane{lane};
        });
        auto h = compose(f, g);
        
        TestMultiLane in = {
            TestLane{{TestPayload{3}}},
            TestLane{{TestPayload{5}}}
        };
        auto out = h(in);
        ASSERT(out.size() == 1);
        ASSERT(out[0].size() == 2);
        ASSERT(out[0][0].payload.value == 6);
        ASSERT(out[0][1].payload.value == 10);
    PASS()
}

void testMorphismTensor() {
    TEST("tensor two morphisms")
        auto f = makeLeaf<TestEvent>(1, 1, [](TestMultiLane in) {
            TestLane lane = in[0];
            for (auto& e : lane) e.payload.value += 10;
            return TestMultiLane{lane};
        });
        auto g = makeLeaf<TestEvent>(1, 1, [](TestMultiLane in) {
            TestLane lane = in[0];
            for (auto& e : lane) e.payload.value *= 3;
            return TestMultiLane{lane};
        });
        auto t = tensor(f, g);
        
        TestMultiLane in = {
            TestLane{{TestPayload{2}}},
            TestLane{{TestPayload{4}}}
        };
        auto out = t(in);
        ASSERT(out.size() == 2);
        ASSERT(out[0][0].payload.value == 12);  // 2 + 10
        ASSERT(out[1][0].payload.value == 6);  // 2 * 3
    PASS()
}

void testSourceApply() {
    TEST("apply morphism to source")
        TestSource src{2, []() {
            return TestMultiLane{
                TestLane{{TestPayload{1}}, {TestPayload{2}}},
                TestLane{{TestPayload{3}}}
            };
        }};
        auto m = makeLeaf<TestEvent>(2, 1, [](TestMultiLane in) {
            TestLane merged;
            for (auto& lane : in)
                for (auto& e : lane) merged.push_back(e);
            return TestMultiLane{merged};
        });
        auto result = apply(src, m);
        
        auto out = result.fn();
        ASSERT(out.size() == 1);
        ASSERT(out[0].size() == 3);
        ASSERT(out[0][0].payload.value == 1);
        ASSERT(out[0][1].payload.value == 2);
        ASSERT(out[0][2].payload.value == 3);
    PASS()
}

void testMerge() {
    TEST("merge sources")
        TestSource s1{1, []() {
            return TestMultiLane{TestLane{{TestPayload{10}}}};
        }};
        TestSource s2{1, []() {
            return TestMultiLane{TestLane{{TestPayload{20}}}};
        }};
        auto merged = merge<TestEvent>({s1, s2});
        
        ASSERT(merged.outArity == 2);
        auto out = merged.fn();
        ASSERT(out.size() == 2);
        ASSERT(out[0][0].payload.value == 10);
        ASSERT(out[1][0].payload.value == 20);
    PASS()
}

void testProject() {
    TEST("project lanes")
        TestSource src{3, []() {
            return TestMultiLane{
                TestLane{{TestPayload{1}}},
                TestLane{{TestPayload{2}}},
                TestLane{{TestPayload{3}}}
            };
        }};
        auto m = project<TestEvent>(std::vector<int>{0, 2});
        auto result = apply(src, m);
        
        auto out = result.fn();
        ASSERT(out.size() == 2);
        ASSERT(out[0][0].payload.value == 1);
        ASSERT(out[1][0].payload.value == 3);
    PASS()
}

void testSignalConstant() {
    TEST("constant signal")
        auto sig = constant<int, TestEvent>(42);
        auto frozen = sig();
        TestEvent e{TestPayload{0}};
        ASSERT(frozen(e) == 42);
    PASS()
}

void testSignalMap() {
    TEST("map over signal")
        auto sig = constant<int, TestEvent>(10);
        auto mapped = map<int, int, TestEvent>(
            std::function<int(int)>([](int x){ return x * 2; }), sig);
        auto frozen = mapped();
        TestEvent e{TestPayload{0}};
        ASSERT(frozen(e) == 20);
    PASS()
}

void testSignalPushforward() {
    TEST("pushforward binary function")
        auto a = constant<int, TestEvent>(3);
        auto b = constant<int, TestEvent>(4);
        auto sum = pushforward<int, TestEvent, int, int>(
            std::function<int(int,int)>([](int x, int y){ return x + y; }), a, b);
        auto frozen = sum();
        TestEvent e{TestPayload{0}};
        ASSERT(frozen(e) == 7);
    PASS()
}

void testSignalArithmetic() {
    TEST("signal arithmetic operators")
        auto a = constant<int, TestEvent>(10);
        auto b = constant<int, TestEvent>(3);
        auto sum  = a + b;
        auto diff = a - b;
        auto prod = a * b;
        auto quot = a / b;
        
        TestEvent e{TestPayload{0}};
        ASSERT(sum()(e) == 13);
        ASSERT(diff()(e) == 7);
        ASSERT(prod()(e) == 30);
        ASSERT(quot()(e) == 3);
    PASS()
}

void testConstantSignal() {
    TEST("ConstantSignal type")
        auto sig = constant<float, TestEvent>(3.14f);
        auto frozen = sig();  // Returns ConstantFrozenSignal
        ASSERT(std::abs(frozen() - 3.14f) < 0.001f);  // Can call without event
        
        TestEvent e{TestPayload{0}};
        ASSERT(std::abs(frozen(e) - 3.14f) < 0.001f);  // Also works with event
    PASS()
}

void testTimedEvent() {
    TEST("TimedEvent construction")
        TimedEvent<TestPayload> e{0.5f, TestPayload{42}};
        ASSERT(std::abs(e.time - 0.5f) < 0.001f);
        ASSERT(e.payload.value == 42);
    PASS()
}

void testEvaluate() {
    TEST("evaluate source with multiple loops")
        TimedSource<TestPayload> src{1, []() {
            return TimedMultiLane<TestPayload>{
                TimedLane<TestPayload>{{0.0f, TestPayload{1}}, {0.5f, TestPayload{2}}}
            };
        }};
        auto lane = evaluate(src, 1.0f, 3);
        
        ASSERT(lane.size() == 6);  // 2 events × 3 loops
        ASSERT(std::abs(lane[0].time - 0.0f) < 0.001f);
        ASSERT(std::abs(lane[1].time - 0.5f) < 0.001f);
        ASSERT(std::abs(lane[2].time - 1.0f) < 0.001f);
        ASSERT(std::abs(lane[3].time - 1.5f) < 0.001f);
        ASSERT(std::abs(lane[4].time - 2.0f) < 0.001f);
        ASSERT(std::abs(lane[5].time - 2.5f) < 0.001f);
    PASS()
}

void testJoin() {
    TEST("join multi-lane to single sorted lane")
        TimedSource<TestPayload> src{2, []() {
            return TimedMultiLane<TestPayload>{
                TimedLane<TestPayload>{{0.3f, TestPayload{1}}, {0.7f, TestPayload{2}}},
                TimedLane<TestPayload>{{0.1f, TestPayload{3}}, {0.5f, TestPayload{4}}}
            };
        }};
        auto m = join<TestPayload>();
        auto result = apply(src, m);
        
        auto out = result.fn();
        ASSERT(out.size() == 1);
        ASSERT(out[0].size() == 4);
        // Should be sorted by time
        ASSERT(std::abs(out[0][0].time - 0.1f) < 0.001f);
        ASSERT(out[0][0].payload.value == 3);
        ASSERT(std::abs(out[0][1].time - 0.3f) < 0.001f);
        ASSERT(out[0][1].payload.value == 1);
        ASSERT(std::abs(out[0][2].time - 0.5f) < 0.001f);
        ASSERT(out[0][2].payload.value == 4);
        ASSERT(std::abs(out[0][3].time - 0.7f) < 0.001f);
        ASSERT(out[0][3].payload.value == 2);
    PASS()
}

int main() {
    std::cout << "\n=== MTPL Core Tests ===\n" << std::endl;
    
    testEventBasics();
    testMorphismIdentity();
    testMorphismCompose();
    testMorphismTensor();
    testSourceApply();
    testMerge();
    testProject();
    
    std::cout << "\n=== Signal Tests ===\n" << std::endl;
    
    testSignalConstant();
    testSignalMap();
    testSignalPushforward();
    testSignalArithmetic();
    testConstantSignal();
    
    std::cout << "\n=== Timed Tests ===\n" << std::endl;
    
    testTimedEvent();
    testEvaluate();
    testJoin();
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    
    return failed > 0 ? 1 : 0;
}