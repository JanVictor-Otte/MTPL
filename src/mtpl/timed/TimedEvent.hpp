#pragma once
#include "mtpl/core/Event.hpp"
#include "mtpl/core/Source.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
// ============================================================================
//  TimedEvent<P> — Event<P> + float time
//  evaluate()    — runs N loops, returns flat time-sorted lane
//  join()      — collapses N lanes into 1, sorted by time
// ============================================================================

namespace mtpl {
template<typename P>
struct TimedEvent : Event<P> {
    float time = 0.f;
    TimedEvent() = default;
    TimedEvent(float t, P p) : Event<P>{p}, time(t) {}
};

template<typename P> using TimedLane      = Lane<TimedEvent<P>>;
template<typename P> using TimedMultiLane = MultiLane<TimedEvent<P>>;
template<typename P> using TimedMorphism  = Morphism<TimedEvent<P>>;
template<typename P> using TimedSource    = Source<TimedEvent<P>>;

template<typename P>
TimedMorphism<P> join() {
    return makeLeaf<TimedEvent<P>>(
        Variadic,
        std::function<int(int)>([](int){ return 1; }),
        [](TimedMultiLane<P> in) {
            TimedLane<P> joined;
            for (const auto& lane : in)
                joined.insert(joined.end(), lane.begin(), lane.end());
            std::sort(joined.begin(), joined.end(),
                [](const TimedEvent<P>& a, const TimedEvent<P>& b) { return a.time < b.time; });
            return TimedMultiLane<P>{ joined };
        });
}

template<typename P>
TimedLane<P> evaluate(TimedSource<P> src, float period, int loops = 1) {
    TimedLane<P> joined;
    for (int i = 0; i < loops; ++i) {
        auto result = src.fn();
        float offset = i * period;
        for (const auto& lane : result)
            for (const auto& e : lane)
                joined.push_back(TimedEvent<P>{ e.time + offset, e.payload });
    }
    std::sort(joined.begin(), joined.end(),
        [](const TimedEvent<P>& a, const TimedEvent<P>& b) { return a.time < b.time; });
    return joined;
}

}