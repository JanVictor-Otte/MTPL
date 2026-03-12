#pragma once
#include <vector>

// ============================================================================
//  Event<P> — pure payload wrapper, no time
//  Lane<E>  — sequence of events of type E
//  MultiLane<E> — multiple parallel lanes, the currency between morphisms
//
//  E is the concrete event type (e.g. Event<P>, TimedEvent<P>, AudioEvent)
//  This lets Morphism<E>, Source<E> etc. work with any event type directly.
// ============================================================================
namespace mtpl {
template<typename P>
struct Event {
    P payload;
};

template<typename E> using Lane      = std::vector<E>;
template<typename E> using MultiLane = std::vector<Lane<E>>;
}