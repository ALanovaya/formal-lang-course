import networkx as nx

from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    EpsilonNFA,
)


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: nx.MultiDiGraph, start_states: set[int], final_states: set[int]
) -> NondeterministicFiniteAutomaton:
    all_states = set(State(node) for node in graph.nodes)
    start_states_set = (
        set(State(node) for node in start_states) if start_states else all_states
    )
    final_states_set = (
        set(State(node) for node in final_states) if final_states else all_states
    )

    epsilon_nfa = EpsilonNFA(
        states=all_states,
        start_state=start_states_set,
        final_states=final_states_set,
    )

    for u, v, edge_data in graph.edges(data=True):
        epsilon_nfa.add_transition(State(u), edge_data["label"], State(v))

    return epsilon_nfa.remove_epsilon_transitions()
