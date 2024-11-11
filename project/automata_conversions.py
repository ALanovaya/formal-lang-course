import networkx as nx

from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    EpsilonNFA,
)
from pyformlang.rsa import RecursiveAutomaton


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


def rsm_to_nfa(automaton: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    result_nfa = NondeterministicFiniteAutomaton()

    for box_name, dfa_container in automaton.boxes.items():
        deterministic_automaton = dfa_container.dfa
        start_end_states = deterministic_automaton.start_states.union(
            deterministic_automaton.final_states
        )

        for state in start_end_states:
            combined_state = State((box_name, state))
            add_state_to_nfa(deterministic_automaton, state, combined_state, result_nfa)

        add_transitions_to_nfa(deterministic_automaton, box_name, result_nfa)

    return result_nfa


def add_state_to_nfa(deterministic_automaton, state, combined_state, result_nfa):
    if state in deterministic_automaton.final_states:
        result_nfa.add_final_state(combined_state)
    if state in deterministic_automaton.start_states:
        result_nfa.add_start_state(combined_state)


def add_transitions_to_nfa(deterministic_automaton, box_name, result_nfa):
    transitions = deterministic_automaton.to_networkx().edges(data="label")
    for origin, destination, transition_label in transitions:
        initial_state = State((box_name, origin))
        target_state = State((box_name, destination))
        result_nfa.add_transition(initial_state, transition_label, target_state)
