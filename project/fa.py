from collections import defaultdict
from typing import Iterable
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton as NFA,
    Symbol,
    State,
)
from scipy.sparse import csr_array, kron, csr_matrix, eye
import scipy.sparse.linalg as spla


class AdjacencyMatrixFA:
    def __init__(self, fa: NFA | None):
        """Initialize the automaton with either a given NFA or empty state"""
        if fa is None:
            self._initialize_empty_automaton()
        else:
            self._initialize_from_nfa(fa)

    def _initialize_empty_automaton(self):
        """Initialize an empty automaton."""
        self.start_states: set[State] = set()
        self.final_states: set[State] = set()
        self.states: dict[State, int] = {}
        self.boolean_decomposition: dict[Symbol, csr_array] = {}

    def _initialize_from_nfa(self, fa: NFA):
        """Initialize the automaton from a given NFA."""
        graph = fa.to_networkx()
        self.states = {st: i for i, st in enumerate(fa.states)}
        self.start_states: set[State] = fa.start_states
        self.final_states: set[State] = fa.final_states

        self.boolean_decomposition = self._build_boolean_decomposition(graph)

    def _build_boolean_decomposition(
        self, graph: MultiDiGraph
    ) -> dict[Symbol, csr_array]:
        """Build the boolean decomposition from the graph."""
        transitions = defaultdict(
            lambda: csr_matrix((len(self.states), len(self.states)), dtype=bool)
        )

        for st1, st2, label in graph.edges(data="label"):
            if label:
                sym = Symbol(label)
                transitions[sym][self.states[st1], self.states[st2]] = True

        return {sym: csr_array(matrix) for sym, matrix in transitions.items()}

    def accepts(self, input_word: Iterable[Symbol]) -> bool:
        """Determine if the automaton accepts the provided input word."""
        final_states_set = {self.states[state] for state in self.final_states}
        initial_states_set = {self.states[state] for state in self.start_states}

        initial_configuration = csr_matrix((1, len(self.states)), dtype=bool)
        for initial_state in initial_states_set:
            initial_configuration[0, initial_state] = True

        final_configuration = csr_matrix((1, len(self.states)), dtype=bool)
        for final_state in final_states_set:
            final_configuration[0, final_state] = True

        current_configuration = initial_configuration

        for symbol in input_word:
            if symbol not in self.boolean_decomposition:
                return False
            transition_matrix = self.boolean_decomposition[symbol]
            current_configuration = current_configuration.dot(transition_matrix)

        return (current_configuration.multiply(final_configuration)).nnz > 0

    def transitive_closure(self) -> csr_matrix:
        """Compute the transitive closure of the automaton."""
        number_of_states = len(self.states)
        if not self.boolean_decomposition:
            return eye(number_of_states, dtype=bool).tocsr()

        combined: csr_array = sum(self.boolean_decomposition.values())
        combined.setdiag(True)
        return spla.matrix_power(combined, number_of_states).astype(bool)

    def is_empty(self) -> bool:
        """Check if the automaton is empty."""
        final_states_indices = set(self.states[st] for st in self.final_states)
        start_states_indices = set(self.states[st] for st in self.start_states)
        transitive_closure = self.transitive_closure()
        empty = True

        for start in start_states_indices:
            for final in final_states_indices:
                empty = False if transitive_closure[start, final] else empty
        return empty


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Constructs a new automaton's adjacency matrix that is the intersection of two input automata.

    Args:
        automaton1 (AdjacencyMatrixFA): The first automaton adjacency matrix.
        automaton2 (AdjacencyMatrixFA): The second automaton adjacency matrix.

    Returns:
        AdjacencyMatrixFA: The resulting automaton adjacency matrix after intersection.
    """
    intersection_automaton = AdjacencyMatrixFA(None)

    # Create a mapping for new states based on combinations of the original states
    for state1 in automaton1.states:
        for state2 in automaton2.states:
            combined_state = State((state1, state2))
            combined_index = (
                len(automaton2.states) * automaton1.states[state1]
            ) + automaton2.states[state2]

            intersection_automaton.states[combined_state] = combined_index

            # Determine start states for the new automaton
            if state1 in automaton1.start_states and state2 in automaton2.start_states:
                intersection_automaton.start_states.add(combined_state)

            # Determine final states for the new automaton
            if state1 in automaton1.final_states and state2 in automaton2.final_states:
                intersection_automaton.final_states.add(combined_state)

    # Construct the boolean decomposition for the new automaton
    for transition_symbol in automaton1.boolean_decomposition:
        if transition_symbol in automaton2.boolean_decomposition:
            adj1 = automaton1.boolean_decomposition[transition_symbol]
            adj2 = automaton2.boolean_decomposition[transition_symbol]

            combined_adj = kron(adj1, adj2, format="csc")
            intersection_automaton.boolean_decomposition[transition_symbol] = (
                combined_adj
            )

    return intersection_automaton
