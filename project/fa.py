from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton as NFA, Symbol
import numpy as np
from numpy import bool_
from scipy.sparse import csr_array, kron, csr_matrix
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
        self.start_states: set[int] = set()
        self.final_states: set[int] = set()
        self.states: dict[Any, int] = {}
        self.boolean_decomposition: dict[Symbol, csr_array] = {}

    def _initialize_from_nfa(self, fa: NFA):
        """Initialize the automaton from a given NFA."""
        graph = fa.to_networkx()
        self.states = {st: i for i, st in enumerate(graph.nodes)}
        self.start_states = {self.states[st] for st in fa.start_states}
        self.final_states = {self.states[st] for st in fa.final_states}

        self.boolean_decomposition = self._build_boolean_decomposition(graph)

    def _build_boolean_decomposition(
        self, graph: MultiDiGraph
    ) -> dict[Symbol, csr_array]:
        """Build the boolean decomposition from the graph."""
        transitions = defaultdict(
            lambda: np.zeros((len(self.states), len(self.states)), dtype=bool_)
        )

        for st1, st2, label in graph.edges(data="label"):
            if label:
                sym = Symbol(label)
                transitions[sym][self.states[st1], self.states[st2]] = True

        return {sym: csr_array(matrix) for sym, matrix in transitions.items()}

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Check if the automaton accepts the given word."""
        word = list(word)
        number_of_states = len(self.states)

        stack = [self._create_config(st, word) for st in self.start_states]
        while stack:
            cfg = stack.pop()
            if not cfg.word and cfg.state in self.final_states:
                return True
            self._process_word(cfg, stack, number_of_states)

        return False

    @dataclass(frozen=True)
    class Config:
        state: int
        word: list[Symbol]

    def _create_config(self, state: int, word: list[Symbol]) -> Config:
        """Create a configuration for the automaton."""
        return self.Config(state, word)

    def _process_word(self, cfg: Config, stack: list[Config], number_of_states: int):
        """Process the current word configuration."""
        sym = cfg.word[0]
        adj = self.boolean_decomposition.get(sym)
        if adj is not None:
            for next_state in range(number_of_states):
                if adj[cfg.state, next_state]:
                    stack.append(self._create_config(next_state, cfg.word[1:]))

    def transitive_closure(self) -> csr_matrix:
        """Compute the transitive closure of the automaton."""
        number_of_states = len(self.states)
        if not self.boolean_decomposition:
            return np.eye(number_of_states, dtype=bool_)

        combined: csr_array = sum(self.boolean_decomposition.values())
        combined.setdiag(True)
        return spla.matrix_power(combined, number_of_states).astype(bool_)

    def is_empty(self) -> bool:
        """Check if the automaton is empty."""
        tc = self.transitive_closure()
        return not any(
            tc[start, final]
            for start, final in product(self.start_states, self.final_states)
        )


def intersect_automata(
    fa1: AdjacencyMatrixFA, fa2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Intersect two automata and return the resulting automaton."""
    result = AdjacencyMatrixFA(None)

    for st1, st2 in product(fa1.states, fa2.states):
        idx1, idx2 = fa1.states[st1], fa2.states[st2]
        result_idx = len(fa2.states) * idx1 + idx2

        if idx1 in fa1.start_states and idx2 in fa2.start_states:
            result.start_states.add(result_idx)
        if idx1 in fa1.final_states and idx2 in fa2.final_states:
            result.final_states.add(result_idx)

        result.states[(st1, st2)] = result_idx

    # Build the boolean decomposition for the resulting automaton
    for sym, adj1 in fa1.boolean_decomposition.items():
        if sym in fa2.boolean_decomposition:
            adj2 = fa2.boolean_decomposition[sym]
            result.boolean_decomposition[sym] = kron(adj1, adj2, format="csr")

    return result
