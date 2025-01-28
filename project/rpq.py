import itertools

from networkx import MultiDiGraph
import numpy as np
from scipy.sparse import csr_matrix, vstack, spmatrix, lil_matrix
from itertools import product
from typing import Type

from project.fa import AdjacencyMatrixFA, intersect_automata
from project.automata_conversions import graph_to_nfa, regex_to_dfa


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    matrix_type: Type[spmatrix] = csr_matrix,
) -> set[tuple[int, int]]:
    """
    Compute the regular path query (RPQ) using tensor-based approach.

    Args:
    regex (str): The regular expression.
    graph (MultiDiGraph): The graph.
    start_nodes (set[int], optional): The set of start nodes.
    final_nodes (set[int], optional): The set of final nodes. Defaults to None.

    Returns:
    set[tuple[int, int]]: The set of pairs of nodes that satisfy the RPQ.
    """

    # Convert the graph to an NFA and the regex to a DFA
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    regex_dfa = regex_to_dfa(regex)

    # Convert the NFA and the DFA to an adjacency matrix FA
    graph_mfa = AdjacencyMatrixFA(graph_nfa, matrix_type)
    regex_mfa = AdjacencyMatrixFA(regex_dfa, matrix_type)

    # Intersect the two automata
    inter_mfa = intersect_automata(graph_mfa, regex_mfa, matrix_type)

    # Compute the transitive closure of the intersected automaton
    inter_tc = inter_mfa.transitive_closure()

    # Compute the RPQ
    rpq = set()

    if start_nodes and final_nodes:
        for start, final in itertools.product(start_nodes, final_nodes):
            if regex_dfa.start_states and regex_dfa.final_states:
                for regex_start, regex_final in itertools.product(
                    regex_dfa.start_states, regex_dfa.final_states
                ):
                    if inter_tc[
                        inter_mfa.states[(start, regex_start)],
                        inter_mfa.states[(final, regex_final)],
                    ]:
                        rpq.add((start, final))
    return rpq


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    matrix_type: Type[spmatrix] = csr_matrix,
) -> set[tuple[int, int]]:
    """
    Compute the regular path query (RPQ) using a matrix-based breadth-first search.

    Args:
    regex (str): The regular expression.
    graph (MultiDiGraph): The graph.
    start_nodes (set[int]): The set of start nodes.
    final_nodes (set[int]): The set of final nodes.

    Returns:
    set[tuple[int, int]]: The set of pairs of nodes that satisfy the RPQ.
    """
    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes), matrix_type)
    dfa = AdjacencyMatrixFA(regex_to_dfa(regex), matrix_type)

    front = initialize_front(start_nodes, dfa, nfa)
    visited = front.copy()
    symbols = set(nfa.boolean_decomposition.keys()) & set(
        dfa.boolean_decomposition.keys()
    )
    dfa_transposed = {
        sym: m.transpose() for sym, m in dfa.boolean_decomposition.items()
    }

    while front.sum() > 0:
        new_fronts = []
        for sym in symbols:
            new_front = front @ nfa.boolean_decomposition[sym]
            parts = []
            for i in range(len(start_nodes)):
                part = new_front[len(dfa.states) * i : len(dfa.states) * (i + 1)]
                parts.append(dfa_transposed[sym] @ part)
            new_fronts.append(vstack(parts))
        front = sum(new_fronts) > visited
        visited += front

    result = collect_results(visited, start_nodes, final_nodes, dfa, nfa)

    return result


def initialize_front(
    start_nodes: set[int], dfa: AdjacencyMatrixFA, nfa: AdjacencyMatrixFA
):
    queue = []
    dfa_start_states_indices: set[int] = set(dfa.states[st] for st in dfa.start_states)
    for start in start_nodes:
        matrix = lil_matrix((len(dfa.states), len(nfa.states)), dtype=np.bool_)
        for dfa_start_state in dfa_start_states_indices:
            matrix[dfa_start_state, nfa.states[start]] = True
        queue.append(matrix)
    return nfa.matrix_type(vstack(queue))


def collect_results(
    visited: lil_matrix,
    start_nodes: set[int],
    final_nodes: set[int],
    dfa: AdjacencyMatrixFA,
    nfa: AdjacencyMatrixFA,
) -> set[tuple[int, int]]:
    result = set()
    dfa_final_states_indices: set[int] = set(dfa.states[st] for st in dfa.final_states)
    for i, start in enumerate(start_nodes):
        visited_part = visited[len(dfa.states) * i : len(dfa.states) * (i + 1)]
        if dfa_final_states_indices and final_nodes:
            for dfa_final, final in product(dfa_final_states_indices, final_nodes):
                if visited_part[dfa_final, nfa.states[final]]:
                    result.add((start, final))
    return result
