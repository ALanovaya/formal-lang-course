import itertools

from networkx import MultiDiGraph
import numpy as np
from scipy.sparse import lil_array, vstack
from itertools import product

from project.fa import AdjacencyMatrixFA, intersect_automata
from project.automata_conversions import graph_to_nfa, regex_to_dfa


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """
    Compute the regular path query (RPQ) using tensor-based approach.

    Args:
    regex (str): The regular expression.
    graph (MultiDiGraph): The graph.
    start_nodes (set[int], optional): The set of start nodes. Defaults to None.
    final_nodes (set[int], optional): The set of final nodes. Defaults to None.

    Returns:
    set[tuple[int, int]]: The set of pairs of nodes that satisfy the RPQ.
    """

    # Convert the graph to an NFA and the regex to a DFA
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    regex_dfa = regex_to_dfa(regex)

    # Convert the NFA and the DFA to an adjacency matrix FA
    graph_mfa = AdjacencyMatrixFA(graph_nfa)
    regex_mfa = AdjacencyMatrixFA(regex_dfa)

    # Intersect the two automata
    inter_mfa = intersect_automata(graph_mfa, regex_mfa)

    # Compute the transitive closure of the intersected automaton
    inter_tc = inter_mfa.transitive_closure()

    # Compute the RPQ
    rpq = {
        (start, final)
        for start, final in itertools.product(start_nodes, final_nodes)
        for regex_start, regex_final in itertools.product(
            regex_dfa.start_states, regex_dfa.final_states
        )
        if inter_tc[
            inter_mfa.states[(start, regex_start)],
            inter_mfa.states[(final, regex_final)],
        ]
    }

    return rpq


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
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
    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    dfa = AdjacencyMatrixFA(regex_to_dfa(regex))

    front = initialize_front(start_nodes, dfa, nfa)
    visited = front.copy()
    symbols = set(nfa.boolean_decomposition.keys()) & set(
        dfa.boolean_decomposition.keys()
    )
    dfa_transposed = {
        sym: m.transpose() for sym, m in dfa.boolean_decomposition.items()
    }

    while front.sum() > 0:
        new_fronts = compute_new_front(
            front, nfa, dfa, symbols, start_nodes, dfa_transposed
        )
        front = update_front(front, visited, new_fronts)

    result = collect_results(visited, start_nodes, final_nodes, dfa, nfa)

    return result


def initialize_front(
    start_nodes: set[int], dfa: AdjacencyMatrixFA, nfa: AdjacencyMatrixFA
) -> lil_array:
    queue = []
    dfa_start_states_indices: set[int] = set(dfa.states[st] for st in dfa.start_states)
    for start in start_nodes:
        matrix = lil_array((len(dfa.states), len(nfa.states)), dtype=np.bool_)
        for dfa_start_state in dfa_start_states_indices:
            matrix[dfa_start_state, nfa.states[start]] = True
        queue.append(matrix)
    return vstack(queue, format="lil")


def compute_new_front(
    front: lil_array,
    nfa: AdjacencyMatrixFA,
    dfa: AdjacencyMatrixFA,
    symbols: set,
    start_nodes: set[int],
    dfa_transposed: dict,
) -> list:
    new_fronts = []
    for sym in symbols:
        new_front = front @ nfa.boolean_decomposition[sym]
        parts = []
        for i in range(len(start_nodes)):
            part = new_front[len(dfa.states) * i : len(dfa.states) * (i + 1)]
            parts.append(dfa_transposed[sym] @ part)
        new_fronts.append(vstack(parts, format="lil"))
    return new_fronts


def update_front(front: lil_array, visited: lil_array, new_fronts: list) -> lil_array:
    front = sum(new_fronts) > visited
    visited += front
    return front


def collect_results(
    visited: lil_array,
    start_nodes: set[int],
    final_nodes: set[int],
    dfa: AdjacencyMatrixFA,
    nfa: AdjacencyMatrixFA,
) -> set[tuple[int, int]]:
    result = set()
    dfa_final_states_indices: set[int] = set(dfa.states[st] for st in dfa.final_states)
    for i, start in enumerate(start_nodes):
        visited_part = visited[len(dfa.states) * i : len(dfa.states) * (i + 1)]
        for dfa_final, final in product(dfa_final_states_indices, final_nodes):
            if visited_part[dfa_final, nfa.states[final]]:
                result.add((start, final))
    return result
