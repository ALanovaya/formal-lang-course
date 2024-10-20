import itertools

from networkx import MultiDiGraph

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
