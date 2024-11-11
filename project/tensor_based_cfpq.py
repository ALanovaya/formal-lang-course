import networkx as nx
from scipy.sparse import csc_array
from pyformlang.rsa import RecursiveAutomaton
from project.fa import AdjacencyMatrixFA, intersect_automata
from project.automata_conversions import graph_to_nfa, rsm_to_nfa


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Perform a context-free path query (CFPQ) using the Tensor based algorithm."""

    # Convert the input graph to a MultiDiGraph
    directed_graph = nx.MultiDiGraph(graph)

    # Create NFA from the directed graph and from the recursive automaton
    nfa_graph = graph_to_nfa(directed_graph, start_nodes, final_nodes)
    adjacency_matrix_graph = AdjacencyMatrixFA(nfa_graph)
    adjacency_matrix_rsm = AdjacencyMatrixFA(rsm_to_nfa(rsm))

    # Initialize boolean decomposition
    initialize_boolean_decomposition(rsm, adjacency_matrix_graph, adjacency_matrix_rsm)

    # Compute intersections and update adjacency matrix
    adjacency_matrix_graph = compute_intersections(
        adjacency_matrix_rsm, adjacency_matrix_graph, rsm
    )

    # Collect valid start-end node pairs
    return collect_valid_pairs(adjacency_matrix_graph, rsm)


def initialize_boolean_decomposition(
    rsm: RecursiveAutomaton,
    adjacency_matrix_graph: AdjacencyMatrixFA,
    adjacency_matrix_rsm: AdjacencyMatrixFA,
):
    """Initialize boolean decomposition for nonterminals in RSM."""
    for nonterminal in rsm.boxes:
        for matrix in (adjacency_matrix_graph, adjacency_matrix_rsm):
            if nonterminal not in matrix.boolean_decomposition:
                matrix.boolean_decomposition[nonterminal] = csc_array(
                    (len(matrix.states), len(matrix.states)), dtype=bool
                )


def compute_intersections(
    adjacency_matrix_rsm: AdjacencyMatrixFA,
    adjacency_matrix_graph: AdjacencyMatrixFA,
    rsm: RecursiveAutomaton,
):
    """Compute intersections until convergence."""
    previous_count = 0
    current_count = None

    while previous_count != current_count:
        previous_count = current_count
        intersection_matrix = intersect_automata(
            adjacency_matrix_rsm, adjacency_matrix_graph
        )

        transitive_closure_matrix = intersection_matrix.transitive_closure()
        index_to_state_dict = {
            index: state for state, index in intersection_matrix.states.items()
        }

        update_boolean_decomposition(
            transitive_closure_matrix, index_to_state_dict, adjacency_matrix_graph, rsm
        )
        current_count = sum(
            adjacency_matrix_graph.boolean_decomposition[nonterminal].count_nonzero()
            for nonterminal in adjacency_matrix_graph.boolean_decomposition
        )

    return adjacency_matrix_graph


def update_boolean_decomposition(
    transitive_closure_matrix,
    index_to_state_dict,
    adjacency_matrix_graph: AdjacencyMatrixFA,
    rsm: RecursiveAutomaton,
):
    """Update boolean decomposition based on the transitive closure matrix."""
    source_indices, destination_indices = transitive_closure_matrix.nonzero()
    for index in range(len(source_indices)):
        src_idx = source_indices[index]
        dest_idx = destination_indices[index]

        src_state, src_node = index_to_state_dict[src_idx].value
        src_symbol, src_rsm_node = src_state.value
        dest_state, dest_node = index_to_state_dict[dest_idx].value
        dest_symbol, dest_rsm_node = dest_state.value

        if src_symbol != dest_symbol:
            continue

        valid_src_states = rsm.boxes[src_symbol].dfa.start_states
        valid_dest_states = rsm.boxes[src_symbol].dfa.final_states

        if src_rsm_node in valid_src_states and dest_rsm_node in valid_dest_states:
            adjacency_matrix_graph.boolean_decomposition[src_symbol][
                adjacency_matrix_graph.states[src_node],
                adjacency_matrix_graph.states[dest_node],
            ] = True


def collect_valid_pairs(
    adjacency_matrix_graph: AdjacencyMatrixFA, rsm: RecursiveAutomaton
) -> set[tuple[int, int]]:
    """Collect valid start-end node pairs."""
    valid_pairs = set()
    for start in adjacency_matrix_graph.start_states:
        for end in adjacency_matrix_graph.final_states:
            if adjacency_matrix_graph.boolean_decomposition[rsm.initial_label][
                adjacency_matrix_graph.states[start], adjacency_matrix_graph.states[end]
            ]:
                valid_pairs.add((start, end))
    return valid_pairs
