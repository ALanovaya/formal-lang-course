from typing import Tuple, Set
import cfpq_data
import networkx as nx

class Graph:
    def __init__(self, node_count: int, edge_count: int, edge_labels: Set[str]):
        self.node_count = node_count
        self.edge_count = edge_count
        self.edge_labels = edge_labels

    def __eq__(self, other):
        return (
            self.node_count == other.node_count
            and self.edge_count == other.edge_count
            and self.edge_labels == other.edge_labels
        )

def get_graph(graph_name: str) -> Graph:
    graph = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph)
    edge_labels = {data["label"] for _, _, data in graph.edges.data()}
    return Graph(graph.number_of_nodes(), graph.number_of_edges(), edge_labels)

def lift_labeled_two_cycles_graph_to_dot(
    first_cycle_size: int,
    second_cycle_size: int,
    labels: Tuple[str, str],
    path: str
) -> None:
    graph = cfpq_data.labeled_two_cycles_graph(
        n=first_cycle_size,
        m=second_cycle_size,
        labels=labels
    )
    pydot_graph = nx.drawing.nx_pydot.to_pydot(graph)
    with open(path, "w") as f:
        f.write(pydot_graph.to_string().replace("\n", ""))
