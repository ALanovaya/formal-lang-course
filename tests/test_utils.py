import pytest
import networkx as nx
from project.utils import get_graph, lift_labeled_two_cycles_graph_to_dot, Graph


@pytest.mark.parametrize(
    "graph_name, expected_metrics, expect_exception",
    [
        ("bzip", Graph(node_count=632, edge_count=556, edge_labels={"d", "a"}), False),
        ("invalid", None, True),
    ],
)
def test_get_graph(graph_name, expected_metrics, expect_exception):
    if expect_exception:
        with pytest.raises(Exception):
            get_graph(graph_name)
    else:
        actual_metrics = get_graph(graph_name)
        assert actual_metrics == expected_metrics


@pytest.mark.parametrize(
    "first_cycle_size, second_cycle_size, labels, expected_edges",
    [
        (
            2,
            2,
            ("a", "b"),
            [
                (0, 1, dict(label="a")),
                (1, 2, dict(label="a")),
                (2, 0, dict(label="a")),
                (0, 3, dict(label="b")),
                (3, 4, dict(label="b")),
                (4, 0, dict(label="b")),
            ],
        ),
        (-1, -1, ("a", "b"), pytest.raises(Exception)),
    ],
)
def test_lift_labeled_two_cycles_graph_to_dot(
    tmp_path, first_cycle_size, second_cycle_size, labels, expected_edges
):
    path_to_actual = tmp_path / "actual.dot"

    if isinstance(expected_edges, list):
        expected_graph = nx.DiGraph(expected_edges)
        lift_labeled_two_cycles_graph_to_dot(
            first_cycle_size, second_cycle_size, labels, path_to_actual
        )
        actual_graph = nx.DiGraph(nx.drawing.nx_pydot.read_dot(path_to_actual))
        assert nx.is_isomorphic(
            actual_graph, expected_graph, edge_match=dict.__eq__, node_match=dict.__eq__
        )
    else:
        with expected_edges:
            lift_labeled_two_cycles_graph_to_dot(
                first_cycle_size, second_cycle_size, labels, path_to_actual
            )
