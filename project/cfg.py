import pyformlang.cfg
from typing import List
import networkx as nx


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    """
    Converts a context-free grammar (CFG) to its weak normal form.

    Args:
        cfg (pyformlang.cfg.CFG): The input context-free grammar.

    Returns:
        pyformlang.cfg.CFG: A new context-free grammar in weak normal form.
    """
    # Get nullable symbols
    nullable = cfg.get_nullable_symbols()

    # Convert the grammar to normal form
    normal_form_cfg = cfg.to_normal_form()

    # Create new productions
    new_productions = set(normal_form_cfg.productions)
    for var in nullable:
        # Add epsilon production for each nullable variable
        new_productions.add(
            pyformlang.cfg.Production(
                pyformlang.cfg.Variable(var.value), [pyformlang.cfg.Epsilon()]
            )
        )

    # Remove useless symbols
    weak_normal_form_cfg = pyformlang.cfg.CFG(
        start_symbol=cfg.start_symbol, productions=new_productions
    )
    weak_normal_form_cfg = weak_normal_form_cfg.remove_useless_symbols()

    return weak_normal_form_cfg


def hellings_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """
    Perform a context-free path query (CFPQ) using the Hellings algorithm.
    Args:
        cfg (CFG): Context-free grammar defining the language.
        graph (nx.DiGraph): Directed graph to search for paths.
        start_nodes (set[int], optional): Nodes to start the path search from. Defaults to None, which considers all nodes.
        final_nodes (set[int], optional): Nodes to end the path search. Defaults to None, which considers all nodes.
    Returns:
        set[tuple[int, int]]: Pairs of nodes (start, end) representing paths that
        match the start symbol of the grammar.
    """
    wcnf = cfg_to_weak_normal_form(cfg)
    triples = set()

    # Add triples (terminal, start, end) for each edge in the graph
    for u, v, label in graph.edges(data="label"):
        if label:
            for var in _label_to_variables(wcnf, label):
                triples.add((var, u, v))

    # Add triples (variable, node, node) for each nullable variable
    nullable = wcnf.get_nullable_symbols()
    for var in nullable:
        for node in graph.nodes:
            triples.add((var, node, node))

    # Update triples until no new ones are added
    new_triples = set()
    while True:
        for var1, u1, v1 in triples:
            for var2, u2, v2 in triples:
                if v2 == u1:
                    for head in _body_to_head(wcnf, [var2, var1]):
                        new_triple = (head, u2, v1)
                        if new_triple not in triples:
                            new_triples.add(new_triple)
                if v1 == u2:
                    for head in _body_to_head(wcnf, [var1, var2]):
                        new_triple = (head, u1, v2)
                        if new_triple not in triples:
                            new_triples.add(new_triple)
        if not new_triples:
            break
        triples.update(new_triples)
        new_triples = set()

    # Return pairs of nodes corresponding to the start symbol of the grammar
    result = set()
    for var, u, v in triples:
        if var == wcnf.start_symbol:
            if (not start_nodes or u in start_nodes) and (
                not final_nodes or v in final_nodes
            ):
                result.add((u, v))
    return result


def _label_to_variables(
    wcnf: pyformlang.cfg.CFG, label: str
) -> List[pyformlang.cfg.Variable]:
    terminal = pyformlang.cfg.Terminal(label)
    return [
        prod.head
        for prod in wcnf.productions
        if terminal in prod.body and len(prod.body) == 1
    ]


def _body_to_head(wcnf: pyformlang.cfg.CFG, body):
    return {prod.head for prod in wcnf.productions if prod.body == body}
