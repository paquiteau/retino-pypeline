"""Base builder function."""


def add2sinker(wf, connections, folder=None):
    """Add connections to sinker.

    connections should be a list of (node_name, edge, output_name)

    """
    if folder is None:
        folder = ""
    else:
        folder += ".@"
    sinker = wf.get_node("sinker")

    for con in connections:
        wf.connect(wf.get_node(con[0]), con[1], sinker, f"{folder}{con[2]}")

    return wf


def add2wf(wf, after_node, edge_out, node, edge_in):
    """Add node to wf after node connecting edge_out and edge_in."""
    if isinstance(after_node, str):
        after_node = wf.get_node(after_node)
    wf.connect([(after_node, node, [(edge_out, edge_in)])])
    return wf


def add2wf_dwim(wf, node_out, node_in, edges):
    """Connect two node with same edge label."""
    if isinstance(node_out, str):
        node_out = wf.get_node(node_out)
    if isinstance(node_in, str):
        node_in = wf.get_node(node_in)

    if not isinstance(edges, list):
        edges = [edges]
    for edge in edges:
        if isinstance(edge, str):
            wf.connect(node_out, edge, node_in, edge)
        elif isinstance(edge, tuple) and len(edge) == 2:
            wf.connect(node_out, edge[0], node_in, edge[1])
        else:
            raise ValueError(f"Unsupported edge config {edge}")
    return wf
