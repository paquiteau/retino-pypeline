"""Builder function, they extend a workflow to add nodes."""
from retino.workflows.preprocessing.nodes import (
    conditional_topup_task,
    coregistration_task,
    denoise_node,
    realign_task,
)


def _getsubid(i):
    return f"sub_{i:02d}"


def _subid_varname(i):
    return f"_sub_id_{i}"


def _get_key(d, k):
    return d[k]


def _add_to_wf(wf, after_node, edge_out, node, edge_in):
    if isinstance(after_node, str):
        after_node = wf.get_node(after_node)
    wf.connect(after_node, edge_out, node, edge_in)
    return wf


def add_realign(wf, name, after_node, edge):
    """Add a Realignment node."""
    realign = realign_task(name=name)
    return _add_to_wf(wf, after_node, edge, realign, "in_files")


def add_denoise_mag(wf, name, after_node, edge):
    """Add denoising step for magnitude input."""
    denoise = denoise_node(name)
    input_node = wf.get_node("input")
    wf.connect(input_node, "denoise_config", denoise, "denoise_str")
    return _add_to_wf(wf, after_node, edge, denoise, "in_file_mag")


def add_topup(wf, name, after_node, edge):
    """Add conditional topup correction."""
    input_node = wf.get_node("input")
    selectfiles = wf.get_node("selectfiles")
    condtopup = conditional_topup_task(name)
    # also adds mandatory connections
    wf.connect(input_node, "sequence", condtopup, "sequence")
    wf.connect(selectfiles, "data_opposite", condtopup, "data_opposite")
    return _add_to_wf(wf, after_node, edge, condtopup, "data")


def add_coreg(wf, name, after_node, edge):
    """Add coregistration step."""
    coreg = coregistration_task(name)
    # also add mandatory connections:
    wf.connect(wf.get_node("selectfiles"), "anat", coreg, "in.anat")
    return _add_to_wf(wf, after_node, edge, coreg, "in.func")


def add_to_sinker(wf, connections, folder=None):
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
