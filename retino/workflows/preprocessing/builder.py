"""Builder function, they extend a workflow to add nodes."""
from retino.workflows.preprocessing.nodes import (
    conditional_topup_task,
    coregistration_task,
    cond_denoise_task,
    realign_task,
)


from ..base.builder import add2wf, add2wf_dwim

from .workflow_manager import (
    REALIGN,
    INPUT,
    FILES,
)


def add_realign(wf, name, after_node, edge):
    """Add a Realignment node."""
    realign = realign_task(name=name)
    add2wf(wf, after_node, edge, realign, "in_files")


def add_denoise_mag(wf, name, after_node, edge):
    """Add denoising step for magnitude input."""
    denoise = cond_denoise_task(name)
    add2wf_dwim(wf, FILES, denoise, ["noise_std_map", "mask"])
    add2wf_dwim(wf, INPUT, denoise, "denoise_str")
    add2wf(wf, after_node, edge, denoise, "data")


def add_denoise_cpx(wf, name, after_realign=False):
    """Add denoising step for magnitude input."""
    denoise = cond_denoise_task(name)
    add2wf_dwim(wf, FILES, denoise, ["noise_std_map", "mask", "data", "data+_phase"])
    add2wf_dwim(wf, INPUT, denoise, "denoise_str")
    if after_realign:
        add2wf_dwim(wf, REALIGN, denoise, ("realignment_parameters", "motion"))


def add_topup(wf, name, after_node, edge):
    """Add conditional topup correction."""
    condtopup = conditional_topup_task(name)
    # also adds mandatory connections
    add2wf_dwim(wf, INPUT, condtopup, "sequence")
    add2wf_dwim(FILES, condtopup, "data_opposite")
    add2wf(wf, after_node, edge, condtopup, "data")


def add_coreg(wf, name, after_node, edge):
    """Add coregistration step."""
    coreg = coregistration_task(name)
    # also add mandatory connections:
    wf.connect(wf.get_node(FILES), "anat", coreg, "in.anat")
    add2wf(wf, after_node, edge, coreg, "in.func")
