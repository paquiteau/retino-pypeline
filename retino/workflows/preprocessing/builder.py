"""Builder function, they extend a workflow to add nodes."""
from retino.workflows.preprocessing.nodes import (
    conditional_topup_task,
    coregistration_task,
    cond_denoise_task,
    realign_task,
)

from ..base.builder import add2wf, add2wf_dwim


def add_realign(wf, name, after_node, edge):
    """Add a Realignment node."""
    realign = realign_task(name=name)
    return add2wf(wf, after_node, edge, realign, "in_files")


def add_denoise_mag(wf, name, after_node, edge):
    """Add denoising step for magnitude input."""
    denoise = cond_denoise_task(name)
    add2wf_dwim(wf, "selectfiles", denoise, ["noise_std_map", "mask"])
    add2wf_dwim(wf, "input_node", denoise, "denoise_str")
    return add2wf(wf, after_node, edge, denoise, "data")


def add_denoise_cpx_after_r(wf, name, after_node, edge):
    """Add denoising step for magnitude input."""
    denoise = cond_denoise_task(name)
    add2wf_dwim(wf, "selectfiles", denoise, ["noise_std_map", "mask"])
    add2wf_dwim(wf, "input_node", denoise, "denoise_str")
    add2wf_dwim(wf, "selectfiles", denoise, "data")
    add2wf_dwim(wf, "selectfiles", denoise, "data_phase")
    add2wf_dwim(wf, "realign", denoise, ("realignment_parameters", "motion"))
    return wf


def add_topup(wf, name, after_node, edge):
    """Add conditional topup correction."""
    input_node = wf.get_node("input")
    selectfiles = wf.get_node("selectfiles")
    condtopup = conditional_topup_task(name)
    # also adds mandatory connections
    wf.connect(input_node, "sequence", condtopup, "sequence")
    wf.connect(selectfiles, "data_opposite", condtopup, "data_opposite")
    return add2wf(wf, after_node, edge, condtopup, "data")


def add_coreg(wf, name, after_node, edge):
    """Add coregistration step."""
    coreg = coregistration_task(name)
    # also add mandatory connections:
    wf.connect(wf.get_node("selectfiles"), "anat", coreg, "in.anat")
    return add2wf(wf, after_node, edge, coreg, "in.func")
