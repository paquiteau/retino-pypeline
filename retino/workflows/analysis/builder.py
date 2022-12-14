"""Builder function for the analysis."""

from ..base.builder import add2wf, add2wf_dwim

from .nodes import (
    cond_design_matrix_node,
    design_matrix_node,
    contrast_node,
    contrast_glob_node,
    phase_map_node,
    tsnr_map_node,
)


def connect_volumetric_tr(wf, node):
    """Link volumetric_tr from input_node to node."""
    input_node = wf.get_node("input")
    wf.connect(input_node, "volumetric_tr", node, "volumetric_tr")
    return wf


def add_design_matrix(wf, n_cycles, mode="clock", tr_unit="s"):
    """Add Design Matrix Interface to workflow."""
    file_node = wf.get_node("selectfiles")
    dm = design_matrix_node(
        n_cycles,
        clockwise=(mode == "clock"),
        extra_name=f"_{mode}",
    )
    dm.inputs.tr_unit = tr_unit

    add2wf(wf, file_node, f"data_{mode}", dm, "data_file")
    add2wf(wf, file_node, f"motion_{mode}", dm, "motion_file")

    return connect_volumetric_tr(wf, dm)


def add_cond_design_matrix(wf, n_cycles, mode="clock", tr_units="s"):
    """Add a conditional design matrix node."""
    file_node = wf.get_node("selectfiles")
    dm = cond_design_matrix_node(
        n_cycles,
        clockwise=(mode == "clock"),
        extra_name=f"_{mode}",
    )
    dm.inputs.tr_unit = tr_units

    add2wf(wf, file_node, f"data_{mode}", dm, "data_file")
    add2wf(wf, file_node, f"motion_{mode}", dm, "motion_file")
    add2wf_dwim(wf, "input", dm, ["volumetric_tr", "preproc_code"])

    return wf


def add_contrast(wf, mode="clock"):
    """Add contrast Interface for specific mode to workflow."""
    contrast = contrast_node(extra_name=f"_{mode}")
    add2wf(wf, f"design_{mode}", "design_matrix", contrast, "design_matrices")
    add2wf(wf, "selectfiles", f"data_{mode}", contrast, "fmri_timeseries")

    return connect_volumetric_tr(wf, contrast)


def add_contrast_clock(wf):
    """Add contrast Interface for clockwise to workflow."""
    return add_contrast(wf, mode="clock")


def add_contrast_anticlock(wf):
    """Add contrast Interface for anticlockwise to workflow."""
    return add_contrast(wf, mode="anticlock")


def add_contrast_glob(wf):
    """Add contrast for fixed effect stats."""
    contrast = contrast_glob_node()

    for mode in ["clock", "anticlock"]:
        add2wf(wf, f"design_{mode}", "design_matrix", contrast, f"dm_{mode}")
        add2wf(wf, "selectfiles", f"data_{mode}", contrast, f"fmri_{mode}")

    return connect_volumetric_tr(wf, contrast)


def _gk(d, k):
    return d[k]


def add_phase_map(wf, threshold):
    """Add phase map to workflow."""
    phase_map = phase_map_node(threshold=threshold)
    # fill all connections.
    for mode in ["clock", "anticlock"]:
        for op in ["cos", "sin"]:
            wf = add2wf(
                wf,
                f"contrast_{mode}",
                (f"{op}_stat", _gk, "z_score"),
                phase_map,
                f"{op}_{mode}",
            )
    for op in ["cos", "rot"]:
        wf = add2wf(
            wf,
            "contrast_glob",
            (f"{op}_stat", _gk, "z_score"),
            phase_map,
            f"{op}_glob",
        )
    phase_map.inputs.threshold = threshold
    return wf


def add_tsnr_map(wf):
    """Add a tSNR node."""
    tsnr_node = tsnr_map_node()
    add2wf(wf, "selectfiles", "data_clock", tsnr_node, "in_file")
    add2wf(wf, "selectfiles", "mask", tsnr_node, "mask_file")
    return wf
