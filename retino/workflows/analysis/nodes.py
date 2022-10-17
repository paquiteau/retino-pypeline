"""Create Node for analysis workflow."""
from nipype import Node

from retino.interfaces.glm import DesignMatrixRetino, ContrastRetino, PhaseMap
from retino.interfaces.tools import TSNR

from ..tools import func2node


def design_matrix_node(n_cycles, clockwise, extra_name=""):
    """Design Matrix Node."""
    return Node(
        DesignMatrixRetino(
            n_cycles=n_cycles,
            clockwise_rotation=clockwise,
        ),
        name="design" + extra_name,
    )


def contrast_node(extra_name):
    """Contrast Node."""
    return Node(ContrastRetino(), name="contrast" + extra_name)


def contrast_glob_node():
    """Contrast Node for fixed Effect."""

    def func_node(fmri_clock, fmri_anticlock, dm_clock, dm_anticlock, volumetric_tr):
        from retino.interfaces.glm import ContrastRetino

        contrast = ContrastRetino()
        contrast.inputs.fmri_timeseries = [fmri_clock, fmri_anticlock]
        contrast.inputs.design_matrices = [dm_clock, dm_anticlock]
        contrast.inputs.volumetric_tr = volumetric_tr
        results = contrast.run()
        return (
            results.outputs.cos_stat,
            results.outputs.sin_stat,
            results.outputs.rot_stat,
        )

    return func2node(
        func_node,
        name="contrast_glob",
        output_names=["cos_stat", "sin_stat", "rot_stat"],
    )


def phase_map_node(threshold, extra_name=""):
    """Phase Map Node."""
    return Node(PhaseMap(threshold=threshold), name="phase_map" + extra_name)


def tsnr_map_node(extra_name):
    """TSNR Node."""
    return Node(TSNR(), name="tsnr" + extra_name)
