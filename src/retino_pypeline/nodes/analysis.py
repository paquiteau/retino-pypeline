"""Create Node for analysis workflow."""
from nipype import Node

from retino_pypeline.interfaces.glm import DesignMatrixRetino, ContrastRetino, PhaseMap
from retino_pypeline.interfaces.tools import TSNR

from .base import func2node


def design_matrix_node(n_cycles, clockwise, extra_name=""):
    """Design Matrix Node."""
    return Node(
        DesignMatrixRetino(
            n_cycles=n_cycles,
            clockwise_rotation=clockwise,
        ),
        name="design" + extra_name,
    )


def cond_design_matrix_node(n_cycles, clockwise, extra_name=""):
    """Compute a design matrix with motion regressor or not."""

    def func_node(
        data_file,
        n_cycles,
        motion_file=None,
        min_onset=0,
        volumetric_tr=1.0,
        tr_unit="s",
        clockwise_rotation=True,
        preproc_code="v",
    ):
        from retino_pypeline.interfaces.glm import DesignMatrixRetino

        dm = DesignMatrixRetino(
            data_file=data_file,
            n_cycles=n_cycles,
            volumetric_tr=volumetric_tr,
            tr_unit=tr_unit,
            min_onset=min_onset,
            clockwise_rotation=clockwise_rotation,
        )

        if "r" in preproc_code:
            dm.inputs.motion_file = motion_file
        return dm.run().outputs.design_matrix

    node = func2node(
        func_node, output_names=["design_matrix"], name="design" + extra_name
    )
    node.inputs.n_cycles = n_cycles
    node.inputs.clockwise = clockwise
    return node


def contrast_node(extra_name):
    """Contrast Node."""
    return Node(ContrastRetino(), name="contrast" + extra_name)


def contrast_glob_node():
    """Contrast Node for fixed Effect."""

    def func_node(
        fmri_clock,
        fmri_anticlock,
        dm_clock,
        dm_anticlock,
        volumetric_tr,
        noise_model="ar1",
    ):
        from retino_pypeline.interfaces.glm import ContrastRetino

        contrast = ContrastRetino()
        contrast.inputs.fmri_timeseries = [fmri_clock, fmri_anticlock]
        contrast.inputs.design_matrices = [dm_clock, dm_anticlock]
        contrast.inputs.volumetric_tr = volumetric_tr
        contrast.inputs.noise_model = noise_model
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


def tsnr_map_node(extra_name=""):
    """TSNR Node."""
    return Node(TSNR(), name="tsnr" + extra_name)
