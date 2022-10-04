"""Create Node for analysis workflow."""
from nipype import Node

from retino.interfaces.glm import DesignMatrixRetino, ContrastRetino, PhaseMap
from retino.interfaces.tools import TSNR


def design_matrix_node(n_cycles, TR, clockwise, extra_name=""):
    """Design Matrix Node."""
    return Node(
        DesignMatrixRetino(
            n_cycles=n_cycles,
            volumetric_tr=TR,
            clockwise_rotation=clockwise,
        ),
        name="design" + extra_name,
    )


def contrast_node(TR, extra_name):
    """Contrast Node."""
    return Node(ContrastRetino(volumetric_tr=TR), name="contrast" + extra_name)


def phase_map_node(threshold, extra_name):
    """Phase Map Node."""
    return Node(PhaseMap(threshold=threshold), name="phase_map" + extra_name)


def tsnr_map_node(threshodl, extra_name):
    """TSNR Node."""
    return Node(TSNR(), name="tsnr" + extra_name)
