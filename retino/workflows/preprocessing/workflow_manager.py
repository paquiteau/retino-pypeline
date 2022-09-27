from retino.workflows.preprocessing.nodes import (
    realign_node,
    topup_node,
    coregistration_node,
)
from nipype import Workflow

from .base import DenoiseParameters
from builder import PreprocBuilder


class PreprocessingManager:
    """Manager for preprocessing workflow."""

    def __init__(self, base_data_dir, working_dir):
        self.base_data_dir = base_data_dir
        self.working_dir = working_dir

    def check(wf, draw=None):
        """Check the workflow. Also draws a representation."""
        # ascii plot: https://github.com/ggerganov/dot-to-ascii
        #
        #


class RetinotopyPreprocessingManager(PreprocessingManager):
    """Manager for Retinotopy base workflow.

    There is two task: Clockwise and Anticlockwise.
    """

    def build(name="preprocess", realign=""):
        """create the workflow."""

        wf = Workflow(name=name, base_dir=self.working_dir)

        wf = PreprocBuilder.add_base(
            wf,
            basedata_dir=self.base_data_dir,
            cached_realignment=realign == "cached")


        if realign == "cached":

        elif realign is True:
            ...
        elif realign is False:
            ...
        else:
            raise ValueError("Realign should be 'cached', True, or False")

        return workflow

    def run(wf, **kwargs):
        """Run the workflow with iterables parametrization defined in kwargs"""
