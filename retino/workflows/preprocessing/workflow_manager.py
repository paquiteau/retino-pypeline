from retino.workflows.preprocessing.nodes import (
    realign_node,
    topup_node,
    coregistration_node,
)
from nipype import Workflow

from .base import DenoiseParameters
from builder import (
    add_base,
    add_coreg,
    add_realign,
    add_topup,
    add_sinker,
)


class PreprocessingManager:
    """Manager for preprocessing workflow."""

    def __init__(self, base_data_dir, working_dir):
        self.base_data_dir = base_data_dir
        self.working_dir = working_dir

    def show_graph(wf):
        """Check the workflow. Also draws a representation."""
        # ascii plot: https://github.com/ggerganov/dot-to-ascii

        fname = wf.write_graph(dotfilename="graph.dot", graph2use="colored")
        return fname

    def run(wf, multi_proc = False,  **kwargs):
        """Run the workflow with iterables parametrization defined in kwargs"""
        inputnode = wf.get_node("input")
        inputnode.iterables = []
        for key in kwargs:
            inputnode.iterables.append((key, kwargs[key]))


        wf.run(plugin="MultiProc" if multi_proc else None)



class RetinotopyPreprocessingManager(PreprocessingManager):
    """Manager for Retinotopy base workflow.

    There is two task: Clockwise and Anticlockwise.
    """

    def build(name="preprocess", realign=""):
        """create the workflow."""

        wf = Workflow(name=name, base_dir=self.working_dir)

        wf = add_base(
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





class RealignmentPreprocessingManager(PreprocessingManager):
    def build(name="realign"):
        wf = Workflow(name=name, base_dir=self.working_dir)
        wf = add_base(
            wf,
            base_data_dir=self.base_data_dir,
            cached_realignment=False,
        )
        wf = add_realign(wf, name="realign",  after_node="selectfiles", edge="data")
        wf = add_sinker(wf, [("realign", "realigned_files", "realign.@data"),
                             ("realign", "realignment_parameters", "realign.@motion")])

        # configure sinker
        sinker=wf.get_node("sinker")
        return wf




class NoisePreprocessingManager(PreprocessingManager):
    ...
