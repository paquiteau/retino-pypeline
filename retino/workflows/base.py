"""Base Workflow Manager.

A workflow manager is responsible for the creation and execution of a Nipype workflow.

The creation of a workflow is done using the `get_workflow` method, which create a workflow in 3 main step

1. add the input and sinker node with the input fields. `_base_build`
2. add the files grabber nodes `_build_files`
3. add the processing nodes   `_build`

"""
from nipype import Workflow

from .base_nodes import input_task, sinker_task
from .tools import _getsubid


class WorkflowManager:
    """Base Workflow Managers."""

    input_fields = []
    workflow_name = ""

    def __init__(self, base_data_dir, working_dir):
        self.base_data_dir = base_data_dir
        self.working_dir = working_dir

    def _base_build(self, extra_wfname=""):
        """
        Build the base of preprocessing workflow.

        4 nodes are created and added:
        - Input -> template_node -> selectfiles
           |-> files
           |-> sinker
        """
        wf = Workflow(name=self.workflow_name + extra_wfname, base_dir=self.working_dir)

        input_node = input_task(self.input_fields)
        sinker = sinker_task(self.base_data_dir)

        wf.connect(
            [
                (input_node, sinker, [(("sub_id", _getsubid), "container")]),
            ]
        )
        return wf

    def _build_files(self, wf):
        """Add the files templates nodes."""
        raise NotImplementedError

    def _build(self, wf):
        return wf

    def get_workflow(self, *args, extra_wfname="", **kwargs):
        """Get a preprocessing workflow."""
        wf = self._base_build(extra_wfname)
        wf = self._build_files(wf)
        wf = self._build(wf, *args, **kwargs)
        return wf

    def show_graph(self, wf, graph2use="colored"):
        """Check the workflow. Also draws a representation."""
        # TODO ascii plot: https://github.com/ggerganov/dot-to-ascii

        fname = wf.write_graph(
            dotfilename=f"graph_{graph2use}.dot", graph2use=graph2use
        )
        return fname

    def show_graph_nb(self, wf, graph2use="colored", detailed=False):
        from IPython.display import Image

        if detailed:
            return Image(
                self.show_graph(wf, graph2use=graph2use).split(".")[0] + "_detailed.png"
            )
        return Image(self.show_graph(wf))

    def configure(self, wf, **kwargs):
        inputnode = wf.get_node("input")
        inputnode.iterables = []
        for key in kwargs:
            inputnode.iterables.append((key, kwargs[key]))
        return wf

    def run(self, wf, multi_proc=False, dry=False, **kwargs):
        """Run the workflow with iterables parametrization defined in kwargs."""
        wf = self.configure(wf, **kwargs)
        if not dry:
            wf.run(plugin="MultiProc" if multi_proc else None)
        return wf
