"""Base classes for workflow controls."""
import os
from nipype import Workflow, Node
from typing import Literal
from .tools import _get_num_thread, _getsubid, dict2cli
from ..nodes.base import selectfile_task, input_task, sinker_task


class WorkflowScenario:
    """Base class for workflow scenarios.

    A workflow scenario is in charge of generating a workflow.

    """

    WF_NAME: str = None
    INPUT_FIELDS: list = None
    INPUT_NODE = "input"
    FILES_NODES = "selectfiles"
    SINKER_NODE = "sinker"

    def __init__(self, base_data_dir, working_dir):
        self.base_data_dir: str = base_data_dir
        self.working_dir: str = working_dir

        self.wf: Workflow = None

    def get_workflow(self) -> Workflow:
        """Return the workflow."""
        raise NotImplementedError
        return self.wf

    def get_node(self, node: Node | str) -> Node:
        """Get node from workflow or raise error."""
        if isinstance(node, Node):
            return node
        if isinstance(node, str):
            node_obj = self.wf.get_node(node)
        if node_obj is None:
            raise ValueError(f"Node {node} not found.")
        return node_obj

    def add2wf(
        self, after_node: Node | str, edge_out: str, node: Node | str, edge_in: str
    ) -> None:
        """Add node to wf after node connecting edge_out and edge_in."""
        if isinstance(after_node, str):
            after_node = self.wf.get_node(after_node)
        if isinstance(node, str):
            node = self.wf.get_node(node)
        self.wf.connect([(after_node, node, [(edge_out, edge_in)])])

    def add2wf_dwim(
        self,
        node_out: Node | str,
        node_in: Node | str,
        edges: list[tuple[str, str] | str] | tuple[str, str] | str,
    ) -> None:
        """Connect two node with same edge label."""
        if isinstance(node_in, str):
            node_in = self.get_node(node_in)

        if isinstance(node_out, str):
            node_out = self.get_node(node_out)

        if not isinstance(edges, list):
            edges = [edges]
        for edge in edges:
            if isinstance(edge, str):
                self.wf.connect(node_out, edge, node_in, edge)
            elif isinstance(edge, tuple) and len(edge) == 2:
                self.wf.connect(node_out, edge[0], node_in, edge[1])
            else:
                raise ValueError(f"Unsupported edge config {edge}")

    def add2sinker(self, connections: list[tuple[str | Node, str, str]], folder=None):
        """Add connections to sinker.

        connections should be a list of (node_name, edge, output_name)

        """
        if folder is None:
            folder = ""
        else:
            folder += ".@"
        sinker = self.wf.get_node("sinker")

        for con in connections:
            self.wf.connect(self.get_node(con[0]), con[1], sinker, f"{folder}{con[2]}")

    def show_graph(self, graph2use="colored") -> str:
        """Check the workflow. Also draws a representation."""
        # TODO ascii plot: https://github.com/ggerganov/dot-to-ascii

        fname = self.wf.write_graph(
            dotfilename=f"graph_{graph2use}.dot", graph2use=graph2use
        )
        return fname

    def show_graph_nb(self, graph2use: str = "colored", detailed: bool = False) -> str:
        from IPython.display import Image

        if detailed:
            return Image(
                self.show_graph(graph2use=graph2use).split(".")[0] + "_detailed.png"
            )
        return Image(self.show_graph())


class BaseWorkflowScenario(WorkflowScenario):

    _REGEX_SINKER = [
        (r"_sub_id_(.*?)([_/])", r"\g<2>"),
        (r"_denoise_str_", ""),
        (r"_task_(.*?)([_/])", r"\g<2>"),
        (r"rsub", "sub"),
        (r"rrsub", "sub"),
        (r"rpsub", "sub"),
    ]

    def get_workflow(self, extra_wfname="") -> Workflow:
        """Return the workflow."""
        self.wf = Workflow(name=self.WF_NAME + extra_wfname)
        self.wf.base_dir = self.working_dir
        self.wf.config = {
            "execution": {"crashdump_dir": os.path.join(self.working_dir, "crash")}
        }

        input_node = input_task(self.INPUT_FIELDS)
        sinker = sinker_task(self.base_data_dir)
        template, template_args = self._get_file_templates()
        file_node = selectfile_task(
            template=template,
            template_args=template_args,
            base_data_dir=self.base_data_dir,
            infields=["sub_id", "task"],
        )

        self.add2wf(input_node, ("sub_id", _getsubid), sinker, "container")
        self.add2wf_dwim(input_node, file_node, ["sub_id", "task"])

        sinker.inputs.regexp_substitutions = self._REGEX_SINKER
        return self.wf

    def _get_file_templates(self) -> tuple[dict[str, str], dict[str, list[list[str]]]]:
        raise NotImplementedError


class WorkflowDispatcher:
    """Dispatche workflow and manage them."""

    def __init__(self, base_data_dir, working_dir, sequence="EPI3D"):
        self.base_data_dir: str = base_data_dir
        self.working_dir: str = working_dir
        self.wf: Workflow = None
        self.wf_scenario: WorkflowScenario = None

    def get_workflow(self, scenario: WorkflowScenario) -> Workflow:
        """Return the workflow."""
        self.wf_scenario = scenario
        self.wf = scenario.get_workflow(self.base_data_dir, self.working_dir)
        return self.wf

    def run(
        self,
        task: str,
        sub_id: int | list[int] | Literal["all"],
        denoise_str,
        plugin="MultiProc",
        plugin_args: dict = None,
        node_args: dict = None,
        nipype_config=None,
    ) -> None:
        """Run a workflow after configuring it."""
        if self.wf is None:
            raise ValueError("Workflow is not defined.")

        inputnode = self.wf.get_node(self.wf_scenario.INPUT_NODE)
        inputnode.iterables = []
        for key, iterable in zip(
            ["task", "sub_id", "denoise_str"], [task, sub_id, denoise_str]
        ):
            if iterable is not None:
                if not isinstance(iterable, (list, tuple)):
                    iterable = [iterable]
                inputnode.iterables.append((key, iterable))

        if nipype_config is not None:
            self.wf.config = nipype_config
        if plugin == "MultiProc":
            if plugin_args["n_procs"] in [None, -1]:
                plugin_args["n_procs"] = _get_num_thread()
        elif "SLURM" in plugin:
            # Translate the n_procs directive to a plugin_args for slurm.
            for node in self.wf._graph.nodes():
                if node.name in node_args:
                    for key, value in node_args[node.name].items():
                        if key == "sbatch_args":
                            value = dict2cli(value)
                        setattr(node, key, value)
                if hasattr(node, "n_procs"):
                    if hasattr(node, "plugin_args"):
                        if "sbatch_args" in node.plugin_args:
                            node.plugin_args["sbatch_args"] += f"-c {node.n_procs}"
                        else:
                            node.plugin_args["sbatch_args"] = f"-c {node.n_procs}"
                    else:
                        node.plugin_args = {"sbatch_args": f"-c {node.n_procs}"}

            # format the sbatch_args
            sbatch_args = plugin_args.get("sbatch_args", "")
            sbatch_args = dict2cli(sbatch_args)
            plugin_args["sbatch_args"] = sbatch_args
        elif plugin is not None:
            raise ValueError(f"Plugin {plugin} is not supported.")

        self.wf.run(plugin=plugin, plugin_args=plugin_args)
