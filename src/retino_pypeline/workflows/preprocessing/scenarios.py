"""Scenarios for preprocessing workflows."""


from nipype import Workflow, Node
from nipype.interfaces import fsl

from .nodes import (
    apply_xfm_node,
    mask_node,
    run_topup,
    noise_std_node,
    denoise_magnitude_task,
    denoise_complex_task,
    realign_complex_task,
    realign_fsl_task,
    mp2ri_task,
)
from ..base.nodes import selectfile_task, input_task, sinker_task
from ..base.builder import add2wf_dwim, add2wf, add2sinker
from retino_pypeline.workflows.tools import func2node

from ..tools import _getsubid, _get_num_thread

FILES_NODE = "selectfiles"
INPUT_NODE = "input"
SINKER_NODE = "sinker"

_REGEX_SINKER = [
    (r"_sub_id_(.*?)([_/])", r"\g<2>"),
    (r"_denoise_str_", ""),
    (r"_task_(.*?)([_/])", r"\g<2>"),
    (r"rsub", "sub"),
    (r"rrsub", "sub"),
    (r"rpsub", "sub"),
]


def add_coregistration(wf, prev_node, func_out):
    extract_first = Node(fsl.ExtractROI(), name="extract_first")
    extract_first.inputs.t_min = 0
    extract_first.inputs.t_size = 1

    flirt = Node(fsl.FLIRT(), name="flirt")

    split = Node(fsl.Split(), name="split")
    split.inputs.dimension = "t"

    apply_xfm = apply_xfm_node()

    merge = Node(fsl.Merge(), name="merge")
    merge.inputs.dimension = "t"
    add2wf(wf, FILES_NODE, "anat", flirt, "reference")
    add2wf(wf, prev_node, func_out, split, "in_file")
    add2wf(wf, prev_node, func_out, extract_first, "in_file")
    add2wf(wf, extract_first, "roi_file", flirt, "in_file")
    add2wf(wf, split, "out_files", apply_xfm, "in_files")
    add2wf(wf, flirt, "out_matrix_file", apply_xfm, "in_matrix_file")
    add2wf(wf, FILES_NODE, "anat", apply_xfm, "reference")
    add2wf(wf, apply_xfm, "out_files", merge, "in_files")
    return merge, "merged_file"


def add_topup(wf, prev_node, func_out, sequence="EPI3D"):
    if sequence == "EPI3D":
        topup = func2node(run_topup, output_names=["out"], name="topup")
        add2wf(wf, prev_node, func_out, topup, "in_file")

        return topup, "out"
    return prev_node, func_out


class BasePreprocessingScenario:

    input_fields = ["sub_id", "task", "denoise_str"]
    workflow_name = "preproc"

    def __init__(self, base_data_dir, working_dir, sequence="EPI3D"):
        self.base_data_dir: str = base_data_dir
        self.working_dir: str = working_dir
        self.sequence: str = sequence
        self.wf: Workflow = None

    def get_workflow(self, extra_wfname="") -> Workflow:
        """
        Add the input and sinker nodes to the workflow.
        """

        wf = Workflow(name=self.workflow_name + extra_wfname, base_dir=self.working_dir)

        input_node = input_task(self.input_fields)
        sinker = sinker_task(self.base_data_dir)

        file_node = selectfile_task(
            template={
                "anat": "sub_%02i/anat/*_T1.nii",
                "data": f"sub_%02i/func/*{self.sequence}_%sTask.nii",
                "data_phase": f"sub_%02i/func/*{self.sequence}_%sTask_phase.nii",
                "noise_std_map": f"sub_%02i/preproc_extra/*{self.sequence}-0v_std.nii",
                "mask": f"sub_%02i/preproc_extra/*{self.sequence}_%sTask_mask.nii",
            },
            template_args={
                "anat": [["sub_id"]],
                "data": [["sub_id", "task"]],
                "data_phase": [["sub_id", "task"]],
                "noise_std_map": [["sub_id"]],
                "mask": [["sub_id", "task"]],
            },
            base_data_dir=self.base_data_dir,
            infields=["sub_id", "task"],
        )

        add2wf(wf, input_node, ("sub_id", _getsubid), sinker, "container")
        add2wf_dwim(wf, input_node, file_node, ["sub_id", "task"])

        sinker.inputs.regexp_substitutions = _REGEX_SINKER
        self.wf = wf
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


class RealignOnlyScenario(BasePreprocessingScenario):
    BUILDCODE = "r"

    def get_workflow(self, extra_wfname="") -> Workflow:
        to_sink = []
        wf = super().get_workflow(extra_wfname=extra_wfname)

        realign = realign_fsl_task("realign")
        to_sink.append(("realign", "motionparams"))

        add2wf(wf, FILES_NODE, "data", realign, "in_file")

        prev_node, func_out = add_topup(wf, realign, "out_file", sequence=self.sequence)
        # No denoising is performed, so we are going to do the coregistration here
        prev_node, func_out = add_coregistration(wf, prev_node, func_out)
        add2wf(wf, prev_node, func_out, SINKER_NODE, "coreg_data")
        # Finished connections
        add2sinker(wf, to_sink, folder=f"preproc.{self.BUILDCODE}")

        self.wf = wf
        return wf


class RealignMagnitudeDenoiseScenario(BasePreprocessingScenario):
    """REealing and Magnitude denoising of the data."""

    BUILDCODE = "rd"

    def get_workflow(self, extra_wfname="") -> Workflow:
        to_sink = []
        wf = super().get_workflow(extra_wfname=extra_wfname)
        # Realign
        realign = realign_fsl_task("realign")
        to_sink.append(("realign", "par_file", "motionparams"))

        add2wf(wf, FILES_NODE, "data", realign, "in_file")
        # MagDenoise
        denoise = denoise_magnitude_task("denoise_mag")

        add2wf(wf, "realign", "out_file", denoise, "data")
        add2wf(wf, FILES_NODE, "noise_std_map", denoise, "noise_std_map")
        add2wf(wf, FILES_NODE, "mask", denoise, "mask")

        prev_node, func_out = add_topup(wf, realign, "out_file", sequence=self.sequence)

        to_sink.append((prev_node, func_out, "data"))
        add2sinker(wf, to_sink, folder=f"preproc.{self.BUILDCODE}")

        self.wf = wf
        return wf


class RealignComplexDenoiseScenario(BasePreprocessingScenario):
    BUILDCODE = "rD"

    def get_workflow(self, extra_wfname="") -> Workflow:
        to_sink = []
        wf = super().get_workflow(extra_wfname=extra_wfname)
        # Realign
        realign = realign_fsl_task("realign")
        to_sink.append(("realign", "par_file", "motionparams"))

        add2wf(wf, FILES_NODE, "data", realign, "in_file")

        # Complex Denoise
        # Apply the motion correction on real and imaginary part.

        realign_cpx = realign_complex_task("realign_cpx")
        realign_cpx.n_procs = _get_num_thread()

        add2wf(wf, FILES_NODE, "data", realign_cpx, "data")
        add2wf(wf, "realign", "par_file", realign_cpx, "trans_files")
        add2wf(wf, FILES_NODE, "data_phase", realign_cpx, "data_phase")

        # Provide the real and imaginary part to the complex denoising
        # Retrieve the magnitude of the denoised complex image.
        denoise = denoise_complex_task("denoise_cpx")
        add2wf_dwim(
            wf,
            realign_cpx,
            denoise,
            ["real_file", "imag_file", "mag_file", "phase_file"],
        )

        add2wf(wf, FILES_NODE, "noise_std_map", denoise, "noise_std_map")
        add2wf(wf, FILES_NODE, "mask", denoise, "mask")

        prev_node, func_out = add_topup(wf, realign, "out_file", sequence=self.sequence)
        to_sink.append((prev_node, func_out, "data"))
        add2sinker(wf, to_sink, folder=f"preproc.{self.BUILDCODE}")

        self.wf = wf
        return wf


class MagnitudeDenoiseRealignScenario(BasePreprocessingScenario):
    """Denoise and then realign the data."""

    BUILDCODE = "dr"

    def get_workflow(self, extra_wfname="") -> Workflow:

        to_sink = []
        wf = super().get_workflow(extra_wfname=extra_wfname)

        denoise = denoise_magnitude_task("denoise_mag")
        add2wf_dwim(wf, FILES_NODE, denoise, ["data", "noise_std_map", "mask"])

        realign = realign_fsl_task(self.realign_backend)
        add2wf(wf, denoise, "denoised_file", realign, "in_file")
        to_sink.append(("realign", "motionparams"))

        prev_node, func_out = add_topup(wf, realign, "out_file", sequence=self.sequence)
        to_sink.append((prev_node, func_out, "data"))

        add2sinker(wf, to_sink, folder=f"preproc.{self.BUILDCODE}")

        self.wf = wf
        return wf


class ComplexDenoiseRealignScenario(BasePreprocessingScenario):
    BUILDCODE = "Dr"

    def get_workflow(self, extra_wfname="") -> Workflow:

        to_sink = []
        wf = super().get_workflow(extra_wfname=extra_wfname)
        mp2ri = mp2ri_task("mp2ri")
        denoise = denoise_complex_task("denoise_cpx")

        add2wf_dwim(wf, FILES_NODE, denoise, ["noise_std_map", "mask"])
        add2wf(wf, FILES_NODE, "data", mp2ri, "mag_file")
        add2wf(wf, FILES_NODE, "data_phase", mp2ri, "phase_file")

        add2wf_dwim(wf, mp2ri, denoise, ["real_file", "imag_file"])
        add2wf(wf, FILES_NODE, "data", denoise, "mag_file")
        add2wf(wf, FILES_NODE, "data_phaes", denoise, "phase_file")

        realign = realign_fsl_task("realign")
        add2wf(wf, denoise, "denoised_file", realign, "in_file")
        to_sink.append(("realign", "motionparams"))

        prev_node, func_out = add_topup(wf, realign, "out_file", sequence=self.sequence)
        to_sink.append((prev_node, func_out, "data"))

        add2sinker(wf, to_sink, folder=f"preproc.{self.BUILDCODE}")

        self.wf = wf
        return wf


class MagnitudeDenoiseScenario(BasePreprocessingScenario):
    BUILDCODE = "d"

    def get_workflow(self, extra_wfname=""):
        raise NotImplementedError


class ComplexDenoiseScenario(BasePreprocessingScenario):
    BUILDCODE = "D"

    def get_workflow(self, extra_wfname=""):
        raise NotImplementedError


# Noise Preprocessing
class NoisePreprocessingScenario(BasePreprocessingScenario):

    BUILDCODE = "n"

    def get_workflow(self, extra_wfname="") -> Workflow:

        wf = super().get_workflow(extra_wfname=extra_wfname)

        mask = mask_node(name="mask")
        noise_std = noise_std_node(name="noise_std")

        files = wf.get_node(FILES_NODE)
        sinker = wf.get_node("sinker")

        wf.connect(files, "data", mask, "in_file")
        wf.connect(files, "noise", noise_std, "noise_map_file")
        wf.connect(mask, "mask", sinker, "preproc_extra.@mask")
        wf.connect(noise_std, "noise_std_map", sinker, "preproc_extra.@noise_std")
        return wf


class BaseWorkflowDispatcher:
    """Dispatche workflow and manage them."""

    def __init__(self, base_data_dir, working_dir, sequence="EPI3D"):
        self.base_data_dir: str = base_data_dir
        self.working_dir: str = working_dir
        self.sequence: str = sequence
        self.wf: Workflow = None
        self.wf_scenario: BasePreprocessingScenario = None

    def run(
        self,
        task,
        sub_id,
        denoise_str,
        plugin="MultiProc",
        plugin_args=None,
        nipype_config=None,
    ) -> None:
        """Run a workflow for a given subject."""
        if self.wf is None:
            raise ValueError("Workflow is not defined.")

        inputnode = self.wf.get_node(INPUT_NODE)
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
            self.wf.run(plugin=plugin, plugin_args=plugin_args)

        elif plugin == "SLURMGraph":
            # Translate the n_procs directive to a plugin_args for slurm.
            for node in self.wf._graph.nodes():
                if hasattr(node, "n_procs"):
                    node.plugin_args = {"sbatch_args": f"-c {node.n_procs}"}
            self.wf.run(plugin=plugin, plugin_args=plugin_args)
        elif plugin is None:
            ...
        else:
            raise ValueError(f"Plugin {plugin} is not supported.")


class PreprocessingWorkflowDispatcher(BaseWorkflowDispatcher):
    """Dispatche Preprocessing workflow and manage them."""

    SCENARIOS = {
        "n": NoisePreprocessingScenario,
        "r": RealignOnlyScenario,
        "rd": RealignMagnitudeDenoiseScenario,
        "rD": RealignComplexDenoiseScenario,
        "dr": MagnitudeDenoiseRealignScenario,
        "Dr": ComplexDenoiseRealignScenario,
        "d": MagnitudeDenoiseScenario,
        "D": ComplexDenoiseScenario,
    }

    def __init__(self, base_data_dir, working_dir, sequence="EPI3D"):
        self.base_data_dir: str = base_data_dir
        self.working_dir: str = working_dir
        self.sequence: str = sequence
        self.wf: Workflow = None
        self.wf_scenario: BasePreprocessingScenario = None

    def get_workflow(self, scenario, extra_wfname=""):
        """Return a workflow for the given scenario."""
        self.wf_scenario = self.SCENARIOS[scenario](
            self.base_data_dir, self.working_dir, self.sequence
        )
        self.wf = self.wf_scenario.get_workflow(extra_wfname=extra_wfname)

        return self.wf
