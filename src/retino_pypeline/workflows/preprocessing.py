"""Scenarios for preprocessing workflows."""


from nipype import Workflow, Node
from nipype.interfaces import fsl

from ..nodes.preprocessing import (
    apply_xfm_node,
    mask_node,
    topup_node_task,
    noise_std_node,
    denoise_magnitude_task,
    denoise_complex_task,
    realign_complex_task,
    realign_fsl_task,
    mp2ri_task,
)
from .base import BaseWorkflowScenario, WorkflowDispatcher

from .tools import _get_num_thread

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


class BasePreprocessingScenario(BaseWorkflowScenario):
    BUILD_CODE = None

    INPUT_FIELDS = ["sub_id", "task", "denoise_str"]
    WF_NAME = "preproc"

    def __init__(self, base_data_dir, working_dir, sequence="EPI3D"):
        super().__init__(base_data_dir, working_dir)
        self.sequence: str = sequence
        self.wf: Workflow = None

    def _add_coregistration(self, prev_node, func_out) -> tuple[Node, str]:
        extract_first = Node(fsl.ExtractROI(), name="extract_first")
        extract_first.inputs.t_min = 0
        extract_first.inputs.t_size = 1

        flirt = Node(fsl.FLIRT(), name="flirt")

        split = Node(fsl.Split(), name="split")
        split.inputs.dimension = "t"

        apply_xfm = apply_xfm_node()

        merge = Node(fsl.Merge(), name="merge")
        merge.inputs.dimension = "t"
        self.add2wf(FILES_NODE, "anat", flirt, "reference")
        self.add2wf(prev_node, func_out, split, "in_file")
        self.add2wf(prev_node, func_out, extract_first, "in_file")
        self.add2wf(extract_first, "roi_file", flirt, "in_file")
        self.add2wf(split, "out_files", apply_xfm, "in_files")
        self.add2wf(flirt, "out_matrix_file", apply_xfm, "in_matrix_file")
        self.add2wf(FILES_NODE, "anat", apply_xfm, "reference")
        self.add2wf(apply_xfm, "out_files", merge, "in_files")
        return merge, "merged_file"

    def _add_topup(self, prev_node, func_out) -> tuple[Node, str]:
        if self.sequence == "EPI3D":
            topup = topup_node_task()
            self.add2wf(prev_node, func_out, topup, "in_file")
            self.add2wf(prev_node, func_out, topup, "data")
            self.add2wf(FILES_NODE, "data_opposite", topup, "data_opposite")
            return topup, "out"

        return prev_node, func_out

    def _get_file_templates(self):
        template = {
            "anat": "sub_%02i/anat/*_T1.nii",
            "data": f"sub_%02i/func/*{self.sequence}_%sTask.nii",
            "data_phase": f"sub_%02i/func/*{self.sequence}_%sTask_phase.nii",
            "noise_std_map": f"sub_%02i/preproc_extra/*{self.sequence}-0v_std.nii",
            "mask": f"sub_%02i/preproc_extra/*{self.sequence}_%sTask_mask.nii",
        }
        template_args = {
            "anat": [["sub_id"]],
            "data": [["sub_id", "task"]],
            "data_phase": [["sub_id", "task"]],
            "noise_std_map": [["sub_id"]],
            "mask": [["sub_id", "task"]],
        }
        if self.sequence == "EPI3D":
            template["data_opposite"] = "sub_%02i/func/*EPI3D_Clockwise_1rep_PA.nii"
            template_args["data_opposite"] = [["sub_id"]]
        return template, template_args

    def get_workflow(self, extra_wfname="") -> Workflow:
        """
        Add the input and sinker nodes to the workflow.
        """
        self.wf = super().get_workflow()
        self.wf.name = f"{self.WF_NAME}_{self.BUILDCODE}" + extra_wfname
        return self.wf


class RealignOnlyScenario(BasePreprocessingScenario):
    BUILDCODE = "r"

    def get_workflow(self, extra_wfname="") -> Workflow:
        to_sink = []
        self.wf = super().get_workflow(extra_wfname=extra_wfname)

        realign = realign_fsl_task("realign")
        to_sink.append(("realign", "motionparams"))

        self.add2wf(FILES_NODE, "data", realign, "in_file")

        prev_node, func_out = self._add_topup(realign, "out_file")
        # No denoising is performed, so we are going to do the coregistration here
        prev_node, func_out = self._add_coregistration(prev_node, func_out)
        self.add2wf(prev_node, func_out, SINKER_NODE, "coreg_data")
        # Finished connections
        self.add2sinker(to_sink, folder=f"preproc.{self.BUILDCODE}")

        return self.wf


class RealignMagnitudeDenoiseScenario(BasePreprocessingScenario):
    """REealing and Magnitude denoising of the data."""

    BUILDCODE = "rd"

    def get_workflow(self, extra_wfname="") -> Workflow:
        to_sink = []
        super().get_workflow(extra_wfname=extra_wfname)
        # Realign
        realign = realign_fsl_task("realign")
        to_sink.append(("realign", "par_file", "motionparams"))

        self.add2wf(FILES_NODE, "data", realign, "in_file")
        # MagDenoise
        denoise = denoise_magnitude_task("denoise_mag")
        denoise.n_procs = _get_num_thread()
        to_sink.append(("denoise_mag", "rank_map", "rank_map"))
        self.add2wf("realign", "out_file", denoise, "data")
        self.add2wf(FILES_NODE, "noise_std_map", denoise, "noise_std_map")
        self.add2wf(FILES_NODE, "mask", denoise, "mask")
        self.add2wf(INPUT_NODE, "denoise_str", denoise, "denoise_str")

        prev_node, func_out = self._add_topup(denoise, "denoised_file")

        to_sink.append((prev_node, func_out, "data"))
        self.add2sinker(to_sink, folder=f"preproc.{self.BUILDCODE}")

        return self.wf


class RealignComplexDenoiseScenario(BasePreprocessingScenario):
    BUILDCODE = "rD"

    def get_workflow(self, extra_wfname="") -> Workflow:
        to_sink = []
        wf = super().get_workflow(extra_wfname=extra_wfname)
        # Realign
        realign = realign_fsl_task("realign")
        to_sink.append(("realign", "par_file", "motionparams"))

        self.add2wf(FILES_NODE, "data", realign, "in_file")

        # Complex Denoise
        # Apply the motion correction on real and imaginary part.

        realign_cpx = realign_complex_task("realign_cpx")
        realign_cpx.n_procs = _get_num_thread()

        self.add2wf(FILES_NODE, "data", realign_cpx, "data")
        self.add2wf("realign", "par_file", realign_cpx, "trans_files")
        self.add2wf(FILES_NODE, "data_phase", realign_cpx, "data_phase")

        # Provide the real and imaginary part to the complex denoising
        # Retrieve the magnitude of the denoised complex image.
        denoise = denoise_complex_task("denoise_cpx")
        denoise.n_procs = _get_num_thread()
        self.add2wf_dwim(
            wf,
            realign_cpx,
            denoise,
            ["real_file", "imag_file", "mag_file", "phase_file"],
        )

        to_sink.append(("denoise_cpx", "rank_map", "rank_map"))
        self.add2wf(FILES_NODE, "noise_std_map", denoise, "noise_std_map")
        self.add2wf(FILES_NODE, "mask", denoise, "mask")

        prev_node, func_out = self._add_topup(wf, denoise, "denoised_file")
        to_sink.append((prev_node, func_out, "data"))
        self.add2sinker(to_sink, folder=f"preproc.{self.BUILDCODE}")

        self.wf = wf
        return wf


class MagnitudeDenoiseRealignScenario(BasePreprocessingScenario):
    """Denoise and then realign the data."""

    BUILDCODE = "dr"

    def get_workflow(self, extra_wfname="") -> Workflow:

        to_sink = []
        wf = super().get_workflow(extra_wfname=extra_wfname)

        denoise = denoise_magnitude_task("denoise_mag")
        denoise.n_procs = _get_num_thread()
        to_sink.append(("denoise_mag", "rank_map", "rank_map"))
        self.add2wf_dwim(FILES_NODE, denoise, ["data", "noise_std_map", "mask"])

        realign = realign_fsl_task(self.realign_backend)
        self.add2wf(denoise, "denoised_file", realign, "in_file")
        to_sink.append(("realign", "motionparams"))

        prev_node, func_out = self._add_topup(realign, "out_file")
        to_sink.append((prev_node, func_out, "data"))

        self.add2sinker(to_sink, folder=f"preproc.{self.BUILDCODE}")

        self.wf = wf
        return wf


class ComplexDenoiseRealignScenario(BasePreprocessingScenario):
    BUILDCODE = "Dr"

    def get_workflow(self, extra_wfname="") -> Workflow:

        to_sink = []
        super().get_workflow(extra_wfname=extra_wfname)
        mp2ri = mp2ri_task("mp2ri")
        denoise = denoise_complex_task("denoise_cpx")
        denoise.n_procs = _get_num_thread()
        to_sink.append(("denoise_cpx", "rank_map", "rank_map"))

        self.add2wf_dwim(FILES_NODE, denoise, ["noise_std_map", "mask"])
        self.add2wf(FILES_NODE, "data", mp2ri, "mag_file")
        self.add2wf(FILES_NODE, "data_phase", mp2ri, "phase_file")

        self.add2wf_dwim(mp2ri, denoise, ["real_file", "imag_file"])
        self.add2wf(FILES_NODE, "data", denoise, "mag_file")
        self.add2wf(FILES_NODE, "data_phaes", denoise, "phase_file")

        realign = realign_fsl_task("realign")
        self.add2wf(denoise, "denoised_file", realign, "in_file")
        to_sink.append(("realign", "motionparams"))

        prev_node, func_out = self._add_topup(realign, "out_file")
        to_sink.append((prev_node, func_out, "data"))

        self.add2sinker(to_sink, folder=f"preproc.{self.BUILDCODE}")

        return self.wf


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

        super().get_workflow(extra_wfname=extra_wfname)

        mask = mask_node(name="mask")
        noise_std = noise_std_node(name="noise_std")

        files = self.wf.get_node(FILES_NODE)
        sinker = self.wf.get_node("sinker")

        self.wf.connect(files, "data", mask, "in_file")
        self.wf.connect(files, "noise", noise_std, "noise_map_file")
        self.wf.connect(mask, "mask", sinker, "preproc_extra.@mask")
        self.wf.connect(noise_std, "noise_std_map", sinker, "preproc_extra.@noise_std")
        return self.wf


class PreprocessingWorkflowDispatcher(WorkflowDispatcher):
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
