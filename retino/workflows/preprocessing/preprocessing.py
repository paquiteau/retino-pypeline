"""Preprocessing Workflow Manager.

The management of workflow happens at two levels:
First on the built time,  where the overall design of the workflow is selected.
Then at the run time,  where the parameters for selecting the data and
the denoising methods are provided.

"""

from retino.workflows.preprocessing.nodes import (
    cond_denoise_task,
    conditional_topup_task,
    coregistration_task,
    realign_task,
)

from ..base.builder import add2sinker, add2wf_dwim, add2wf
from ..base.nodes import file_task
from ..base.workflow_manager import WorkflowManager
from ..tools import func2node

_REGEX_SINKER = [
    (r"_sequence_(.*?)([_/])", r"\g<2>"),
    (r"_sub_id_(.*?)([_/])", r"\g<2>"),
    (r"_denoise_str_", ""),
    (r"_task_(.*?)([_/])", r"\g<2>"),
    (r"rsub", "sub"),
    (r"rrsub", "sub"),
    (r"rpsub", "sub"),
]

DENOISE = "denoise"
REALIGN = "realign"
TOPUP = "topup"
COREG = "coreg"
FILES = "selectfiles"
SINKER = "sinker"
INPUT = "input"
TEMPLATE = "template_node"

#######################
#  Builder functions  #
#######################


def add_realign(wf, name, after_node, edge):
    """Add a Realignment node."""
    realign = realign_task(name=name)
    add2wf(wf, after_node, edge, realign, "in_files")


def add_denoise_mag(wf, name, after_node, edge):
    """Add denoising step for magnitude input."""
    denoise = cond_denoise_task(name)
    add2wf_dwim(wf, FILES, denoise, ["noise_std_map", "mask"])
    add2wf_dwim(wf, INPUT, denoise, "denoise_str")
    add2wf(wf, after_node, edge, denoise, "data")


def add_denoise_cpx(wf, name, after_realign=False):
    """Add denoising step for magnitude input."""
    denoise = cond_denoise_task(name)
    add2wf_dwim(wf, FILES, denoise, ["noise_std_map", "mask", "data", "data_phase"])
    add2wf_dwim(wf, INPUT, denoise, "denoise_str")
    if after_realign:
        add2wf_dwim(wf, REALIGN, denoise, ("realignment_parameters", "motion"))


def add_topup(wf, name, after_node, edge):
    """Add conditional topup correction."""
    condtopup = conditional_topup_task(name)
    # also adds mandatory connections
    add2wf_dwim(wf, INPUT, condtopup, "sequence")
    add2wf_dwim(wf, FILES, condtopup, "data_opposite")
    add2wf_dwim(wf, after_node, condtopup, (edge, "data"))


def add_coreg(wf, name, after_node, edge):
    """Add coregistration step."""
    coreg = coregistration_task(name)
    # also add mandatory connections:
    wf.connect(wf.get_node(FILES), "anat", coreg, "in.anat")
    add2wf_dwim(wf, after_node, coreg, (edge, "in.func"))


########################################
#  Base Preprocessing Worflow  Manager #
########################################


def _tplt_node(sequence, cached_realignment):
    """Template node as a Function to handle cached realignment.

    TODO add support for complex according to the denoising config string.
    """
    template = {
        "anat": "sub_%02i/anat/*_T1.nii",
        "data": "sub_%02i/func/*%s_%sTask.nii",
        "data_phase": "sub_%02i/func/*%s_%sTask_phase.nii",
        "noise_std_map": "sub_%02i/preproc_extra/*%s-0v_std.nii",
        "mask": "sub_%02i/preproc_extra/*%s_%sTask_mask.nii",
    }
    file_template_args = {
        "anat": [["sub_id"]],
        "data": [["sub_id", "sequence", "task"]],
        "data_phase": [["sub_id", "sequence", "task"]],
        "noise_std_map": [["sub_id", "sequence"]],
        "mask": [["sub_id", "sequence", "task"]],
    }

    if cached_realignment:
        template["data"] = "sub_%02i/realign/*%s_%sTask.nii"
        template["motion"] = "sub_%02i/realign/*%s_%sTask.txt"
        file_template_args["motion"] = [["sub_id", "sequence", "task"]]
    if "EPI" in sequence:
        template["data_opposite"] = "sub_%02i/func/*%s_Clockwise_1rep_PA.nii"
        file_template_args["data_opposite"] = [["sub_id", "sequence"]]
    else:
        template.pop("data_phase")
    return template, file_template_args


class PreprocessingWorkflowManager(WorkflowManager):
    """Manager for preprocessing workflow."""

    input_fields = ["sub_id", "sequence", "denoise_str", "task"]

    def _build_files(self, wf):
        """
        Build the base of preprocessing workflow.

        4 nodes are created and added:
        - Input -> template_node -> selectfiles
           |-> files
           |-> sinker
        """
        templates_args = ["sub_id", "sequence", "task"]

        # template node needs to be implemented in child classes.
        tplt_node = func2node(
            _tplt_node,
            name="template_node",
            output_names=["field_template", "template_args"],
        )
        tplt_node.inputs.cached_realignment = False
        files = file_task(
            infields=templates_args,
            outfields=[
                "data",
                "data_phase",
                "anat",
                "noise_std_map",
                "mask",
                "motion",
                "data_opposite",
            ],
            base_data_dir=self.base_data_dir,
        )
        wf = add2wf_dwim(wf, INPUT, tplt_node, "sequence")
        wf = add2wf_dwim(
            wf,
            tplt_node,
            files,
            ["field_template", "template_args"],
        )
        wf = add2wf_dwim(wf, INPUT, files, templates_args)
        return wf


###########################################
#  Modular Preprocessing Worflow  Manager #
###########################################


class RetinotopyPreprocessingManager(PreprocessingWorkflowManager):
    """Manager for Retinotopy base workflow.

    There is two task: Clockwise and Anticlockwise.
    """

    workflow_name = "preprocessing"

    def get_workflow(self, build_code="v"):
        """Get a Retinotopy workflow."""
        return super().get_workflow(
            extra_wfname=f"_{build_code}",
            build_code=build_code,
        )

    def _build(self, wf, build_code="v"):
        """Create a Retinotopy Workflow with option for the order of steps.

        Parameters
        ----------
        build_code: str

        Returns
        -------
        wf : a nipype workflow ready to be run.
        """
        if len(build_code) == 1:
            build_code += "_"
        if build_code[0] == "v":
            # cached realignment or nothing
            nxt = (FILES, "data")
        elif build_code[0] == "r":
            add_realign(wf, REALIGN, FILES, "data")
            nxt = (REALIGN, "realigned_files")
            if build_code[1] == "d":
                add_denoise_mag(wf, DENOISE, *nxt)
                nxt = (DENOISE, "denoised_file")
            elif build_code[1] == "D":
                add_denoise_cpx(wf, DENOISE, after_realign=True)
                nxt = (DENOISE, "denoised_file")
        elif build_code[0] == "d":
            # denoising, realigment
            add_denoise_mag(wf, DENOISE, FILES, "data")
            nxt = (DENOISE, "denoised_file")
            if build_code[1] == "r":
                add_realign(wf, REALIGN, *nxt)
                nxt = (REALIGN, "realigned_files")
        elif build_code[0] == "D":
            # complex denoising and realignment
            add_denoise_cpx(wf, DENOISE, after_realign=False)
            nxt = (DENOISE, "denoised_file")
            if build_code[1] == "r":
                add_realign(wf, REALIGN, *nxt)
                nxt = (REALIGN, "realigned_files")
        else:
            raise ValueError("Unsupported build code.")

        add_topup(wf, TOPUP, nxt[0], nxt[1])
        to_sink = []
        # no denoising <=> coregistration
        if "d" not in build_code.lower():
            add_coreg(wf, COREG, TOPUP, "out")
            to_sink.extend(
                [
                    (COREG, "out.coreg_func", "coreg_func"),
                    (COREG, "out.coreg_anat", "coreg_anat"),
                ]
            )
        else:
            to_sink.append((TOPUP, "out", "func_out"))

        # retrieve realignment parameters if available.
        if "r" in build_code:
            to_sink.append((REALIGN, "realignment_parameters", "motionparams"))
        else:
            print("no realignment parameters available")
        if "d" in build_code:
            to_sink.append((DENOISE, "noise_std_map", "noise_map"))

        wf = add2sinker(wf, to_sink, folder=f"preproc.{build_code}")
        # extra configuration for  sinker
        sinker = wf.get_node(SINKER)

        sinker.inputs.regexp_substitutions = _REGEX_SINKER
        return wf

    def run(
        self,
        wf,
        task=None,
        denoise_str=None,
        sub_id=None,
        sequence=None,
        multi_proc=False,
        dry=False,
    ):
        if task is None:
            task = ["AntiClockwise", "Clockwise"]

        # get build code back:
        bc = wf.name.split("_")[1]
        if bc in ["v", "r"]:
            denoise_str = None

        return super().run(
            wf,
            task=task,
            denoise_str=denoise_str,
            sub_id=sub_id,
            sequence=sequence,
            multi_proc=multi_proc,
            dry=dry,
        )
