"""Preprocessing Workflow Manager.

The management of workflow happens at two levels:
First on the built time,  where the overall design of the workflow is selected.
Then at the run time,  where the parameters for selecting the data and the denoising methods are provided.

"""

from ..base import WorkflowManager
from ..base_nodes import selectfile_task, file_task
from ..tools import func2node
from .builder import (
    add_coreg,
    add_denoise_mag,
    add_realign,
    add_to_sinker,
    add_topup,
)
from .nodes import mask_node, noise_std_node

_REGEX_SINKER = [
    (r"_sequence_(.*?)([_/])", r"\g<2>"),
    (r"_sub_id_(.*?)([_/])", r"\g<2>"),
    (r"_denoise_config_(.*?)([_/])", r"\g<2>"),
    (r"_task_(.*?)([_/])", r"\g<2>"),
    (r"rsub", "sub"),
    (r"rrsub", "sub"),
    (r"rpsub", "sub"),
]


def _tplt_node(sequence, cached_realignment):
    """Template node as a Function to handle cached realignment.

    TODO add support for complex according to the denoising config string.
    """
    template = {
        "anat": "sub_%02i/anat/*_T1.nii",
        "data": "sub_%02i/func/*%s_%sTask.nii",
    }
    file_template_args = {
        "anat": [["sub_id"]],
        "data": [["sub_id", "sequence", "task"]],
    }

    if cached_realignment == "cached":
        template["data"] = "sub_%02i/realign/*%s_%sTask.nii"
        template["motion"] = "sub_%02i/realign/*%s_%sTask.txt"
        file_template_args["motion"] = [["sub_id", "sequence", "task"]]
    if "EPI" in sequence:
        template["data_opposite"] = "sub_%02i/func/*%s_Clockwise_1rep_PA.nii"
        file_template_args["data_opposite"] = [["sub_id", "sequence"]]
    return template, file_template_args


class PreprocessingWorkflowManager(WorkflowManager):
    """Manager for preprocessing workflow."""

    input_fields = ["sub_id", "sequence", "denoise_config", "task"]

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
        tplt_node = func2node(_tplt_node, output_names=["template", "template_args"])
        tplt_node.inputs.cached_realignment = False
        files = file_task(
            infields=templates_args,
            outfields=["data", "anat", "motion", "data_opposite"],
            base_data_dir=self.base_data_dir,
        )
        input_node = wf.get_node("input")
        wf.connect(
            [
                (input_node, tplt_node, [("sequence", "sequence")]),
                (
                    tplt_node,
                    files,
                    [
                        ("template", "field_template"),
                        ("template_args", "template_args"),
                    ],
                ),
                (
                    input_node,
                    files,
                    [("sub_id", "sub_id"), ("sequence", "sequence"), ("task", "task")],
                ),
            ]
        )

        return wf


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
        if build_code in ["v", "R"]:
            # cached realignment or nothing
            nxt = ("selectfiles", "data")
        elif build_code == "r":
            # realignment
            wf = add_realign(wf, "realign", "selectfiles", "data")
            nxt = ("realign", "realigned_files")
        elif build_code == "rd":
            # realignment, denoising
            wf = add_realign(wf, "realign", "selectfiles", "data")
            wf = add_denoise_mag(wf, "denoise", "realign", "realigned_files")
            nxt = ("denoise", "denoised_file")
        elif build_code == "Rd":
            # cached realignement, denoising
            wf.template_node.cached_realignmnent = True
            wf = add_denoise_mag(wf, "denoise", "selectfiles", "data")
            nxt = ("denoise", "denoised_file")
        elif build_code == "dr":
            # denoising, realigment
            wf = add_denoise_mag(wf, "denoise", "selectfiles", "data")
            wf = add_realign(wf, "realign", "denoise", "denoised_files")
            nxt = ("realign", "realigned_files")
        elif build_code == "d":
            # denoising only
            wf = add_denoise_mag(wf, "denoise", "selectfiles", "data")
        else:
            raise ValueError("Unsupported build code.")

        wf = add_topup(wf, "topup", nxt[0], nxt[1])
        wf = add_coreg(wf, "coreg", "cond_topup", "out")

        to_sink = [
            ("coreg", "out.coreg_func", "coreg_func"),
            ("coreg", "out.coreg_anat", "coreg_anat"),
        ]

        if "r" in build_code.lower():
            to_sink.append(("realign", "realignment_parameters", "motionparams"))
        if "d" in build_code:
            to_sink.append(("denoise", "noise_std_map", "noise_map"))

        wf = add_to_sinker(wf, to_sink, folder=f"preproc.{build_code}")
        # extra configuration for  sinker
        sinker = wf.get_node("sinker")

        sinker.inputs.regexp_substitutions = _REGEX_SINKER
        return wf

    def run(
        self,
        wf,
        task=None,
        denoise_config=None,
        sub_id=None,
        sequence=None,
        multi_proc=False,
    ):
        if task is None:
            task = ["AntiClockwise", "Clockwise"]

        return super().run(
            wf,
            task=task,
            denoise_config=denoise_config,
            sub_id=sub_id,
            sequence=sequence,
            multi_proc=multi_proc,
        )


class RealignmentPreprocessingManager(PreprocessingWorkflowManager):
    """Manager for a simple Realignment Only Workflow."""

    workflow_name = "cached_realign"

    def _build(self, wf):
        wf = add_realign(wf, name="realign", after_node="selectfiles", edge="data")
        wf = add_to_sinker(
            wf,
            [
                ("realign", "realigned_files", "realign.@data"),
                ("realign", "realignment_parameters", "realign.@motion"),
            ],
        )

        # configure sinker
        sinker = wf.get_node("sinker")
        sinker.inputs.regexp_substitutions = _REGEX_SINKER + [
            (r"realign/_", "realign/"),
            (r"rp_sub", "sub"),
        ]
        return wf


class NoisePreprocManager(PreprocessingWorkflowManager):
    """Workflow Manager for Noise Preprocessing steps (noise map, mask, G-Map)."""

    input_fields = ["sub_id", "sequence"]
    workflow_name = "noise_preprocessing"

    def get_workflow(self):
        return super().get_workflow(self)

    def _build_files(self, wf):
        """Return a Workflow with minimal nodes."""
        template_args = {
            "noise": [["sub_id", "sequence"]],
            # "smaps": [["sub_id"]],
            "data": [["sub_id", "sequence"]],
        }

        template = {
            "noise": "sub_%02i/extra/*%s-0v.nii",
            # "smaps": "sub_%02i/extra/*",
            "data": "sub_%02i/func/*%s_ClockwiseTask.nii",
        }
        files = selectfile_task(
            infields=self.input_fields,
            template=template,
            template_args=template_args,
            base_data_dir=self.base_data_dir,
        )
        input_node = wf.get_node("input")
        wf.connect(
            [
                (input_node, files, [("sub_id", "sub_id"), ("sequence", "sequence")]),
            ]
        )
        return wf

    def _build(self, wf, *args, **kwargs):

        mask = mask_node(name="mask")
        noise_std = noise_std_node(name="noise_std")

        files = wf.get_node("selectfiles")
        sinker = wf.get_node("sinker")

        sinker.inputs.regexp_substitutions = _REGEX_SINKER

        wf.connect(files, "data", mask, "in_file")
        wf.connect(files, "noise", noise_std, "noise_map_file")
        wf.connect(mask, "mask", sinker, "preproc_extra.@mask")
        wf.connect(noise_std, "noise_std_map", sinker, "preproc_extra.@noise_std")
        return wf
