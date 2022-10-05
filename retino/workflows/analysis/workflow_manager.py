"""Analysis Workflow Manager."""

from ..base import WorkflowManager
from ..tools import func2node
from ..base.nodes import file_task

from .builder import add_design_matrix, add_contrast, add_contrast_glob, add_phase_map


def _tplt_node(preproc_code):
    """Template creation node for analysis workflow."""
    args = [["sub_id", "preproc_code", "sequence", "denoise_code"]]
    template = {
        "data_clock": "sub_%02i/preproc/%s/*%s_ClockwiseTask_d_%s_corrected.nii",
        "data_anticlock": "sub_%02i/preproc/%s/*%s_AntiClockwiseTask_d_%s_corrected.nii",
    }

    template_args = {
        "data_clock": args,
        "data_anticlock": args,
    }
    if "r" in preproc_code:
        template["motion_clock"] = "sub_%02i/preproc/%s/*%s_ClockwiseTask*.txt"
        template["motion_anticlock"] = "sub_%02i/preproc/%s/*%s_ClockwiseTask*.txt"
        template_args["motion_clock"] = args
        template_args["motion_anticlock"] = args

    return template, template_args


class AnalysisWorkflowManager(WorkflowManager):
    """Manager for Analysis Workflow."""

    workflow_name = "analysis"
    input_fields = [
        "sub_id",
        "sequence",
        "preproc_code",
        "denoise_code",
    ]

    def _build_files(self, wf):
        """
        Build the base of analysis workflow.

        4 nodes are created and added:
        - Input -> template_node -> selectfiles
           |-> files
           |-> sinker
        """
        templates_args = ["sub_id", "sequence", "preproc_code", "denoise_code"]

        # template node needs to be implemented in child classes.
        tplt_node = func2node(
            _tplt_node,
            name="template_node",
            output_names=["template", "template_args"],
        )
        tplt_node.inputs.cached_realignment = False
        files = file_task(
            infields=templates_args,
            outfields=["data", "anat", "noise", "mask", "motion", "data_opposite"],
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
                (input_node, files, [(a, a) for a in templates_args]),
            ]
        )

        return wf


class RetinoAnalysisWorkflowManager(AnalysisWorkflowManager):
    """Analysis workflow for retinotopy."""

    workflow_name = "glm_analysis"
    input_fields = [
        "sub_id",
        "sequence",
        "preproc_code",
        "denoise_code",
        "volumetric_tr",
    ]

    def _build(self, wf, n_cycles, threshold=None):

        for mode in ["clock", "anticlock"]:
            wf = add_design_matrix(wf, n_cycles, mode)
            wf = add_contrast(wf, mode)

        wf = add_contrast_glob(wf)
        wf = add_phase_map(wf, threshold)

        return wf

    ...


class FirstLevelStats(AnalysisWorkflowManager):
    """Get first level stats on the preprocessing, eg TSNR."""

    ...
