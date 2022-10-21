"""Analysis Workflow Manager."""

from ..base.workflow_manager import WorkflowManager
from ..tools import func2node, _get_key
from ..base.builder import add2wf_dwim, add2sinker
from ..base.nodes import file_task

from .builder import (
    add_cond_design_matrix,
    add_contrast,
    add_contrast_glob,
    add_phase_map,
)


def _tplt_node(preproc_code):
    """Template creation node for analysis workflow."""
    args = [["sub_id", "preproc_code", "denoise_str", "sequence"]]
    template = {
        "data_clock": "sub_%02i/preproc/%s/%s/*%s_ClockwiseTask*.nii",
        "data_anticlock": "sub_%02i/preproc/%s/%s/*%s_AntiClockwiseTask*.nii",
        "motion_clock": "%s*",
        "motion_anticlock": "%s*",
    }

    template_args = {
        "data_clock": args,
        "data_anticlock": args,
        "motion_clock": [[""]],
        "motion_anticlock": [[""]],
    }
    if "r" in preproc_code:
        template["motion_clock"] = "sub_%02i/preproc/%s/%s/*%s_ClockwiseTask.txt"
        template[
            "motion_anticlock"
        ] = "sub_%02i/preproc/%s/%s/*%s_AntiClockwiseTask.txt"
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
        "denoise_str",
    ]

    def _build_files(self, wf):
        """
        Build the base of analysis workflow.

        4 nodes are created and added:
        - Input -> template_node -> selectfiles
           |-> files
           |-> sinker
        """
        tplt_node = func2node(
            _tplt_node,
            name="template_node",
            output_names=["field_template", "template_args"],
        )
        files = file_task(
            infields=self.input_fields,
            outfields=[
                "data_clock",
                "data_anticlock",
                "motion_clock",
                "motion_anticlock",
            ],
            base_data_dir=self.base_data_dir,
        )
        input_node = wf.get_node("input")
        add2wf_dwim(wf, input_node, tplt_node, "preproc_code")
        add2wf_dwim(wf, tplt_node, files, ["field_template", "template_args"])
        add2wf_dwim(wf, input_node, files, self.input_fields)
        return wf


class RetinoAnalysisWorkflowManager(AnalysisWorkflowManager):
    """Analysis workflow for retinotopy."""

    workflow_name = "glm_analysis"
    input_fields = [
        "sub_id",
        "sequence",
        "preproc_code",
        "denoise_str",
        "volumetric_tr",
    ]

    def _build(self, wf, n_cycles, threshold=None):
        to_sink = []
        for mode in ["clock", "anticlock"]:
            wf = add_cond_design_matrix(wf, n_cycles, mode)
            wf = add_contrast(wf, mode)
            to_sink.extend(
                [
                    (f"contrast_{mode}", "cos_stat", f"cos_stat_{mode}"),
                    (f"contrast_{mode}", "sin_stat", f"sin_stat_{mode}"),
                    (f"contrast_{mode}", "rot_stat", f"rot_stat_{mode}"),
                ]
            )

        wf = add_contrast_glob(wf)
        to_sink.append(("contrast_glob", "rot_stat", "rot_stat_glob"))
        wf = add_phase_map(wf, threshold)
        ph = wf.get_node("phase_map")
        print(ph)
        add2sinker(wf, [("phase_map", "phase_map", "phase_map")], "stats.@")
        sinker = wf.get_node("sinker")

        sinker.inputs.regexp_substitutions = [
            (
                r"_denoise_str_(.*?)_preproc_code_(.*?)_sequence_(.*?)_sub_id(.*?)/",
                r"\g<2>_\g<1>/",
            ),
        ]

        for node_name, stat, stat_out in to_sink:
            node = wf.get_node(node_name)
            wf.connect(
                [(node, sinker, [((stat, _get_key, "z_score"), "stats.@" + stat_out)])]
            )
        return wf


class FirstLevelStats(AnalysisWorkflowManager):
    """Get first level stats on the preprocessing, eg TSNR."""

    workflow_name = "first_stats"

    def _build(self, wf):
        ...
