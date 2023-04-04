"""Analysis workflows for retino_pypeline."""
from nipype import Workflow
from .base import BaseWorkflowScenario, WorkflowDispatcher

from ..tools import _get_key

from ..nodes.analysis import (
    cond_design_matrix_node,
    design_matrix_node,
    contrast_node,
    contrast_glob_node,
)


class RetinoAnalysisScenario(BaseWorkflowScenario):

    WF_NAME = "retino_analysis"

    INPUT_FIELDS = [
        "sub_id",
        "preproc_code",
        "denoise_str",
        "volumetric_tr",
    ]

    def _get_file_templates(self):
        args = [["sub_id", "preproc_code", "denoise_str"]]
        base_str = "sub_%02i/preproc/%s/%s/*"
        template = {
            "data_clock": f"{base_str}{self.sequence}_ClockwiseTask*_corrected.nii",
            "data_anticlock": f"{base_str}{self.sequence}_AntiClockwiseTask*_corrected.nii",
            "motion_clock": f"{base_str}{self.sequence}_ClockwiseTask*.txt",
            "motion_anticlock": f"{base_str}{self.sequence}_AntiClockwiseTask*.txt",
        }

        template_args = {
            #       "mask": [["sub_id", "sequence"]],
            "data_clock": args,
            "data_anticlock": args,
            "motion_clock": args,
            "motion_anticlock": args,
        }
        return template, template_args

    def get_workflow(self, n_cycles, threshold=0.05, extra_wfname="") -> Workflow:
        """Get the workflow."""
        super().get_workflow(extra_wfname="")
        to_sink = []
        for mode in ["clock", "anticlock"]:
            self.add_cond_design_matrix(n_cycles, mode)
            self.add_contrast(mode)
            to_sink.extend(
                [
                    (f"contrast_{mode}", "cos_stat", f"cos_stat_{mode}"),
                    (f"contrast_{mode}", "sin_stat", f"sin_stat_{mode}"),
                    (f"contrast_{mode}", "rot_stat", f"rot_stat_{mode}"),
                ]
            )

        self.add_contrast_glob()
        to_sink.append(("contrast_glob", "rot_stat", "rot_stat_glob"))
        self.add_phase_map(threshold)
        ph = self.get_node("phase_map")
        print(ph)
        self.add2sinker([("phase_map", "phase_map", "phase_map")], "stats.@")
        sinker = self.get_node("sinker")

        sinker.inputs.regexp_substitutions = [
            (
                r"_denoise_str_(.*?)_preproc_code_(.*?)_sequence_(.*?)_sub_id(.*?)/",
                r"\g<2>_\g<1>/",
            ),
        ]

        for node_name, stat, stat_out in to_sink:
            node = self.get_node(node_name)
            self.wf.connect(
                [(node, sinker, [((stat, _get_key, "z_score"), "stats.@" + stat_out)])]
            )
        return self.wf

    def _add_design_matrix(self, n_cycles, mode="clock", tr_unit="s"):
        """Add Design Matrix Interface to workflow."""
        file_node = self.wf.get_node("selectfiles")
        dm = design_matrix_node(
            n_cycles,
            clockwise=(mode == "clock"),
            extra_name=f"_{mode}",
        )
        dm.inputs.tr_unit = tr_unit

        self.add2wf(file_node, f"data_{mode}", dm, "data_file")
        self.add2wf(file_node, f"motion_{mode}", dm, "motion_file")

        return self._connect_volumetric_tr(dm)

    def _add_cond_design_matrix(self, n_cycles, mode="clock", tr_units="s"):
        """Add a conditional design matrix node."""
        file_node = self.get_node("selectfiles")
        dm = cond_design_matrix_node(
            n_cycles,
            clockwise=(mode == "clock"),
            extra_name=f"_{mode}",
        )
        dm.inputs.tr_unit = tr_units

        self.add2wf(file_node, f"data_{mode}", dm, "data_file")
        self.add2wf(file_node, f"motion_{mode}", dm, "motion_file")
        self.add2wf_dwim("input", dm, ["volumetric_tr", "preproc_code"])

    def _add_contrast(self, mode="clock"):
        """Add contrast Interface for specific mode to workflow."""
        contrast = contrast_node(extra_name=f"_{mode}")
        self.add2wf(f"design_{mode}", "design_matrix", contrast, "design_matrices")
        self.add2wf("selectfiles", f"data_{mode}", contrast, "fmri_timeseries")

        self.connect_volumetric_tr(contrast)

    def _connect_volumetric_tr(self, node):
        """Link volumetric_tr from input_node to node."""
        input_node = self.wf.get_node("input")
        self.wf.connect(input_node, "volumetric_tr", node, "volumetric_tr")
        return self.wf

    def _add_contrast_glob(self):
        """Add contrast for fixed effect stats."""
        contrast = contrast_glob_node()

        for mode in ["clock", "anticlock"]:
            self.add2wf(f"design_{mode}", "design_matrix", contrast, f"dm_{mode}")
            self.add2wf("selectfiles", f"data_{mode}", contrast, f"fmri_{mode}")

        return self.connect_volumetric_tr(contrast)


class RetinoAnalysisDispatcher(WorkflowDispatcher):
    ...
