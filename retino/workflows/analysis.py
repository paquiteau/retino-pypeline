from retino.workflows.base import (
    BaseWorkflowFactory,
    node_name,
    getsubid,
    subid_varname,
    get_key,
)

from retino.interfaces.glm import DesignMatrixRetino, PhaseMap, ContrastRetino

import nipype.interfaces.io as nio
from nipype import Node, Workflow, Function, IdentityInterface


class AnalysisWorkflowFactory(BaseWorkflowFactory):
    def __init__(self, basedata_dir, working_dir, n_cycles, TR, threshold=0.001):
        self.basedata_dir = basedata_dir
        self.working_dir = working_dir
        self.n_cycles = n_cycles
        self.TR = TR
        self.threshold = threshold

    def _add_design_matrix(self, clockwise=True, extra_name=""):
        return Node(
            DesignMatrixRetino(
                n_cycles=self.n_cycles,
                volumetric_tr=self.TR,
                clockwise_rotation=clockwise,
            ),
            node_name("design", extra_name),
        )

    def _add_contrast(self, extra_name=""):
        return Node(
            ContrastRetino(volumetric_tr=self.TR),
            name=node_name("contrast", extra_name),
        )

    def _add_phase_map(self):
        return Node(PhaseMap(threshold=self.threshold), "phase_map")

    def build(self, denoise_setting):
        final_wf = Workflow(name="analysis", base_dir=self.working_dir)

        in_fields = ["sub_id", "sequence", "denoise_method"]

        input_node = Node(IdentityInterface(fields=in_fields), name="infosource")
        input_files = [
            "data_clock",
            "motion_clock",
            "data_anticlock",
            "motion_anticlock",
        ]
        files = Node(
            nio.DataGrabber(
                infields=in_fields[:-1] if denoise_setting == "noisy" else in_fields,
                outfields=input_files,
                base_directory=self.basedata_dir,
                sort_filelist=True,
                template="*",
            ),
            name="selectfiles",
        )
        files.inputs.template_args = {
            key: [["sub_id", "sequence"]] for key in input_files
        }
        if denoise_setting == "noisy":
            files.inputs.field_template = {
                "data_clock": f"sub_%02i/preprocess/noisy/*%s_ClockwiseTask_corrected.nii",
                "motion_clock": f"sub_%02i/preprocess/noisy/*%s_ClockwiseTask.txt",
                "data_anticlock": f"sub_%02i/preprocess/noisy/*%s_AntiClockwiseTask_corrected.nii",
                "motion_anticlock": f"sub_%02i/preprocess/noisy/*%s_AntiClockwiseTask.txt",
            }
            # don't consider the denoise method input params for template format
        else:
            files.inputs.field_template = {
                "data_clock": f"sub_%02i/preprocess/{denoise_setting}/*%s_ClockwiseTask_d_%s_corrected.nii",
                "motion_clock": f"sub_%02i/preprocess/{denoise_setting}/*%s_ClockwiseTask.txt",
                "data_anticlock": f"sub_%02i/preprocess/{denoise_setting}/*%s_AntiClockwiseTask_d_%s_corrected.nii",
                "motion_anticlock": f"sub_%02i/preprocess/{denoise_setting}/*%s_AntiClockwiseTask.txt",
            }
            files.inputs.template_args["data_clock"] = [
                ["sub_id", "sequence", "denoise_method"]
            ]
            files.inputs.template_args["data_anticlock"] = [
                ["sub_id", "sequence", "denoise_method"]
            ]

        c_glob = self._add_contrast(extra_name="glob")

        def merge_list(clock, anticlock):
            return [clock, anticlock]

        list_data = Node(
            Function(input_names=["clock", "anticlock"], function=merge_list),
            "merge_data",
        )
        list_design = Node(
            Function(input_names=["clock", "anticlock"], function=merge_list),
            "merge_design",
        )

        phase = self._add_phase_map()

        sinker = Node(nio.DataSink(), name="sinker")
        sinker.inputs.base_directory = self.basedata_dir
        sinker.parameterization = False
        sinker.inputs.substitutions = [("_denoise_method_", "")]
        out_dir = f"stats.{denoise_setting}.@"

        connect_list = [
            (
                input_node,
                files,
                [(a, a) for a in in_fields],
            ),
            (list_design, c_glob, [("out", "design_matrices")]),
            (list_data, c_glob, [("out", "fmri_timeseries")]),
            (
                c_glob,
                sinker,
                [(("rot_stat", get_key, "z_score"), out_dir + "rot_z")],
            ),
            (
                c_glob,
                phase,
                [
                    (("rot_stat", get_key, "z_score"), "rot_glob"),
                    (("cos_stat", get_key, "z_score"), "cos_glob"),
                ],
            ),
            (phase, sinker, [("phase_map", out_dir + "phase_map")]),
            (
                input_node,
                sinker,
                [
                    (("sub_id", getsubid), "container"),
                    (("sub_id", subid_varname), "strip_dir"),
                ],
            ),
        ]

        for name in ["clock", "anticlock"]:
            d_node = self._add_design_matrix(clockwise=name == "clock", extra_name=name)
            c_node = self._add_contrast(extra_name=name)
            connect_list.append(
                (
                    files,
                    d_node,
                    [
                        (f"data_{name}", "data_file"),
                        (f"motion_{name}", "motion_file"),
                    ],
                )
            )
            connect_list.append(
                (d_node, c_node, [("design_matrix", "design_matrices")])
            )
            connect_list.append((files, c_node, [(f"data_{name}", "fmri_timeseries")]))
            connect_list.append(
                (
                    c_node,
                    phase,
                    [
                        (("cos_stat", get_key, "z_score"), f"cos_{name}"),
                        (("sin_stat", get_key, "z_score"), f"sin_{name}"),
                    ],
                )
            )
            connect_list.append(
                (
                    c_node,
                    sinker,
                    [
                        (("cos_stat", get_key, "z_score"), out_dir + f"cos_{name}_z"),
                        (("sin_stat", get_key, "z_score"), out_dir + f"sin_{name}_z"),
                    ],
                )
            )
            # merge design_matrix and data to list for global contrast input.
            connect_list.append((d_node, list_design, [("design_matrix", name)]))
            connect_list.append((files, list_data, [(f"data_{name}", name)]))

        final_wf.connect(connect_list)
        return final_wf

    def run(self, wf, iter_on=None, sequence=None, plugin=None):
        wf.get_node("infosource").inputs.sequence = sequence
        if iter_on is not None:
            wf.get_node("infosource").iterables = iter_on
        return wf.run(plugin)
