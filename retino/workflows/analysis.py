
from retino.workflows.base import BaseWorkflowFactory, node_name, getsubid, subid_varname, get_key

from retino.interfaces.glm import DesignMatrixRetino, PhaseMap, ContrastRetino

import nipype.interfaces.io as nio
from nipype import Node, Workflow, Function, IdentityInterface



class AnalysisWorkflowFactory(BaseWorkflowFactory):
    def __init__(self, basedata_dir, working_dir, n_cycles, TR, threshold=0.001):
        self.basedata_dir = basedata_dir
        self.working_dir = working_dir
        self.n_cycles = n_cycles
        self.TR = TR
        self.threshold=threshold

    def _add_design_matrix(self, clockwise=True, extra_name=""):
        return Node(DesignMatrixRetino(
            n_cycles = self.n_cycles,
            volumetric_tr = self.TR,
            clockwise_rotation=clockwise
        ), node_name("design", extra_name))

    def _add_contrast(self, extra_name=""):
        return Node(ContrastRetino(volumetric_tr=self.TR),
               name=node_name("contrast", extra_name))

    def _add_phase_map(self):
        return Node(PhaseMap(threshold= self.threshold), "phase_map")


    def build(self, preprocessing_subfolder):
        self._wf = Workflow(name="analysis", base_dir=self.working_dir)

        input_node = Node(
            IdentityInterface(fields=["sub_id", "sequence"]), name="infosource"
        )
        input_files = ["data_clock", "motion_clock", "data_anticlock", "motion_anticlock"]
        files = Node(
            nio.DataGrabber(
                infields=["sub_id", "sequence"],
                outfields=input_files,
                base_directory=self.basedata_dir,
                sort_filelist=True,
                template="*",
            ),
            name="selectfiles",
        )
        files.inputs.field_template = {
            "data_clock": f"sub_%02i/{preprocessing_subfolder}/*%s_Clock*.nii",
            "motion_clock": f"sub_%02i/{preprocessing_subfolder}/*%s_Clock*.txt",
            "data_anticlock": f"sub_%02i/{preprocessing_subfolder}/*%s_AntiClock*.nii",
            "motion_anticlock": f"sub_%02i/{preprocessing_subfolder}/*%s_AntiClock*.txt",

        }

        files.inputs.template_args = {key: [["sub_id", "sequence"]] for key in input_files}

        c_glob = self._add_contrast(extra_name="glob")


        list_data = Node(Function(input_names=["clock", "anticlock"],
                                  function=lambda clock, anticlock: [clock, anticlock]), "merge_data")
        list_design = Node(Function(input_names=["clock", "anticlock"],
                                  function=lambda clock, anticlock: [clock, anticlock]), "merge_design")

        phase = self._add_phase_map()

        sinker = Node(nio.DataSink(), name="sinker")
        sinker.inputs.base_directory = self.basedata_dir
        sinker.parameterization = False


        connect_list=[
            (
                input_node,
                files,
                [("sub_id", "sub_id"), ("sequence", "sequence")],
            ),
            (list_design, c_glob, [("out", "design_matrices")]),
            (list_data, c_glob, [("out", "fmri_timeseries")]),
            (
                c_glob,
                sinker,
                [(("rot_stat", get_key, "z_score"), "stats.@rot_z")],
            ),
            (
                input_node,
                sinker,
                [
                    (("sub_id", getsubid), "container"),
                    (("sub_id", subid_varname), "strip_dir"),
                ],
            ),
        ]

        for name in ["clock","anticlock"]:
            d_node = self._add_design_matrix(clockwise= name == "clock", extra_name=name)
            c_node = self._add_contrast(extra_name=name)
            connect_list.append((files, d_node,
                                 [
                                     (f"data_{name}", "data_file"),
                                     (f"motion_{name}", "motion_file"),
                                 ]
                                 ))
            connect_list.append((d_node, c_node, [("design_matrix", "design_matrices")]))
            connect_list.append((files, c_node, [(f"data_{name}", "fmri_timeseries")]))
            connect_list.append((c_node,
                                 phase,
                                 [
                                     (("cos_stat", get_key, "z_score"), f"cos_{name}"),
                                     (("sin_stat", get_key, "z_score"), f"sin_{name}"),
                                 ]
                                 ))
            connect_list.append((c_node, sinker,
                                [
                                    (("cos_stat", get_key, "z_score"), f"stats.@cos_{name}_z"),
                                    (("sin_stat", get_key, "z_score"), f"stats.@sin_{name}_z"),
                                ]))
            # merge design_matrix and data to list for global contrast input.
            connect_list.append((d_node, list_design, [("design_matrix", name)]))
            connect_list.append((files, list_data, [(f"data_{name}", name)]))

        self._wf.connect(connect_list)
        return self._wf

    def run(self, iter_on=None, plugin=None):
        if iter_on is not None:
            self._wf.get_node("infosource").iterables = iter_on
        self._wf.run(plugin)
