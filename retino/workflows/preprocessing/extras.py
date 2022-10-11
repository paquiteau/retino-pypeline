"""Extra workflows for preprocessing."""
from ..base.nodes import selectfile_task
from .nodes import mask_node, noise_std_node
from .preprocessing import (
    _REGEX_SINKER,
    FILES,
    SINKER,
    PreprocessingWorkflowManager,
    add2sinker,
    add_realign,
)


class RealignmentPreprocessingManager(PreprocessingWorkflowManager):
    """Manager for a simple Realignment Only Workflow."""

    workflow_name = "cached_realign"

    def _build(self, wf):
        wf = add_realign(wf, name="realign", after_node="selectfiles", edge="data")
        wf = add2sinker(
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

    input_fields = ["sub_id", "sequence", "task"]
    workflow_name = "noise_preprocessing"

    def get_workflow(self):
        return super().get_workflow(self)

    def _build_files(self, wf):
        """Return a Workflow with minimal nodes."""
        template = {
            "data": "sub_%02i/func/*%s_%sTask.nii",
            "noise": "sub_%02i/extra/*%s-0v.nii",
            # "smaps": "sub_%02i/extra/*",
        }
        template_args = {
            "data": [["sub_id", "sequence", "task"]],
            "noise": [["sub_id", "sequence"]],
            # "smaps": [["sub_id"]],
        }

        files = selectfile_task(
            infields=["sub_id", "sequence", "task"],
            template=template,
            template_args=template_args,
            base_data_dir=self.base_data_dir,
        )
        input_node = wf.get_node("input")
        wf.connect([(input_node, files, [(a, a) for a in self.input_fields])])
        return wf

    def _build(self, wf, *args, **kwargs):

        mask = mask_node(name="mask")
        noise_std = noise_std_node(name="noise_std")

        files = wf.get_node(FILES)
        sinker = wf.get_node(SINKER)

        sinker.inputs.regexp_substitutions = _REGEX_SINKER

        wf.connect(files, "data", mask, "in_file")
        wf.connect(files, "noise", noise_std, "noise_map_file")
        wf.connect(mask, "mask", sinker, "preproc_extra.@mask")
        wf.connect(noise_std, "noise_std_map", sinker, "preproc_extra.@noise_std")
        return wf
