"""Preprocessing Workflow Manager.

The management of workflow happens at two levels:
First on the built time,  where the overall design of the workflow is selected.
Then at the run time,  where the parameters for selecting the data and the denoising methods are provided.

"""
from nipype import Workflow
from retino.workflows.preprocessing.builder import (
    add_coreg,
    add_denoise_mag,
    add_realign,
    add_sinker,
    add_topup,
)


class PreprocessingManager:
    """Manager for preprocessing workflow."""

    def __init__(self, base_data_dir, working_dir):
        self.base_data_dir = base_data_dir
        self.working_dir = working_dir
        self._workflow_name = ""

    def set_workflow_name(self, name):
        self._workflow_name = name

    def show_graph(self, wf):
        """Check the workflow. Also draws a representation."""
        # TODO ascii plot: https://github.com/ggerganov/dot-to-ascii

        fname = wf.write_graph(dotfilename="graph.dot", graph2use="colored")
        return fname

    def run(self, wf, multi_proc=False, **kwargs):
        """Run the workflow with iterables parametrization defined in kwargs."""
        inputnode = wf.get_node("input")
        inputnode.iterables = []
        for key in kwargs:
            inputnode.iterables.append((key, kwargs[key]))

        wf.run(plugin="MultiProc" if multi_proc else None)


class RetinotopyPreprocessingManager(PreprocessingManager):
    """Manager for Retinotopy base workflow.

    There is two task: Clockwise and Anticlockwise.
    """

    def build(self, build_code=""):
        """Create a Retinotopy Workflow with option for the order of steps.

        Parameters
        ----------
        build_code: str

        Returns
        -------
        wf : a nipype workflow ready to be run.
        """
        wf = Workflow(name="preprocess_" + build_code, base_dir=self.working_dir)
        wf = add_base(
            wf,
            base_data_dir=self.base_data_dir,
            cached_realignment="R" in build_code,
        )

        if build_code in ["", "R"]:
            # cached realignment or nothing
            nxt = ("selectfiles", "data")
        elif build_code == "r":
            # realignment
            wf = add_realign(wf, "realign", "selectfiles", "data")
            nxt = ("realign", "realigned_files")
        elif build_code == "rd":
            # realignment, denoising
            wf = add_realign(wf, "realign", "selectfiles", "data")
            wf = add_denoise(wf, "denoise", "realign", "realigned_files")
            nxt = ("denoise", "denoised_file")
        elif build_code == "Rd":
            # cached realignement, denoising
            wf = add_denoise(wf, "denoise", "selectfiles", "data")
            nxt = ("denoise", "denoised_file")
        elif build_code == "dr":
            # denoising, realigment
            wf = add_denoise(wf, "denoise", "selectfiles", "data")
            wf = add_realign(wf, "realign", "denoise", "denoised_files")
            nxt = ("realign", "realigned_files")
        elif build_code == "d":
            # denoising only
            wf = add_denoise(wf, "denoise", "selectfiles", "data")
        else:
            raise ValueError("Unsupported build code.")

        wf = add_topup(wf, "topup", nxt[0], nxt[1])
        wf = add_coreg(wf, "coreg", "topup", "out")

        to_sink = [
            ("coreg", "out.func", "coreg_func"),
            ("coreg", "out.anat", "coreg_anat"),
        ]

        if "r" in build_code.lower():
            to_sink.append(("realign", "realignment_parameters", "motionparams"))
        if "d" in build_code:
            to_sink.append(("denoise", "noise_std_map", "noise_map"))

        wf = add_sinker(wf, to_sink)
        # extra configuration for  sinker
        sinker = wf.get_node("sinker")

        sinker.inputs.regexp_substitutions = [
            (r"rp_sub", "sub"),
            (r"rrsub", "sub"),
            (r"rsub", "sub"),
            (r"_sequence_(.*?)[_/]", "_"),
            (r"_sub_id_(.*?)[_/]", "_"),
            (r"_task_(.*?)[_/]", "_"),
        ]
        return wf

    def _build_minimal(self, wf, after_node, edge):
        """Minimal setup with conditional topup and coregistration."""
        wf = add_topup(wf, "topup", after_node, edge)
        wf = add_coreg(wf, "coreg", after_node="topup", edge="out")
        wf = add_sinker(
            wf,
            [
                ("coreg", "out.func", "preprocess.@coreg_func"),
                ("coreg", "out.anat", "preprocess.@coreg_anat"),
            ],
        )
        return wf

    def run(self, wf, task=["Clockwise", "AntiClockwise"], **kwargs):
        """Rune the workflow, with iterables kwargs."""
        return super().run(wf, task=task, **kwargs)


class RealignmentPreprocessingManager(PreprocessingManager):
    """Manager for a simple Realignment Only Workflow."""

    def build(self, name="realign"):
        wf = Workflow(name=name, base_dir=self.working_dir)
        wf = add_base(
            wf,
            base_data_dir=self.base_data_dir,
            cached_realignment=False,
        )
        wf = add_realign(wf, name="realign", after_node="selectfiles", edge="data")
        wf = add_sinker(
            wf,
            [
                ("realign", "realigned_files", "realign.@data"),
                ("realign", "realignment_parameters", "realign.@motion"),
            ],
        )

        # configure sinker
        sinker = wf.get_node("sinker")
        sinker.inputs.regexp_substitutions = [
            (r"_sequence_(.*?)[_/]", "_"),
            (r"_sub_id_(.*?)[_/]", "_"),
            (r"_task_(.*?)[_/]", "_"),
            (r"realign/_", "realign/"),
        ]
        return wf


class NoisePreprocessingManager(PreprocessingManager):
    """Workflow Manager for Noise Preprocessing steps (noise map estimation, G-Map)."""
    """Workflow Manager for Noise Preprocessing steps (noise map, mask, G-Map)."""

    def build(self, name="noise_preprocessing"):
        wf = Workflow(name=name, base_dir=self.working_dir)

    ...
