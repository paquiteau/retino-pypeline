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
    add_to_sinker,
    add_topup,
)

from retino.workflows.tools import func2node, _getsubid

from retino.workflows.preprocessing.nodes import (
    file_task,
    sinker_task,
    selectfile_task,
    mask_node,
    noise_std_node,
    input_task,
)

_REGEX_SINKER = [
    (r"_sequence_(.*?)([_/])", r"\g<2>"),
    (r"_sub_id_(.*?)([_/])", r"\g<2>"),
    (r"_denoise_config_(.*?)([_/])", r"\g<2>"),
    (r"_task_(.*?)([_/])", r"\g<2>"),
    (r"rsub", "sub"),
    (r"rrsub", "sub"),
    (r"rpsub", "sub"),
]


def template_node(sequence, cached_realignment):
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


class PreprocessingManager:
    """Manager for preprocessing workflow."""

    def __init__(self, base_data_dir, working_dir):
        self.base_data_dir = base_data_dir
        self.working_dir = working_dir
        self._workflow_name = ""

    def set_workflow_name(self, name):
        self._workflow_name = name

    def show_graph(self, wf, graph2use="colored"):
        """Check the workflow. Also draws a representation."""
        # TODO ascii plot: https://github.com/ggerganov/dot-to-ascii

        fname = wf.write_graph(dotfilename="graph.dot", graph2use=graph2use)
        return fname

    def show_graph_nb(self, wf, graph2use="colored", detailed=False):
        from IPython.display import Image

        if detailed:
            return Image(
                self.show_graph(wf, graph2use=graph2use).split(".")[0] + "_detailed.png"
            )
        return Image(self.show_graph(wf))

    def _base_build(self):
        """
        Build the base of preprocessing workflow.

        4 nodes are created and added:
        - Input -> template_node -> selectfiles
           |-> files
           |-> sinker
        """
        wf = Workflow(self._workflow_name, base_dir=self.working_dir)

        in_fields = ["sub_id", "sequence", "denoise_config", "task"]
        templates_args = ["sub_id", "sequence", "task"]

        input_node = input_task(in_fields)
        tplt_node = func2node(template_node, output_names=["template", "template_args"])
        tplt_node.inputs.cached_realignment = False
        files = file_task(
            infields=templates_args,
            outfields=["data", "anat", "motion", "data_opposite"],
            base_data_dir=self.base_data_dir,
        )
        sinker = sinker_task(self.base_data_dir)

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
                (input_node, sinker, [(("sub_id", _getsubid), "container")]),
            ]
        )

        return wf

    def _build():
        raise NotImplementedError()

    def get_workflow(self, *args, **kwargs):
        """Get a preprocessing workflow."""
        wf = self._base_build()
        wf = self._build(wf, *args, **kwargs)
        return wf

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

    def get_workflow(self, name="preprocessing", build_code="v"):
        """Get a Retinotopy workflow."""
        wf_name = name + f"_{build_code}"
        self.set_workflow_name(wf_name)
        return super().get_workflow(build_code)

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

    def run(self, wf, task=None, denoise_config=None, sub_id=None, sequence=None):
        if task is None:
            task = ["AntiClockwise", "Clockwise"]

        return super().run(
            wf,
            task=task,
            denoise_config=denoise_config,
            sub_id=sub_id,
            sequence=sequence,
        )


class RealignmentPreprocessingManager(PreprocessingManager):
    """Manager for a simple Realignment Only Workflow."""

    def _build(self, wf, name="cached_realign"):
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
            (r"realign/_", "realign/")
        ]
        return wf


class NoisePreprocManager(PreprocessingManager):
    """Workflow Manager for Noise Preprocessing steps (noise map, mask, G-Map)."""

    def get_workflow(self):
        self.set_workflow_name("noise_preprocessing")
        return super().get_workflow(self)

    def _base_build(self):
        """Return a Workflow with minimal nodes."""
        wf = Workflow(name=self._workflow_name, base_dir=self.working_dir)

        infields = ["sub_id", "sequence"]
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
        input_node = input_task(infields)
        files = selectfile_task(
            infields=infields,
            template=template,
            template_args=template_args,
            base_data_dir=self.base_data_dir,
        )
        sinker = sinker_task(self.base_data_dir)

        wf.connect(
            [
                (input_node, files, [("sub_id", "sub_id"), ("sequence", "sequence")]),
                (input_node, sinker, [(("sub_id", _getsubid), "container")]),
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
