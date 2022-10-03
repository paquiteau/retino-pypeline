"""Collections of function to create ready to connect nodes to use in preprocessin workflow.

Some Nodes are implemented as Nipype workflow but don't worry about that.
"""

import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio
import nipype.interfaces.matlab as mlab
import nipype.interfaces.spm as spm
from nipype import Function, IdentityInterface, Node, Workflow

from retino.workflows.tools import func2node, _get_matlab_cmd, _get_num_thread

from retino.interfaces.denoise import NoiseStdMap, PatchDenoise
from retino.interfaces.tools import Mask
from retino.interfaces.topup import myTOPUP


def input_task(in_fields):
    """Return input node."""
    return Node(IdentityInterface(fields=in_fields), "input")


def sinker_task(base_data_dir):
    """Return Sinker node."""
    sinker = Node(nio.DataSink(), name="sinker")
    sinker.inputs.base_directory = base_data_dir
    sinker.parameterization = False
    return sinker


def file_task(infields, outfields, base_data_dir):
    """Return a file selector node."""
    return Node(
        nio.DataGrabber(
            infields=infields,
            outfields=outfields,
            base_directory=base_data_dir,
            template="*",
            sort_filelist=True,
        ),
        name="selectfiles",
    )


def selectfile_task(template, base_data_dir, template_args=None, infields=None):
    """Create basic Select file node, requires template."""
    files = Node(
        nio.DataGrabber(
            infields=infields,
            outfields=list(template.keys()),
            base_directory=base_data_dir,
            template="*",
            sort_filelist=True,
        ),
        name="selectfiles",
    )
    files.inputs.field_template = template
    files.inputs.templates_args = template_args
    return files


def realign_task(matlab_cmd=None, name="realign"):
    """Create a realign node."""
    matlab_cmd = _get_matlab_cmd(matlab_cmd)
    realign = Node(spm.Realign(), name=name)
    realign.inputs.separation = 1.0
    realign.inputs.fwhm = 1.0
    realign.inputs.register_to_mean = False
    realign.interface.mlab = mlab.MatlabCommand(
        matlab_cmd=matlab_cmd, resource_monitor=False, single_comp_thread=False
    )
    realign.n_procs = 3
    return realign


def topup_task(name="topup", base_dir=None):
    """Topup Task."""
    topup = Node(myTOPUP(), name=name, base_dir=base_dir)
    topup.inputs.fwhm = 0
    topup.inputs.subsamp = 1
    topup.inputs.out_base = "topup_out"
    topup.inputs.encoding_direction = ["y-", "y"]
    topup.inputs.readout_times = 1.0
    topup.inputs.output_type = "NIFTI"
    return topup


def applytopup_task(name="applytopup", base_dir=None):
    """Apply topup Task."""
    applytopup = Node(fsl.ApplyTOPUP(), name=name, base_dir=base_dir)
    applytopup.inputs.in_index = [1]
    applytopup.inputs.method = "jac"
    applytopup.inputs.output_type = "NIFTI"
    return applytopup


def alltopup_task(name="", base_dir=None):
    """Return a Topup Workflow (with inner workflow).

    Input: "in.blips" and "in.blip_opposite"
    Output: "out.out"
    """
    in_topup = Node(IdentityInterface(fields=["blips", "blip_opposite"]), name="input")
    out_topup = Node(IdentityInterface(fields=["out"]), name="output")
    roi_ap = Node(fsl.ExtractROI(t_min=3, t_size=1), name="roi_ap")

    def fsl_merge(in1, in2):
        import nipype.interfaces.fsl as fsl

        merger = fsl.Merge(in_files=[in1, in2], dimension="t")
        results = merger.run()
        return results.outputs.merged_file

    fsl_merger = Node(
        Function(inputs_name=["in1", "in2"], function=fsl_merge), name="merger"
    )
    topup = topup_task("topup")
    applytopup = applytopup_task("applytopup")
    topup_wf = Workflow(name=name, base_dir=base_dir)

    topup_wf.connect(
        [
            (in_topup, roi_ap, [("blips", "in_file")]),
            (in_topup, fsl_merger, [("blip_opposite", "in2")]),
            (roi_ap, fsl_merger, [("roi_file", "in1")]),
            (fsl_merger, topup, [("out", "in_file")]),
            (
                topup,
                applytopup,
                [
                    ("out_fieldcoef", "in_topup_fieldcoef"),
                    ("out_movpar", "in_topup_movpar"),
                    ("out_enc_file", "encoding_file"),
                ],
            ),
            (in_topup, applytopup, [("blips", "in_files")]),
            (applytopup, out_topup, [("out_corrected", "out")]),
        ]
    )
    return topup_wf


def conditional_topup_task(name):
    """Return a Node with the conditional execution of a topup workflow."""

    def run_topup(sequence, data, data_opposite):
        from nipype.interfaces import fsl
        from nipype import Node
        from retino.workflows.preprocessing.nodes import topup_task, applytopup_task
        import os

        base_dir = os.getcwd()

        if "EPI" in sequence:
            roi_ap = Node(
                fsl.ExtractROI(t_min=3, t_size=1),
                name="roi_ap",
                base_dir=base_dir,
            )
            roi_ap.inputs.in_file = data
            res = roi_ap.run()
            merger = Node(
                fsl.Merge(dimension="t"),
                name="merger",
                base_dir=base_dir,
            )
            merger.inputs.in_files = [data_opposite, res.outputs.roi_file]
            res = merger.run()

            topup = topup_task("topup", base_dir=base_dir)
            applytopup = applytopup_task("applytopup", base_dir=base_dir)
            topup.inputs.in_file = res.outputs.merged_file
            res = topup.run()

            applytopup.inputs.in_topup_fieldcoef = res.outputs.out_fieldcoef
            applytopup.inputs.in_topup_movpar = res.outputs.out_movpar
            applytopup.inputs.encoding_file = res.outputs.out_enc_file
            applytopup.inputs.in_files = data

            res = applytopup.run()
            return res.outputs.out_corrected
        else:
            return data

    return func2node(run_topup, output_names=["out"], name="cond_topup")


def coregistration_task(name, working_dir=None, matlab_cmd=None):
    """Coregistration Node.

    Input: in.func, in.anat
    Output: out.coreg_func, out.coreg_anat
    """
    matlab_cmd = _get_matlab_cmd(matlab_cmd)
    in_node = Node(IdentityInterface(fields=["func", "anat"]), name="in")
    out_node = Node(IdentityInterface(fields=["coreg_func", "coreg_anat"]), name="out")

    roi_coreg = Node(fsl.ExtractROI(t_min=0, t_size=1), name="roi_coreg")
    roi_coreg.inputs.output_type = "NIFTI"

    coreg = Node(spm.Coregister(), name="coregister")
    coreg.inputs.separation = [1, 1]
    coreg.interface.mlab = mlab.MatlabCommand(
        matlab_cmd=matlab_cmd, resource_monitor=False, single_comp_thread=False
    )

    coreg_wf = Workflow(name=name, base_dir=working_dir)

    coreg_wf.connect(
        [
            (in_node, roi_coreg, [("func", "in_file")]),
            (in_node, coreg, [("anat", "source"), ("func", "apply_to_files")]),
            (roi_coreg, coreg, [("roi_file", "target")]),
            (
                coreg,
                out_node,
                [
                    ("coregistered_files", "coreg_func"),
                    ("coregistered_source", "coreg_anat"),
                ],
            ),
        ]
    )
    return coreg_wf


def denoise_node(name):
    """Noise Node.

    Input: in_file_mag, in_file_real, in_file_imag, mask, denoise_str
    Output: denoised_file
    """
    d_node = Node(PatchDenoise(), name=name)
    d_node.n_procs = _get_num_thread()

    return d_node


def mask_node(name):
    """Mask Node."""
    return Node(
        Mask(
            convex_mask=False,
            use_mean=True,
            method="otsu",
        ),
        name=name,
    )


def noise_std_node(name):
    """Estimator for noise std."""
    return Node(NoiseStdMap(fft_scale=100, block_size=5), name=name)
