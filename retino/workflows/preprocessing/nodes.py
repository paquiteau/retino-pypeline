"""Elementary nodes to use in preprocessin workflow.

Some Nodes are implemented as Nipype workflow but don't worry about that.
"""
import os

import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio
import nipype.interfaces.matlab as mlab
import nipype.interfaces.spm as spm
from nipype import Function, IdentityInterface, Node, Workflow

from retino.interfaces.denoise import NoiseStdMap, PatchDenoise
from retino.interfaces.tools import Mask
from retino.interfaces.topup import myTOPUP


def get_matlab_cmd(matlab_cmd):
    if matlab_cmd:
        return matlab_cmd
    else:
        return "matlab -nodesktop -nosplash"


def get_num_thread(n=None):
    if n:
        return n
    else:
        return len(os.sched_getaffinity(0))


def selectfile_node(template, basedata_dir, template_args=None):
    if template_args is None:
        template_args = ["sub_id"]
    files = Node(
        nio.DataGrabber(
            infields=template_args,
            outfields=list(template.keys()),
            base_directory=basedata_dir,
            template="*",
            sort_filelist=True,
        ),
        name="selectfiles",
    )
    files.inputs.field_template = template
    files.inputs.templates_args = {k: [template_args] for k in template.keys()}
    return files


def realign_node(matlab_cmd=None, name="realign"):
    """Create a realign node."""
    matlab_cmd = get_matlab_cmd(matlab_cmd)
    realign = Node(spm.Realign(), name=name)
    realign.inputs.separation = 1.0
    realign.inputs.fwhm = 1.0
    realign.inputs.register_to_mean = False
    realign.interface.mlab = mlab.MatlabCommand(
        matlab_cmd=matlab_cmd, resource_monitor=False, single_comp_thread=False
    )
    realign.n_procs = 3
    return realign


def topup_node(name="", working_dir=None):
    """Return a Topup node (with inner workflow).

    Input: "in.blips" and "in.blip_opposite"
    Output: "out.out"
    """
    in_topup = Node(IdentityInterface(fields=["blips", "blip_opposite"]), name="input")
    out_topup = Node(IdentityInterface(fields=["out"]), name="output")
    roi_ap = Node(fsl.ExtractROI(t_min=5, t_size=1), name="roi_ap")

    def fsl_merge(in1, in2):
        import nipype.interfaces.fsl as fsl

        merger = fsl.Merge(in_files=[in1, in2], dimension="t")
        results = merger.run()
        return results.outputs.merged_file

    fsl_merger = Node(
        Function(inputs_name=["in1", "in2"], function=fsl_merge), name="merger"
    )
    # 2.3 Topup Estimation
    topup = Node(myTOPUP(), name="topup")
    topup.inputs.fwhm = 0
    topup.inputs.subsamp = 1
    topup.inputs.out_base = "topup_out"
    topup.inputs.encoding_direction = ["y-", "y"]
    topup.inputs.readout_times = 1.0
    topup.inputs.output_type = "NIFTI"
    # 2.4 Topup correction
    applytopup = Node(fsl.ApplyTOPUP(), name="applytopup")
    applytopup.inputs.in_index = [1]
    applytopup.inputs.method = "jac"
    applytopup.inputs.output_type = "NIFTI"

    topup_wf = Workflow(name=name, working_dir=working_dir)

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


def conditional_topup(name, working_dir=None):
    """Return a Node with the conditional execution of a topup workflow."""

    def run_topup(sequence, data, data_oppposite_encoding):
        if "EPI" in sequence:
            topup_wf = topup_node(name="topup_cond", working_dir=working_dir)
            topup_wf.inputs.input.blips = data
            topup_wf.inputs.input.blip_opposite = data_oppposite_encoding
            topup_wf.run()
            return topup_wf.outputs.out.out
        else:
            return data

    return Node(
        Function(function=run_topup, input_name=["sequence", "data", "data_opposite"]),
        name="cond_topup",
    )


def coregistration_node(name, working_dir=None, matlab_cmd=None):
    """Coregistration Node.
    Input: in.func, in.anat
    Output: out.coreg_func, out.coreg_anat
    """
    matlab_cmd = get_matlab_cmd(matlab_cmd)
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


def noise_node(name):
    """Noise Node.

    Input: in_file_mag, in_file_real, in_file_imag, mask, denoise_method
    Output: denoised_file
    """
    d_node = Node(PatchDenoise(), name=name)
    d_node.n_procs = get_num_thread()

    return d_node


def preproc_noise_node(
    patch_shape,
    file_template,
    basedata_dir,
    template_args=["sub_id"],
    working_dir="",
    extra_name="",
):
    """Preprocessing noise workflow.

    file_template is a dict with key "noise" and "data".
    """
    in_node = Node(IdentityInterface(fields=template_args), name="in")
    out_node = Node(IdentityInterface(fields=["mask", "noise_std_map"]), name="out")
    files = selectfile_node(file_template)

    noise_map = Node(NoiseStdMap(), name="noise_map")
    noise_map.inputs.block_size = patch_shape
    noise_map.inputs.fft_scale = 100  # Magic Number, needs to be configured elsewhere

    brain_mask = Node(Mask(use_mean=False), name="mask")
    brain_mask.n_procs = 10
    wf = Workflow(name="mask_std", base_dir=working_dir)
    wf.connect(
        [
            (in_node, files, [("sub_id", "sub_id")]),
            (files, brain_mask, [("data", "in_file")]),
            (files, noise_map, [("noise", "noise_map_file")]),
            (brain_mask, out_node, [("mask", "mask")]),
            (noise_map, out_node, [("noise_std_map", "noise_std_map")]),
        ]
    )

    return wf
