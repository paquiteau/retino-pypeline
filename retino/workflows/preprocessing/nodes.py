"""Basic function to yield nodes."""
import os

import nipype.interfaces.fsl as fsl
import nipype.interfaces.matlab as mlab
import nipype.interfaces.spm as spm
import nipype.interfaces.io as nio

from nipype import IdentityInterface, Node, Workflow, Function
from retino.interfaces.topup import myTOPUP
from retino.interfaces.denoise import PatchDenoise, NoiseStdMap
from retino.interfaces.tools import Mask


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


def realign_node(matlab_cmd=None):
    """Create a realign node."""
    matlab_cmd = get_matlab_cmd(matlab_cmd)
    realign = Node(spm.Realign(), name="realign")
    realign.inputs.separation = 1.0
    realign.inputs.fwhm = 1.0
    realign.inputs.register_to_mean = False
    realign.interface.mlab = mlab.MatlabCommand(
        matlab_cmd=matlab_cmd, resource_monitor=False, single_comp_thread=False
    )
    realign.n_procs = 3
    return realign


def topup_node(extra_name="", working_dir=None):
    """Return a Topup node (with inner workflow).

    Input: "blips" and "blip_opposite"
    Output: "out"
    """
    in_topup = Node(IdentityInterface(fields=["blips", "blip_opposite"]), name="in")
    out_topup = Node(IdentityInterface(fields=["out"]), name="out")
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

    topup_wf = Workflow(name="topup" + extra_name, base_dir=working_dir)

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


def coregistration_node(extra_name, working_dir=None, matlab_cmd=None):
    """Coregistration Node.

    Input: func, anat
    Output: coreg_func, coreg_anat
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

    coreg_wf = Workflow(name="coreg" + extra_name, base_dir=working_dir)

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


def noise_node(denoise_parameters, extra_names=""):
    """Noise Node.

    Input: in_file_mag, in_file_real, in_file_imag, mask, denoise_method
    Output: denoised_file
    """
    d_node = Node(PatchDenoise(), name="denoise")
    if denoise_parameters.method:
        d_node.n_procs = get_num_thread()

    # match input parameters to denoise node interface
    for attr in ["patch_shape", "patch_overlap", "recombination", "mask_threshold"]:
        setattr(d_node.inputs, attr, getattr(denoise_parameters, attr))

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
