"""Collections of function to create nodes to use in preprocessin workflow.

Some Nodes are implemented as Nipype workflow but don't worry about that.
"""
import os
import nipype.interfaces.fsl as fsl
import nipype.interfaces.spm as spm
from nipype import Function, IdentityInterface, Node, Workflow

from retino_pypeline.workflows.tools import func2node, _setup_matlab, _get_num_thread

from patch_denoise.bindings.nipype import NoiseStdMap
from retino_pypeline.interfaces.tools import Mask
from retino_pypeline.interfaces.topup import CustomTOPUP

from retino_pypeline.interfaces.motion import (
    ApplyMotion,
    MagPhase2RealImag,
)


def realign_task(matlab_cmd=None, name="realign", spm_path=None):
    """Create a realign node."""
    spm_path = spm_path or os.env.get("SPM_PATH", None)
    realign = Node(spm.Realign(paths=spm_path), name=name)
    _setup_matlab(realign)
    realign.inputs.separation = 1.0
    realign.inputs.fwhm = 1.0
    realign.inputs.register_to_mean = False
    realign.n_procs = _get_num_thread()
    return realign


def topup_task(name="topup", base_dir=None):
    """Topup Task."""
    topup = Node(CustomTOPUP(), name=name, base_dir=base_dir)
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


def run_topup(data, data_opposite):
    """A Function running the topup steps sequentially."""
    base_dir = os.getcwd()

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


def conditional_topup_task(name):
    """Return a Node with the conditional execution of a topup workflow."""

    def _func(sequence, data, data_opposite):
        from retino_pypeline.workflows.preprocessing.nodes import run_topup

        if "EPI" in sequence:
            return run_topup(data, data_opposite)
        else:
            return data

    return func2node(_func, output_names=["out"], name=name)


def run_coregistration(name, func, anat):

    extract_roi = Node(fsl.ExtractROI(t_min=0, t_size=1), name="roi_coreg")
    extract_roi.inputs.in_files = func
    extract_roi.inputs.output_type = "NIFTI"
    res = extract_roi.run()

    coreg = Node(fsl.FLIRT(dof=6), name="coregister")
    coreg.inputs.reference = anat
    coreg.inputs.in_file = res.outputs.roi_file

    res = coreg.run()
    return res.outputs.out_file




def cond_denoise_task(name):
    """Smart denoising node.

    This node will:
    1) select the proper interface for denoising
    2) use complex valued data if available
    3) apply motion correction to complex data if available.

    """

    def cond_node(denoise_str, mask, noise_std_map, data, data_phase=None, motion=None):

        from retino_pypeline.interfaces.nordic import NORDICDenoiser
        from patch_denoise.bindings.nipype import PatchDenoise
        from retino_pypeline.interfaces.motion import RealImag2MagPhase
        from retino_pypeline.workflows.preprocessing.nodes import (
            _apply_cplx_realignment,
        )

        # denoise string is defined as method-name_patch-size_patch-overlap
        code = denoise_str.split("_")
        if code[0] == "nordic-mat":
            denoiser = NORDICDenoiser()
            if data_phase is not None and motion is not None:
                real_file, imag_file = _apply_cplx_realignment(data, data_phase, motion)
                ri2mp = RealImag2MagPhase()
                ri2mp.inputs.real_file = real_file
                ri2mp.inputs.imag_file = imag_file
                results = ri2mp.run().outputs

                denoiser.inputs.file_mag = results.mag_file
                denoiser.inputs.file_phase = results.phase_file
            elif data_phase is not None:
                denoiser.inputs.file_mag = data
                denoiser.inputs.file_phase = data_phase
            else:
                denoiser.inputs.file_mag = data
            denoiser.inputs.arg_kernel_size_PCA = int(code[1])
            if int(code[2]) > 0:
                denoiser.inputs.arg_NORDIC_patch_overlap = int(code[1]) / int(code[2])
            else:
                denoiser.inputs.arg_NORDIC_patch_overlap = 1
            results = denoiser.run()
            return results.outputs.file_out_mag, noise_std_map

        denoiser = PatchDenoise()
        if data_phase is not None:
            real_file, imag_file = _apply_cplx_realignment(data, data_phase, motion)
            denoiser.inputs.in_real = real_file
            denoiser.inputs.in_imag = imag_file
        else:
            denoiser.inputs.in_mag = data
        denoiser.inputs.denoise_str = denoise_str
        denoiser.inputs.mask = mask
        denoiser.inputs.noise_std_map = noise_std_map
        results = denoiser.run()
        return results.outputs.denoised_file, results.outputs.noise_std_map

    node = func2node(
        cond_node,
        output_names=["denoised_file", "noise_std_map"],
        name=name,
    )
    node.n_procs = _get_num_thread()
    return node


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
