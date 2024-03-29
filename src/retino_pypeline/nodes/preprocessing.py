"""Collections of function to create nodes to use in preprocessin workflow.

Some Nodes are implemented as Nipype workflow but don't worry about that.
"""
import nipype.interfaces.fsl as fsl
from nipype import Function, IdentityInterface, Node, Workflow
from patch_denoise.bindings.nipype import NoiseStdMap

from retino_pypeline.interfaces.motion import MagPhase2RealImag
from retino_pypeline.interfaces.tools import Mask
from retino_pypeline.interfaces.topup import CustomTOPUP

from .base import func2node


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


def topup_node_task(name="topup", base_dir=None):
    def run_topup(data, data_opposite):
        """A Function running the topup steps sequentially."""
        import os

        from nipype import Node
        from nipype.interfaces import fsl

        from retino_pypeline.nodes.preprocessing import applytopup_task, topup_task

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

    return func2node(run_topup, output_names=["out"], name=name)


def conditional_topup_task(name):
    """Return a Node with the conditional execution of a topup workflow."""

    def _func(sequence, data, data_opposite):
        from retino_pypeline.nodes.preprocessing import run_topup

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


def denoise_magnitude_task(name="denoise_mag"):
    def denoise(denoise_str, mask, noise_std_map, data):

        from patch_denoise.bindings.nipype import PatchDenoise

        from retino_pypeline.interfaces.nordic import NORDICDenoiser

        code = denoise_str.split("_")
        if code[0] == "nordic-mat":
            denoiser = NORDICDenoiser()
            denoiser.inputs.file_mag = data
            denoiser.inputs.arg_kernel_size_PCA = int(code[1])
            if int(code[2]) > 0:
                denoiser.inputs.arg_NORDIC_patch_overlap = int(code[1]) / int(code[2])
            else:
                denoiser.inputs.arg_NORDIC_patch_overlap = 1
            results = denoiser.run()
            return results.outputs.file_out_mag, noise_std_map
        else:
            denoiser = PatchDenoise()
            denoiser.inputs.in_mag = data
            denoiser.inputs.denoise_str = denoise_str
            denoiser.inputs.mask = mask
            denoiser.inputs.noise_std_map = noise_std_map
        results = denoiser.run()
        return (
            results.outputs.denoised_file,
            results.outputs.noise_std_map,
            results.outputs.rank_map,
        )

    node = func2node(
        denoise, output_names=["denoised_file", "noise_std_map", "rank_map"], name=name
    )
    return node


def denoise_complex_task(name="denoise_cpx"):
    def denoise(
        denoise_str, mask, noise_std_map, real_file, imag_file, mag_file, phase_file
    ):
        print(real_file, imag_file, mag_file, phase_file)

        from patch_denoise.bindings.nipype import PatchDenoise

        from retino_pypeline.interfaces.nordic import NORDICDenoiser

        code = denoise_str.split("_")
        if code[0] == "nordic-mat":
            denoiser = NORDICDenoiser()
            denoiser.inputs.file_mag = mag_file
            denoiser.inputs.file_phase = phase_file
            denoiser.inputs.arg_kernel_size_PCA = int(code[1])
            if int(code[2]) > 0:
                denoiser.inputs.arg_NORDIC_patch_overlap = int(code[1]) / int(code[2])
            else:
                denoiser.inputs.arg_NORDIC_patch_overlap = 1
            results = denoiser.run()
            return results.outputs.file_out_mag, noise_std_map
        else:
            denoiser = PatchDenoise()
            denoiser.inputs.in_real = real_file
            denoiser.inputs.in_imag = imag_file
            denoiser.inputs.denoise_str = denoise_str
            denoiser.inputs.mask = mask
            denoiser.inputs.noise_std_map = noise_std_map
        results = denoiser.run()
        return (
            results.outputs.denoised_file,
            results.outputs.noise_std_map,
            results.outputs.rank_map,
        )

    node = func2node(
        denoise, output_names=["denoised_file", "noise_std_map", "rank_map"], name=name
    )
    return node


def realign_complex_task(name="denoise_complex_preprocess"):
    """Prepare the file for the complex denoising"""

    def realign_complex(data, data_phase, trans_files, num_threads=1):
        import pathlib

        import nipype.interfaces.fsl as fsl
        from nipype import IdentityInterface, MapNode, Node, Workflow
        from nipype.utils.filemanip import split_filename

        from retino_pypeline.interfaces.motion import (
            ApplyXfm4D,
            MagPhase2RealImag,
            RealImag2MagPhase,
        )

        mp2ri = Node(MagPhase2RealImag(), name="mp2ri")
        ri2mp = Node(RealImag2MagPhase(), name="ri2mp")  # TODO use fslcomplex ?

        mp2ri.inputs.mag_file = data
        mp2ri.inputs.phase_file = data_phase

        extract_ref = Node(fsl.ExtractROI(t_min=0, t_size=1), name="extract_ref")

        split_real = Node(fsl.Split(dimension="t"), name="split_real")
        split_imag = Node(fsl.Split(dimension="t"), name="split_imag")
        merge_real = Node(fsl.Merge(dimension="t"), name="merge_real")
        merge_imag = Node(fsl.Merge(dimension="t"), name="merge_imag")

        applyxfm_real = MapNode(
            ApplyXfm4D(),
            iterfield=[
                "in_file",
                "trans_file",
            ],
            name="applyxfm",
        )
        applyxfm_real.inputs.single_matrix = True
        applyxfm_real.inputs.trans_file = trans_files
        applyxfm_imag = applyxfm_real.clone("applyxfm_imag")

        for n in [
            extract_ref,
            split_real,
            split_imag,
            merge_real,
            merge_imag,
            applyxfm_real,
            applyxfm_imag,
        ]:
            setattr(n, "n_procs", 1)
        output_node = Node(
            IdentityInterface(
                fields=["real_file", "imag_file", "mag_file, phase_file"]
            ),
            name="out",
        )
        wf = Workflow(name="realign_cpx_wf", base_dir=".")
        wf.connect(
            [
                (mp2ri, extract_ref, [("real_file", "in_file")]),
                (extract_ref, applyxfm_real, [("roi_file", "ref_vol")]),
                (extract_ref, applyxfm_imag, [("roi_file", "ref_vol")]),
                (mp2ri, split_real, [("real_file", "in_file")]),
                (mp2ri, split_imag, [("imag_file", "in_file")]),
                (split_real, applyxfm_real, [("out_files", "in_file")]),
                (split_imag, applyxfm_imag, [("out_files", "in_file")]),
                (applyxfm_real, merge_real, [("out_file", "in_files")]),
                (applyxfm_imag, merge_imag, [("out_file", "in_files")]),
                (merge_real, ri2mp, [("merged_file", "real_file")]),
                (merge_imag, ri2mp, [("merged_file", "imag_file")]),
                (merge_real, output_node, [("merged_file", "real_file")]),
                (merge_imag, output_node, [("merged_file", "imag_file")]),
                (ri2mp, output_node, [("mag_file", "mag_file")]),
                (ri2mp, output_node, [("phase_file", "phase_file")]),
            ]
        )
        wf.run(plugin="MultiProc", plugin_args={"n_procs": num_threads})

        # Workflow don't return the output files, so we need to do it manually
        cwd = pathlib.Path.cwd() / wf.name
        results = {
            "real": None,
            "imag": None,
            "mag": None,
            "phase": None,
        }
        for key, node, filename in zip(
            results,
            ["merge_real", "merge_imag", "ri2mp", "ri2mp"],
            [
                "vol0000_warp4D_merged.nii.gz",
                "vol0000_warp4D_merged.nii.gz",
                "vol0000_warp4D_merged_mag.nii.gz",
                "vol0000_warp4D_merged_phase.nii.gz",
            ],
        ):
            results[key] = cwd / node / filename

        _, fname, ext = split_filename(data)
        print(fname, ext)
        for key, value in results.items():
            value.rename(value.parent / f"{fname}_{key}{ext}")
            results[key] = str(value)
        return tuple(results.values())

    return func2node(
        realign_complex,
        output_names=["real_file", "imag_file", "mag_file", "phase_file"],
        name=name,
    )


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


def apply_xfm_node(name="apply_xfm"):
    # Create an embedded workflow, and run it inside the node.
    def vectorized_applyxfm(in_file, ref_vol, trans_file):

        from nipype import MapNode, Workflow

        from retino_pypeline.interfaces.motion import ApplyXfm4D

        mnode = MapNode(ApplyXfm4D(), iterfield=["in_file"], name="applyxfm")
        mnode.inputs.single_matrix = True
        mnode.inputs.in_file = in_file
        mnode.inputs.ref_vol = ref_vol
        mnode.inputs.trans_file = trans_file
        mnode.n_procs = 1

        wf = Workflow(name="vectorize_apply")
        wf.add_nodes([mnode])
        wf.run(plugin="MultiProc", plugin_args={"n_procs": 10})
        return wf.outputs.applyxfm.out_file

    return func2node(
        vectorized_applyxfm,
        output_names=["out_files"],
        name="apply_xfm",
    )


def realign_fsl_task(name="realign_fsl"):
    """Realign using FSL MCFLIRT."""
    return Node(
        fsl.MCFLIRT(
            save_plots=True,
            save_mats=True,
            save_rms=True,
            interpolation="spline",
            ref_vol=0,
            cost="mutualinfo",
            bins=256,
            dof=6,
            mean_vol=False,
            output_type="NIFTI_GZ",
        ),
        name=name,
    )


def mp2ri_task(name="mp2ri"):
    return Node(MagPhase2RealImag(), name=name)
