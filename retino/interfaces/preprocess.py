"""Nipype  bindings for retinotopic processing."""

import os  # system functions
import io

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.spm as spm  # spm
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.matlab as mlab  # how to run matlab
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.rapidart as ra  # artifact detection
import nipype.algorithms.modelgen as model  # model specification
from nipype import SelectFiles, Node, MapNode, Workflow


from retino.topup import myTOPUP


# Setup matlab command
mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
# Tell fsl to generate all output in uncompressed nifti format
fsl.FSLCommand.set_default_output_type("NIFTI")


def retino_preprocess(working_dir, output_dir):

    preproc_input_node = Node(
        util.IdentityInterface(fields=["AP", "PA", "T1w"]), name="inputspec"
    )
    # 1. Realign  with SPM
    realign = Node(spm.Realign(), name="realign")
    realign.inputs.separation = 1.0
    realign.inputs.fwhm = 1.0
    realign.inputs.register_to_mean = False
    # 2. Motion estimation and correction
    # 2.1 Extract Roi with fsl
    roi_ap = Node(fsl.ExtractROI(t_min=5, t_size=1), name="roi_ap")

    merge = Node(util.Merge(2), name="input_merge")
    # 2.2 Merge Roi in a single file for topup.
    fsl_merger = Node(fsl.Merge(), name="merger")
    fsl_merger.inputs.dimension = "t"

    # 2.3 Topup Estimation
    topup = Node(myTOPUP(), name="topup")
    topup.inputs.fwhm = 0
    topup.inputs.subsamp = 1
    topup.inputs.out_base = "topup_out"
    topup.inputs.encoding_file = os.path.join(
        os.path.abspath(os.getcwd()), "acq_params.txt"
    )
    # 2.4 Topup correction
    applytopup = Node(fsl.ApplyTOPUP(), name="applytopup")
    applytopup.inputs.in_index = [1]
    applytopup.inputs.method = "jac"
    applytopup.inputs.encoding_file = os.path.join(
        os.path.abspath(os.getcwd()), "acq_params.txt"
    )
    # 3. Coregistration with the T1 image
    roi_coreg = Node(fsl.ExtractROI(t_min=0, t_size=1), name="roi_coreg")

    coreg = Node(spm.Coregister(), name="coregister")
    coreg.inputs.separation = [1, 1]

    output_node = Node(
        util.IdentityInterface(fields=["preprocessed", "motion_par"]), name="outputspec"
    )
    # Connect Workflow
    preprocessing = Workflow(name="preprocessing", base_dir=working_dir)

    preprocessing.connect(
        [
            (preproc_input_node, realign, [("AP", "in_files")]),
            (realign, roi_ap, [("realigned_files", "in_file")]),
            (roi_ap, merge, [("roi_file", "in1")]),
            (preproc_input_node, merge, [("PA", "in2")]),
            (merge, fsl_merger, [("out", "in_files")]),
            (fsl_merger, topup, [("merged_file", "in_file")]),
            (
                topup,
                applytopup,
                [
                    ("out_fieldcoef", "in_topup_fieldcoef"),
                    ("out_movpar", "in_topup_movpar"),
                ],
            ),
            (realign, applytopup, [("realigned_files", "in_files")]),
            (applytopup, roi_coreg, [("out_corrected", "in_file")]),
            (roi_coreg, coreg, [("roi_file", "target")]),
            (preproc_input_node, coreg, [("T1w", "source")]),
            (applytopup, coreg, [("out_corrected", "apply_to_files")]),
            (coreg, output_node, [("coregistered_files", "preprocessed")]),
            (realign, output_node, [("realignment_parameters", "motion_par")]),
        ]
    )

    # Duplicate workflow for the clockwise and anti-clockwise acquisitions.
    templates = {
        "t1w": "{date}/rec/anat/t1_mp2rage_1mmiso.nii",
        "clockwise": "{date}/rec/func/EPI3D-1mmiso-tr2-4s-120rep-AP-Clock_mag.nii",
        "anticlockwise": "{date}/rec/func/EPI3D-1mmiso-tr2-4s-120rep-AP-AntiClock_mag.nii",
        "pa": "{date}/rec/func/EPI3D-1mmiso-tr2-4s-001rep-PA-AntiClock_mag.nii",
    }

    retino_input_node = Node(
        nio.SelectFiles(templates, base_directory=basedata_dir, sort_filelist=True),
        name="selectfiles",
    )

    roi_pa = Node(fsl.ExtractROI(t_min=0, t_size=1), name="roi_pa")

    preproc_clock = preprocessing.clone("preprocess_clock")
    preproc_anticlock = preprocessing.clone("preprocess_anticlock")

    sinker = pe.Node(nio.DataSink(), name="sinker")
    sinker.inputs.base_directory = os.path.realpath(
        os.path.join(output_dir, os.path.pardir)
    )
    sinker.inputs.substitutions = [
        ("rrEPI", "EPI"),
        ("rp_EPI", "EPI"),
    ]

    retino_preprocess = Workflow(name="retino_preprocess", base_dir=working_dir)

    retino_preprocess.connect(
        retino_input_node, "clockwise", preproc_clock, "inputspec.AP"
    )
    retino_preprocess.connect(retino_input_node, "t1w", preproc_clock, "inputspec.T1w")

    retino_preprocess.connect(
        retino_input_node, "anticlockwise", preproc_anticlock, "inputspec.AP"
    )
    retino_preprocess.connect(
        retino_input_node, "t1w", preproc_anticlock, "inputspec.T1w"
    )

    retino_preprocess.connect(retino_input_node, "pa", roi_pa, "in_file")
    retino_preprocess.connect(roi_pa, "roi_file", preproc_clock, "inputspec.PA")
    retino_preprocess.connect(roi_pa, "roi_file", preproc_anticlock, "inputspec.PA")
    retino_preprocess.connect(
        preproc_clock, "outputspec.preprocessed", sinker, "preproc.@clock_preprocessed"
    )
    retino_preprocess.connect(
        preproc_anticlock,
        "outputspec.preprocessed",
        sinker,
        "preproc.@anticlock_preprocessed",
    )

    retino_preprocess.connect(
        preproc_anticlock,
        "outputspec.motion_par",
        sinker,
        "preproc.@anticlock_motion_par",
    )
    retino_preprocess.connect(
        preproc_clock, "outputspec.motion_par", sinker, "preproc.@clock_motion_clock"
    )

    return retino_preprocess
