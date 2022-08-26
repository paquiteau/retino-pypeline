"""worflow creation for fmri preprocessing and retinotopy."""

import os

import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio
import nipype.interfaces.matlab as mlab
import nipype.interfaces.spm as spm
import nipype.interfaces.utility as util
from nipype import IdentityInterface, Node, Workflow

from retino.interfaces.denoise import NoiseStdMap, PatchDenoise
from retino.interfaces.glm import ContrastRetino, DesignMatrixRetino, PhaseMap
from retino.interfaces.topup import myTOPUP

MATLAB_CMD = "matlab -nosplash -nodesktop   "


def getsubid(i):
    return f"sub_{i:02d}"


def strip_id(i):
    return f"_sub_id_{i}"


def get_key(d, k):
    return d[k]


def get_preprocessing_workflow_topup(working_dir):
    """Workflow for preprocessing of fmri data.

    It uses fsl and spm.
    """
    # Node Setup

    preproc_input_node = Node(
        util.IdentityInterface(fields=["AP", "PA", "T1w"]), name="input"
    )
    # 1. Realign  with SPM
    realign = Node(spm.Realign(), name="realign")
    realign.inputs.separation = 1.0
    realign.inputs.fwhm = 1.0
    realign.inputs.register_to_mean = False
    realign.interface.mlab = mlab.MatlabCommand(
        matlab_cmd=MATLAB_CMD, resource_monitor=False, single_comp_thread=False
    )
    realign.n_procs = 3

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
    topup.inputs.encoding_direction = ["y-", "y"]
    topup.inputs.readout_times = 1.0
    topup.inputs.output_type = "NIFTI"
    # topup.inputs.encoding_file = os.path.join(
    #     os.path.abspath(os.getcwd()), "acq_params.txt"
    # )
    # 2.4 Topup correction
    applytopup = Node(fsl.ApplyTOPUP(), name="applytopup")
    applytopup.inputs.in_index = [1]
    applytopup.inputs.method = "jac"
    applytopup.inputs.output_type = "NIFTI"
    applytopup.inputs.encoding_file = os.path.join(
        os.path.abspath(os.getcwd()), "acq_params.txt"
    )
    roi_coreg = Node(fsl.ExtractROI(t_min=0, t_size=1), name="roi_coreg")
    roi_coreg.inputs.output_type = "NIFTI"

    coreg = Node(spm.Coregister(), name="coregister")
    coreg.inputs.separation = [1, 1]
    coreg.interface.mlab = mlab.MatlabCommand(
        matlab_cmd=MATLAB_CMD, resource_monitor=False, single_comp_thread=False
    )

    output_node = Node(
        util.IdentityInterface(
            fields=["preprocessed", "motion_par", "preprocessed_anat"]
        ),
        name="output",
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
                    ("out_enc_file", "encoding_file"),
                ],
            ),
            (realign, applytopup, [("realigned_files", "in_files")]),
            (realign, output_node, [("realignment_parameters", "motion_par")]),
            (applytopup, roi_coreg, [("out_corrected", "in_file")]),
            (roi_coreg, coreg, [("roi_file", "target")]),
            (preproc_input_node, coreg, [("T1w", "source")]),
            (applytopup, coreg, [("out_corrected", "apply_to_files")]),
            (
                coreg,
                output_node,
                [
                    ("coregistered_files", "preprocessed"),
                    ("coregistered_source", "preprocessed_anat"),
                ],
            ),
        ]
    )

    return preprocessing


def get_preprocessing_workflow_notopup(working_dir):
    """Workflow for preprocessing of fmri data.

    It uses fsl and spm.
    """
    # Node Setup

    preproc_input_node = Node(
        util.IdentityInterface(fields=["AP", "T1w"]), name="input"
    )
    # 1. Realign  with SPM
    realign = Node(spm.Realign(), name="realign")
    realign.inputs.separation = 1.0
    realign.inputs.fwhm = 1.0
    realign.inputs.register_to_mean = False
    realign.interface.mlab = mlab.MatlabCommand(
        matlab_cmd=MATLAB_CMD, resource_monitor=False, single_comp_thread=False
    )

    roi_coreg = Node(fsl.ExtractROI(t_min=0, t_size=1), name="roi_coreg")
    roi_coreg.inputs.output_type = "NIFTI"

    coreg = Node(spm.Coregister(), name="coregister")
    coreg.inputs.separation = [1, 1]
    coreg.interface.mlab = mlab.MatlabCommand(
        matlab_cmd=MATLAB_CMD, resource_monitor=False, single_comp_thread=False
    )
    output_node = Node(
        util.IdentityInterface(
            fields=["preprocessed", "motion_par", "preprocessed_anat"]
        ),
        name="output",
    )
    # Connect Workflow
    preprocessing = Workflow(name="preprocessing", base_dir=working_dir)

    preprocessing.connect(
        [
            (preproc_input_node, realign, [("AP", "in_files")]),
            (
                realign,
                output_node,
                [
                    ("realignment_parameters", "motion_par"),
                    ("realigned_files", "preprocessed"),
                ],
            ),
            (realign, roi_coreg, [("realigned_files", "in_file")]),
            (roi_coreg, coreg, [("realigned_files", "target")]),
            (preproc_input_node, coreg, [("t1w", "source")]),
            (realign, coreg, [("out_corrected", "apply_to_files")]),
            (
                coreg,
                output_node,
                [
                    ("coregistered_files", "preprocessed"),
                    ("coregistered_source", "preprocessed_anat"),
                ],
            ),
        ]
    )

    return preprocessing


def get_retino_preprocessing_workflow(
    working_dir,
    output_dir,
    basedata_dir,
    sequence="EPI3D",
    field_template=None,
):
    """
    Create a Retinotopy preprocessing pipeline

    Parameters
    ----------
    working_dir: str
        The temporary directory to store all intermediate results.
    output_dir: str
        The final location of the processed dataset
    basedata_dir: str
        The localtion of the original data.
    type: "EPI" or "NC"
        if "EPI", a ific workflow using TOPUP will be used.
    """

    input_node = Node(
        IdentityInterface(fields=["sub_id", "sequence"]), name="infosource"
    )
    input_node.inputs.sequence = sequence

    in_files = Node(
        nio.DataGrabber(
            infields=["sub_id", "sequence"],
            outfields=["t1w", "clockwise", "anticlockwise", "pa"],
            base_directory=basedata_dir,
            sort_filelist=True,
            template="*",
        ),
        name="selectfiles",
    )
    in_files.inputs.base_directory = basedata_dir
    if field_template is None:
        in_files.inputs.field_template = {
            "t1w": "sub_%02i/anat/sub_%02i_T1.nii",
            "clockwise": "sub_%02i/func/sub_%02i_%s_ClockwiseTask.nii",
            "anticlockwise": "sub_%02i/func/sub_%02i_%s_AntiClockwiseTask.nii",
        }
    else:
        in_files.inputs.field_template = field_template

    in_files.inputs.template_args = {
        "t1w": [["sub_id", "sub_id"]],
        "clockwise": [["sub_id", "sub_id", "sequence"]],
        "anticlockwise": [["sub_id", "sub_id", "sequence"]],
    }

    sinker = Node(nio.DataSink(), name="sinker")
    sinker.inputs.base_directory = output_dir
    sinker.parameterization = False
    sinker.inputs.substitutions = [
        ("rp_sub", "sub"),
        ("rrsub", "sub"),
    ]
    if sequence == "EPI3D":
        preprocessing = get_preprocessing_workflow_topup(working_dir)
    else:
        preprocessing = get_preprocessing_workflow_notopup(working_dir)

    p_clock = preprocessing.clone("preprocess_clock")
    p_anticlock = preprocessing.clone("preprocess_anticlock")

    wf = Workflow(name="retino_preprocess", base_dir=working_dir)

    wf.connect(
        [
            (
                input_node,
                in_files,
                [("sub_id", "sub_id"), ("sequence", "sequence")],
            ),
            (
                in_files,
                p_clock,
                [
                    ("clockwise", "input.AP"),
                    ("t1w", "input.T1w"),
                ],
            ),
            (
                in_files,
                p_anticlock,
                [
                    ("anticlockwise", "input.AP"),
                    ("t1w", "input.T1w"),
                ],
            ),
            (
                input_node,
                sinker,
                [
                    (("sub_id", getsubid), "container"),
                    (("sub_id", strip_id), "strip_dir"),
                ],
            ),
            (
                p_clock,
                sinker,
                [
                    ("output.motion_par", "preproc.@clock_motion"),
                    ("output.preprocessed", "preproc.@clock_data"),
                    ("output.preprocessed_anat", "preproc.@anat_realign"),
                ],
            ),
            (
                p_anticlock,
                sinker,
                [
                    ("output.motion_par", "preproc.@anticlock_motion"),
                    ("output.preprocessed", "preproc.@anticlock_data"),
                ],
            ),
        ]
    )

    if sequence == "EPI3D":
        # add the PA  specific component.

        in_files.inputs.template_args["pa"] = [["sub_id", "sub_id", "sequence"]]
        in_files.inputs.field_template[
            "pa"
        ] = "sub_%02i/func/sub_%02i_%s_Clockwise_1rep_PA.nii"
        roi_pa = Node(fsl.ExtractROI(t_min=0, t_size=1), name="roi_pa")

        wf.connect(
            [
                (in_files, roi_pa, [("pa", "in_file")]),
                (roi_pa, p_clock, [("roi_file", "input.PA")]),
                (roi_pa, p_anticlock, [("roi_file", "input.PA")]),
            ]
        )
    return wf


def get_retino_workflow(
    basedata_dir, working_dir, n_cycles, TR, threshold=0.001, sequence="EPI3D"
):
    """
    Retinotopy analysis workflow .

    The data has been alreadly preprocessed, and we can then perform the analysis.

    Parameters
    ----------
    basedata_dir: str
        The location of the preprocessed dataset
    working_dir: str
        Temporary working directory
    sequence: str
        Name of the sequence use to select the files.
    n_cycles: int
        Number of cycle in the retinotopy
    TR: float
        Time to acquire a single volume
    threshold: float, default 0.001
        Probability threshold for the activation detection

    Returns
    -------
    nipype.Workflow: The retinotopy analysis workflow.
    """
    input_node = Node(
        IdentityInterface(fields=["sub_id", "sequence"]), name="infosource"
    )
    input_node.inputs.sequence = sequence

    # Duplicate workflow for the clockwise and anti-clockwise acquisitions.
    preproc_files = Node(
        nio.DataGrabber(
            infields=["sub_id", "sequence"],
            outfields=[
                "data_clock",
                "motion_clock",
                "data_anticlock",
                "motion_anticlock",
            ],
            base_directory=basedata_dir,
            sort_filelist=True,
            template="*",
        ),
        name="selectfiles",
    )
    preproc_files.inputs.field_template = {
        "data_clock": "sub_%02i/preproc/sub_%02i_%s_ClockwiseTask_corrected.nii",
        "motion_clock": "sub_%02i/preproc/sub_%02i_%s_ClockwiseTask.txt",
        "data_anticlock": "sub_%02i/preproc/sub_%02i_%s_AntiClockwiseTask_corrected.nii",
        "motion_anticlock": "sub_%02i/preproc/sub_%02i_%s_AntiClockwiseTask.txt",
    }
    preproc_files.inputs.template_args = {
        "data_clock": [["sub_id", "sub_id", "sequence"]],
        "motion_clock": [["sub_id", "sub_id", "sequence"]],
        "data_anticlock": [["sub_id", "sub_id", "sequence"]],
        "motion_anticlock": [["sub_id", "sub_id", "sequence"]],
    }

    design_clock = Node(DesignMatrixRetino(), name="design_clock")
    design_clock.inputs.n_cycles = n_cycles
    design_clock.inputs.volumetric_tr = TR
    design_clock.clockwise_rotation = True

    design_anticlock = design_clock.clone("design_anticlock")
    design_anticlock.inputs.clockwise_rotation = False

    contrast_clock = Node(ContrastRetino(), name="contrast_clock")
    contrast_clock.inputs.volumetric_tr = TR

    contrast_anticlock = contrast_clock.clone("contrast_anticlock")
    contrast_glob = contrast_clock.clone("contrast_glob")

    list_data = Node(util.Merge(2), "merge_data")
    list_design = Node(util.Merge(2), "merge_design")

    phase = Node(PhaseMap(), "phase_map")
    phase.inputs.threshold = threshold

    sinker = Node(nio.DataSink(), name="sinker")
    sinker.inputs.base_directory = basedata_dir
    sinker.parameterization = False
    wf = Workflow(name="glm_retino", base_dir=working_dir)

    wf.connect(
        [
            (
                input_node,
                preproc_files,
                [("sub_id", "sub_id"), ("sequence", "sequence")],
            ),
            (
                preproc_files,
                design_clock,
                [
                    ("data_clock", "data_file"),
                    ("motion_clock", "motion_file"),
                ],
            ),
            (
                preproc_files,
                design_anticlock,
                [
                    ("data_anticlock", "data_file"),
                    ("motion_anticlock", "motion_file"),
                ],
            ),
            (
                preproc_files,
                list_data,
                [("data_clock", "in1"), ("data_anticlock", "in2")],
            ),
            (design_clock, list_design, [("design_matrix", "in1")]),
            (design_anticlock, list_design, [("design_matrix", "in2")]),
            (design_clock, contrast_clock, [("design_matrix", "design_matrices")]),
            (
                design_anticlock,
                contrast_anticlock,
                [("design_matrix", "design_matrices")],
            ),
            (list_design, contrast_glob, [("out", "design_matrices")]),
            (list_data, contrast_glob, [("out", "fmri_timeseries")]),
            (preproc_files, contrast_clock, [("data_clock", "fmri_timeseries")]),
            (
                preproc_files,
                contrast_anticlock,
                [("data_anticlock", "fmri_timeseries")],
            ),
            (
                contrast_clock,
                phase,
                [
                    (("cos_stat", get_key, "z_score"), "cos_clock"),
                    (("sin_stat", get_key, "z_score"), "sin_clock"),
                ],
            ),
            (
                contrast_anticlock,
                phase,
                [
                    (("cos_stat", get_key, "z_score"), "cos_anticlock"),
                    (("sin_stat", get_key, "z_score"), "sin_anticlock"),
                ],
            ),
            (
                contrast_glob,
                phase,
                [
                    (("rot_stat", get_key, "z_score"), "rot_glob"),
                    (("cos_stat", get_key, "z_score"), "cos_glob"),
                ],
            ),
            (phase, sinker, [("phase_map", "stats.@phase_map")]),
            (
                contrast_glob,
                sinker,
                [(("rot_stat", get_key, "z_score"), "stats.@rot_z")],
            ),
            (
                contrast_clock,
                sinker,
                [
                    (("cos_stat", get_key, "z_score"), "stats.@cos_clock_z"),
                    (("sin_stat", get_key, "z_score"), "stats.@sin_clock_z"),
                ],
            ),
            (
                contrast_anticlock,
                sinker,
                [
                    (("cos_stat", get_key, "z_score"), "stats.@cos_anticlock_z"),
                    (("sin_stat", get_key, "z_score"), "stats.@sin_anticlock_z"),
                ],
            ),
            (
                input_node,
                sinker,
                [
                    (("sub_id", getsubid), "container"),
                    (("sub_id", strip_id), "strip_dir"),
                ],
            ),
        ]
    )

    return wf


def get_denoising_workflow(
    basedata_dir,
    working_dir,
    output_dir,
    sequence,
    patch_shape,
    denoise_method="nordic",
    **kwargs,
):

    input_node = Node(
        IdentityInterface(fields=["sub_id", "sequence"]), name="infosource"
    )
    input_node.inputs.sequence = sequence

    in_files = Node(
        nio.DataGrabber(
            infields=["sub_id", "sequence"],
            outfields=["data_clock_mag", "data_clock_phase",
                       "data_anticlock_mag", "data_anticlock_phase",
                       "noise_map"],
            base_directory=basedata_dir,
            sort_filelist=True,
            template="*",
        ),
        name="selectfiles",
    )
    in_files.inputs.field_template = {
        "data_clock_mag": "sub_%02i/func_noisy/sub_%02i_%s_ClockwiseTask_mag.nii",
        "data_clock_phase": "sub_%02i/func_noisy/sub_%02i_%s_ClockwiseTask_pha.nii",
        "data_anticlock_mag": "sub_%02i/func_noisy/sub_%02i_%s_AntiClockwiseTask_mag.nii",
        "data_anticlock_phase": "sub_%02i/func_noisy/sub_%02i_%s_ClockwiseTask_pha.nii",
        "noise_map": "sub_%02i/extra/sub_%02i_%s-0v_mag.nii",
    }
    in_files.inputs.template_args = {
        "data_clock_mag": [["sub_id", "sub_id", "sequence"]],
        "data_clock_phase": [["sub_id", "sub_id", "sequence"]],
        "data_anticlock_mag": [["sub_id", "sub_id", "sequence"]],
        "data_anticlock_phase": [["sub_id", "sub_id", "sequence"]],
        "noise_map": [["sub_id", "sub_id", "sequence"]],
    }

    noise_map = Node(NoiseStdMap(), name="noise_map")
    noise_map.inputs.block_size = patch_shape
    noise_map.inputs.fft_scale = 100  # magic number from subject 5.

    denoise_clock = Node(PatchDenoise(), name="noise_anticlock")
    denoise_clock.inputs.patch_shape = patch_shape
    denoise_clock.inputs.patch_overlap = patch_shape//2
    denoise_clock.inputs.recombination = kwargs.pop("recombination", "weighted")
    denoise_clock.inputs.extra_kwargs = kwargs
    denoise_clock.inputs.method = denoise_method

    denoise_anticlock = denoise_clock.clone("denoise_anticlock")

    sinker = Node(nio.DataSink(), name="sinker")
    sinker.inputs.base_directory = output_dir


    wf = Workflow("patch_denoising", base_dir=working_dir)

    wf.connect(
        [
            (input_node, in_files, [("sub_id", "sub_id"), ("sequence", "sequence")]),
            (in_files, noise_map, [("noise_map", "noise_map_file")]),
            (
                in_files,
                denoise_clock,
                [
                    ("data_clock_mag", "in_file_mag"),
                    ("data_clock_phase", "in_file_phase"),
                ],
            ),
            (
                in_files,
                denoise_anticlock,
                [
                    ("data_anticlock_mag", "in_file_mag"),
                    ("data_anticlock_phase", "in_file_phase"),
                ],
            ),
            (noise_map, denoise_clock, [("noise_std_map", "noise_std")]),
            (noise_map, denoise_anticlock, [("noise_std_map", "noise_std")]),
            (denoise_clock, sinker, [("denoised_file", "denoise.@clock_denoise")]),
            (denoise_anticlock, sinker, [("denoised_file", "denoise.@anticlock_denoise")]),
            (noise_map, sinker, [("noise_std_map", "denoise.@noise_std_map")]),
            (
                input_node,
                sinker,
                [
                    (("sub_id", getsubid), "container"),
                    (("sub_id", strip_id), "strip_dir"),
                ],
            ),
        ]
    )
    return wf
