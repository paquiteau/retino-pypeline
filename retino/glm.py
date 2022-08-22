import nibabel as nib
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
import os

from nilearn.glm import threshold_stats_img
import pandas as pd


def make_design_matrix(
    fmri_timeserie, motion_file, n_cycles, clockwise=True, TR=1.0, min_onset=44
):

    sign = clockwise * 2 - 1  # clockwise = 1, anticlockwise = -1

    n_scans = fmri_timeserie.shape[-1]
    cos_reg = np.cos(
        -np.pi / 2 + sign * np.arange(n_scans) * 2 * n_cycles * np.pi / n_scans
    )
    sin_reg = np.sin(
        -np.pi / 2 + sign * np.arange(n_scans) * 2 * n_cycles * np.pi / n_scans
    )

    motion = np.loadtxt(motion_file)
    print(cos_reg.shape, sin_reg.shape, motion.shape)

    regs = np.hstack((cos_reg[:, None], sin_reg[:, None], motion))
    print(regs.shape)

    return make_first_level_design_matrix(
        np.arange(n_scans) * TR,
        drift_model="polynomial",
        drift_order=1,
        hrf_model="spm",
        add_regs=regs,
        add_reg_names=["cos", "sin", "tx", "ty", "tz", "rx", "ry", "rz"],
        min_onset=min_onset,
    )


def get_contrast_zscore(fmri_timeseries, design_matrices, TR, first_level_kwargs=None):

    first_level_kwargs = first_level_kwargs or dict()

    fmri_glm_model = FirstLevelModel(t_r=TR, **first_level_kwargs)
    fmri_glm_model = fmri_glm_model.fit(
        fmri_timeseries, design_matrices=design_matrices
    )

    # get a single dataframe to retrieve headers.
    if isinstance(design_matrices, (tuple, list)):
        dm = design_matrices[0]
    else:
        dm = design_matrices
    if isinstance(dm, str):
        dm = pd.read_csv(dm, index_col=0)
    reg_list = dm.keys()
    print(reg_list)
    n_reg = len(reg_list)

    elementary_contrasts = {reg_list[i]: np.eye(n_reg)[i] for i in range(len(reg_list))}
    contrasts = {
        "cos": elementary_contrasts["cos"],
        "sin": elementary_contrasts["sin"],
        "rot": np.vstack((elementary_contrasts["cos"], elementary_contrasts["sin"])),
    }

    output_cos = fmri_glm_model.compute_contrast(
        contrasts["cos"], stat_type="t", output_type="z_score"
    )
    output_sin = fmri_glm_model.compute_contrast(
        contrasts["sin"], stat_type="t", output_type="z_score"
    )
    output_rot = fmri_glm_model.compute_contrast(
        contrasts["rot"], stat_type="F", output_type="z_score"
    )

    return output_cos, output_sin, output_rot


def glm_phase_map(cos_clock, sin_clock, cos_anticlock, sin_anticlock, threshold=0.001):

    phase_clock = np.arctan2(-cos_clock, -sin_clock)
    phase_anticlock = np.arctan2(cos_anticlock, -sin_anticlock)
    # from Zaineb code.
    hemo = 0.5 * (phase_clock-phase_anticlock)
    hemo += np.pi * (hemo < 0 )
    hemo += np.pi * (hemo < 0 )
    pr1 = -phase_clock + hemo
    pr2 = hemo + phase_anticlock
    pr2[(pr1 - pr2) > np.pi] += (2 * np.pi)
    pr2[(pr1 - pr2) > np.pi] += (2 * np.pi)
    pr1[(pr2 - pr1) > np.pi] += (2 * np.pi)
    pr1[(pr2 - pr1) > np.pi] += (2 * np.pi)

    phase = 0.5 * (phase_clock -phase_anticlock)
    phase += np.pi/2
    phase += 2 * np.pi * (phase < - np.pi)
    phase += 2 * np.pi * (phase < - np.pi)
    phase -= 2 * np.pi * (phase > np.pi)
    phase -= 2 * np.pi * (phase > np.pi)

    _, threshold = threshold_stats_img(rot_glob, alpha=threshold, height_control='fpr')
    phase[rot_glob.get_fdata() < threshold] = 0
    return phase
