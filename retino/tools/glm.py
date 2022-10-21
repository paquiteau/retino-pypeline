"""Function to perform first level glm analysis."""
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix


def make_design_matrix(
    n_scans,
    n_cycles,
    motion=None,
    clockwise=True,
    TR=1.0,
    min_onset=0,
    TR_unit="s",
):
    """Create a design matrix for retinotopy.

    Parameters
    ----------
    n_scans: int
        Number of time point
    n_cycle: int
        The number of turn of the retinotopy.
    motion: np.ndarray, optional
        Motion estimation parameters, used as regressors.
    clockwise: bool
        the rotation sens for the retinotopy
    TR: float
        Volumetric TR (time for a scan)
    min_onset: int
        start of the experiment
    """
    # TODO: add option to set unit time in second or millisecond.
    if TR_unit == "ms":
        TR = TR / 1000
    elif TR_unit != "s":
        raise ValueError("unknown unit for TR. supported are `ms`, `s`")
    sign = clockwise * 2 - 1  # clockwise = 1, anticlockwise = -1

    cos_reg = np.cos(
        -np.pi / 2 + sign * np.arange(n_scans) * 2 * n_cycles * np.pi / n_scans
    )
    sin_reg = np.sin(
        -np.pi / 2 + sign * np.arange(n_scans) * 2 * n_cycles * np.pi / n_scans
    )

    if motion is not None:
        regs = np.hstack((cos_reg[:, None], sin_reg[:, None], motion))
        reg_names = ["cos", "sin", "tx", "ty", "tz", "rx", "ry", "rz"]
    else:
        regs = np.hstack((cos_reg[:, None], sin_reg[:, None]))
        reg_names = ["cos", "sin"]
    return make_first_level_design_matrix(
        np.arange(n_scans) * TR,
        drift_model="polynomial",
        drift_order=1,
        hrf_model="spm",
        add_regs=regs,
        add_reg_names=reg_names,
        min_onset=min_onset,
    )


def get_contrast_zscore(fmri_timeseries, design_matrices, TR, **first_level_kwargs):
    """Get contrast for the retinotopy."""
    if isinstance(design_matrices, list):
        design_matrices = [pd.read_csv(dm, index_col=0) for dm in design_matrices]
    else:
        design_matrices = pd.read_csv(design_matrices, index_col=0)
    fmri_glm_model = FirstLevelModel(t_r=TR, **first_level_kwargs)
    fmri_glm_model = fmri_glm_model.fit(
        fmri_timeseries, design_matrices=design_matrices
    )

    # get a single dataframe to retrieve headers.
    if isinstance(design_matrices, list):
        reg_list = design_matrices[0].keys()
    else:
        reg_list = design_matrices.keys()
    n_reg = len(reg_list)
    elementary_contrasts = {reg_list[i]: np.eye(n_reg)[i] for i in range(len(reg_list))}
    contrasts = {
        "cos": elementary_contrasts["cos"],
        "sin": elementary_contrasts["sin"],
        "rot": np.vstack((elementary_contrasts["cos"], elementary_contrasts["sin"])),
    }

    output_cos = fmri_glm_model.compute_contrast(
        contrasts["cos"], stat_type="t", output_type="all"
    )
    output_sin = fmri_glm_model.compute_contrast(
        contrasts["sin"], stat_type="t", output_type="all"
    )
    output_rot = fmri_glm_model.compute_contrast(
        contrasts["rot"], stat_type="F", output_type="all"
    )

    return output_cos, output_sin, output_rot


def glm_phase_map(cos_clock, sin_clock, cos_anticlock, sin_anticlock):
    """Compute the phase map."""
    phase_clock = np.arctan2(-cos_clock, -sin_clock)
    phase_anticlock = np.arctan2(cos_anticlock, -sin_anticlock)
    print(
        phase_clock.min(),
        phase_clock.max(),
        phase_anticlock.min(),
        phase_anticlock.max(),
        np.pi / 2,
    )
    # from Zaineb code.

    phase = 0.5 * (phase_clock - phase_anticlock)
    phase += np.pi / 2
    phase += 2 * np.pi * (phase < -np.pi)
    phase += 2 * np.pi * (phase < -np.pi)
    phase -= 2 * np.pi * (phase > np.pi)
    phase -= 2 * np.pi * (phase > np.pi)
    return phase
