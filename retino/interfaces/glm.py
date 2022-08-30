"""Retinotopic Model using a GLM. Interface for nipype."""

import os

import nibabel as nib
import numpy as np
from nilearn.glm import threshold_stats_img
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import split_filename

from retino.tools.glm import get_contrast_zscore, glm_phase_map, make_design_matrix


class DesignMatrixRetinoInputSpec(BaseInterfaceInputSpec):
    """Input Specification for Design Matrix Interface"""

    data_file = File(exists=True, mandatory=True, desc="the input data")
    motion_file = File(exists=True, mandatory=True, desc="motion regressor from spm")
    n_cycles = traits.Int(mandatory=True, desc="number of cycle for the retinotopy.")
    clockwise_rotation = traits.Bool(
        False,
        usedefault=True,
        desc="The direction of rotation, default: anticlockwise (False)",
    )
    volumetric_tr = traits.Float(1.0, desc="the time to acquire a single volume.")
    min_onset = traits.Int(
        10,
        desc=(
            "the first frame to consider for the design matrix. "
            "Set this parameter to the first frame that reaches the steady state."
        ),
    )


class DesignMatrixRetinoOuputSpec(TraitedSpec):
    """Output Specification for DesignMatrix"""

    design_matrix = File()


class DesignMatrixRetino(BaseInterface):
    """Design matrix for retinotopy interface."""

    input_spec = DesignMatrixRetinoInputSpec
    output_spec = DesignMatrixRetinoOuputSpec

    def _run_interface(self, runtime):

        fmri_timeserie = nib.load(self.inputs.data_file)

        motion = np.loadtxt(self.inputs.motion_file)

        design_matrix = make_design_matrix(
            fmri_timeserie,
            motion,
            n_cycles=self.inputs.n_cycles,
            clockwise=self.inputs.clockwise_rotation,
            TR=self.inputs.volumetric_tr,
        )

        _, base, _ = split_filename(self.inputs.data_file)
        print(base)
        design_matrix.to_csv(f"{base}_dm.csv")

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, base, _ = split_filename(self.inputs.data_file)
        outputs["design_matrix"] = os.path.abspath(f"{base}_dm.csv")
        return outputs


class ContrastRetinoInputSpec(BaseInterfaceInputSpec):
    fmri_timeseries = traits.Union(
        File(exists=True),
        traits.List(
            File,
            exists=True,
        ),
    )
    design_matrices = traits.Union(
        File(exists=True),
        traits.List(
            File,
            exists=True,
        ),
    )
    volumetric_tr = traits.Float(1.0, desc="the time to acquire a single volume.")

    first_level_kwargs = traits.Dict(
        desc="extra kwargs for the first level Model of nilearn"
    )


class ContrastRetinoOutputSpec(TraitedSpec):
    cos_stat = traits.Dict(desc="statistics for cosinus contrast")
    sin_stat = traits.Dict(desc="statistics for sinus contrast")
    rot_stat = traits.Dict(desc="statistics for rotation contrast")


class ContrastRetino(BaseInterface):
    input_spec = ContrastRetinoInputSpec
    output_spec = ContrastRetinoOutputSpec
    available_stats = ["z_score", "stat", "p_value", "effect_size", "effect_variance"]

    def _run_interface(self, runtime):

        cos, sin, rot = get_contrast_zscore(
            self.inputs.fmri_timeseries,
            self.inputs.design_matrices,
            self.inputs.volumetric_tr,
            self.inputs.first_level_kwargs,
        )

        basename = self._get_base_name()
        for arr, suffix in zip([cos, sin, rot], ["cos", "sin", "rot"]):
            for key in self.available_stats:
                arr[key].to_filename(f"{basename}_{suffix}_{key}.nii")

        return runtime

    def _list_outputs(self):

        basename = self._get_base_name()

        outputs = self._outputs().get()
        for suffix in ["cos", "sin", "rot"]:
            outputs[f"{suffix}_stat"] = {
                key: os.path.abspath(f"{basename}_{suffix}_{key}.nii")
                for key in self.available_stats
            }

        return outputs

    def _get_base_name(self):

        if isinstance(self.inputs.fmri_timeseries, list):
            _, basename, _ = split_filename(self.inputs.fmri_timeseries[0])
            basename.replace("AntiClock", "Global")
            basename.replace("Clock", "Global")
        else:
            _, basename, _ = split_filename(self.inputs.fmri_timeseries)

        return basename


class PhaseMapInputSpec(BaseInterfaceInputSpec):
    cos_clock = File(mandatory=True, desc="cos z-score for clockwise data")
    sin_clock = File(mandatory=True, desc="cos z-score for clockwise data")
    cos_anticlock = File(mandatory=True, desc="cos z-score for clockwise data")
    sin_anticlock = File(mandatory=True, desc="cos z-score for clockwise data")
    cos_glob = File(mandatory=True, desc="cos z-score fixed effect map.")
    rot_glob = File(mandatory=True, desc="rot z-score fixed effect map.")
    threshold = traits.Float(0.001, desc="the threshold for significance")


class PhaseMapOutputSpec(TraitedSpec):
    phase_map = File(desc="the phase map")


class PhaseMap(BaseInterface):
    input_spec = PhaseMapInputSpec
    output_spec = PhaseMapOutputSpec

    def _run_interface(self, runtime):
        cos_clock = nib.load(self.inputs.cos_clock).get_fdata()
        sin_clock = nib.load(self.inputs.sin_clock).get_fdata()
        cos_anticlock = nib.load(self.inputs.cos_anticlock).get_fdata()
        sin_anticlock = nib.load(self.inputs.sin_anticlock).get_fdata()

        phase_map = glm_phase_map(cos_clock, sin_clock, cos_anticlock, sin_anticlock)

        rot_glob = nib.load(self.inputs.rot_glob)
        _, threshold = threshold_stats_img(
            rot_glob, alpha=self.inputs.threshold, height_control="fpr"
        )
        phase_map[rot_glob.get_fdata() <= threshold] = np.NaN

        phase_map = nib.Nifti1Image(phase_map, nib.load(self.inputs.cos_glob).affine)

        out_name = (
            "_".join(os.path.basename(self.inputs.rot_glob).split("_")[:3])
            + "_phasemap.nii"
        )

        phase_map.to_filename(out_name)

        return runtime

    def _list_outputs(self):
        out_name = (
            "_".join(os.path.basename(self.inputs.rot_glob).split("_")[:3])
            + "_phasemap.nii"
        )
        outputs = {"phase_map": os.path.abspath(out_name)}
        return outputs
