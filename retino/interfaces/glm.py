"""Retinotopic Model using a GLM. Interface for nipype."""

import os

import numpy as np

import nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
)


from nipype.utils.filemanip import split_filename
from pandas._libs.tslibs.timestamps import integer_op_not_supported

from retino.glm import make_design_matrix, get_contrast_zscore


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
            clockwise=self.inputs.clockwise,
            TR=self.inputs.volumetric_tr,
            file_suffix="design_matrix.csv",
        )

        _, base, _ = split_filename(self.inputs.data_file)

        design_matrix.to_csv(base + "_design_matrix.csv")

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, base, _ = split_filename(self.inputs.data_file)
        outputs["design_matrix"] = os.path.abspath(base + "_design_matrix.csv")
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

    first_level_kwargs = traits.Dict()


class ContrastRetinoOutputSpec(TraitedSpec):
    cos_z = File(desc="Z-score after t test on cos regressor")
    sin_z = File(desc="Z-score after t test on sin regressor")
    rot_z = File(desc="Z-score after t test on rot regressor")


class ContrastRetino(BaseInterface):
    input_spec = ContrastRetinoInputSpec
    output_spec = ContrastRetinoOutputSpec

    def _run_interface(self, runtime):

        cos, sin , rot = get_contrast_zscore(
            self.inputs.fmri_timeserie,
            self.inputs.design_matrices,
            self.inputs.volumetric_tr,
            self.inputs.first_level_kwargs,
        )

        basename = self._get_base_name()
        for arr, suffix in zip([cos, sin ,rot], ['cos', 'sin', 'rot']):
            arr.to_filename(f'{basename}_{suffix}_zscore.nii')

        return runtime

    def _list_outputs(self):

        basename = self._get_base_name()

        outputs = self._outputs().get()
        for suffix in ['cos', 'sin', 'rot']:
            outputs[f"{suffix}_z"] = os.path.abspath(f'{basename}_{suffix}_zscore.nii')
        return outputs

    def _get_base_name(self):

        if isinstance(self.inputs.fmri_series, list):
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
    threshold = traits.Float(0.001)


class PhaseMapOutputSpec(TraitedSpec):
    phase_map = File(desc="the phase map")

class PhaseMap(BaseInterface):
    input_spec = PhaseMapInputSpec
    output_spec = PhaseMapOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        return
