import os
import numpy as np

import nibabel as nib
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
)

from retino.tools.motion import apply_motion


class MotionRealImagInputSpec(BaseInterfaceInputSpec):
    mag_file = File(exists=True, desc="input sequence magnitude")
    phase_file = File(exists=True, desc="input sequence phase")
    motion_file = File(exists=True, desc="motion csv.")


class MotionRealImagOutputSpec(TraitedSpec):
    out_real = File(desc="Real part of Sequence which have been motion corrected")
    out_imag = File(desc="Imaginary part of Sequence which have been motion corrected")


class MotionRealImag(BaseInterface):
    input_spec = MotionRealImagInputSpec
    output_spec = MotionRealImagOutputSpec

    def _run_interface(self, runtime):

        image_nii = nib.load(self.inputs.in_file)
        images = image_nii.get_fdata()
        motions = np.genfromtxt(self.inputs.motion_file, delimiter="  ")
        corrected_images = np.zeros_like(images)

        for i in range(images.shape[-1]):
            corrected_images[..., i] = apply_motion(
                images[..., i], motions[i], reverse=True
            )

        corrected_nii = nib.Nifti1Image(corrected_images, affine=image_nii.affine)
        corrected_nii.to_filename(
            os.path.abspath("r" + os.path.basename(self.inputs.in_file))
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out"] = os.path.abspath("r" + os.path.basename(self.inputs.in_file))
        return outputs
