"""Motion Recalibration Interfaces."""
import os
import numpy as np

import nibabel as nib
from nipype.interfaces.base import (
    SimpleInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
)

from retino.tools.motion import apply_motion


class ApplyMotionInputSpec(BaseInterfaceInputSpec):
    """InputSpec for ApplyMotion."""

    in_file = File(exists=True, desc="input sequence magnitude")
    motion_file = File(exists=True, desc="motion csv.")


class ApplyMotionOutputSpec(TraitedSpec):
    """OutputSpec for ApplyMotion."""

    out_file = File(desc="motion corrected file")


class ApplyMotion(SimpleInterface):
    """Apply Motion Parameters to data."""

    input_spec = ApplyMotionInputSpec
    output_spec = ApplyMotionOutputSpec

    def _run_interface(self, runtime):

        image_nii = nib.load(self.inputs.in_file)
        images = image_nii.get_fdata()
        motions = np.genfromtxt(self.inputs.motion_file, delimiter="  ")
        corrected_images = np.zeros_like(images)

        for i in range(images.shape[-1]):
            corrected_images[..., i] = apply_motion(images[..., i], motions[i])

        filename = os.path.abspath("r" + os.path.basename(self.inputs.in_file))
        corrected_nii = nib.Nifti1Image(corrected_images, affine=image_nii.affine)
        corrected_nii.to_filename(filename)
        self._results["out_file"] = filename
        return runtime


class MagPhase2RealImagInputSpec(BaseInterfaceInputSpec):
    """InputSpec for MagPhase2RealImag."""

    mag_file = File(exists=True)
    phase_file = File(exists=True)


class MagPhase2RealImagOutputSpec(TraitedSpec):
    """OutputSpect for MagPhase2RealImag."""

    real_file = File()
    imag_file = File()


class MagPhase2RealImag(SimpleInterface):
    """Get Real and imaginary part of data from magnitude and phase files."""

    input_spec = MagPhase2RealImagInputSpec
    output_spec = MagPhase2RealImagOutputSpec

    def _run_interface(self, runtime):

        mag_nii = nib.load(self.inputs.mag_file)
        pha_nii = nib.load(self.inputs.pha_file)

        mag_data = mag_nii.get_fdata(dtype=np.float32)
        pha_data = pha_nii.get_fdata(dtype=np.float32)

        pha_min = np.min(pha_data)
        pha_max = np.max(pha_data)
        # normalizing to -pi, pi
        pha_data = (2 * np.pi * (pha_data - pha_min) / (pha_max - pha_min)) + np.pi

        real_data = mag_data * np.cos(pha_data)
        imag_data = mag_data * np.sin(pha_data)

        basename = os.path.bsename(self.inputs.mag_file).split(".")[0]

        real_nii = nib.Nifti1Image(real_data, mag_nii.affine)
        imag_nii = nib.Nifti1Image(imag_data, mag_nii.affine)
        real_fname = basename + "_real.nii"
        imag_fname = basename + "_imag.nii"

        real_nii.to_filename(real_fname)
        imag_nii.to_filename(imag_fname)

        self._results["real_file"] = os.path.abspath(real_fname)
        self._results["imag_file"] = os.path.abspath(imag_fname)

        return runtime


class RealImag2MagPhaseInputSpec(BaseInterfaceInputSpec):
    """OutputSpect for MagPhase2RealImag."""

    real_file = File()
    imag_file = File()


class RealImag2MagPhaseOutputSpec(TraitedSpec):
    """InputSpec for RealImag2MagPhase."""

    mag_file = File()
    phase_file = File()


class RealImag2MagPhase(SimpleInterface):
    """Get Real and imaginary part of data from magnitude and phase files."""

    input_spec = RealImag2MagPhaseInputSpec
    output_spec = RealImag2MagPhaseOutputSpec

    def _run_interface(self, runtime):

        real_nii = nib.load(self.inputs.real_file)
        imag_nii = nib.load(self.inputs.imag_file)

        real_data = real_nii.get_fdata(dtype=np.float32)
        imag_data = imag_nii.get_fdata(dtype=np.float32)

        mag_data = np.sqrt(real_data**2 + imag_data**2)
        phase_data = np.atan2(imag_data, real_data)
        basename = os.path.bsename(self.inputs.real_file).split(".")[0]
        basename.replace("real", "")
        basename.replace("imag", "")

        mag_nii = nib.Nifti1Image(mag_data, real_nii.affine)
        phase_nii = nib.Nifti1Image(phase_data, real_nii.affine)
        mag_fname = basename + "_mag.nii"
        phase_fname = basename + "_phase.nii"

        mag_nii.to_filename(mag_fname)
        phase_nii.to_filename(phase_fname)

        self._results["mag_file"] = os.path.abspath(mag_fname)
        self._results["phase_file"] = os.path.abspath(phase_fname)

        return runtime
