"""Motion Recalibration Interfaces."""
import os
from os import path

import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec
from nipype.utils.filemanip import split_filename

from retino_pypeline.tools.motion import apply_motion

############################
# Manual Motion Correction #
############################


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
        images = image_nii.get_fdata(dtype=np.float32)
        motions = np.genfromtxt(self.inputs.motion_file, delimiter="  ")
        corrected_images = np.zeros_like(images)

        for i in range(images.shape[-1]):
            corrected_images[..., i] = apply_motion(images[..., i], motions[i])

        filename = os.path.abspath("r" + os.path.basename(self.inputs.in_file))
        corrected_nii = nib.Nifti1Image(corrected_images, affine=image_nii.affine)
        corrected_nii.to_filename(filename)
        self._results["out_file"] = filename
        return runtime


###############################
# Motion Correction using FSL #
###############################

# https://neurostars.org/t/correct-way-to-combine-mcflirt-and-applyxfm4d-in-nipype/4324/2
#
class ApplyXfm4DInputSpec(FSLCommandInputSpec):
    in_file = File(
        exists=True,
        position=0,
        argstr="%s",
        mandatory=True,
        desc="timeseries to motion-correct",
    )
    ref_vol = File(
        exists=True,
        position=1,
        argstr="%s",
        mandatory=True,
        desc="volume with final FOV and resolution",
    )
    out_file = File(
        exists=True,
        position=2,
        argstr="%s",
        genfile=True,
        desc="file to write",
        hash_files=False,
    )
    trans_file = File(
        argstr="%s",
        position=3,
        desc="single tranformation matrix",
        xor=["trans_dir"],
        requires=["single_matrix"],
    )
    trans_dir = File(
        argstr="%s",
        position=3,
        desc="folder of transformation matricies",
        xor=["trans_file"],
    )
    single_matrix = traits.Bool(
        argstr="-singlematrix", desc="true if applying one volume to all timepoints"
    )
    four_digit = traits.Bool(
        argstr="-fourdigit", desc="true mat names have four digits not five"
    )
    user_prefix = traits.Str(
        argstr="-userprefix %s", desc="supplied prefix if mats don't start with 'MAT_'"
    )


class ApplyXfm4DOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="transform applied timeseries")


class ApplyXfm4D(FSLCommand):
    """
    Wraps the applyxfm4D command line tool for applying one 3D transform to every volume in a 4D image OR
    a directory of 3D tansforms to a 4D image of the same length.

    Examples
    ---------
    >>> import nipype.interfaces.fsl as fsl
    >>> from nipype.testing import example_data
    >>> applyxfm4d = fsl.ApplyXfm4D()
    >>> applyxfm4d.inputs.in_file = example_data('functional.nii')
    >>> applyxfm4d.inputs.in_matrix_file = example_data('functional_mcf.mat')
    >>> applyxfm4d.inputs.out_file = 'newfile.nii'
    >>> applyxfm4d.inputs.reference = example_data('functional_mcf.nii')
    >>> result = applyxfm.run() # doctest: +SKIP

    """

    _cmd = "applyxfm4D"
    input_spec = ApplyXfm4DInputSpec
    output_spec = ApplyXfm4DOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._gen_outfilename()
        return None

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        if isdefined(out_file):
            out_file = path.realpath(out_file)
        if not isdefined(out_file) and isdefined(self.inputs.in_file):
            out_file = self._gen_fname(self.inputs.in_file, suffix="_warp4D")
        return path.abspath(out_file)


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
        pha_nii = nib.load(self.inputs.phase_file)

        mag_data = mag_nii.get_fdata(dtype=np.float32)
        pha_data = pha_nii.get_fdata(dtype=np.float32)

        pha_min = np.min(pha_data)
        pha_max = np.max(pha_data)
        # normalizing to [0, 2*pi]
        pha_data = 2 * np.pi * (pha_data - pha_min) / (pha_max - pha_min)

        real_data = mag_data * np.cos(pha_data)
        imag_data = mag_data * np.sin(pha_data)

        _, basename, ext = split_filename(self.inputs.mag_file)

        real_nii = nib.Nifti1Image(real_data, mag_nii.affine)
        imag_nii = nib.Nifti1Image(imag_data, mag_nii.affine)
        real_fname = f"{basename}_real{ext}"
        imag_fname = f"{basename}_imag{ext}"

        real_nii.to_filename(real_fname)
        imag_nii.to_filename(imag_fname)

        self._results["real_file"] = os.path.abspath(real_fname)
        self._results["imag_file"] = os.path.abspath(imag_fname)

        return runtime


class RealImag2MagPhaseInputSpec(BaseInterfaceInputSpec):
    """OutputSpect for MagPhase2RealImag."""

    real_file = File(exists=True)
    imag_file = File(exists=True)


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
        phase_data = np.arctan2(imag_data, real_data)
        basename = os.path.basename(self.inputs.real_file).split(".")[0]
        basename.replace("real", "")
        basename.replace("imag", "")

        _, basename, ext = split_filename(self.inputs.real_file)

        mag_nii = nib.Nifti1Image(mag_data, real_nii.affine)
        phase_nii = nib.Nifti1Image(phase_data, real_nii.affine)
        mag_fname = f"{basename}_mag{ext}"
        phase_fname = f"{basename}_phase{ext}"

        mag_nii.to_filename(mag_fname)
        phase_nii.to_filename(phase_fname)

        self._results["mag_file"] = os.path.abspath(mag_fname)
        self._results["phase_file"] = os.path.abspath(phase_fname)

        return runtime
