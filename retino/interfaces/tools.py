"""Basic Processing for fMRI volumes."""
import os

import nibabel as nib
import numpy as np
import scipy as sp
from nipype.interfaces.base import (
    SimpleInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
    isdefined,
)

from skimage.morphology import convex_hull_image
from skimage.filters import threshold_otsu
from nipy.labs.mask import compute_mask


class MaskInputSpec(BaseInterfaceInputSpec):
    """InputSpec for Mask Interface."""

    in_file = File(exists=True, mandatory=True, desc="A fMRI input file.")
    use_mean = traits.Bool(
        True,
        mandatory=False,
        desc="Compute the brain mask using the mean image (over the last dimension)",
    )
    method = traits.Enum(
        "otsu",
        "nichols",
        desc="thresholding method for the brain segmentation. otsu recommended",
    )
    convex_mask = traits.Bool(False, desc="Should the mask be convex, default False")


class MaskOutputSpec(TraitedSpec):
    """OutputSpec for Mask Interface."""

    mask = File(desc="the mask of a ROI")


class Mask(SimpleInterface):
    """Compute a Brain Mask."""

    input_spec = MaskInputSpec
    output_spec = MaskOutputSpec

    def _run_interface(self, runtime):
        data = nib.load(self.inputs.in_file)

        if self.inputs.use_mean:
            avg = np.mean(data.get_fdata(), axis=-1)
        else:
            avg = data.get_fdata()

        if self.inputs.method == "otsu":
            mask = np.zeros(avg.shape, dtype=bool)
            for i in range(avg.shape[-1]):
                mask[..., i] = avg[..., i] > threshold_otsu(avg[..., i])
        elif self.inputs.method == "nipy":
            mask = np.uint8(compute_mask(avg))

        if self.inputs.convex_mask:
            for i in range(mask.shape[-1]):
                mask[..., i] = convex_hull_image(mask[..., i])

        mask_nii = nib.Nifti1Image(np.uint8(mask), affine=data.affine)

        filename = os.path.basename(self.inputs.in_file).split(".")[0] + "_mask.nii"
        mask_nii.to_filename(filename)
        self._results["mask"] = os.path.abspath(filename)
        return runtime


class TSNRInputSpec(BaseInterfaceInputSpec):
    """InputSpec for tsnr map estimation."""

    in_file = File(exists=True, mandatory=True, desc="A fMRI Input file.")
    mask_file = File(
        exists=True,
        mandatory=False,
        desc="A spatial mask on which the TSNR will be computed.",
    )


class TSNROutputSpec(TraitedSpec):
    """OutputSpec for tsnr map estimation."""

    tsnr_file = File(desc="The tSNR map")


class TSNR(SimpleInterface):
    """tSNR estimation."""

    input_spec = TSNRInputSpec
    output_spec = TSNROutputSpec

    def _run_interface(self, runtime):

        nii = nib.load(self.inputs.in_file)
        data = nii.get_fdata()
        tsnr = np.zeros(data.shape[:-1], dtype=np.float32)

        if isdefined(self.inputs.mask_file) and self.inputs.mask_file:
            mask = nib.load(self.inputs.mask_file).get_fdata() > 0
        else:
            mask = np.ones(data.shape[:-1], dtype=np.int16)
        roi_data = data[mask]
        roi_data[np.isnan(roi_data)] = 0
        detrended = sp.signal.detrend(roi_data)
        # detrended = roi_data
        std = np.std(detrended, axis=-1)
        tsnr[mask] = np.mean(detrended, axis=-1) / std
        tsnr_nii = nib.Nifti1Image(tsnr, affine=nii.affine)

        filename = os.path.abspath(os.path.basename(self.inputs.in_file) + "_tsnr.nii")
        tsnr_nii.to_filename(filename)
        self._results["tsnr_file"] = filename
        return runtime
