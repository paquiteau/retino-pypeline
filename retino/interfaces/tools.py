"""Basic Processing for fMRI volumes."""
import os

import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
    isdefined,
)

from skimage.morphology import convex_hull_image
from nipy.labs.mask import compute_mask


class MaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="A fMRI input file.")


class MaskOutputSpec(TraitedSpec):
    mask = File(desc="the mask of a ROI")


class Mask(BaseInterface):
    """Compute a Brain Mask."""

    input_spec = MaskInputSpec
    output_spec = MaskOutputSpec

    def _run_interface(self, runtime):
        data = nib.load(self.inputs.in_file)

        avg = np.mean(data.get_fdata(), axis=-1)

        mask = np.uint8(compute_mask(avg))
        for i in range(mask.shape[-1]):
            mask[..., i] = convex_hull_image(mask[..., i])

        mask_nii = nib.Nifti1Image(mask, affine=data.affine)

        self._output_name = os.path.basename(self.inputs.in_file) + "_mask.nii"

        mask_nii.to_filename(self._output_name)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["mask"] = os.path.abspath(self._output_name)
        return outputs


class TSNRInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="A fMRI Input file.")
    mask_file = File(
        exists=True,
        mandatory=False,
        desc="A spatial mask on which the TSNR will be computed.",
    )


class TSNROutputSpec(TraitedSpec):
    tsnr_file = File(desc="The tSNR map")


class TSNR(BaseInterface):
    input_spec = TSNRInputSpec
    output_spec = TSNROutputSpec

    def _run_interface(self, runtime):

        nii = nib.load(self.inputs.in_file)
        data = nii.get_fdata()
        avg = np.mean(data, axis=-1)
        tsnr = np.empty_like(avg)

        if isdefined(self.inputs.mask_file) and self.inputs.mask_file:
            mask = nib.load(self.inputs.mask_file).get_fdata() > 0
            data = data * mask[..., None]
        else:
            mask = np.ones_like(avg, dtype=np.int8)

        std = np.std(data[mask], axis=-1)
        tsnr[mask] = avg[mask] / std

        tsnr_nii = nib.Nifti1Image(tsnr, affine=nii.affine)

        self._output_name = os.path.basename(self.inputs.in_file) + "_tsnr.nii"
        tsnr_nii.to_filename(self._output_name)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["tsnr_file"] = os.path.abspath(self._output_name)
        return outputs
