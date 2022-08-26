"""Interface for patch denoising methods."""

import os
import numpy as np

import nibabel as nib
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
    isdefined,
)
from nipype.utils.filemanip import split_filename

from denoiser.denoise import hybrid_pca, mp_pca, nordic, optimal_thresholding, raw_svt
from denoiser.space_time.utils import estimate_noise

DENOISER_MAP = {
    "mp-pca": mp_pca,
    "hybrid-pca": hybrid_pca,
    "raw": raw_svt,
    "optimal-fro": lambda *args, **kwargs: optimal_thresholding(
        *args, loss="fro", **kwargs
    ),
    "optimal-nuc": lambda *args, **kwargs: optimal_thresholding(
        *args, loss="nuc", **kwargs
    ),
    "optimal-ope": lambda *args, **kwargs: optimal_thresholding(
        *args, loss="ope", **kwargs
    ),
    "nordic": nordic,
}


class PatchDenoiseInputSpec(BaseInterfaceInputSpec):

    in_file_mag = File(
        exists=True, mandatory=True, desc="magnitude input file to denoise."
    )
    in_file_phase = File(
        exists=True, mandatory=False, desc="phase input file to denoise."
    )
    noise_std = File(desc="noise_std_map")
    method = traits.Enum(*DENOISER_MAP.keys())
    patch_shape = traits.Union(
        traits.Int(), traits.List(traits.Int(), minlen=3, maxlen=3)
    )
    patch_overlap = traits.Union(
        traits.Int(), traits.List(traits.Int(), minlen=3, maxlen=3)
    )
    mask_file = File(exists=True)
    recombination = traits.Enum("weighted", "mean")
    extra_kwargs = traits.Dict()


class PatchDenoiseOutputSpec(TraitedSpec):
    denoised_file = File(desc="denoised file")
    noise_std_map = File(desc="a map of the noise variance.")
    pass


class PatchDenoise(BaseInterface):
    input_spec = PatchDenoiseInputSpec
    output_spec = PatchDenoiseOutputSpec

    def _run_interface(self, runtime):

        data_mag = nib.load(self.inputs.in_file_mag)

        data = data_mag.get_fdata()
        if isdefined(self.inputs.in_file_phase) and self.inputs.in_file_phase:

            phase = nib.load(self.inputs.in_file_phase).get_fdata()
            # put in [0, 2*pi], some stretching may happen...
            phase = (
                2 * np.pi * (phase - np.min(phase)) / (np.max(phase) - np.min(phase))
            )
            # combine to get complex data
            data = np.complex64(data) * np.exp(1j * phase)

        if isdefined(self.inputs.mask_file) and self.inputs.mask_file:
            mask = np.abs(nib.load(self.inputs.mask_file).get_fdata()) > 0
        else:
            mask = None
        try:
            denoise_func = DENOISER_MAP[self.inputs.method]
        except KeyError:
            raise ValueError(
                f"unknown denoising method '{self.inputs.method}', available are {list(DENOISER_MAP.keys())}"
            )
        extra_kwargs = self.inputs.extra_kwargs
        if self.inputs.method in ["nordic"]:
            extra_kwargs["noise_std"] = nib.load(self.inputs.noise_std).get_fdata()

        denoised_data, _, noise_std_map = denoise_func(
            data,
            patch_shape=self.inputs.patch_shape,
            patch_overlap=self.inputs.patch_overlap,
            mask=mask,
            recombination=self.inputs.recombination,
            **extra_kwargs,
        )
        denoise_filename, noise_map_filename = self._get_filenames()

        denoised_data_img = nib.Nifti1Image(abs(denoised_data), affine=data_mag.affine)
        denoised_data_img.to_filename(denoise_filename)

        noise_map_img = nib.Nifti1Image(noise_std_map, affine=data_mag.affine)
        noise_map_img.to_filename(noise_map_filename)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        denoise_filename, noise_map_filename = self._get_filenames()

        outputs["denoised_file"] = os.path.abspath(denoise_filename)
        outputs["noise_std_map"] = os.path.abspath(noise_map_filename)
        return outputs

    def _get_filenames(self):
        _, base, _ = split_filename(self.inputs.in_file_mag)
        base = base.replace("_mag", "")
        return f"{base}_d_{self.inputs.method}.nii", f"{base}_noise_map.nii"


class NoiseStdMapInputSpec(BaseInterfaceInputSpec):
    noise_map_file = File(exists=True, desc="A 0-Volt volume acquisition")
    fft_scale = traits.Int(100)
    block_size = traits.Int(3)


class NoiseStdMapOutputSpec(TraitedSpec):
    noise_std_map = File(desc="Spatial variation of noise variance")


class NoiseStdMap(BaseInterface):
    input_spec = NoiseStdMapInputSpec
    output_spec = NoiseStdMapOutputSpec

    def _run_interface(self, runtime):

        noise_map = nib.load(self.inputs.noise_map_file)
        noise_std_map = estimate_noise(
            noise_map.get_fdata() / self.inputs.fft_scale, self.inputs.block_size
        )
        noise_std_map_img = nib.Nifti1Image(noise_std_map, affine=noise_map.affine)
        noise_std_map_img.to_filename(
            os.path.basename(self.inputs.noise_map_file) + "_std.nii"
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["noise_std_map"] = os.path.abspath(
            os.path.basename(self.inputs.noise_map_file) + "_std.nii"
        )
        return outputs
