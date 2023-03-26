"""Interface for patch denoising methods."""

from pathlib import Path
from nipype.interfaces.base import (
    File,
    TraitedSpec,
    traits,
    isdefined,
)
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec


class NORDICInputSpec(MatlabInputSpec):
    """InputSpec for NORDIC, see NIFTI_NORDIC docstring."""

    file_mag = File(mandatory=True, exists=True, desc="the magnitude nifti file.")
    file_phase = File(exists=True, desc="the magnitude nifti file.")
    file_out_mag = File(desc="output_name for the magnitude file")
    nordic_path = File(desc="location of NIFTI_NORDIC.m file.")
    arg_mp = traits.Enum(
        0,
        1,
        2,
        desc=(
            "0: default,"
            " 1: NORDIC gfactor with MP estimation"
            " 2: MP without gfactor correction",
        ),
    )
    arg_nordic = traits.Enum(1, 0, desc="1 Default")
    arg_kernel_size_gfactor = traits.Tuple(
        (traits.Int, traits.Int, traits.Int), default=(14, 14, 1)
    )
    arg_kernel_size_PCA = traits.Int(
        desc="default: ratio 11:1 between spatial and temporal voxels",
    )

    arg_NORDIC_patch_overlap = traits.Union(
        traits.Int(),
        traits.Float(),
        desc="overlap is used to determined step_size = floor(patch_size/overlap)",
    )
    arg_save_add_info = traits.Bool(
        desc=" If 1, then an extra file is being saved with degress removed etc."
    )
    arg_make_complex_nii = traits.Bool(
        "If defined, the phase is being saved in a similar format as the input phase"
    )
    arg_save_gfactor_map = traits.Enum(
        0,
        1,
        2,
        desc=(
            "1 save the relative gfactor,"
            " 2 saves the gfactor and does not compute the nordic processing",
        ),
    )


class NORDICOutputSpec(TraitedSpec):
    """Output of NORDIC Denoiser.

    TODO add files for g-map, noise etc
    """

    file_out_mag = File()
    file_out_phase = File()
    matlab_output = traits.Str()
    matlab_output_err = traits.Str()


class NORDICDenoiser(MatlabCommand):
    """NORDIC Denoiser.

    Inputs
    ------

    file_mag
    file_phase
    arg_{key}
    """

    input_spec = NORDICInputSpec
    output_spec = NORDICOutputSpec

    def _nordic_script(self):

        arg_name = [name[4:] for name in self.inputs.__dict__ if "arg_" in name]
        # fill the argument MATLAB Struct.
        mstruct = f"ARG.DIROUT = '{Path.cwd()}/';\n ARG.temporal_phase = 0;\n"
        for name in arg_name:
            val = getattr(self.inputs, "arg_" + name)
            if not isdefined(val):
                continue
            if name == "kernel_size_PCA":
                val = [val, val, val]
            mstruct += f"ARG.{name} = {val};\n"

        file_mag = Path(self.inputs.file_mag)

        file_phase = []
        if isdefined(self.inputs.file_phase):
            file_phase = f"'{Path(self.inputs.file_phase).resolve()}'"
        else:
            mstruct += "ARG.magnitude_only = 1;\n"

        file_out_base = file_mag.stem + "_NORDIC"
        self._file_out = Path(file_out_base + ".nii").resolve()

        if isdefined(self.inputs.nordic_path):
            nordic_path = self.inputs.nordic_path
        else:
            nordic_path = Path(__file__).parents[1] / "libs/NORDIC_Raw"
        script = (
            "\n{mstruct}\n"
            "addpath('{nordic_path}');\n"
            "disp(ARG);\n"
            "NIFTI_NORDIC('{file_mag}', {file_phase}, '{file_out}', ARG);"
        ).format(
            mstruct=mstruct,
            file_mag=self.inputs.file_mag,
            file_phase=file_phase,
            file_out=file_out_base,
            nordic_path=nordic_path,
        )
        print(script)
        return script

    def run(self, **inputs):
        self.inputs.single_comp_thread = False
        self.inputs.script = self._nordic_script()
        results = super(MatlabCommand, self).run(**inputs)
        stdout = results.runtime.stdout
        stderr = results.runtime.stderr
        results.outputs.matlab_output = stdout
        results.outputs.matlab_output_err = stderr

        return results

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["file_out_mag"] = Path(self._file_out).resolve()
        return outputs
