import os

import nipype.interfaces.fsl as fsl  # fsl
import numpy as np
from nipype.interfaces.base import File, InputMultiPath, TraitedSpec, isdefined, traits
from nipype.utils.filemanip import split_filename


class TOPUPInputSpec(fsl.base.FSLCommandInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="--imain=%s",
        desc="name of the file containing two different blips oriented images",
    )

    encoding_direction = traits.List(
        traits.Enum("y", "x", "z", "x-", "y-", "z-"),
        mandatory=True,
        argstr="--datain=%s",
        desc=("encoding direction for automatic " "generation of encoding_file"),
    )
    readout_times = InputMultiPath(
        traits.Float,
        mandatory=True,
        desc=("readout times (dwell times by # " "phase-encode steps minus 1)"),
    )
    out_base = File(
        desc=(
            "base-name of output files (spline "
            "coefficients (Hz) and movement parameters)"
        ),
        name_source=["in_file"],
        name_template="%s_topup_out",
        argstr="--out=%s",
        hash_files=False,
    )
    subsamp = traits.Int(argstr="--subsamp=%d", desc="sub-sampling scheme")
    fwhm = traits.Float(
        argstr="--fwhm=%f", desc="FWHM (in mm) of gaussian smoothing kernel"
    )
    output_type = traits.Enum(
        "NIFTI", list(fsl.Info.ftypes.keys()), desc="FSL output type"
    )


class TOPUPOutputSpec(TraitedSpec):
    out_fieldcoef = File(exists=True, desc="file containing the field coefficients")
    out_movpar = File(exists=True, desc="movpar.txt output file")
    out_enc_file = File(exists=True, desc="file containing the encoding parameters.")
    # out_field = File(desc="name of image file with field (Hz)")
    # out_warps = traits.List(File(exists=True), desc="warpfield images")
    # out_jacs = traits.List(File(exists=True), desc="Jacobian images")
    # out_mats = traits.List(File(exists=True), desc="realignment matrices")
    # out_corrected = File(desc="name of 4D image file with unwarped images")
    # out_logfile = File(desc="name of log-file")


class myTOPUP(fsl.base.FSLCommand):
    _cmd = "topup"
    input_spec = TOPUPInputSpec
    output_spec = TOPUPOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == "encoding_direction":
            return trait_spec.argstr % self._generate_encfile()
        if name == "out_base":
            path, name, ext = split_filename(value)
            if path != "":
                if not os.path.exists(path):
                    raise ValueError("out_base path must exist if provided")
        return super(myTOPUP, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = super(myTOPUP, self)._list_outputs()
        del outputs["out_base"]
        base_path = None
        if isdefined(self.inputs.out_base):
            base_path, base, _ = split_filename(self.inputs.out_base)
            if base_path == "":
                base_path = None
        else:
            base = split_filename(self.inputs.in_file)[1] + "_base"
        outputs["out_fieldcoef"] = self._gen_fname(
            base, suffix="_fieldcoef", cwd=base_path
        )
        outputs["out_movpar"] = self._gen_fname(
            base, suffix="_movpar", ext=".txt", cwd=base_path
        )

        if isdefined(self.inputs.encoding_direction):
            outputs["out_enc_file"] = self._get_encfilename()

        return outputs

    def _overload_extension(self, value, name=None):
        if name == "out_base":
            return value
        return super(myTOPUP, self)._overload_extension(value, name)

    def _get_encfilename(self):
        out_file = os.path.join(os.getcwd(), "acquisition_encfile.txt")
        return out_file

    def _generate_encfile(self):
        """Generate a topup compatible encoding file based on given directions"""
        out_file = self._get_encfilename()
        durations = self.inputs.readout_times
        if len(self.inputs.encoding_direction) != len(durations):
            if len(self.inputs.readout_times) != 1:
                raise ValueError(
                    (
                        "Readout time must be a float or match the"
                        "length of encoding directions"
                    )
                )
            durations = durations * len(self.inputs.encoding_direction)

        lines = []
        for idx, encdir in enumerate(self.inputs.encoding_direction):
            direction = 1.0
            if encdir.endswith("-"):
                direction = -1.0
            line = [
                float(val[0] == encdir[0]) * direction for val in ["x", "y", "z"]
            ] + [durations[idx]]
            lines.append(line)
        np.savetxt(out_file, np.array(lines), fmt=b"%d %d %d %.8f")
        return out_file
