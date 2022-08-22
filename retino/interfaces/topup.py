from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
    InputMultiPath,
    isdefined,
)
import nipype.interfaces.fsl as fsl  # fsl
from nipype.utils.filemanip import split_filename, fname_presuffix


class TransformInfoInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="--imain=%s",
        desc="name of the file containing two different blips oriented images",
    )
    encoding_file = File(
        exists=True,
        mandatory=True,
        desc="name of text file with PE directions/times",
        argstr="--datain=%s",
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
    # out_field = File(desc="name of image file with field (Hz)")
    # out_warps = traits.List(File(exists=True), desc="warpfield images")
    # out_jacs = traits.List(File(exists=True), desc="Jacobian images")
    # out_mats = traits.List(File(exists=True), desc="realignment matrices")
    # out_corrected = File(desc="name of 4D image file with unwarped images")
    # out_logfile = File(desc="name of log-file")


class myTOPUP(fsl.base.FSLCommand):
    _cmd = "topup"
    input_spec = TransformInfoInputSpec
    output_spec = TOPUPOutputSpec

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
        return outputs

    def _overload_extension(self, value, name=None):
        if name == "out_base":
            return value
        return super(myTOPUP, self)._overload_extension(value, name)
