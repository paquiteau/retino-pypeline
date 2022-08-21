from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, BaseInterface, traits


class DesignMatrixRetinoInputSpec(BaseInterfaceInputSpec):
    data_file = File(exists=True, mandatory=True, desc="the input data")
    motion_file = File(exists=True, mandatory=True, desc="motion regressor from spm")


class DesignMatrixRetinoOuputSpec(TraitedSpec):

class DesignMatrixRetino(BaseInterface):
