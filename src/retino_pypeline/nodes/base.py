"""Base Nodes."""
import inspect
import glob
import distutils.spawn

from nipype import Node, IdentityInterface, Function
from nipype.interfaces import io as nio
from nipype.interfaces import matlab as mlab
from nipype.interfaces import spm


def _setup_matlab(node):

    if distutils.spawn.find_executable("matlab"):
        matlab_cmd = "matlab -nodesktop -nosplash"
    else:
        print("no matlab command found, use MCR.")
        mcr_path = glob.glob("/opt/mcr*/v*")[0]
        spm_path = glob.glob("/opt/spm12*/run_spm12.sh")[0]
        matlab_cmd = f"{spm_path} {mcr_path} script"

    if "mcr" in matlab_cmd:
        spm.SPMCommand.set_mlab_paths(matlab_cmd, use_mcr=True)
        # node.inputs.matlab_cmd = matlab_cmd
        # node.inputs.use_mcr = True
    elif node is not None:
        node.interface.mlab = mlab.MatlabCommand(
            matlab_cmd=matlab_cmd,
            resource_monitor=False,
            single_comp_thread=False,
        )


def func2node(func, output_names, name=None, input_names=None):
    """Return a Node created encapsulating a function.

    If not provided, input_names and name are determined using inspect module.
    """
    if name is None:
        name = func.__name__
    if input_names is None:
        input_names = inspect.getfullargspec(func).args
    return Node(
        Function(function=func, input_names=input_names, output_names=output_names),
        name=name,
    )


def input_task(in_fields):
    """Return input node."""
    return Node(IdentityInterface(fields=in_fields), "input")


def sinker_task(base_data_dir):
    """Return Sinker node."""
    sinker = Node(nio.DataSink(), name="sinker")
    sinker.inputs.base_directory = base_data_dir
    sinker.parameterization = False
    return sinker


def file_task(infields, outfields, base_data_dir):
    """Return a file selector node."""
    return Node(
        nio.DataGrabber(
            infields=infields,
            outfields=outfields,
            base_directory=base_data_dir,
            template="*",
            sort_filelist=True,
        ),
        name="selectfiles",
    )


def selectfile_task(template, base_data_dir, template_args=None, infields=None):
    """Create basic Select file node, requires template."""
    files = Node(
        nio.DataGrabber(
            infields=infields,
            outfields=list(template.keys()),
            base_directory=base_data_dir,
            template="*",
            sort_filelist=True,
        ),
        name="selectfiles",
    )
    files.inputs.field_template = template
    files.inputs.template_args = template_args
    return files
