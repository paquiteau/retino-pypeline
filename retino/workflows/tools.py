"""Utility functions for all workflows."""
import os
import inspect
import distutils.spawn
import glob

from nipype import Node, Function
import nipype.interfaces.matlab as mlab


def _setup_matlab(node, matlab_cmd):

    if not matlab_cmd:
        if distutils.spawn.find_executable("matlab"):
            matlab_cmd = "matlab -nodesktop -nosplash"
        else:
            print("no matlab command found, use MCR.")
            mcr_path = glob.glob("/opt/matlabmcr-*/v*")[0]
            matlab_cmd = f"/opt/spm12/run_spm12.sh {mcr_path} script"

    if "matlabmcr" in matlab_cmd:
        node.inputs.matlab_cmd = matlab_cmd
        node.inputs.use_mcr = True
    else:
        node.interface.mlab = mlab.MatlabCommand(
            matlab_cmd=matlab_cmd,
            resource_monitor=False,
            single_comp_thread=False,
        )


def _get_num_thread(n=None):
    if n:
        return n
    else:
        return len(os.sched_getaffinity(0))


def _getsubid(i):
    return f"sub_{i:02d}"


def _subid_varname(i):
    return f"_sub_id_{i}"


def _get_key(d, k):
    return d[k]


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
