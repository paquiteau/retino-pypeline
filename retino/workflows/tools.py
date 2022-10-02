"""Utility functions for all workflows."""
import os
import inspect

from nipype import Node, Function


def _get_matlab_cmd(matlab_cmd):
    if matlab_cmd:
        return matlab_cmd
    else:
        return "matlab -nodesktop -nosplash"


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