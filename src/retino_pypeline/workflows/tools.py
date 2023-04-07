"""Utility functions for all workflows."""
import os


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


def dict2cli(d):
    """Convert a dictionary to a command line string."""
    if isinstance(d, str):
        return d
    elif isinstance(d, dict):
        return " ".join([f"--{k}={v}" for k, v in d.items()])
    else:
        raise ValueError(f"Cannot convert {d} to command line arguments.")


def show_graph(wf, graph2use="colored"):
    """Check the workflow. Also draws a representation."""
    # TODO ascii plot: https://github.com/ggerganov/dot-to-ascii

    fname = wf.write_graph(dotfilename=f"graph_{graph2use}.dot", graph2use=graph2use)
    return fname


def show_graph_nb(wf, graph2use="colored", detailed=False):
    """Show the workflow in a Ipython Notebook."""
    from IPython.display import Image

    if detailed:
        return Image(
            show_graph(wf, graph2use=graph2use).split(".")[0] + "_detailed.png"
        )
    return Image(show_graph(wf))
