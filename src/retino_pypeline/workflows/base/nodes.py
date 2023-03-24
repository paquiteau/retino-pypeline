"""Base Nodes."""

from nipype import Node, IdentityInterface
from nipype.interfaces import io as nio


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
