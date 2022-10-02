"""Builder function, they extend a workflow to add nodes."""
from retino.workflows.preprocessing.nodes import (
    conditional_topup_task,
    coregistration_task,
    denoise_node,
    realign_task,
)


def _getsubid(i):
    return f"sub_{i:02d}"


def _subid_varname(i):
    return f"_sub_id_{i}"


def _get_key(d, k):
    return d[k]


def add_base(wf, base_data_dir, cached_realignment):
    """Add the  basic nodes for the workflow.

    "input" node with [sub_id, sequence, denoise_config, task]
    "selectfiles" node with templates for magnitude images
    "sinker" node for the output. The sub_id is use a parent folder (container).
    to select files and output to sink files in.
    """
    in_fields = ["sub_id", "sequence", "denoise_config", "task"]
    templates_args = ["sub_id", "sequence", "task"]

    input_node = Node(IdentityInterface(fields=in_fields), "input")

    def template_node(sequence, cached_realignment):
        """Template node as a Function to handle cached realignment.

        TODO add support for complex according to the denoising config string.
        """
        template = {
            "anat": "sub_%02i/anat/*_T1.nii",
            "data": "sub_%02i/func/*%s_%sTask.nii",
        }
        file_template_args = {
            "anat": [["sub_id"]],
            "data": [["sub_id", "sequence", "task"]],
        }

        if cached_realignment == "cached":
            template["data"] = "sub_%02i/realign/*%s_%sTask.nii"
            template["motion"] = "sub_%02i/realign/*%s_%sTask.txt"
            file_template_args["motion"] = [["sub_id", "sequence", "task"]]
        if "EPI" in sequence:
            template["data_pa"] = "sub_%02i/func/*%s_Clockwise_1rep_PA.nii"
            file_template_args["data_pa"] = [["sub_id", "sequence"]]
        return template, file_template_args

    tplt_node = Node(
        Function(
            function=template_node,
            input_names=["sequence", "cached_realignment"],
            output_names=["template", "template_args"],
        ),
        name="template_node",
    )
    tplt_node.inputs.cached_realignment = cached_realignment
    files = Node(
        nio.DataGrabber(
            infields=templates_args,
            outfields=["data", "anat", "motion", "data_pa"],
            base_directory=base_data_dir,
            template="*",
            sort_filelist=True,
        ),
        name="selectfiles",
    )
    sinker = Node(nio.DataSink(), name="sinker")
    sinker.inputs.base_directory = base_data_dir
    sinker.parameterization = False

    wf.connect(
        [
            (input_node, tplt_node, [("sequence", "sequence")]),
            (
                tplt_node,
                files,
                [
                    ("template", "field_template"),
                    ("template_args", "template_args"),
                ],
            ),
            (
                input_node,
                files,
                [("sub_id", "sub_id"), ("sequence", "sequence"), ("task", "task")],
            ),
            (input_node, sinker, [(("sub_id", _getsubid), "container")]),
        ]
    )

    return wf


def _add_to_wf(wf, after_node, edge_out, node, edge_in):
    if isinstance(after_node, str):
        after_node = wf.get_node(after_node)
    wf.connect(after_node, edge_out, node, edge_in)
    return wf


def add_realign(wf, name, after_node, edge):
    """Add a Realignment node."""
    realign = realign_task(name=name)
    return _add_to_wf(wf, after_node, edge, realign, "in_files")


def add_denoise_mag(wf, name, after_node, edge):
    """Add denoising step for magnitude input."""
    denoise = denoise_node(name)
    input_node = wf.get_node("input")
    wf.connect(input_node, "denoise_config", denoise, "denoise_str")
    return _add_to_wf(wf, after_node, edge, denoise, "in_file_mag")


def add_topup(wf, name, after_node, edge):
    """Add conditional topup correction."""
    input_node = wf.get_node("input")
    selectfiles = wf.get_node("selectfiles")
    condtopup = conditional_topup_task(name)
    # also adds mandatory connections
    wf.connect(input_node, "sequence", condtopup, "sequence")
    wf.connect(selectfiles, "data_opposite", condtopup, "data_opposite")
    return _add_to_wf(wf, after_node, edge, condtopup, "data")


def add_coreg(wf, name, after_node, edge):
    """Add coregistration step."""
    coreg = coregistration_task(name)
    # also add mandatory connections:
    wf.connect(wf.get_node("selectfiles"), "anat", coreg, "in.anat")
    return _add_to_wf(wf, after_node, edge, coreg, "in.func")


def add_sinker(wf, connections, folder=None):
    """Add connections to sinker.

    connections should be a list of (node_name, edge, output_name)

    """
    if folder is None:
        folder = ""
    else:
        folder += ".@"
    sinker = wf.get_node("sinker")

    for con in connections:
        wf.connect(wf.get_node(con[0]), con[1], sinker, f"{folder}{con[2]}")

    return wf
