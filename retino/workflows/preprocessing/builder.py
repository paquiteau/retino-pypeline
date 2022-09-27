"""Builder function, they extend a workflow to add nodes."""
from warnings import warn

from retino.workflows.preprocessing.nodes import (
    realign_node,
    coregistration_node,
    topup_node,
    conditional_topup,
    noise_node,
    selectfile_node,
    preproc_noise_node,
)

from nipype import IdentityInterface, Node, Function



def add_base(workflow, base_data_dir, cached_realignment):
    """add the minimum base node for the workflow
    to select files and output to sink files in."""

    in_fields = ["sub_id", "sequence", "denoise_config", "task"]
    templates_args = ["sub_id", "sequence", "task"]

    input_node = Node(IdentityInterface(fields=in_fields), "input")

    def template_node(sequence, cached_realignment):
        """test"""
        template = {
            "anat": "sub_%02i/anat/*_T1.nii",
            "data": f"sub_%02i/func/*%s_%sTask.nii",
        }

        if cached_realignment == "cached":
            template["data"] = f"sub_%02i/realign/*%s_%sTask.nii"
            template["motion"] = f"sub_%02i/realign/*%s_%sTask.txt"

        if "EPI" in sequence:
            template["data_pa"] = f"sub_%02i/func/*%s_Clockwise_1rep_PA.nii"

        return template

    tplt_node = Node(
        Function(
            function=template_node,
            input_name=["sequence", "realignment"],
            output_name=["template"],
        ),
        name="template_node",
    )
    tplt_node.inputs.cached_realignment = cached_realignment
    files = Node(
        nio.DataGrabber(
            in_fields=templates_args,
            base_directory=base_data_dir,
            template="*",
            sort_filelist=True,
        ),
        name="selectfiles",
    )

    sinker = Node(nio.DataSink(), name="sinker")
    sinker.inputs.base_directory = base_data_dir
    sinker.parameterization = False
    sinker.inputs.substitutions = [
        ("rp_sub", "sub"),
        ("rrsub", "sub"),
        ("rsub", "sub"),
        ("_denoise_method_", "denoise_method-"),
    ]

    sinker.inputs.regexp_substitutions = [(r"denoise_method-(.*?)/(.*)", r"\2")]

    wf.connect(
        [
            (
                input_node,
                tplt_node,
                [("sequence", "sequence"), ("realignment", "realignment")],
            ),
            (
                tplt_node,
                files,
                [
                    ("template", "field_template"),
                    ("templates_args", "templates_args"),
                ],
            ),
            (
                input_node,
                files,
                [("sub_id", "sub_id"), ("sequence", "sequence"), ("task", "task")],
            ),
            (input_node, sinker, [(("sub_id", getsubid), "container")]),
        ]
    )

    return wf


def add_realign(wf, name="realign", after_node, edge):
    """Add a Realignment node after_node,  withedge """
    if isinstance(after_node, str):
        after_node = wf.get_node(after_node)

    realign = realign_node(name=name)
    wf.connect(after_node, edge, realign, "in_files")
    return wf



def add_denoise(wf, after_node, edge):




def add_topup(wf, after_node, edge):
    """Add conditional topup correction """
    input_node = wf.get_node("input")
    condtopup = conditional_topup("conditional_topup")


    if isinstance(after_node, str):
        after_node = wf.get_node(after_node)

    wf.connect(after_node, edge, condtopup, "data")
    # also adds mandatory connections
    wf.connect(
        [
            (
                input_node
                condtopup,
                [
                    ("sequence", "sequence"),
                    ("data_opposite", "input.data_opposite"),
                ],
            ),
        ])
    return wf



def add_coreg(wf, after_node, edge):

    if isinstance(after_node, str):
        after_node = wf.get_node(after_node)

    coreg = coregistration_node("coregistration")

    wf.connect(after_node, edge, coreg, "in.func")
    #also add mandatory connections:
    wf.connect(wf.get_node("selectfiles"), "anat", coreg, "in.anat")
    return wf


def add_sinker(wf, connections):
    """connections for sinker to add to workflow.

    connections hould be a list of (node_name, edge, output_name)

    """
