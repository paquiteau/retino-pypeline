#!/usr/bin/env python3
"""Script to perform the analysis of data."""

from .base import base_parser


def get_parser():
    """Get parser."""
    parser = base_parser("Compute the analysis of data")
    parser.add_argument(
        "--tsnr",
        action="store_true",
        help="compute only the tsnr",
    )
    return parser


def analyse(ns):
    """Perform analysis."""
    from retino.workflows.analysis import RetinoAnalysisWorkflowManager

    mgr = RetinoAnalysisWorkflowManager(ns.dataset, ns.tmpdir)
    print(ns)
    wf = mgr.get_workflow(n_cycles=9, threshold=0.001)
    wf.inputs.input.volumetric_tr = 2.4
    mgr.run(
        wf,
        multi_proc=True,
        sub_id=ns.sub,
        sequence=ns.sequence,
        preproc_code=ns.build_code,
        denoise_str=ns.denoise_str,
    )


def tsnr(ns):
    """Compute tSNR map."""
    from retino.workflows.analysis import FirstLevelStats

    mgr = FirstLevelStats(ns.dataset, ns.tmpdir)
    print(ns)
    wf = mgr.get_workflow()
    mgr.run(
        wf,
        multi_proc=True,
        sub_id=ns.sub,
        sequence=ns.sequence,
        preproc_code=ns.build_code,
        denoise_str=ns.denoise_str,
    )


def main_cli():
    """Run cli."""
    parser = get_parser()

    ns = parser.parse_args()
    if ns.tsnr:
        tsnr(ns)
    else:
        analyse(ns)


if __name__ == "__main__":
    main_cli()
