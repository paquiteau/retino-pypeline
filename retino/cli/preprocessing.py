#!/usr/bin/env python3

"""Script to run preprocessing workflow."""

from retino.cli.base import base_parser, get_namespace, TASKS


def get_parser():
    """Get parser for preprocessing."""
    parser = base_parser(
        prog="benchmark script for patch denoising method."
        "this script execute a standard retinotopy pipeline with denoising for a set "
        "of subject, but with a single configuration of denoising"
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=TASKS,
        help="task to preprocess",
    )
    parser.add_argument(
        "--noise", action="store_true", help="Perform the preprocesssing of noise data."
    )
    return parser


def noise_preprocessing(ns):
    """Perform preprocessing of noise data."""
    from retino.workflows.preprocessing import NoisePreprocManager

    noise_prep_mgr = NoisePreprocManager(ns.dataset, ns.tmpdir)
    noise_prep_wf = noise_prep_mgr.get_workflow()
    noise_prep_mgr.run(
        noise_prep_wf,
        multi_proc=True,
        dry=ns.dry,
        sub_id=ns.sub,
        sequence=ns.sequence,
        task=ns.task,
    )


def preprocessing(ns):
    """Perform preprocessing of data."""
    from retino.workflows.preprocessing import RetinotopyPreprocessingManager

    prep_mgr = RetinotopyPreprocessingManager(ns.dataset, ns.tmpdir)
    for bc in ns.build_code:
        prep_mgr_wf = prep_mgr.get_workflow(build_code=bc)

        prep_mgr.run(
            prep_mgr_wf,
            multi_proc=True,
            dry=ns.dry,
            sub_id=ns.sub,
            task=ns.task,
            sequence=ns.sequence,
            denoise_str=ns.denoise_str,
        )


def main_cli():
    """Run cli."""
    parser = get_parser()
    ns = get_namespace(parser)

    if ns.noise:
        noise_preprocessing(ns)
    preprocessing(ns)


if __name__ == "__main__":
    main_cli()
