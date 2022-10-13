#!/usr/bin/env python3

"""Script to run preprocessing workflow."""

import argparse


DATA_DIR = "/neurospin/optimed/pierre-antoine/dataset/data"

TMP_DIR = "/neurospin/optimed/pierre-antoine/dataset/tmp"
TASKS = ["Clockwise", "AntiClockwise"]
SEQUENCES = ["EPI3D"]


def get_parser():
    parser = argparse.ArgumentParser(
        prog="benchmark script for patch denoising method."
        "this script execute a standard retinotopy pipeline with denoising for a set "
        "of subject, but with a single configuration of denoising"
    )

    parser.add_argument(
        "--dataset",
        default=DATA_DIR,
        help="location of a BIDS like dataset",
    )
    parser.add_argument(
        "--tmpdir",
        default=TMP_DIR,
        help="location for temp files",
    )
    parser.add_argument(
        "--sequence",
        help="fMRI sequence used",
        nargs="+",
        default=SEQUENCES,
    )
    parser.add_argument(
        "--denoise-str",
        type=str,
        nargs="+",
        help="denoiser config string",
    )
    parser.add_argument(
        "--build-code",
        type=str,
        nargs="+",
        help="build code for the preprocessing pipeline, eg `Rd`, `rD`, `rd` etc...",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="dry mode:  the workflow is not run, instead exec graph is produced",
    )
    parser.add_argument(
        "--sub",
        type=int,
        nargs="+",
        default=list(range(1, 7)),
        help="list of subject to process.",
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


def noise_prepocessing(ns):

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
    parser = get_parser()

    ns = parser.parse_args()

    if ns.noise:
        noise_preprocessing(ns)
    preprocessing(ns)


if __name__ == "__main__":
    main_cli()
