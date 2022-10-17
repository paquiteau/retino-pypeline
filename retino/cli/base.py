#!/usr/bin/env python3
"""Base configuration for the cli tools."""

import argparse


DATA_DIR = "/neurospin/optimed/pierre-antoine/dataset/data"
TMP_DIR = "/neurospin/optimed/pierre-antoine/dataset/tmp"
TASKS = ["Clockwise", "AntiClockwise"]
SEQUENCES = ["EPI3D"]


def base_parser(prog):
    """Base Parser configuration."""
    parser = argparse.ArgumentParser(prog=prog)

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
    return parser
